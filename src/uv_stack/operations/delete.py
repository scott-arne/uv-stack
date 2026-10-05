"""The ``stack delete`` operations: remove a profile, bundle, env, or project.

The inverse of :mod:`uv_stack.operations.scaffold` and
:mod:`uv_stack.operations.project`: each function takes the lock its creating
counterpart takes, so a delete cannot interleave with a create of the same
name.

Each takes an optional ``confirm`` callback, invoked with the result the call
would return — after every check has passed and the lock is held, and before
anything irreversible. It is how the CLI asks the question only when there is
something to delete, with the forced-delete warnings already in view. A callback
that answers ``False`` ends the call with nothing done and ``None`` returned.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from uv_stack.commands import micromamba_remove, uv_remove, uv_sync
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.fsutil import name_lock, require_regular_file
from uv_stack.hints import escape_controls, render_positional_arg
from uv_stack.operations.create import env_micromamba_exists
from uv_stack.operations.project import _project_lock, _with_cwd, require_tracking
from uv_stack.operations.pyproject import read_project_dependency_names, remove_tracking
from uv_stack.operations.scaffold import validate_name
from uv_stack.parse import canonical_name, read_clean_lines, requirement_name
from uv_stack.runner import Runner

_FORCE_HINT = (
    "Remove those references first, or pass --force to delete it anyway. A "
    "bare token then resolves to the pip package of the same name, and a "
    "qualified one fails to resolve."
)


@dataclass
class DeleteResult:
    """Outcome of deleting a profile or bundle.

    :param path: The file, under the root as given.
    :param warnings: Non-fatal advisories for the CLI to print.
    """

    path: Path
    warnings: list[str] = field(default_factory=list)


@dataclass
class EnvDeleteResult:
    """Outcome of deleting a shared environment.

    :param directory: ``envs/<name>``, under the root as given.
    :param removed_micromamba: Whether a built micromamba environment was
        there to remove.
    """

    directory: Path
    removed_micromamba: bool


@dataclass
class WithdrawResult:
    """Outcome of :func:`withdraw_project`.

    :param removed: Distribution names handed to ``uv remove``.
    :param skipped_removals: Ledger entries left in ``[project.dependencies]``
        because they cannot be removed by name (direct references, editables,
        paths). The CLI prints :data:`~uv_stack.operations.project.SKIPPED_REMOVAL_NOTICE`
        for each.
    """

    removed: list[str]
    skipped_removals: list[str]


@dataclass(frozen=True)
class Reference:
    """One token in the root that names a profile or bundle.

    :param location: The holding file, relative to the root.
    :param token: The token as written.
    :param bare: Whether the token is unqualified. A bare token survives the
        deletion with a new meaning; a qualified one stops resolving.
    """

    location: str
    token: str
    bare: bool


@dataclass(frozen=True)
class References:
    """What the reference scan found.

    :param found: Tokens that name the resource.
    :param unreadable: Sources the scan could not read, with the reason. Their
        tokens are unknown, so the scan is inconclusive while any is listed.
    """

    found: list[Reference]
    unreadable: list[str]


def _names_same_file(config: ConfigRoot, kind: str, written: str, name: str) -> bool:
    """Whether the name a token carries reaches the file ``name`` does.

    Spelling is not enough either way: on a case-folding filesystem ``DS``
    reaches ``ds.yaml``, and on a case-sensitive one ``DS.yaml`` and ``ds.yaml``
    are two files. Identity settles both.
    """
    if written == name:
        return True
    if written.casefold() != name.casefold():
        return False
    path_of = config.profile_path if kind == "profile" else config.bundle_path
    try:
        return os.path.samefile(path_of(written), path_of(name))
    except OSError:
        return False


def _token_reference(
    config: ConfigRoot, kind: str, name: str, token: str
) -> tuple[str, bool] | None:
    """Return ``(token, bare)`` when ``token`` names the ``kind`` called ``name``.

    A bare token names a profile when a profile file answers to it, and a
    bundle only when no profile does, which is the resolver's own order.
    """
    token = token.strip()
    if not token:
        return None
    if kind == "profile":
        if token.startswith("profile:"):
            written, bare = token[len("profile:") :], False
        elif token.startswith(("@", "bundle:", "package:", "pkg:")):
            return None
        else:
            written, bare = token, True
    else:
        if token.startswith("@"):
            written, bare = token[1:], False
        elif token.startswith("bundle:"):
            written, bare = token[len("bundle:") :], False
        elif token.startswith(("profile:", "package:", "pkg:")):
            return None
        elif config.profile_exists(token):
            return None
        else:
            written, bare = token, True
    if _names_same_file(config, kind, written.strip(), name):
        return token, bare
    return None


def references_to(config: ConfigRoot, kind: str, name: str) -> References:
    """Find every direct reference to a profile or bundle in the root.

    Direct means written in an env's ``stack.txt`` or a bundle's ``includes``;
    an env that reaches a profile through a bundle is not listed, because the
    bundle is what would need editing. A bundle is never a reference to itself.

    :param config: Configuration root.
    :param kind: ``"profile"`` or ``"bundle"``.
    :param name: The resource name.
    :returns: The references, and the sources that could not be read.
    """
    found: list[Reference] = []
    unreadable: list[str] = []

    def _record(location: str, tokens: list[str]) -> None:
        for token in tokens:
            match = _token_reference(config, kind, name, token)
            if match is not None:
                found.append(Reference(location, match[0], match[1]))

    try:
        envs = config.list_envs()
    except OSError as error:
        envs = []
        unreadable.append(f"envs/: {error}")
    for env in envs:
        location = f"envs/{env}/stack.txt"
        try:
            _record(location, read_clean_lines(config.env_stack_path(env)))
        except (UvStackError, OSError) as error:
            unreadable.append(f"{location}: {_reason(error)}")
    try:
        bundles = config.list_bundles()
    except OSError as error:
        bundles = []
        unreadable.append(f"bundles/: {error}")
    for bundle in bundles:
        if kind == "bundle" and bundle == name:
            continue
        location = f"bundles/{bundle}.yaml"
        try:
            _record(location, config.load_bundle(bundle).includes)
        except (UvStackError, OSError, UnicodeDecodeError) as error:
            unreadable.append(f"{location}: {_reason(error)}")
    return References(found, unreadable)


def _reason(error: Exception) -> str:
    return error.message if isinstance(error, UvStackError) else str(error)


def _refuse_or_warn(kind: str, name: str, references: References, *, force: bool) -> list[str]:
    """Turn the scan into a refusal, or into the warnings a forced delete prints.

    :returns: The warnings; empty when nothing refers to the resource.
    :raises ConfigError: When something does, or might, and ``force`` is off.
    """
    if not references.found and not references.unreadable:
        return []
    shown = escape_controls(name)
    if not force:
        lines = [f"{kind.capitalize()} '{shown}' is still referenced:"]
        lines += [
            f"  {escape_controls(ref.location)}: {escape_controls(ref.token)}"
            for ref in references.found
        ]
        if references.unreadable:
            lines.append("These could not be checked for references:")
            lines += [f"  {escape_controls(entry)}" for entry in references.unreadable]
        raise ConfigError("\n".join(lines), hint=_FORCE_HINT)
    warnings: list[str] = []
    for ref in references.found:
        where = f"{escape_controls(ref.location)}: '{escape_controls(ref.token)}'"
        if ref.bare:
            warnings.append(
                f"{where} now resolves to the pip package '{escape_controls(ref.token)}'."
            )
        else:
            warnings.append(f"{where} no longer resolves; the {kind} is gone.")
    for entry in references.unreadable:
        warnings.append(f"Not checked for references: {escape_controls(entry)}")
    return warnings


def _delete_file(
    config: ConfigRoot,
    kind: str,
    name: str,
    *,
    force: bool,
    confirm: Callable[[DeleteResult], bool] | None,
) -> DeleteResult | None:
    """Delete a profile or bundle file under the stem lock its create takes."""
    validate_name(kind, name)
    path = config.profile_path(name) if kind == "profile" else config.bundle_path(name)
    other = "bundle" if kind == "profile" else "profile"
    # Messages and the returned path name the root as given; the lock, the
    # checks and the unlink all go through one resolution of it.
    locked = config.resolved()
    with name_lock(locked.stem_lock_path(name), name, action="deleting"):
        target = locked.profile_path(name) if kind == "profile" else locked.bundle_path(name)
        require_regular_file(target)
        if not target.is_file():
            other_exists = (
                locked.bundle_exists(name) if kind == "profile" else locked.profile_exists(name)
            )
            hint = (
                # The hint is a command the user is meant to paste.
                f"There is a {other} of that name: stack delete {other} "
                f"{render_positional_arg(name)}"
                if other_exists
                else "Check the name."
            )
            raise ConfigError(f"Missing {kind}: {path}", hint=hint)
        warnings = _refuse_or_warn(kind, name, references_to(locked, kind, name), force=force)
        result = DeleteResult(path=path, warnings=warnings)
        if confirm is not None and not confirm(result):
            return None
        target.unlink()
    return result


def delete_profile(
    config: ConfigRoot,
    name: str,
    *,
    force: bool = False,
    confirm: Callable[[DeleteResult], bool] | None = None,
) -> DeleteResult | None:
    """Delete ``profiles/<name>.yaml``.

    :param config: Configuration root.
    :param name: Profile name (file stem).
    :param force: Delete even when something in the root still refers to the
        profile, or when a source could not be checked.
    :param confirm: Asked last, with the result below; see the module docstring.
    :returns: What was deleted, with one warning per reference left behind, or
        ``None`` when ``confirm`` declined.
    :raises ConfigError: If ``name`` is not a valid profile name, the profile
        does not exist, the stem lock cannot be used (see
        :func:`uv_stack.fsutil.name_lock`), or — without ``force`` — the root
        still refers to it or has a source the scan could not read.
    :raises OSError: If the lock file cannot be opened for a reason ``name_lock``
        neither refuses nor degrades to no locking on; its docstring has the rule.
    """
    return _delete_file(config, "profile", name, force=force, confirm=confirm)


def delete_bundle(
    config: ConfigRoot,
    name: str,
    *,
    force: bool = False,
    confirm: Callable[[DeleteResult], bool] | None = None,
) -> DeleteResult | None:
    """Delete ``bundles/<name>.yaml``.

    :param config: Configuration root.
    :param name: Bundle name (file stem).
    :param force: Delete even when something in the root still refers to the
        bundle, or when a source could not be checked.
    :param confirm: Asked last, with the result below; see the module docstring.
    :returns: What was deleted, with one warning per reference left behind, or
        ``None`` when ``confirm`` declined.
    :raises ConfigError: As :func:`delete_profile`, for a bundle.
    :raises OSError: As :func:`delete_profile`.
    """
    return _delete_file(config, "bundle", name, force=force, confirm=confirm)


def delete_env(
    config: ConfigRoot,
    runner: Runner,
    name: str,
    *,
    confirm: Callable[[EnvDeleteResult], bool] | None = None,
) -> EnvDeleteResult | None:
    """Delete environment ``name``: its micromamba environment, then ``envs/<name>/``.

    The micromamba environment goes first, and a failure there leaves the
    sources in place: with them, a retry is the same command again, whereas
    sources removed ahead of a failed ``micromamba remove`` would leave an
    environment nothing in the root remembers.

    :param config: Configuration root.
    :param runner: Command runner.
    :param name: Environment name.
    :param confirm: Asked once the micromamba environment has been probed, so
        the question can say whether one is there; see the module docstring.
    :returns: What was removed, or ``None`` when ``confirm`` declined.
    :raises ConfigError: If ``name`` is not a valid environment name, the
        environment has no ``stack.txt`` (so uv-stack does not manage it),
        ``envs/<name>`` is a symbolic link, or the per-name lock cannot be used
        (see :func:`uv_stack.fsutil.name_lock`).
    :raises ToolError: If ``micromamba`` cannot be started or its remove exits
        non-zero. The sources are untouched in either case.
    :raises OSError: If the lock file cannot be opened for a reason ``name_lock``
        neither refuses nor degrades to no locking on, or if the directory
        cannot be removed.
    """
    validate_name("environment", name)
    directory = config.env_dir(name)
    locked = config.resolved()
    with name_lock(locked.env_lock_path(name), name, action="deleting"):
        locked.require_env(name)
        target = locked.env_dir(name)
        if target.is_symlink():
            # rmtree refuses a link anyway; refusing first says what to do, and
            # keeps the micromamba environment from being removed ahead of a
            # directory removal that cannot succeed.
            raise ConfigError(
                f"Environment directory {directory} is a symbolic link.",
                hint="Remove the link, or the directory it points to, by hand.",
            )
        result = EnvDeleteResult(
            directory=directory,
            removed_micromamba=env_micromamba_exists(locked, runner, name),
        )
        if confirm is not None and not confirm(result):
            return None
        if result.removed_micromamba:
            runner.run(micromamba_remove(name))
        shutil.rmtree(target)
    return result


def withdraw_project(
    config: ConfigRoot,
    runner: Runner,
    *,
    cwd: Path,
    no_sync: bool = False,
    confirm: Callable[[WithdrawResult], bool] | None = None,
) -> WithdrawResult | None:
    """Withdraw uv-stack from the tracked project in ``cwd``.

    Removes the packages uv-stack applied and drops the ``[tool.uv-stack]``
    table, leaving a plain uv project with only the dependencies the user added
    themselves. The removal set follows the rule ``stack refresh`` enforces: an
    entry of the ``applied`` ledger, or of a leftover ``pending`` record, whose
    name is still in ``[project.dependencies]``. Nothing else is named to uv.

    Failure semantics: ``uv remove`` runs before anything is written, so its
    failure leaves the table intact and a retry is the same command again. A
    failure after it leaves a table whose entries are no longer present, which
    a retry simply drops, and which ``stack refresh`` would re-apply.

    The final ``uv sync`` is plain — not pinned to the recorded interpreter —
    because once the table is gone the interpreter is no longer uv-stack's to
    choose, and uv validates the existing ``.venv`` itself.

    :param config: Configuration root, for the project lock.
    :param runner: Command runner.
    :param cwd: The project directory.
    :param no_sync: Remove the dependencies but skip the final ``uv sync``.
    :param confirm: Asked with the removal set worked out, so the question can
        list it; see the module docstring.
    :returns: What was removed and what was left for the user, or ``None`` when
        ``confirm`` declined.
    :raises ConfigError: When no tracked project is present, the schema is
        newer than this uv-stack, the pyproject cannot be rewritten, or the
        project lock cannot be used (see :func:`uv_stack.fsutil.name_lock`).
    :raises ToolError: When ``uv remove`` or ``uv sync`` fails.
    :raises OSError: If the lock file cannot be opened for a reason ``name_lock``
        neither refuses nor degrades to no locking on; its docstring has the rule.
    """
    pyproject = cwd / "pyproject.toml"
    with _project_lock(config, cwd):
        tracking = require_tracking(pyproject, cwd)
        names: list[str] = []
        skipped: list[str] = []
        for entry in [*tracking.applied, *(tracking.pending or [])]:
            # Direct references (name @ url) contain '@'; refresh skips them
            # before requirement_name for the same reason.
            name = None if "@" in entry else requirement_name(entry)
            if name is None:
                skipped.append(entry)
            else:
                names.append(name)
        present = {canonical_name(n) for n in read_project_dependency_names(pyproject)}
        removable = [n for n in dict.fromkeys(names) if canonical_name(n) in present]
        result = WithdrawResult(removed=removable, skipped_removals=list(dict.fromkeys(skipped)))
        if confirm is not None and not confirm(result):
            return None
        if removable:
            runner.run(_with_cwd(uv_remove(removable), cwd))
        remove_tracking(pyproject)
        if not no_sync:
            runner.run(_with_cwd(uv_sync(), cwd))
    return result
