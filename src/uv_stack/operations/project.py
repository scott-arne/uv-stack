"""The ``project init`` operation: scaffold a uv project from a stack.

The resolved stack is flattened (profiles expanded inline) so the project's
dependencies do not reference files under the config root. Dependencies are
written to a temp requirements file and added via ``uv add``.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from uv_stack.commands import micromamba_python_path, uv_add, uv_init, uv_remove, uv_sync
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, EnvError
from uv_stack.models import ProjectTracking
from uv_stack.operations.pyproject import (
    read_project_dependency_names,
    read_tracking,
    remove_tracking,
    write_tracking,
)
from uv_stack.parse import canonical_name, ownership_name, requirement_name
from uv_stack.resolver import Resolver
from uv_stack.runner import Command, Runner

#: Environment variable consulted between the ``--python`` flag and the
#: config-root default when selecting the project interpreter.
PROJECT_PYTHON_ENV = "UV_STACK_PROJECT_PYTHON"

#: Fallback interpreter spec when nothing else is configured.
DEFAULT_PROJECT_PYTHON = "3.12"

_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")
_IMPLEMENTATION_RE = re.compile(r"^(cpython|pypy|graalpy)[-@]")


@dataclass
class ProjectOptions:
    """Options for ``project init``.

    :param python: Interpreter spec — a Python version, a micromamba env name,
        or ``None`` to fall back to ``UV_STACK_PROJECT_PYTHON``, then the
        config-root default, then ``"3.12"``.
    :param name: Optional project name passed to ``uv init``.
    :param no_sync: Add dependencies but skip ``uv sync``.
    :param force: Allow adding to an existing ``pyproject.toml``.
    :param strict: Fail when a bare token falls through to a literal package.
    :param track: Record [tool.uv-stack] tracking metadata.
    """

    python: str | None = None
    name: str | None = None
    no_sync: bool = False
    force: bool = False
    strict: bool = False
    track: bool = True


def init_project(
    config: ConfigRoot,
    runner: Runner,
    tokens: list[str],
    options: ProjectOptions,
    *,
    cwd: Path,
) -> list[str]:
    """Initialize a uv project in ``cwd`` from resolved stack ``tokens``.

    Tracking metadata is recorded after ``uv add`` succeeds unless ``track`` is
    false, in which case any existing table is removed.

    :param config: Configuration root.
    :param runner: Command runner.
    :param tokens: Stack tokens (profiles, bundles, packages).
    :param options: Project options.
    :param cwd: Directory in which to create the project.
    :returns: Non-fatal resolution warnings for the CLI to print.
    :raises ConfigError: If a ``pyproject.toml`` exists and ``force`` is False.
    """
    pyproject = cwd / "pyproject.toml"
    if pyproject.is_file() and not options.force:
        raise ConfigError(
            "pyproject.toml already exists.",
            hint="Use --force to add to the existing project.",
        )

    # Ownership snapshot: names already in [project.dependencies] that a
    # PREVIOUS uv-stack create owned stay ours (force over a tracked
    # project); anything else pre-existing is user-owned and must never
    # enter the removal ledger.
    previously_owned: set[str] = set()
    existing_tracking = read_tracking(pyproject) if pyproject.is_file() else None
    if existing_tracking is not None:
        for entry in existing_tracking.applied:
            owned_name = ownership_name(entry)
            if owned_name:
                previously_owned.add(canonical_name(owned_name))
    user_owned = (
        {canonical_name(n) for n in read_project_dependency_names(pyproject)}
        - previously_owned
    )

    # Resolve the stack first so --strict token errors are not preceded by
    # external command execution or an unrelated EnvError.
    resolver = Resolver(config, strict=options.strict)
    stack = resolver.resolve(tokens)
    packages = resolver.flatten(stack)

    # Filter out user-owned dependencies from stack adds BEFORE rendering or writing.
    stack_adds = [
        p
        for p in packages
        if canonical_name(requirement_name(p) or "") not in user_owned
    ]
    # Build warnings for ownership-filtered entries.
    warnings = list(stack.warnings)
    for entry in packages:
        name = requirement_name(entry)
        if name and canonical_name(name) in user_owned:
            warnings.append(
                f"Skipping stack requirement '{entry}': '{name}' is user-owned in this project."
            )

    # Resolve the interpreter so a bad env name fails before any scaffolding
    # runs, and so both uv init and uv sync receive the same value.
    python = resolve_project_python(config, runner, options.python)

    if not options.track:
        # Opting out is authoritative: clear stale metadata from a previous
        # tracked create BEFORE any fallible uv step, so a failed add can
        # never leave an old ledger behind.
        remove_tracking(pyproject)

    fd, tmp_name = tempfile.mkstemp(prefix="uv-stack-stack.", suffix=".txt")
    tmp_req = Path(tmp_name)
    try:
        with open(fd, "w") as handle:
            for entry in stack_adds:
                handle.write(entry)
                handle.write("\n")

        if not pyproject.is_file():
            runner.run(_with_cwd(uv_init(python, options.name), cwd))
        runner.run(_with_cwd(uv_add(tmp_req), cwd))
        if options.track:
            write_tracking(
                pyproject,
                ProjectTracking(
                    stack=list(tokens),
                    python=options.python,
                    applied=stack_adds,
                ),
            )
        if not options.no_sync:
            runner.run(_with_cwd(uv_sync(python), cwd))
    finally:
        if tmp_req.exists():
            tmp_req.unlink()

    return warnings


def select_project_python(config: ConfigRoot, flag: str | None) -> str:
    """Resolve the interpreter spec by precedence (no env-name resolution yet).

    Precedence: the ``--python`` flag, then ``UV_STACK_PROJECT_PYTHON``, then the
    config-root default, then :data:`DEFAULT_PROJECT_PYTHON`. The returned value
    may be a version, a path, or a micromamba env name; resolving an env name to
    an interpreter path is :func:`resolve_project_python`'s job.

    :param config: Configuration root (for the file-backed default).
    :param flag: The raw ``--python`` value, or ``None`` when unset.
    :returns: The selected interpreter spec.
    """
    return (
        flag
        or os.environ.get(PROJECT_PYTHON_ENV)
        or config.default_project_python()
        or DEFAULT_PROJECT_PYTHON
    )


def _is_python_passthrough(spec: str) -> bool:
    """Return whether ``spec`` is a uv interpreter spec rather than an env name.

    Versions, explicit paths, and uv implementation forms pass straight through
    to uv; anything else is treated as a micromamba environment name.

    :param spec: The interpreter spec to classify.
    """
    if "/" in spec or "\\" in spec:
        return True
    if "@" in spec:
        return True
    if _VERSION_RE.match(spec):
        return True
    return bool(_IMPLEMENTATION_RE.match(spec))


def resolve_project_python(
    config: ConfigRoot, runner: Runner, flag: str | None
) -> str:
    """Select and fully resolve the project interpreter.

    Applies :func:`select_project_python`, then resolves a micromamba env name to
    that env's interpreter path (versions and paths are returned unchanged).

    :param config: Configuration root.
    :param runner: Command runner used to probe the micromamba env.
    :param flag: The raw ``--python`` value, or ``None`` when unset.
    :returns: A version, path, or resolved interpreter path suitable for uv.
    :raises EnvError: If the spec names a micromamba env that cannot be probed.
    """
    spec = select_project_python(config, flag)
    if _is_python_passthrough(spec):
        return spec

    result = runner.run(micromamba_python_path(spec), capture=True, check=False)
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        raise EnvError(
            f"Could not resolve micromamba environment '{spec}' to an interpreter.",
            hint=(
                f"Ensure the env exists ('stack create env {spec}') and that "
                "MAMBA_ROOT_PREFIX is set, or pass --python <version>."
            ),
        )
    return path


def _with_cwd(command: Command, cwd: Path) -> Command:
    return Command(command.args, cwd=cwd)


#: Placeholder interpreter in dry-run plans when probing would be required.
_DRY_RUN_PROJECT_PYTHON = "<project-python>"


@dataclass
class RefreshOptions:
    """Options for ``stack refresh``.

    :param python: Override (and record) the project interpreter spec.
    :param strict: Fail when a bare token falls through to a literal package.
    :param no_sync: Apply dependency changes but skip the final ``uv sync``.
    :param dry_run: Plan only — no probe, no uv execution, no writes.
    """

    python: str | None = None
    strict: bool = False
    no_sync: bool = False
    dry_run: bool = False


@dataclass
class RefreshResult:
    """Outcome of :func:`refresh_project`.

    :param warnings: Non-fatal resolution advisories.
    :param added: Flattened requirements new to the applied ledger.
    :param removed: Dropped ledger entries eligible for removal.
    :param skipped_removals: Dropped entries never auto-removed (editables,
        paths, direct references).
    :param planned: The command plan (dry runs only).
    """

    warnings: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped_removals: list[str] = field(default_factory=list)
    planned: list[Command] = field(default_factory=list)


def refresh_project(
    config: ConfigRoot,
    runner: Runner,
    options: RefreshOptions,
    *,
    cwd: Path,
) -> RefreshResult:
    """Re-resolve a tracked project against the current profiles/bundles.

    Refresh is deliberately NOT transactional: uv mutates ``pyproject.toml``
    before the ledger write, so a mid-sequence failure leaves the old ledger
    in place and the next refresh converges — removals are presence-filtered
    and re-adding the full flat list is idempotent.

    :param config: Configuration root.
    :param runner: Command runner.
    :param options: Refresh options.
    :param cwd: The project directory.
    :returns: A :class:`RefreshResult`.
    :raises ConfigError: When no tracked project is present, the schema is
        newer than this uv-stack, or the pyproject cannot be written.
    """
    pyproject = cwd / "pyproject.toml"
    tracking = read_tracking(pyproject)
    if tracking is None:
        raise ConfigError(
            "No tracked project here.",
            hint=(
                "Run inside a project created by 'stack create project' (or "
                "add a [tool.uv-stack] table with 'stack' tokens)."
            ),
        )
    if tracking.version > 1:
        raise ConfigError(
            f"This project was tracked by a newer uv-stack (schema {tracking.version}).",
            hint="Upgrade uv-stack, or edit [tool.uv-stack] manually.",
        )

    resolver = Resolver(config, strict=options.strict)
    stack = resolver.resolve(tracking.stack)
    new_flat = resolver.flatten(stack)

    # Ownership: names currently owned by uv-stack (from the old ledger).
    owned = {
        canonical_name(n)
        for n in (ownership_name(e) for e in tracking.applied)
        if n
    }
    # Names in [project.dependencies] NOT in the old ledger are user-owned.
    user_owned = (
        {canonical_name(n) for n in read_project_dependency_names(pyproject)} - owned
    )

    # Filter out user-owned dependencies from stack adds BEFORE rendering or writing.
    stack_adds = [
        p
        for p in new_flat
        if canonical_name(requirement_name(p) or "") not in user_owned
    ]
    # Build warnings for ownership-filtered entries.
    warnings = list(stack.warnings)
    for entry in new_flat:
        name = requirement_name(entry)
        if name and canonical_name(name) in user_owned:
            warnings.append(
                f"Skipping stack requirement '{entry}': '{name}' is user-owned in this project."
            )

    dropped = [entry for entry in tracking.applied if entry not in stack_adds]
    removable_names: list[str] = []
    skipped: list[str] = []
    removed: list[str] = []
    for entry in dropped:
        # Direct references (name @ url) contain '@' — skip before calling requirement_name.
        if "@" in entry:
            skipped.append(entry)
        else:
            name = requirement_name(entry)
            if name is None:
                skipped.append(entry)
            else:
                removable_names.append(name)
                removed.append(entry)
    current_names = {canonical_name(n) for n in read_project_dependency_names(pyproject)}
    names = [n for n in dict.fromkeys(removable_names) if canonical_name(n) in current_names]
    added = [entry for entry in stack_adds if entry not in tracking.applied]

    spec_flag = options.python if options.python is not None else tracking.python

    if options.dry_run:
        selected = select_project_python(config, spec_flag)
        shown = selected if _is_python_passthrough(selected) else _DRY_RUN_PROJECT_PYTHON
        planned: list[Command] = []
        if names:
            planned.append(uv_remove(names))
        planned.append(uv_add(Path("<stack-requirements>")))
        if not options.no_sync:
            planned.append(uv_sync(shown))
        return RefreshResult(
            warnings=warnings,
            added=added,
            removed=removed,
            skipped_removals=skipped,
            planned=planned,
        )

    python = resolve_project_python(config, runner, spec_flag)
    fd, tmp_name = tempfile.mkstemp(prefix="uv-stack-refresh.", suffix=".txt")
    tmp_req = Path(tmp_name)
    try:
        # Render only stack-owned dependencies to the temp requirements file.
        with open(fd, "w") as handle:
            for entry in stack_adds:
                handle.write(entry)
                handle.write("\n")
        if names:
            runner.run(_with_cwd(uv_remove(names), cwd))
        runner.run(_with_cwd(uv_add(tmp_req), cwd))
        write_tracking(
            pyproject,
            ProjectTracking(
                version=1,
                stack=tracking.stack,
                python=options.python if options.python is not None else tracking.python,
                applied=stack_adds,
            ),
        )
        if not options.no_sync:
            runner.run(_with_cwd(uv_sync(python), cwd))
    finally:
        if tmp_req.exists():
            tmp_req.unlink()
    return RefreshResult(
        warnings=warnings,
        added=added,
        removed=removed,
        skipped_removals=skipped,
    )
