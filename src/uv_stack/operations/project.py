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
    NEWER_SCHEMA_HINT,
    NEWER_SCHEMA_MESSAGE,
    read_project_dependency_names,
    read_tracking,
    remove_tracking,
    validate_tracking_write,
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


def _adopt_orphans(
    pending: list[str],
    applied: list[str],
    stack_adds: list[str],
    dep_names: set[str],
    warnings: list[str],
) -> list[str]:
    """Adopt crashed-run orphans from an old ``pending`` record.

    An orphan is a pending entry whose name is present in the project's
    dependencies but absent from both the old ledger and the fresh stack:
    the interrupted run applied it and the stack no longer provides it.
    Adoption puts it into ``applied`` (durably, from the resuming run's
    FIRST write) so ownership is never lost silently; the warning tells the
    user what happens next. Names never applied (absent from dependencies)
    are dropped without action.

    :param pending: The crashed run's pending entries, verbatim.
    :param applied: The old ledger entries.
    :param stack_adds: The fresh ownership-filtered target list.
    :param dep_names: Canonical names currently in [project.dependencies].
    :param warnings: Warning list to append one message per orphan.
    :returns: The adopted entries, verbatim, in pending order.
    """
    applied_names = {
        canonical_name(n) for n in (ownership_name(e) for e in applied) if n
    }
    stack_names = {
        canonical_name(n) for n in (ownership_name(e) for e in stack_adds) if n
    }
    adopted: list[str] = []
    for entry in pending:
        name = ownership_name(entry)
        if name is None:
            # Name-less entries (editables, bare paths) bypass name-based
            # ownership entirely — nothing to adopt, but never be silent.
            warnings.append(
                f"'{entry}' from an interrupted run cannot be verified by "
                "name; check [project.dependencies] manually if it should "
                "not remain."
            )
            continue
        canon = canonical_name(name)
        if canon in applied_names or canon in stack_names or canon not in dep_names:
            continue
        adopted.append(entry)
        if "@" in entry or requirement_name(entry) is None:
            warnings.append(
                f"'{entry}' was applied by an interrupted run and is no longer "
                "in the stack; it will not be auto-removed — the next refresh "
                "will report it under 'Not auto-removed' for manual cleanup."
            )
        else:
            warnings.append(
                f"'{name}' was applied by an interrupted run and is no longer "
                "in the stack; the next refresh will remove it (delete it from "
                "[tool.uv-stack].applied to keep it)."
            )
    return adopted


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

    Tracking is two-phase: a ``pending`` intent record is written before
    ``uv add`` (after ``uv init`` on fresh projects, which must create
    ``pyproject.toml`` first) and the final ledger clears it after the add
    succeeds, so a crash at any point converges on retry (see
    :func:`refresh_project` for the failure regimes). With ``track`` false,
    any existing table is removed instead.

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
        try:
            interrupted = read_tracking(pyproject)
        except ConfigError as exc:
            # Tolerate corrupt/absent tracking here (the guard only decides
            # which hint to show) — but never mask the newer-schema
            # compatibility error.
            if str(exc).startswith(NEWER_SCHEMA_MESSAGE.split("{", 1)[0]):
                raise
            interrupted = None
        if interrupted is not None and interrupted.pending is not None:
            raise ConfigError(
                "pyproject.toml already exists.",
                hint=(
                    "An interrupted create left pending state; re-run with "
                    "--force to resume it."
                ),
            )
        raise ConfigError(
            "pyproject.toml already exists.",
            hint="Use --force to add to the existing project.",
        )

    # Ownership snapshot: names already in [project.dependencies] that a
    # PREVIOUS uv-stack create owned stay ours (force over a tracked
    # project); anything else pre-existing is user-owned and must never
    # enter the removal ledger.
    existing_tracking = read_tracking(pyproject) if pyproject.is_file() else None
    if existing_tracking is not None:
        if existing_tracking.version > 1:
            raise ConfigError(
                NEWER_SCHEMA_MESSAGE.format(version=existing_tracking.version),
                hint=NEWER_SCHEMA_HINT,
            )
    previous_applied = list(existing_tracking.applied) if existing_tracking else []
    previous_pending = list(existing_tracking.pending or []) if existing_tracking else []
    previously_owned: set[str] = set()
    for entry in [*previous_applied, *previous_pending]:
        owned_name = ownership_name(entry)
        if owned_name:
            previously_owned.add(canonical_name(owned_name))
    dep_names = {canonical_name(n) for n in read_project_dependency_names(pyproject)}
    user_owned = dep_names - previously_owned

    # Resolve the stack first so --strict token errors are not preceded by
    # external command execution or an unrelated EnvError.
    resolver = Resolver(config, strict=options.strict)
    stack = resolver.resolve(tokens)
    packages = resolver.flatten(stack)

    # Filter out user-owned dependencies from stack adds BEFORE rendering or writing.
    stack_adds = [
        p
        for p in packages
        if canonical_name(ownership_name(p) or "") not in user_owned
    ]
    # Build warnings for ownership-filtered entries.
    warnings = list(stack.warnings)
    for entry in packages:
        name = ownership_name(entry)
        if name and canonical_name(name) in user_owned:
            warnings.append(
                f"Skipping stack requirement '{entry}': '{name}' is user-owned in this project."
            )

    # Adopt orphans left by a crashed tracked init/refresh (spec §2.3):
    # durable from the FIRST write below.
    adopted = (
        _adopt_orphans(previous_pending, previous_applied, stack_adds, dep_names, warnings)
        if previous_pending
        else []
    )

    # Resume-only carry: when RESUMING a pending run (not a fresh --force
    # reset), carry forward previously-owned entries from applied that would
    # otherwise vanish silently (the orphan was adopted into applied in a
    # PREVIOUS retry that then crashed, so it is skipped by _adopt_orphans
    # above and not in stack_adds either). The carried entries ride to the next
    # refresh, whose dropped-diff removes or reports them loudly.
    carried = []
    if previous_pending:
        stack_names = {
            canonical_name(n) for n in (ownership_name(e) for e in stack_adds) if n
        }
        carried = [
            e
            for e in previous_applied
            if (n := ownership_name(e))
            and canonical_name(n) in dep_names
            and canonical_name(n) not in stack_names
        ]

    # One target, two table shapes derived from it (spec §2.2).
    pending_tracking = ProjectTracking(
        stack=list(tokens),
        python=options.python,
        applied=[*previous_applied, *adopted],
        pending=stack_adds,
    )
    final_tracking = ProjectTracking(
        stack=list(tokens),
        python=options.python,
        applied=[*stack_adds, *adopted, *carried],
        pending=None,
    )
    if options.track:
        # Pre-flight the FIRST write this run will attempt.
        validate_tracking_write(pyproject, pending_tracking)

    # Resolve the interpreter so a bad env name fails before any scaffolding
    # runs, and so both uv init and uv sync receive the same value.
    python = resolve_project_python(config, runner, options.python)

    # Opting out is authoritative: clear stale metadata BEFORE any fallible
    # uv step — but AFTER the interpreter probe, so a bad env spec cannot
    # destroy the ledger.
    if not options.track:
        remove_tracking(pyproject)

    # write_tracking on a missing pyproject would CREATE it and suppress
    # uv init, so fresh projects must initialize first (spec §2.2).
    fresh = not pyproject.is_file()
    fd, tmp_name = tempfile.mkstemp(prefix="uv-stack-stack.", suffix=".txt")
    tmp_req = Path(tmp_name)
    try:
        with open(fd, "w") as handle:
            for entry in stack_adds:
                handle.write(entry)
                handle.write("\n")

        if fresh:
            runner.run(_with_cwd(uv_init(python, options.name), cwd))
        if options.track:
            write_tracking(pyproject, pending_tracking)
        runner.run(_with_cwd(uv_add(tmp_req), cwd))
        if options.track:
            write_tracking(pyproject, final_tracking)
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

    Failure semantics: an intent record (``pending``) is written atomically
    before any uv mutation, so every crash cut-point converges on retry.
    ``uv remove``/``uv add`` failure leaves the old ``applied`` plus
    ``pending``; the retry's ownership union shields the intent and
    converges via presence-filtered removal + idempotent re-add. After a
    successful add the cleared ledger is written before sync, so a sync
    failure leaves deps and ledger consistent. Pending entries the stack no
    longer provides are adopted into ``applied`` with a warning and removed
    by the following refresh (direct references follow the loud
    skipped-removals path instead). If a run crashed before its add took
    effect and the same name was added manually before the retry, adoption
    cannot distinguish provenance (spec §2.3 accepted corner) — the warning
    names the escape hatch a full refresh before any removal.

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
            NEWER_SCHEMA_MESSAGE.format(version=tracking.version),
            hint=NEWER_SCHEMA_HINT,
        )

    resolver = Resolver(config, strict=options.strict)
    stack = resolver.resolve(tracking.stack)
    new_flat = resolver.flatten(stack)

    # Ownership: names owned by uv-stack — the old ledger PLUS any pending
    # intent record from an interrupted run, so a crash between uv add and
    # the clearing write can never surrender ownership to the user bucket.
    ledger_entries = [*tracking.applied, *(tracking.pending or [])]
    owned = {
        canonical_name(n)
        for n in (ownership_name(e) for e in ledger_entries)
        if n
    }
    # Names in [project.dependencies] NOT in the old ledger are user-owned.
    dep_names = {canonical_name(n) for n in read_project_dependency_names(pyproject)}
    user_owned = dep_names - owned

    # Filter out user-owned dependencies from stack adds BEFORE rendering or writing.
    stack_adds = [
        p
        for p in new_flat
        if canonical_name(ownership_name(p) or "") not in user_owned
    ]
    # Build warnings for ownership-filtered entries.
    warnings = list(stack.warnings)
    for entry in new_flat:
        name = ownership_name(entry)
        if name and canonical_name(name) in user_owned:
            warnings.append(
                f"Skipping stack requirement '{entry}': '{name}' is user-owned in this project."
            )

    adopted = (
        _adopt_orphans(tracking.pending, tracking.applied, stack_adds, dep_names, warnings)
        if tracking.pending
        else []
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

    # Build both tables once, pre-flight the PENDING one (it is the first write
    # attempted), and keep the dry-run path write-free.
    pending_tracking = ProjectTracking(
        version=1,
        stack=tracking.stack,
        python=options.python if options.python is not None else tracking.python,
        applied=[*tracking.applied, *adopted],
        pending=stack_adds,
    )
    final_tracking = ProjectTracking(
        version=1,
        stack=tracking.stack,
        python=options.python if options.python is not None else tracking.python,
        applied=[*stack_adds, *adopted],
        pending=None,
    )
    if not options.dry_run:
        validate_tracking_write(pyproject, pending_tracking)

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
    write_tracking(pyproject, pending_tracking)
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
        write_tracking(pyproject, final_tracking)
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
