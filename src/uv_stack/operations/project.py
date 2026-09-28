"""The ``project init`` operation: scaffold a uv project from a stack.

The resolved stack is flattened (profiles expanded inline) so the project's
dependencies do not reference files under the config root. Dependencies are
written to a temp requirements file and added via ``uv add``.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from uv_stack.commands import micromamba_python_path, uv_add, uv_init, uv_remove, uv_sync
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, EnvError, NewerSchemaError, UvStackError
from uv_stack.fsutil import name_lock
from uv_stack.hints import render_positional_arg
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
from uv_stack.pyversion import (
    PLAIN_VERSION_RE,
    is_near_miss_version,
    near_miss_version_notice,
)
from uv_stack.resolver import Resolver
from uv_stack.runner import Command, Runner
from uv_stack.variables import expand_all

#: Environment variable consulted between the ``--python`` flag and the
#: config-root default when selecting the project interpreter.
PROJECT_PYTHON_ENV = "UV_STACK_PROJECT_PYTHON"

#: Fallback interpreter spec when nothing else is configured.
DEFAULT_PROJECT_PYTHON = "3.12"

#: Wording for a ledger entry uv-stack drops but will not remove from
#: [project.dependencies] itself. The success path prints it from the CLI and
#: the failure path attaches it to the raised error, so the user reads the same
#: sentence either way; operations may not import from ``cli``, so the single
#: source lives here and ``cli/refresh_cmd.py`` imports it.
SKIPPED_REMOVAL_NOTICE = "Not auto-removed (edit pyproject.toml manually): {entry}"

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
    :raises ConfigError: When another stack process holds this project's lock
        past the timeout, or when what stands at the lock path is something
        :func:`uv_stack.fsutil.name_lock` refuses; its docstring has the rule.
    :raises OSError: If the lock file cannot be opened for a reason ``name_lock``
        neither refuses nor degrades to no locking on; its docstring has the rule.
    """
    # The guard below is inside the lock too: two fresh creates would both
    # pass it, and the second would then write over the first's project as
    # if --force had been given.
    with _project_lock(config, cwd):
        return _init_project(config, runner, tokens, options, cwd=cwd)


def _init_project(
    config: ConfigRoot,
    runner: Runner,
    tokens: list[str],
    options: ProjectOptions,
    *,
    cwd: Path,
) -> list[str]:
    """The body of :func:`init_project`, run under the project lock."""
    pyproject = cwd / "pyproject.toml"
    if pyproject.is_file() and not options.force:
        try:
            interrupted = read_tracking(pyproject)
        except NewerSchemaError:
            # Never mask a forward-compatibility refusal.
            raise
        except ConfigError:
            # Tolerate corrupt/absent tracking here: the guard only decides
            # which hint to show.
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
            raise NewerSchemaError(
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

    # Expansion is computed here — after the ownership filter, before the
    # first write and before any subprocess — so an undefined variable aborts
    # a run that has changed nothing. This follows the rule already in this
    # function that resolution errors precede external command execution.
    # Only the temp requirements file below gets the expanded strings; the
    # ledger keeps the unexpanded ones, so it never records this machine's
    # filesystem. expand_all also enforces the placement rule, which is what
    # makes that split safe (spec "Projects and the ledger split").
    expanded_adds = expand_all(stack_adds, config.load_variables())

    # Only when the spec is actually recorded: --no-track removes the table
    # below, so the advisory would name a trip nothing is taking.
    travel_spec = options.python if options.track else None

    # Adopt orphans left by a crashed tracked init/refresh (spec §2.3):
    # durable from the FIRST write below. Skipped entirely with --no-track:
    # the table is deleted below, so there is no ledger to adopt into and the
    # warnings would point at a 'stack refresh' that can no longer run.
    adopted = (
        _adopt_orphans(previous_pending, previous_applied, stack_adds, dep_names, warnings)
        if options.track and previous_pending
        else []
    )

    # Resume-only carry: when RESUMING a pending run (not a fresh --force
    # reset), carry forward previously-owned entries from applied that would
    # otherwise vanish silently (the orphan was adopted into applied in a
    # PREVIOUS retry that then crashed, so it is skipped by _adopt_orphans
    # above and not in stack_adds either). The carried entries ride to the next
    # refresh, whose dropped-diff removes or reports them loudly.
    # A reference-bearing entry (editable, VCS, path) has no name, so
    # [project.dependencies] cannot be consulted to confirm it is still
    # installed and its exact unexpanded spelling is the only identity it has.
    # It is therefore carried unless the stack still supplies that same
    # spelling. That is the recoverable side of the guess: a carried entry the
    # user no longer wants is reported by the next refresh, whereas one
    # dropped from the ledger becomes user-owned and is never mentioned again.
    carried: list[str] = []
    if options.track and previous_pending:
        stack_names = {
            canonical_name(n) for n in (ownership_name(e) for e in stack_adds) if n
        }
        for entry in previous_applied:
            owned = ownership_name(entry)
            if owned is None:
                if entry not in stack_adds:
                    carried.append(entry)
                continue
            canon = canonical_name(owned)
            if canon in dep_names and canon not in stack_names:
                carried.append(entry)

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

    # Classified here, alongside the interpreter probe and above every write:
    # the classifier reads the config root and can raise, and the delivery
    # points below cannot afford to. Only the tense is left to them. This is
    # the last point where a failure costs nothing.
    travel = _travel_advisory(config, travel_spec)

    # Opting out is authoritative once the run commits to touching this
    # project, so this precedes every fallible uv step — but it deliberately
    # FOLLOWS the interpreter probe. A probe failure means nothing has been
    # created or mutated yet and the command aborts whole; a typo in --python
    # must not destroy tracking metadata as a side effect. The boundary is
    # pinned from both sides: test_init_project_no_track_probe_failure_
    # preserves_ledger (probe fails -> table survives) and
    # test_init_project_no_track_removes_table_even_when_add_fails (the run
    # started -> table goes).
    if not options.track:
        remove_tracking(pyproject)

    # write_tracking on a missing pyproject would CREATE it and suppress
    # uv init, so fresh projects must initialize first (spec §2.2).
    fresh = not pyproject.is_file()
    fd, tmp_name = tempfile.mkstemp(prefix="uv-stack-stack.", suffix=".txt")
    tmp_req = Path(tmp_name)
    recorded = False
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            for entry in expanded_adds:
                handle.write(entry)
                handle.write("\n")

        if fresh:
            runner.run(_with_cwd(uv_init(python, options.name), cwd))
        if options.track:
            write_tracking(pyproject, pending_tracking)
            recorded = True
        runner.run(_with_cwd(uv_add(tmp_req), cwd))
        if options.track:
            write_tracking(pyproject, final_tracking)
        if not options.no_sync:
            runner.run(_with_cwd(uv_sync(python), cwd))
    except UvStackError as error:
        # These advisories ride on the returned list, which a raised error
        # never produces. Hand them to the error instead: a failure past the
        # pending write leaves the spec on disk, so the caveat outlives the
        # run that raised.
        error.resolution_warnings = [
            *warnings,
            *_travel_notices(travel, recorded=recorded),
        ]
        raise
    finally:
        if tmp_req.exists():
            tmp_req.unlink()

    return [*warnings, *_travel_notices(travel, recorded=recorded)]


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
    if PLAIN_VERSION_RE.match(spec):
        return True
    return bool(_IMPLEMENTATION_RE.match(spec))


#: Why a recorded interpreter spec will not travel, keyed by
#: :func:`python_travel_problem`'s return value. The opening that precedes it
#: is chosen at delivery, because the same problem is worth saying in two
#: tenses.
PYTHON_TRAVEL_REASON = {
    "path": (
        "a filesystem path does not travel to another machine. A version, a uv "
        "implementation form, or an environment name this root declares does."
    ),
    "undeclared-env": (
        "this config root declares no environment by that name, so the value "
        "will not resolve on a machine that clones it."
    ),
}

#: 'Recording' is only true once the table holding the spec is on disk. Every
#: other delivery point -- a dry run, a failure above the write -- is a
#: prediction, and saying it in the present tense is a false claim about a
#: durable record.
PYTHON_TRAVEL_RECORDED = "Recording interpreter '{spec}' in pyproject.toml: "
PYTHON_TRAVEL_PROSPECTIVE = "Would record interpreter '{spec}' in pyproject.toml: "


def python_travel_problem(config: ConfigRoot, spec: str) -> str | None:
    """Classify why a recorded interpreter spec will not travel, if it will not.

    ``ProjectTracking.python`` records the caller's ``--python`` verbatim, so a
    path or an environment name only this machine has becomes part of a
    committed ``pyproject.toml``. Rewriting or refusing the value is out of
    scope — the tracking contract is settled — but saying so is not.

    The path case is tested independently of :func:`_is_python_passthrough`
    rather than through it, because a path IS passthrough: it reaches uv
    unchanged and works perfectly here. It simply does not travel. Reusing the
    predicate for the environment case (rather than restating it) is what keeps
    uv implementation forms such as ``cpython@3.12`` from being mistaken for
    environment names.

    :param config: Configuration root, for the declared environment names.
    :param spec: The interpreter spec as it would be recorded.
    :returns: ``"path"``, ``"undeclared-env"``, or ``None`` when it travels.
    """
    if "/" in spec or "\\" in spec:
        return "path"
    if not _is_python_passthrough(spec) and spec not in config.list_envs():
        return "undeclared-env"
    return None


@dataclass(frozen=True)
class TravelAdvisory:
    """Both tenses of one travel advisory, rendered before any mutation.

    The classification behind the wording reads the config root, which can
    fail; choosing between two ready strings cannot. Carrying both tenses is
    what lets the delivery points stay infallible while still saying the true
    one (see :func:`_travel_advisory`).

    :param prospective: The wording for a delivery above the tracking write.
    :param recorded: The wording for a delivery below it.
    """

    prospective: str
    recorded: str


def _travel_advisory(config: ConfigRoot, spec: str | None) -> TravelAdvisory | None:
    """Classify ``spec``'s travel problem and render both tenses of it.

    Called ONCE per operation, above every write and every file created,
    because :func:`python_travel_problem` consults
    :meth:`~uv_stack.config.ConfigRoot.list_envs` and an unreadable envs
    directory makes that raise. Classifying at each delivery point instead put
    that OSError on the success returns, where it reported a completed run as
    failed, and inside the error handlers, where it replaced the
    :class:`~uv_stack.errors.UvStackError` actually being reported.

    :param config: Configuration root, for the declared environment names.
    :param spec: The spec about to be recorded, or ``None`` when unset.
    :returns: The advisory in both tenses, or ``None`` when the spec travels.
    :raises OSError: If the declared environments cannot be listed. Left to
        propagate here, above the scaffolding, where the run has changed
        nothing yet. ``init_project`` calls this below the interpreter probe,
        so one read-only micromamba command can precede it.
    """
    if spec is None:
        return None
    problem = python_travel_problem(config, spec)
    if problem is None:
        return None
    reason = PYTHON_TRAVEL_REASON[problem]
    return TravelAdvisory(
        prospective=PYTHON_TRAVEL_PROSPECTIVE.format(spec=spec) + reason,
        recorded=PYTHON_TRAVEL_RECORDED.format(spec=spec) + reason,
    )


def _travel_notices(advisory: TravelAdvisory | None, *, recorded: bool) -> list[str]:
    """Pick ``advisory``'s tense, or an empty list when there is none.

    At most one advisory is ever produced. A list rather than an optional
    string because every caller splices the result into a warning list, and
    five copies of the same ``None`` check read worse than one empty list.

    :param advisory: The pre-classified advisory, or ``None``.
    :param recorded: Whether the tracking write has already put the spec on
        disk. Callers pass this rather than letting the advisory assume,
        because the same problem reads as a false claim when the record it
        describes does not exist yet.
    :returns: The warning text in a one-element list, or an empty list.
    """
    if advisory is None:
        return []
    return [advisory.recorded if recorded else advisory.prospective]


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
    :raises ToolError: If ``micromamba`` cannot be started while probing for
        the environment.
    """
    spec = select_project_python(config, flag)
    if _is_python_passthrough(spec):
        return spec

    result = runner.run(micromamba_python_path(spec), capture=True, check=False)
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        # The default hint reads as an instruction to create an env by this
        # name, which for a botched version sends the user in exactly the wrong
        # direction — nobody wants a micromamba env called '3.12.x'.
        hint = (
            near_miss_version_notice(spec)
            if is_near_miss_version(spec)
            else (
                f"Ensure the env exists ('stack create env {render_positional_arg(spec)}') "
                "and that MAMBA_ROOT_PREFIX is set, or pass --python <version>."
            )
        )
        raise EnvError(
            f"Could not resolve micromamba environment '{spec}' to an interpreter.",
            hint=hint,
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
    :param union: Tokens to union into the recorded stack for this run (see
        :func:`union_stack`). Tokens rather than a finished stack, because the
        union must be taken against the stack read under the project lock; one
        computed before it would write back whatever a concurrent run added in
        between. ``None`` — the default — re-resolves what the project already
        records, which is what keeps ``stack refresh`` byte-identical in
        behavior.
    """

    python: str | None = None
    strict: bool = False
    no_sync: bool = False
    dry_run: bool = False
    union: tuple[str, ...] | None = None


@dataclass
class RefreshResult:
    """Outcome of :func:`refresh_project`.

    :param warnings: Non-fatal resolution advisories.
    :param added: Flattened requirements new to the applied ledger.
    :param removed: Dropped ledger entries eligible for removal.
    :param skipped_removals: Dropped entries never auto-removed (editables,
        paths, direct references).
    :param planned: The command plan (dry runs only).
    :param stack: The target stack, on a dry run given a union; ``None``
        otherwise. The union and the dependency delta are independent — a
        token whose packages are all already present changes ``stack`` and adds
        nothing — so the CLI needs this to avoid printing an empty delta for a
        run that is not a no-op.
    """

    warnings: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped_removals: list[str] = field(default_factory=list)
    planned: list[Command] = field(default_factory=list)
    stack: list[str] | None = None


def require_tracking(pyproject: Path, cwd: Path) -> ProjectTracking:
    """Read a tracked project's table, or raise the shared errors.

    :param pyproject: The project's ``pyproject.toml``.
    :param cwd: The project directory, named in the error message.
    :returns: The recorded tracking table.
    :raises ConfigError: When no tracked project is present.
    :raises NewerSchemaError: When the table declares a newer schema.
    """
    tracking = read_tracking(pyproject)
    if tracking is None:
        raise ConfigError(
            f"No tracked project in {cwd}.",
            hint=(
                "Run inside a project created by 'stack create project' (or "
                "add a [tool.uv-stack] table with 'stack' tokens)."
            ),
        )
    if tracking.version > 1:
        raise NewerSchemaError(
            NEWER_SCHEMA_MESSAGE.format(version=tracking.version),
            hint=NEWER_SCHEMA_HINT,
        )
    return tracking


def union_stack(recorded: Sequence[str], tokens: Sequence[str]) -> list[str]:
    """Return ``recorded`` with ``tokens`` appended, order preserved.

    Union, not replace: this is the sole behavioral difference from
    ``create project --force``, and the reason that command cannot simply be
    re-pointed at this use. Existing entries are never reordered or
    de-duplicated — only genuinely new tokens are appended, each once.

    :param recorded: The stack the project records.
    :param tokens: The tokens to union in, in the order given.
    :returns: The target stack.
    """
    new = [token for token in dict.fromkeys(tokens) if token not in recorded]
    return [*recorded, *new]


def _project_lock(config: ConfigRoot, cwd: Path) -> AbstractContextManager[None]:
    """The lock every run that writes ``cwd``'s ledger holds from first read to last write.

    Without it, two runs that both read the ledger before either wrote it
    each write back their own view, and the last one drops whatever the other
    added. Where :func:`uv_stack.fsutil.name_lock` degrades to no locking,
    that is the residual.
    """
    return name_lock(config.project_lock_path(cwd), str(cwd), action="updating project")


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
    cannot distinguish provenance (spec §2.3 accepted corner) — hence the
    warning, which gives the user a full refresh cycle to intervene before
    anything is removed.

    Advisories on failure: ``warnings`` and the ``skipped_removals`` notice are
    computed before any mutation but would otherwise reach the caller only on
    the ``RefreshResult``, which a raised error never produces, so both are
    attached as ``resolution_warnings`` to any :class:`UvStackError` raised from
    :func:`resolve_project_python` onward, for the CLI to print. What a later
    run recomputes turns on the write the failure got past: resolver and
    ownership warnings survive every write; adoption warnings die at the pending
    write, which folds ``adopted`` into ``applied``; an unverifiable-name
    warning dies there too if the stack has dropped the entry, and otherwise at
    the final ledger write, which clears ``pending``; the notice dies at that
    same write, which lands BEFORE ``uv sync`` already stripped of the skipped
    entries. The earliest boundary is inside the guarded region, so the handler
    cannot tell which are still recoverable; it attaches the whole set.

    :param config: Configuration root.
    :param runner: Command runner.
    :param options: Refresh options.
    :param cwd: The project directory.
    :returns: A :class:`RefreshResult`.
    :raises ConfigError: When no tracked project is present, the schema is
        newer than this uv-stack, or the pyproject cannot be written.
    :raises ConfigError: When another stack process holds this project's lock
        past the timeout, or when what stands at the lock path is something
        :func:`uv_stack.fsutil.name_lock` refuses; its docstring has the rule.
    :raises OSError: If the lock file cannot be opened for a reason ``name_lock``
        neither refuses nor degrades to no locking on; its docstring has the rule.
    """
    # A dry run writes nothing, so it has nothing to serialize and does not
    # wait behind a run that may be minutes into its uv sync.
    lock = nullcontext() if options.dry_run else _project_lock(config, cwd)
    with lock:
        return _refresh_project(config, runner, options, cwd=cwd)


def _refresh_project(
    config: ConfigRoot,
    runner: Runner,
    options: RefreshOptions,
    *,
    cwd: Path,
) -> RefreshResult:
    """The body of :func:`refresh_project`; every run but a dry run holds the project lock."""
    pyproject = cwd / "pyproject.toml"
    tracking = require_tracking(pyproject, cwd)

    # A union (stack sync project) extends the stack just read; stack refresh
    # passes None and re-resolves what is recorded. Computed above the dry-run
    # return at all times, so both paths see one value.
    target_stack = (
        union_stack(tracking.stack, options.union)
        if options.union is not None
        else tracking.stack
    )

    resolver = Resolver(config, strict=options.strict)
    stack = resolver.resolve(target_stack)
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

    # As in init_project, expansion precedes every write. It matters more
    # here: the pending table is written before the temp file is opened, so
    # computing expansion first means an undefined variable cannot leave a
    # pending record behind for a run that never started. It also sits outside
    # the dry-run guard below, so a dry run reports the refusal rather than
    # planning a run that cannot succeed.
    expanded_adds = expand_all(stack_adds, config.load_variables())

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

    # An explicit --python overrides the recorded spec; both written tables and
    # the interpreter resolution below must agree on the value.
    spec_flag = options.python if options.python is not None else tracking.python

    # Build both tables once, pre-flight the PENDING one (it is the first write
    # attempted), and keep the dry-run path write-free.
    pending_tracking = ProjectTracking(
        version=1,
        stack=target_stack,
        python=spec_flag,
        applied=[*tracking.applied, *adopted],
        pending=stack_adds,
    )
    final_tracking = ProjectTracking(
        version=1,
        stack=target_stack,
        python=spec_flag,
        applied=[*stack_adds, *adopted],
        pending=None,
    )

    if not options.dry_run:
        validate_tracking_write(pyproject, pending_tracking)

    # Below the pre-flight and above the dry-run return, which is the only
    # placement that satisfies both constraints. Above the return, so the dry
    # run and the real run share one classification and no delivery point below
    # can raise -- an unreadable envs directory surfacing from the success
    # return would report a completed refresh as failed, and the same failure
    # inside the handler would replace the error being reported. Below the
    # pre-flight, so that when the ledger is also unwritable the user hears
    # about the ledger, which is their actual blocker; init_project orders the
    # two the same way. A dry run that fails here is correct -- a dry run
    # exists to find out.
    travel = _travel_advisory(config, spec_flag)

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
            warnings=[*warnings, *_travel_notices(travel, recorded=False)],
            added=added,
            removed=removed,
            skipped_removals=skipped,
            planned=planned,
            stack=target_stack if options.union is not None else None,
        )

    # The temp file is created and filled before the durable write so that the
    # filesystem failures this run can produce fall above it, where they cost
    # nothing: only a UvStackError can carry the advisories out, so an OSError
    # raised past the write loses them. The two write_tracking calls below are
    # the residual -- atomic_write can still raise, and validate_tracking_write
    # pre-flights only the first of them. init_project is built the same way
    # for the same reason.
    recorded = False
    fd, tmp_name = tempfile.mkstemp(prefix="uv-stack-refresh.", suffix=".txt")
    tmp_req = Path(tmp_name)
    try:
        # Render only stack-owned dependencies to the temp requirements file.
        with open(fd, "w", encoding="utf-8") as handle:
            for entry in expanded_adds:
                handle.write(entry)
                handle.write("\n")
        python = resolve_project_python(config, runner, spec_flag)
        write_tracking(pyproject, pending_tracking)
        recorded = True
        if names:
            runner.run(_with_cwd(uv_remove(names), cwd))
        runner.run(_with_cwd(uv_add(tmp_req), cwd))
        write_tracking(pyproject, final_tracking)
        if not options.no_sync:
            runner.run(_with_cwd(uv_sync(python), cwd))
    except UvStackError as error:
        # These advisories ride on the RefreshResult, which a raised error never
        # produces. Hand them to the error instead: past the pending write some
        # are gone for good, and this handler cannot tell which (see docstring).
        error.resolution_warnings = [
            *warnings,
            *_travel_notices(travel, recorded=recorded),
            *(SKIPPED_REMOVAL_NOTICE.format(entry=entry) for entry in skipped),
        ]
        raise
    finally:
        if tmp_req.exists():
            tmp_req.unlink()
    return RefreshResult(
        warnings=[*warnings, *_travel_notices(travel, recorded=True)],
        added=added,
        removed=removed,
        skipped_removals=skipped,
    )
