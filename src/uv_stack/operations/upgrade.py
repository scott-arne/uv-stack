"""The ``upgrade`` operation: render → compile → sync → check.

Filesystem writes (``requirements.in``, ``environment.yml``, the lock) are
guarded explicitly; ``--dry-run`` renders the two safe generated files, builds
the command plan, and returns without touching the env or the lock.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from uv_stack.commands import (
    micromamba_create,
    micromamba_python_info,
    micromamba_python_path,
    micromamba_remove,
    uv_pip_check,
    uv_pip_compile,
    uv_pip_compile_for_version,
    uv_pip_sync,
)
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, EnvError, ToolError, UvStackError
from uv_stack.fsutil import atomic_write, relax_to_conventional_mode
from uv_stack.hints import render_positional_arg
from uv_stack.operations.create import ensure_env
from uv_stack.pyversion import is_comparable, parse_python_info, satisfies
from uv_stack.render import render_environment_yml, render_requirements_in
from uv_stack.resolver import Resolver
from uv_stack.runner import Command, Runner

_DRY_RUN_PYTHON = "<env-python>"

#: Kept local rather than imported from fsutil, whose flags are bundled with
#: O_NOFOLLOW; seeding deliberately follows a symlinked lock. Where the
#: platform lacks the flag getattr yields 0, which removes the FIFO guard
#: rather than weakening it — the fstat afterwards can still refuse a FIFO,
#: but only once the open it would have hung on has already returned.
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


@dataclass
class UpgradeOptions:
    """Options controlling an environment upgrade.

    :param create: Create the micromamba env if missing.
    :param recreate: Remove and recreate the env first.
    :param dry_run: Render generated files and return the plan without executing.
    :param upgrade_packages: Upgrade only these packages (disables full upgrade).
    :param no_upgrade: Recompile without forcing any upgrade.
    :param strict: Fail when a bare token falls through to a literal package.
    """

    create: bool = False
    recreate: bool = False
    dry_run: bool = False
    upgrade_packages: list[str] = field(default_factory=list)
    no_upgrade: bool = False
    strict: bool = False


@dataclass
class UpgradeResult:
    """Outcome of :func:`upgrade_env`.

    :param env_name: The environment that was upgraded.
    :param planned: The commands that would run (populated for dry runs).
    :param warnings: Non-fatal resolution advisories for the CLI to print.
    """

    env_name: str
    planned: list[Command] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _should_upgrade_all(options: UpgradeOptions) -> bool:
    return not options.no_upgrade and not options.upgrade_packages


def _new_candidate_lock(lock: Path, *, seed: bool) -> tuple[Path, bool]:
    """Create a sibling file to compile a candidate lock into.

    The candidate is seeded from the published lock when one exists. ``uv pip
    compile`` reads prior pins out of its *output* file, so compiling into an
    empty file re-resolves everything: ``--no-upgrade`` would have nothing to
    preserve and ``--upgrade-package X`` would upgrade far more than ``X``.
    Seeding is what gives both flags their documented meaning. A full
    ``--upgrade`` is not seeded at all: uv ignores those pins, so reading the
    lock would only introduce a failure mode that buys nothing.

    Compiling into a copy rather than into the lock itself is what keeps the
    published lock intact when the compile fails; the caller's unwind path
    deletes the candidate and leaves the original untouched.

    :param lock: The lock the candidate will replace once it is complete.
    :param seed: Whether to seed the candidate from the published lock. Pass
        False for a full upgrade, where uv ignores the output file.
    :returns: The newly created temp file, and whether the published lock's
        pins actually reached it. Requesting a seed does not guarantee one --
        an absent lock, or one that is not a regular file, leaves the candidate
        empty -- and callers that describe the candidate must say what it
        holds, not what was asked for.
    """
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=lock.parent, prefix=lock.name + ".", suffix=".tmp"
    )
    os.close(tmp_fd)
    candidate = Path(tmp_name)
    # mkstemp creates the candidate 0600, and the lock is not a secret: it must
    # not be the one generated file a second account cannot read, where the
    # requirements.in written in the same operation lands 0o666 & ~umask
    # through atomic_write. The relax is defensive rather than load-bearing --
    # uv pip compile replaces its output file rather than writing into this
    # inode, so the mode Path.replace publishes is uv's own. Setting it here
    # keeps the invariant local instead of resting on an undocumented uv
    # detail: a uv that wrote in place would put 0600 on the user's root.
    relax_to_conventional_mode(candidate)
    if not seed:
        return candidate, False
    try:
        # One lookup, not two. Checking the pathname and then opening it lets a
        # FIFO swapped in between park the open with no timeout and no unwind,
        # so what gets interrogated is the descriptor actually obtained.
        # O_NONBLOCK without O_NOFOLLOW: a symlinked lock is a topology this
        # project tolerates -- require_regular_file accepts one resolving to a
        # regular file -- and the flag still applies once the link has been
        # followed, so a FIFO is refused through a symlink just as directly.
        try:
            fd = os.open(lock, os.O_RDONLY | _O_NONBLOCK)
        except FileNotFoundError:
            # Covers both a lock never published and one removed between the
            # caller's decision to seed and this open.
            return candidate, False
        copied = False
        try:
            if stat.S_ISREG(os.fstat(fd).st_mode):
                with os.fdopen(fd, "rb") as handle:
                    fd = -1  # ownership transferred to the file object
                    pins = handle.read()
                candidate.write_bytes(pins)
                copied = True
        finally:
            if fd != -1:
                os.close(fd)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    return candidate, copied


def _explain_candidate_lock(
    error: BaseException, lock: Path, env_name: str, *, seeded: bool, copied: bool
) -> None:
    """Point a failed compile at the published lock behind the temp file.

    uv compiles into a ``.tmp`` sibling of the lock that the unwind deletes on
    the way out, so its diagnostics name a path that no longer exists and never
    mention the lock. What that path *was* takes three forms, and the two
    arguments below separate them because the request and the outcome come
    apart: a seeded copy of the lock; a new empty file a full upgrade resolves
    into from scratch; or a new empty file a seeded mode had to settle for
    because there was no lock to read. Calling either empty case a copy would
    describe a read that never happened and would send the user to fix a parse
    error in a file that may not exist, while blaming the third case on the
    mode would claim pins were ignored when there were none. Two CLI commands
    reach this code, so the seeded hint names one command that escapes a
    seeded compile whichever of them ran, scoped to this environment: a bare
    ``stack sync --upgrade`` would force-upgrade every environment in the root.

    :param error: The exception about to be re-raised. Anything that is not a
        :class:`ToolError`, any error that already carries a hint, and any
        error from a command other than the compile, is left alone.
    :param lock: The published lock the candidate stands in for.
    :param env_name: The environment being compiled, named in the seeded hint.
    :param seeded: Whether seeding was REQUESTED — the same value the caller
        passed as ``seed`` to :func:`_new_candidate_lock`.
    :param copied: Whether the lock's pins actually reached the candidate — the
        second element :func:`_new_candidate_lock` returns. False with
        ``seeded`` true is the absent-or-irregular lock.
    """
    # The argv PREFIX, not membership: micromamba_remove and
    # micromamba_python_path both place the env name in argv as a bare element,
    # so an environment called "compile" would otherwise claim this hint for
    # every failure on the recreate path. _compile_args emits this exact prefix
    # for both compile builders and nothing else emits it.
    if (
        isinstance(error, ToolError)
        and error.hint is None
        and error.command[:3] == ["uv", "pip", "compile"]
    ):
        if copied:
            # The flag precedes the name: render_positional_arg prefixes '--'
            # to a name starting with '-', and a flag after that separator
            # would be read as a second environment name.
            error.hint = (
                f"uv compiles into a copy of {lock}, so a '.tmp' path above "
                f"names that copy, not a file you are missing. If {lock.name} "
                "itself cannot be parsed, re-run as a full upgrade, which "
                "ignores the existing pins and rewrites it: 'stack sync env "
                f"--upgrade {render_positional_arg(env_name)}'."
            )
        elif seeded:
            error.hint = (
                f"uv compiles into a new empty file beside {lock}, so a '.tmp' "
                "path above names that file, not one you are missing. There "
                f"was nothing to seed it with: {lock.name} is absent or is not "
                "a regular file, so this run re-resolved from scratch despite "
                "being asked to preserve pins."
            )
        else:
            error.hint = (
                f"uv compiles into a new empty file beside {lock}, so a '.tmp' "
                "path above names that file, not one you are missing. "
                f"{lock.name} itself was not read: a full upgrade ignores the "
                "existing pins."
            )


def _env_python(runner: Runner, env_name: str, *, probed: str | None = None) -> str:
    """Return the env's interpreter path, reusing an earlier probe when given.

    ``probed`` is keyword-only: passing a stale path here syncs the wrong
    interpreter, so supplying it must always be a deliberate act.

    :param runner: Command runner.
    :param env_name: Environment name.
    :param probed: An interpreter path an earlier probe already reported. Omit
        it whenever the env may have been rebuilt since that probe.
    :returns: The interpreter path.
    :raises EnvError: If the interpreter path cannot be determined.
    :raises ToolError: If the probe command fails.
    """
    if probed:
        return probed
    python = runner.run(micromamba_python_path(env_name), capture=True).stdout.strip()
    if not python:
        raise EnvError(
            f"Could not determine the Python interpreter for env '{env_name}'.",
            hint="Verify the micromamba environment was created successfully.",
        )
    return python


def upgrade_env(
    config: ConfigRoot,
    runner: Runner,
    env_name: str,
    options: UpgradeOptions,
) -> UpgradeResult:
    """Render config, then compile, sync, and check an environment.

    :param config: Configuration root.
    :param runner: Command runner.
    :param env_name: Environment name.
    :param options: Upgrade options.
    :returns: An :class:`UpgradeResult`.
    :raises ConfigError: If the environment config is missing or invalid, or if
        a recreate is requested for an env whose ``python.txt`` is not a plain
        version.
    :raises EnvError: If the env is missing and creation was not requested, or
        if a non-recreate upgrade is requested for an env whose running
        interpreter no longer matches ``python.txt``.
    :raises ToolError: If a uv/micromamba command fails.
    """
    env = config.load_env(env_name)
    stack = Resolver(config, strict=options.strict).resolve(env.stack)

    # Guard: refuse when the running Python version differs from python.txt.
    # Placing this before the atomic_write calls ensures generated files are
    # untouched when drift is detected, preserving the "sources changed" signal.
    # A dry run is held to this refusal too, rather than exempted. It is not
    # read-only — it rewrites the two generated files below — and a plan
    # describing commands the real command declines to issue is precisely the
    # dishonesty this guard exists to remove.
    probed_python = None
    if not options.recreate:
        try:
            result = runner.run(
                micromamba_python_info(env_name), capture=True, check=False
            )
            if result.returncode == 0 and result.stdout.strip():
                executable, actual_version = parse_python_info(result.stdout)
                if actual_version and is_comparable(env.python):
                    if not satisfies(env.python, actual_version):
                        error = EnvError(
                            f"Environment '{env_name}' runs Python {actual_version}, "
                            f"but python.txt requests {env.python}.",
                            hint=(
                                "The interpreter is only rebuilt when the environment "
                                "is recreated. Either set "
                                f"{config.env_python_path(env_name)} to "
                                f"{actual_version} to keep the current interpreter, "
                                "or run 'stack create env --recreate "
                                f"{render_positional_arg(env_name)}' to rebuild it "
                                "(this wipes and reinstalls the environment)."
                            ),
                        )
                        error.resolution_warnings = stack.warnings
                        raise error
                # Guard passed; remember the executable to avoid re-probing.
                probed_python = executable
        except EnvError:
            raise
        except Exception:
            # Probe failure (e.g., micromamba not installed): fail open.
            pass

    # Guard: refuse a recreate whose target version uv cannot resolve against.
    # Resolving before the rebuild means the version must be one uv accepts for
    # --python-version. is_comparable is that predicate: it was written to
    # decide whether a drift comparison is meaningful, but the values it
    # admits — plain dotted versions — are exactly the ones the flag takes.
    # Refuse here rather than let a conda match spec, which renders into
    # environment.yml perfectly well, die inside uv with uv's own wording.
    # Like the drift guard, this sits before the atomic_write calls so a
    # refusal leaves the generated files — and the dry-run plan below — alone.
    if options.recreate and not is_comparable(env.python):
        refusal = ConfigError(
            f"Cannot recreate env '{env_name}': python.txt requests "
            f"'{env.python}', which is not a plain version. Recreating "
            "resolves the lock against the target version before "
            "rebuilding, and only a plain version can be resolved "
            "against.",
            hint=(
                "Set a plain version such as 3.14 in "
                f"{config.env_python_path(env_name)}, or upgrade "
                "without --recreate to keep the current interpreter."
            ),
        )
        # Outside the execution try below, so nothing else attaches these.
        refusal.resolution_warnings = stack.warnings
        raise refusal

    atomic_write(
        config.env_requirements_in(env_name),
        render_requirements_in(stack, config, env_name, config.load_variables()),
    )
    atomic_write(
        config.env_environment_yml(env_name),
        render_environment_yml(env),
    )

    upgrade_all = _should_upgrade_all(options)
    # Decided once for all four uses below. Seeding is what gives --no-upgrade
    # and --upgrade-package their meaning, and it is equally what makes the
    # failure hint true; a second spelling of this predicate would let the
    # candidate's contents and the explanation of them drift apart.
    seeded = not upgrade_all
    requirements_in = config.env_requirements_in(env_name)
    lock = config.env_requirements_lock(env_name)

    if options.dry_run:
        planned: list[Command] = []
        if options.recreate:
            # Mirrors the execution order below, including the real target
            # version: a recreate's compile is resolved from python.txt alone,
            # so the plan can name it instead of the placeholder.
            planned.append(
                uv_pip_compile_for_version(
                    env.python,
                    requirements_in,
                    lock,
                    upgrade=upgrade_all,
                    upgrade_packages=options.upgrade_packages,
                )
            )
            planned.append(micromamba_remove(env_name))
            planned.append(micromamba_create(config.env_environment_yml(env_name)))
        else:
            # ensure_env creates only when the env is missing, so planning the
            # create unconditionally described a step the real run declines to
            # issue -- the dishonesty the drift guard above refuses in its own
            # case. The answer is already paid for: that guard probed the
            # interpreter and kept it. None covers both "not there" and "could
            # not ask", so an unknown answer keeps the create and a dry run on
            # a machine without micromamba still plans the whole sequence.
            if options.create and probed_python is None:
                planned.append(micromamba_create(config.env_environment_yml(env_name)))
            planned.append(
                uv_pip_compile(
                    _DRY_RUN_PYTHON,
                    requirements_in,
                    lock,
                    upgrade=upgrade_all,
                    upgrade_packages=options.upgrade_packages,
                )
            )
        planned.append(uv_pip_sync(_DRY_RUN_PYTHON, lock))
        planned.append(uv_pip_check(_DRY_RUN_PYTHON))
        return UpgradeResult(env_name=env_name, planned=planned, warnings=stack.warnings)

    try:
        if options.recreate:
            # The destructive step runs at the last possible moment. An
            # unsatisfiable resolve is the likeliest failure in this sequence,
            # and it is fully detectable up front: uv resolves against
            # --python-version without needing that interpreter to exist, so
            # the candidate lock can be validated while the old environment is
            # still standing. This narrows the window; it does not close it —
            # micromamba create or uv pip sync can still fail once the old
            # environment is gone.
            tmp_lock, copied = _new_candidate_lock(lock, seed=seeded)
            try:
                runner.run(
                    uv_pip_compile_for_version(
                        env.python,
                        requirements_in,
                        tmp_lock,
                        upgrade=upgrade_all,
                        upgrade_packages=options.upgrade_packages,
                    )
                )
                ensure_env(
                    config, runner, env_name, create=options.create, recreate=True
                )
                # Probe the rebuilt env: sync and check need its real
                # interpreter, and any pre-recreate probe describes the env
                # that was just removed.
                python = _env_python(runner, env_name)
                # Publish only now: every failure above unwinds through the
                # handler below, which discards the candidate and leaves the
                # published lock untouched.
                tmp_lock.replace(lock)
            except BaseException as error:
                if tmp_lock.exists():
                    tmp_lock.unlink()
                _explain_candidate_lock(error, lock, env_name, seeded=seeded, copied=copied)
                raise
        else:
            ensure_env(
                config, runner, env_name, create=options.create, recreate=False
            )

            # Reuse the probe from the drift guard when available; otherwise probe now.
            python = _env_python(runner, env_name, probed=probed_python)

            # Compile to a temp lock, then atomically replace, so a failed compile never
            # corrupts an existing lockfile.
            tmp_lock, copied = _new_candidate_lock(lock, seed=seeded)
            try:
                runner.run(
                    uv_pip_compile(
                        python,
                        requirements_in,
                        tmp_lock,
                        upgrade=upgrade_all,
                        upgrade_packages=options.upgrade_packages,
                    )
                )
                tmp_lock.replace(lock)
            except BaseException as error:
                if tmp_lock.exists():
                    tmp_lock.unlink()
                _explain_candidate_lock(error, lock, env_name, seeded=seeded, copied=copied)
                raise

        runner.run(uv_pip_sync(python, lock))
        runner.run(uv_pip_check(python))
    except UvStackError as error:
        error.resolution_warnings = stack.warnings
        raise

    return UpgradeResult(env_name=env_name, warnings=stack.warnings)
