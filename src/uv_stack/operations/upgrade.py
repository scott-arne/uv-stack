"""The ``upgrade`` operation: render → compile → sync → check.

Filesystem writes (``requirements.in``, ``environment.yml``, the lock) are
guarded explicitly; ``--dry-run`` renders the two safe generated files, builds
the command plan, and returns without touching the env or the lock.
"""

from __future__ import annotations

import os
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
from uv_stack.errors import EnvError, UvStackError
from uv_stack.fsutil import atomic_write
from uv_stack.hints import render_positional_arg
from uv_stack.operations.create import ensure_env
from uv_stack.pyversion import is_comparable, parse_python_info, satisfies
from uv_stack.render import render_environment_yml, render_requirements_in
from uv_stack.resolver import Resolver
from uv_stack.runner import Command, Runner

_DRY_RUN_PYTHON = "<env-python>"


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


def _new_candidate_lock(lock: Path) -> Path:
    """Create an empty sibling file to compile a candidate lock into.

    :param lock: The lock the candidate will replace once it is complete.
    :returns: Path to the newly created temp file.
    """
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=lock.parent, prefix=lock.name + ".", suffix=".tmp"
    )
    os.close(tmp_fd)
    return Path(tmp_name)


def _env_python(runner: Runner, env_name: str, probed: str | None = None) -> str:
    """Return the env's interpreter path, reusing an earlier probe when given.

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
    :raises ConfigError: If the environment config is missing or invalid.
    :raises EnvError: If the env is missing and creation was not requested.
    :raises ToolError: If a uv/micromamba command fails.
    """
    env = config.load_env(env_name)
    stack = Resolver(config, strict=options.strict).resolve(env.stack)

    # Guard: refuse when the running Python version differs from python.txt.
    # Placing this before the atomic_write calls ensures generated files are
    # untouched when drift is detected, preserving the "sources changed" signal.
    probed_python = None
    if not options.dry_run and not options.recreate:
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
                                "is recreated. Run "
                                f"'stack create env {render_positional_arg(env_name)} "
                                "--recreate' to rebuild it "
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

    atomic_write(
        config.env_requirements_in(env_name),
        render_requirements_in(stack, config, env_name),
    )
    atomic_write(
        config.env_environment_yml(env_name),
        render_environment_yml(env),
    )

    upgrade_all = _should_upgrade_all(options)
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
            if options.create:
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
            tmp_lock = _new_candidate_lock(lock)
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
                # Publish only now, so a rebuild that failed above leaves the
                # lock still describing the environment that is actually there.
                tmp_lock.replace(lock)
            except BaseException:
                if tmp_lock.exists():
                    tmp_lock.unlink()
                raise
        else:
            ensure_env(
                config, runner, env_name, create=options.create, recreate=False
            )

            # Reuse the probe from the drift guard when available; otherwise probe now.
            python = _env_python(runner, env_name, probed_python)

            # Compile to a temp lock, then atomically replace, so a failed compile never
            # corrupts an existing lockfile.
            tmp_lock = _new_candidate_lock(lock)
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
            except BaseException:
                if tmp_lock.exists():
                    tmp_lock.unlink()
                raise

        runner.run(uv_pip_sync(python, lock))
        runner.run(uv_pip_check(python))
    except UvStackError as error:
        error.resolution_warnings = stack.warnings
        raise

    return UpgradeResult(env_name=env_name, warnings=stack.warnings)
