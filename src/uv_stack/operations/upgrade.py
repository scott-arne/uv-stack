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
            planned.append(micromamba_remove(env_name))
            planned.append(micromamba_create(config.env_environment_yml(env_name)))
        elif options.create:
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
        ensure_env(
            config, runner, env_name, create=options.create, recreate=options.recreate
        )

        # Reuse the probe from the drift guard when available; otherwise probe now.
        if probed_python:
            python = probed_python
        else:
            python = runner.run(
                micromamba_python_path(env_name), capture=True
            ).stdout.strip()
            if not python:
                raise EnvError(
                    f"Could not determine the Python interpreter for env '{env_name}'.",
                    hint="Verify the micromamba environment was created successfully.",
                )

        # Compile to a temp lock, then atomically replace, so a failed compile never
        # corrupts an existing lockfile.
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=lock.parent, prefix=lock.name + ".", suffix=".tmp"
        )
        os.close(tmp_fd)
        tmp_lock = Path(tmp_name)
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
