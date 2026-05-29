"""The ``env update`` operation: render → compile → install → check.

Filesystem writes (``requirements.in``, ``environment.yml``, the lock) are
guarded explicitly; ``--dry-run`` renders the two safe generated files, builds
the command plan, and returns without touching the env or the lock.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from uvstack.commands import (
    micromamba_create,
    micromamba_python_path,
    micromamba_remove,
    uv_pip_check,
    uv_pip_compile,
    uv_pip_install,
)
from uvstack.config import ConfigRoot
from uvstack.errors import EnvError
from uvstack.fsutil import atomic_write
from uvstack.operations.create import ensure_env
from uvstack.render import render_environment_yml, render_requirements_in
from uvstack.resolver import Resolver
from uvstack.runner import Command, Runner

_DRY_RUN_PYTHON = "<env-python>"


@dataclass
class UpdateOptions:
    """Options controlling an environment update.

    :param create: Create the micromamba env if missing.
    :param recreate: Remove and recreate the env first.
    :param dry_run: Render generated files and return the plan without executing.
    :param upgrade_packages: Upgrade only these packages (disables full upgrade).
    :param no_upgrade: Recompile without forcing any upgrade.
    """

    create: bool = False
    recreate: bool = False
    dry_run: bool = False
    upgrade_packages: list[str] = field(default_factory=list)
    no_upgrade: bool = False


@dataclass
class UpdateResult:
    """Outcome of :func:`update_env`.

    :param env_name: The environment that was updated.
    :param planned: The commands that would run (populated for dry runs).
    """

    env_name: str
    planned: list[Command] = field(default_factory=list)


def _should_upgrade_all(options: UpdateOptions) -> bool:
    return not options.no_upgrade and not options.upgrade_packages


def update_env(
    config: ConfigRoot,
    runner: Runner,
    env_name: str,
    options: UpdateOptions,
) -> UpdateResult:
    """Render config, then compile, install, and check an environment.

    :param config: Configuration root.
    :param runner: Command runner.
    :param env_name: Environment name.
    :param options: Update options.
    :returns: An :class:`UpdateResult`.
    :raises ConfigError: If the environment config is missing or invalid.
    :raises EnvError: If the env is missing and creation was not requested.
    :raises ToolError: If a uv/micromamba command fails.
    """
    env = config.load_env(env_name)
    stack = Resolver(config).resolve(env.stack)

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
    lock = config.env_lock(env_name)

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
        planned.append(uv_pip_install(_DRY_RUN_PYTHON, lock))
        planned.append(uv_pip_check(_DRY_RUN_PYTHON))
        return UpdateResult(env_name=env_name, planned=planned)

    ensure_env(
        config, runner, env_name, create=options.create, recreate=options.recreate
    )

    python = runner.run(micromamba_python_path(env_name), capture=True).stdout.strip()
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

    runner.run(uv_pip_install(python, lock))
    runner.run(uv_pip_check(python))

    return UpdateResult(env_name=env_name)
