"""Read-only status computation for named environments.

Drift is detected by re-rendering the generated files in memory (the render
layer is pure) and comparing against what is on disk; nothing here writes.
The micromamba existence probe runs with ``check=False`` and treats any
execution failure (e.g. micromamba not installed) as "unknown" rather than
an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from uv_stack.commands import micromamba_python_info
from uv_stack.config import ConfigRoot
from uv_stack.errors import UvStackError
from uv_stack.pyversion import is_comparable, parse_python_info, satisfies
from uv_stack.render import render_environment_yml, render_requirements_in
from uv_stack.resolver import Resolver
from uv_stack.runner import Runner


def _read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None


def _mtime_or_none(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return None


@dataclass
class EnvStatus:
    """One environment's build state.

    :param name: Environment name.
    :param python: Configured Python version, or ``None`` when the config
        failed to load.
    :param actual_python: The version the env actually runs, or ``None`` when
        the probe could not run or did not report one.
    :param created: Whether the micromamba env exists; ``None`` when the probe
        could not run.
    :param lock_present: Whether ``requirements.lock.txt`` exists.
    :param state: One of ``config error``, ``not created``, ``never built``,
        ``python changed``, ``sources changed``, ``lock stale``, ``ok``.
    :param message: The error text for ``config error`` rows.
    """

    name: str
    python: str | None
    actual_python: str | None
    created: bool | None
    lock_present: bool
    state: str
    message: str | None = None


def _probe_env(runner: Runner, name: str) -> tuple[bool | None, str | None]:
    """Probe existence and running version in one command.

    :returns: ``(created, actual_version)``. ``created`` is ``None`` when the
        probe could not run at all.
    """
    try:
        result = runner.run(micromamba_python_info(name), capture=True, check=False)
    except Exception:
        return (None, None)
    created = result.returncode == 0 and bool(result.stdout.strip())
    _, actual_version = parse_python_info(result.stdout)
    return (created, actual_version)


def env_status(config: ConfigRoot, runner: Runner, name: str) -> EnvStatus:
    """Compute the status of one environment.

    :param config: Configuration root.
    :param runner: Command runner (used only for the existence probe).
    :param name: Environment name.
    :returns: The computed :class:`EnvStatus`.
    """
    lock = config.env_requirements_lock(name)
    lock_present = lock.is_file()

    try:
        env = config.load_env(name)
        stack = Resolver(config).resolve(env.stack)
        expected_req = render_requirements_in(stack, config, name)
        expected_yml = render_environment_yml(env)
    except UvStackError as error:
        return EnvStatus(
            name=name,
            python=None,
            actual_python=None,
            created=None,
            lock_present=lock_present,
            state="config error",
            message=error.message,
        )

    created, actual_python = _probe_env(runner, name)

    if created is False:
        return EnvStatus(
            name=name,
            python=env.python,
            actual_python=actual_python,
            created=created,
            lock_present=lock_present,
            state="not created",
        )

    if not lock_present:
        return EnvStatus(
            name=name,
            python=env.python,
            actual_python=actual_python,
            created=created,
            lock_present=False,
            state="never built",
        )

    # Check for Python version drift.
    python_changed = (
        actual_python is not None
        and is_comparable(env.python)
        and not satisfies(env.python, actual_python)
    )

    req_path = config.env_requirements_in(name)
    yml_path = config.env_environment_yml(name)
    actual_req = _read_text_or_none(req_path)
    actual_yml = _read_text_or_none(yml_path)
    sources_changed = actual_req != expected_req or actual_yml != expected_yml

    # Compute lock-staleness reference as max mtime of requirements.in
    # and requirements.local.in (when present).
    local_req = config.env_local_path(name)
    req_mtime = _mtime_or_none(req_path) or 0
    local_mtime = _mtime_or_none(local_req) or 0
    reference_mtime = max(req_mtime, local_mtime)
    lock_mtime = _mtime_or_none(lock)
    if lock_mtime is None:
        # Lock vanished between the is_file() check and here.
        lock_present = False
        lock_stale = False
    else:
        lock_stale = lock_mtime < reference_mtime if reference_mtime > 0 else False

    if not lock_present:
        state = "never built"
    elif python_changed:
        state = "python changed"
    elif sources_changed:
        state = "sources changed"
    elif lock_stale:
        state = "lock stale"
    else:
        state = "ok"

    return EnvStatus(
        name=name,
        python=env.python,
        actual_python=actual_python,
        created=created,
        lock_present=lock_present,
        state=state,
    )


def compute_status(
    config: ConfigRoot, runner: Runner, names: list[str] | None = None
) -> list[EnvStatus]:
    """Compute statuses for ``names`` (default: every discovered env).

    :param config: Configuration root.
    :param runner: Command runner.
    :param names: Environment names, or ``None`` for all.
    :returns: One :class:`EnvStatus` per environment, in input/discovery order.
    """
    targets = config.list_envs() if names is None else names
    return [env_status(config, runner, name) for name in targets]
