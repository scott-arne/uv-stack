"""Read-only status computation for named environments.

Drift is detected by re-rendering the generated files in memory (the render
layer is pure) and comparing against what is on disk; nothing here writes.
The micromamba existence probe runs with ``check=False`` and treats any
execution failure (e.g. micromamba not installed) as "unknown" rather than
an error.
"""

from __future__ import annotations

from dataclasses import dataclass

from uv_stack.commands import micromamba_python_path
from uv_stack.config import ConfigRoot
from uv_stack.errors import UvStackError
from uv_stack.render import render_environment_yml, render_requirements_in
from uv_stack.resolver import Resolver
from uv_stack.runner import Runner


@dataclass
class EnvStatus:
    """One environment's build state.

    :param name: Environment name.
    :param python: Configured Python version, or ``None`` when the config
        failed to load.
    :param created: Whether the micromamba env exists; ``None`` when the probe
        could not run.
    :param lock_present: Whether ``requirements.lock.txt`` exists.
    :param state: One of ``config error``, ``not created``, ``never built``,
        ``sources changed``, ``lock stale``, ``ok``.
    :param message: The error text for ``config error`` rows.
    """

    name: str
    python: str | None
    created: bool | None
    lock_present: bool
    state: str
    message: str | None = None


def _probe_created(runner: Runner, name: str) -> bool | None:
    try:
        result = runner.run(micromamba_python_path(name), capture=True, check=False)
    except Exception:
        return None
    return result.returncode == 0 and bool(result.stdout.strip())


def env_status(config: ConfigRoot, runner: Runner, name: str) -> EnvStatus:
    """Compute the status of one environment.

    :param config: Configuration root.
    :param runner: Command runner (used only for the existence probe).
    :param name: Environment name.
    :returns: The computed :class:`EnvStatus`.
    """
    lock = config.env_lock(name)
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
            created=None,
            lock_present=lock_present,
            state="config error",
            message=error.message,
        )

    created = _probe_created(runner, name)
    req_path = config.env_requirements_in(name)
    yml_path = config.env_environment_yml(name)
    sources_changed = (
        not req_path.is_file()
        or req_path.read_text() != expected_req
        or not yml_path.is_file()
        or yml_path.read_text() != expected_yml
    )
    lock_stale = (
        lock_present
        and req_path.is_file()
        and lock.stat().st_mtime < req_path.stat().st_mtime
    )

    if created is False:
        state = "not created"
    elif not lock_present:
        state = "never built"
    elif sources_changed:
        state = "sources changed"
    elif lock_stale:
        state = "lock stale"
    else:
        state = "ok"

    return EnvStatus(
        name=name,
        python=env.python,
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
    targets = names if names else config.list_envs()
    return [env_status(config, runner, name) for name in targets]
