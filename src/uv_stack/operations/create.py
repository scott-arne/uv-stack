"""Micromamba environment existence checks and creation/recreation."""

from __future__ import annotations

from uv_stack.commands import (
    micromamba_create,
    micromamba_python_path,
    micromamba_remove,
)
from uv_stack.config import ConfigRoot
from uv_stack.errors import EnvError
from uv_stack.hints import render_positional_arg
from uv_stack.runner import Runner


def env_micromamba_exists(config: ConfigRoot, runner: Runner, env_name: str) -> bool:
    """Return whether a micromamba environment named ``env_name`` exists.

    Probes by running ``python`` inside the env and checking for a usable
    interpreter path.

    :param config: Configuration root (unused beyond signature symmetry).
    :param runner: Command runner.
    :param env_name: Environment name.
    """
    result = runner.run(micromamba_python_path(env_name), capture=True, check=False)
    return result.returncode == 0 and bool(result.stdout.strip())


def ensure_env(
    config: ConfigRoot,
    runner: Runner,
    env_name: str,
    *,
    create: bool,
    recreate: bool,
) -> None:
    """Ensure the micromamba environment is in the desired state.

    Assumes ``environment.yml`` has already been rendered by the caller.

    :param config: Configuration root.
    :param runner: Command runner.
    :param env_name: Environment name.
    :param create: Create the env if it is missing.
    :param recreate: Remove (if present) and recreate the env.
    :raises EnvError: If the env is missing and neither ``create`` nor
        ``recreate`` was requested.
    """
    environment_yml = config.env_environment_yml(env_name)
    exists = env_micromamba_exists(config, runner, env_name)

    if recreate:
        if exists:
            runner.run(micromamba_remove(env_name))
        runner.run(micromamba_create(environment_yml))
        return

    if not exists:
        if create:
            runner.run(micromamba_create(environment_yml))
            return
        raise EnvError(
            f"Micromamba environment '{env_name}' does not exist.",
            hint=f"Run 'stack create env {render_positional_arg(env_name)}' to create it.",
        )


def env_interpreter(
    config: ConfigRoot, runner: Runner, env_name: str
) -> str | None:
    """Return the env's interpreter path, or None when it cannot be probed.

    ``None`` covers both a missing environment and an unrunnable probe (e.g.
    micromamba is not installed) — callers that need to distinguish should use
    a richer probe.

    :param config: Configuration root (unused beyond signature symmetry).
    :param runner: Command runner.
    :param env_name: Environment name.
    """
    try:
        result = runner.run(
            micromamba_python_path(env_name), capture=True, check=False
        )
    except OSError:
        return None
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        return None
    return path
