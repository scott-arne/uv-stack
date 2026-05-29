"""The ``project init`` operation: scaffold a uv project from a stack.

The resolved stack is flattened (profiles expanded inline) so the project's
dependencies do not reference files under the config root. Dependencies are
written to a temp requirements file and added via ``uv add``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from uvstack.commands import uv_add, uv_init, uv_sync
from uvstack.config import ConfigRoot
from uvstack.errors import ConfigError
from uvstack.render import render_requirements_flat
from uvstack.resolver import Resolver
from uvstack.runner import Command, Runner


@dataclass
class ProjectOptions:
    """Options for ``project init``.

    :param python: Python version to pin.
    :param name: Optional project name passed to ``uv init``.
    :param no_sync: Add dependencies but skip ``uv sync``.
    :param force: Allow adding to an existing ``pyproject.toml``.
    """

    python: str = "3.12"
    name: str | None = None
    no_sync: bool = False
    force: bool = False


def init_project(
    config: ConfigRoot,
    runner: Runner,
    tokens: list[str],
    options: ProjectOptions,
    *,
    cwd: Path,
) -> None:
    """Initialize a uv project in ``cwd`` from resolved stack ``tokens``.

    :param config: Configuration root.
    :param runner: Command runner.
    :param tokens: Stack tokens (profiles, bundles, packages).
    :param options: Project options.
    :param cwd: Directory in which to create the project.
    :raises ConfigError: If a ``pyproject.toml`` exists and ``force`` is False.
    """
    pyproject = cwd / "pyproject.toml"
    if pyproject.is_file() and not options.force:
        raise ConfigError(
            "pyproject.toml already exists.",
            hint="Use --force to add to the existing project.",
        )

    stack = Resolver(config).resolve(tokens)
    flat = render_requirements_flat(stack, config)

    fd, tmp_name = tempfile.mkstemp(prefix="uvstack-stack.", suffix=".txt")
    tmp_req = Path(tmp_name)
    try:
        with open(fd, "w") as handle:
            handle.write(flat)

        if not pyproject.is_file():
            runner.run(_with_cwd(uv_init(options.python, options.name), cwd))
        runner.run(_with_cwd(uv_add(tmp_req), cwd))
        if not options.no_sync:
            runner.run(_with_cwd(uv_sync(), cwd))
    finally:
        if tmp_req.exists():
            tmp_req.unlink()


def _with_cwd(command: Command, cwd: Path) -> Command:
    return Command(command.args, cwd=cwd)
