"""``stack create``: create a new environment or project."""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from uv_stack.cli._render import console
from uv_stack.cli.upgrade import _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.operations.project import ProjectOptions, init_project
from uv_stack.operations.upgrade import UpgradeOptions
from uv_stack.runner import SubprocessRunner


@click.group("create")
def create() -> None:
    """Create a new environment or project."""


@create.command("env")
@click.argument("name")
@click.option("--recreate", is_flag=True, help="Remove and recreate the env first.")
@click.pass_obj
def create_env(config: ConfigRoot, name: str, recreate: bool) -> None:
    """Create environment NAME, then upgrade it (``--recreate`` wipes it first)."""
    options = UpgradeOptions(recreate=True) if recreate else UpgradeOptions(create=True)
    _run_upgrade(config, [name], options)


@create.command("project")
@click.argument("tokens", nargs=-1, required=True)
@click.option(
    "--python",
    "python",
    default=None,
    help=(
        "Python version or micromamba env name for the project interpreter. "
        "Defaults to $UV_STACK_PROJECT_PYTHON, the config-root default, then 3.12."
    ),
)
@click.option("--name", "name", default=None, help="Project name for uv init.")
@click.option("--no-sync", is_flag=True, help="Add dependencies but do not sync.")
@click.option("--force", is_flag=True, help="Add to an existing pyproject.toml.")
@click.pass_obj
def create_project(
    config: ConfigRoot,
    tokens: tuple[str, ...],
    python: str | None,
    name: str | None,
    no_sync: bool,
    force: bool,
) -> None:
    """Create a uv project from the resolved stack TOKENS."""
    options = ProjectOptions(python=python, name=name, no_sync=no_sync, force=force)
    init_project(config, SubprocessRunner(), list(tokens), options, cwd=Path.cwd())
    console.print("[green]Project initialized.[/green]")
