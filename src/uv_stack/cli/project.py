"""``stack project`` commands: init a local uv project from a stack."""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from uv_stack.cli._render import console
from uv_stack.config import ConfigRoot
from uv_stack.operations.project import ProjectOptions, init_project
from uv_stack.runner import SubprocessRunner


@click.group()
def project() -> None:
    """Create local uv projects from a stack."""


@project.command("init")
@click.argument("tokens", nargs=-1, required=True)
@click.option("--python", "python", default="3.12", help="Python version to pin.")
@click.option("--name", "name", default=None, help="Project name for uv init.")
@click.option("--no-sync", is_flag=True, help="Add dependencies but do not sync.")
@click.option("--force", is_flag=True, help="Add to an existing pyproject.toml.")
@click.pass_obj
def project_init(
    config: ConfigRoot,
    tokens: tuple[str, ...],
    python: str,
    name: str | None,
    no_sync: bool,
    force: bool,
) -> None:
    """Initialize a uv project from the resolved stack TOKENS."""
    options = ProjectOptions(python=python, name=name, no_sync=no_sync, force=force)
    init_project(config, SubprocessRunner(), list(tokens), options, cwd=Path.cwd())
    console.print("[green]Project initialized.[/green]")
