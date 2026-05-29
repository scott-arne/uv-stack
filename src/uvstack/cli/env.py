"""``uvstack env`` commands."""

from __future__ import annotations

import rich_click as click

from uvstack.cli._render import echo
from uvstack.config import ConfigRoot


@click.group()
def env() -> None:
    """Manage named micromamba + uv environments."""


@env.command("show")
@click.argument("name", default="main")
@click.pass_obj
def env_show(config: ConfigRoot, name: str) -> None:
    """Show an environment's configuration."""
    cfg = config.load_env(name)
    echo(f"Environment: {cfg.name}")
    echo(f"Python: {cfg.python}")
    echo("Stack:")
    for token in cfg.stack:
        echo(f"  {token}")
