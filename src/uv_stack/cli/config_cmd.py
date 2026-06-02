"""``stack config`` commands (init)."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo
from uv_stack.config import ConfigRoot
from uv_stack.operations.init import init_config_root


@click.group()
def config() -> None:
    """Initialize the config tree."""


@config.command("init")
@click.pass_obj
def config_init(config_root: ConfigRoot) -> None:
    """Create missing config directories (profiles/, bundles/, envs/)."""
    created = init_config_root(config_root)
    if not created:
        echo("Nothing to do — all config directories already exist.")
        return
    echo(f"Created {len(created)} directories under {config_root.root}:")
    for path in created:
        echo(f"  {path}")
