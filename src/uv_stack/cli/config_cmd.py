"""``stack config`` commands (init)."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo
from uv_stack.config import ConfigRoot
from uv_stack.operations.init import seed_defaults


@click.group()
def config() -> None:
    """Initialize the config tree."""


@config.command("init")
@click.pass_obj
def config_init(config_root: ConfigRoot) -> None:
    """Seed default profiles and bundles (never clobbers existing files)."""
    written = seed_defaults(config_root)
    if not written:
        echo("Nothing to do — all default files already exist.")
        return
    echo(f"Wrote {len(written)} files under {config_root.root}:")
    for path in written:
        echo(f"  {path}")
