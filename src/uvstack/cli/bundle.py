"""``uvstack bundle`` commands."""

from __future__ import annotations

import rich_click as click

from uvstack.cli._render import echo, render_list
from uvstack.config import ConfigRoot


@click.group()
def bundle() -> None:
    """Inspect composable bundles."""


@bundle.command("list")
@click.pass_obj
def bundle_list(config: ConfigRoot) -> None:
    """List available bundles."""
    render_list("bundles", config.list_bundles())


@bundle.command("show")
@click.argument("name")
@click.pass_obj
def bundle_show(config: ConfigRoot, name: str) -> None:
    """Show the tokens in a bundle."""
    b = config.load_bundle(name)
    echo(f"Bundle: {b.name}")
    for token in b.tokens:
        echo(f"  {token}")
