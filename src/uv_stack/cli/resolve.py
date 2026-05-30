"""``stack resolve`` command (resolver debugging)."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo
from uv_stack.config import ConfigRoot
from uv_stack.resolver import Resolver


@click.command()
@click.argument("tokens", nargs=-1, required=True)
@click.pass_obj
def resolve(config: ConfigRoot, tokens: tuple[str, ...]) -> None:
    """Resolve TOKENS and print the resulting profiles and inline packages."""
    stack = Resolver(config).resolve(list(tokens))
    echo("Profiles:")
    for name in stack.profiles:
        echo(f"  {name}")
    echo("Inline requirements:")
    for req in stack.inline:
        echo(f"  {req}")
