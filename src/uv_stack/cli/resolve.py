"""``stack resolve`` command (resolver debugging)."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo
from uv_stack.config import ConfigRoot
from uv_stack.resolver import Resolver


@click.command()
@click.option(
    "--full",
    is_flag=True,
    help="Expand bundles and profiles to a flat package list (requirements.txt-compatible).",
)
@click.argument("tokens", nargs=-1, required=True)
@click.pass_obj
def resolve(config: ConfigRoot, tokens: tuple[str, ...], full: bool) -> None:
    """Resolve TOKENS and print the result.

    By default each token is classified as ``bundle:``, ``profile:``, or
    ``package:`` without expansion. With ``--full``, bundles and profiles are
    expanded to a raw package list with no prefixes.
    """
    resolver = Resolver(config)
    if full:
        lines = resolver.resolve_packages(list(tokens))
    else:
        lines = resolver.classify(list(tokens))
    for line in lines:
        echo(line)
