"""``stack resolve`` command (resolver debugging)."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo, render_warnings
from uv_stack.config import ConfigRoot
from uv_stack.resolver import Resolver


@click.command()
@click.option(
    "--full",
    is_flag=True,
    help="Expand bundles and profiles to a flat package list (requirements.txt-compatible).",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail if an unqualified token falls through to a literal package.",
)
@click.argument("tokens", nargs=-1, required=True)
@click.pass_obj
def resolve(config: ConfigRoot, tokens: tuple[str, ...], full: bool, strict: bool) -> None:
    """Resolve TOKENS and print the result.

    By default each token is classified as 'bundle:', 'profile:', or
    'package:' without expansion. With '--full', bundles and profiles are
    expanded to a raw package list with no prefixes.
    """
    resolver = Resolver(config, strict=strict)
    if full:
        stack = resolver.resolve(list(tokens))
        render_warnings(stack.warnings)
        lines = resolver.flatten(stack)
    else:
        result = resolver.classify(list(tokens))
        render_warnings(result.warnings)
        lines = result.entries
    for line in lines:
        echo(line)
