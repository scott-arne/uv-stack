"""``stack resolve`` command (resolver debugging)."""

from __future__ import annotations

import json

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
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.argument("tokens", nargs=-1, required=True)
@click.pass_obj
def resolve(
    config: ConfigRoot,
    tokens: tuple[str, ...],
    full: bool,
    strict: bool,
    as_json: bool,
) -> None:
    """Resolve TOKENS and print the result.

    By default each token is classified as 'bundle:', 'profile:', or
    'package:' without expansion. With '--full', bundles and profiles are
    expanded to a raw package list with no prefixes.
    """
    resolver = Resolver(config, strict=strict)
    if full:
        stack = resolver.resolve(list(tokens))
        render_warnings(stack.warnings, styled=not as_json)
        packages = resolver.flatten(stack)
        if as_json:
            echo(json.dumps({"packages": packages}, indent=2))
            return
        for line in packages:
            echo(line)
        return
    result = resolver.classify(list(tokens))
    render_warnings(result.warnings, styled=not as_json)
    if as_json:
        entries = []
        seen: set[str] = set()
        for token in tokens:
            single = resolver.classify([token]).entries
            if not single or single[0] in seen:
                continue
            seen.add(single[0])
            kind, _, target = single[0].partition(":")
            entries.append({"input": token, "kind": kind, "name": target})
        echo(json.dumps(entries, indent=2))
        return
    for line in result.entries:
        echo(line)
