"""``stack show KIND [NAME]``: show an environment, profile, or bundle."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo
from uv_stack.config import ConfigRoot
from uv_stack.render import render_requirements_in
from uv_stack.resolver import Resolver

_KINDS = ("env", "profile", "bundle")


@click.command("show")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("name", required=False)
@click.pass_obj
def show(config: ConfigRoot, kind: str, name: str | None) -> None:
    """Show details of KIND NAME. NAME defaults to ``main`` for ``env``."""
    if kind == "env":
        _show_env(config, name or "main")
    elif name is None:
        raise click.UsageError(f"'show {kind}' requires a NAME.")
    elif kind == "profile":
        _show_profile(config, name)
    else:
        _show_bundle(config, name)


def _show_env(config: ConfigRoot, name: str) -> None:
    cfg = config.load_env(name)
    echo(f"Environment: {cfg.name}")
    echo(f"Python: {cfg.python}")
    echo("Stack:")
    for token in cfg.stack:
        echo(f"  {token}")
    echo("Micromamba packages:")
    for pkg in cfg.micromamba:
        echo(f"  {pkg}")
    stack = Resolver(config).resolve(cfg.stack)
    echo("Resolved profiles:")
    for profile_name in stack.profiles:
        echo(f"  {profile_name}")
    echo("Resolved inline requirements:")
    for req in stack.inline:
        echo(f"  {req}")
    # Touch render to validate it produces text without error.
    render_requirements_in(stack, config, name)


def _show_profile(config: ConfigRoot, name: str) -> None:
    prof = config.load_profile(name)
    echo(f"Profile: {prof.name}")
    for req in prof.requirements:
        echo(f"  {req}")


def _show_bundle(config: ConfigRoot, name: str) -> None:
    b = config.load_bundle(name)
    echo(f"Bundle: {b.name}")
    for token in b.tokens:
        echo(f"  {token}")
