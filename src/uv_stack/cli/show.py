"""``stack show KIND [NAME]``: show an environment, profile, or bundle."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo
from uv_stack.config import ConfigRoot
from uv_stack.operations.create import env_interpreter
from uv_stack.render import render_requirements_in
from uv_stack.resolver import Resolver
from uv_stack.runner import SubprocessRunner

_KINDS = ("env", "profile", "bundle")


def _probe_interpreter(config: ConfigRoot, name: str) -> str | None:
    """Probe the env's interpreter via the operations layer."""
    return env_interpreter(config, SubprocessRunner(), name)


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
    echo(f"Config: {config.env_dir(name)}")
    interpreter = _probe_interpreter(config, name)
    if interpreter:
        echo(f"Interpreter: {interpreter}")
    else:
        echo(f"Interpreter: not created (run 'stack create env {name}')")
    echo("Stack:")
    for token in cfg.stack:
        echo(f"  {token}")
    echo("Channels:")
    echo("  conda-forge")
    for channel in cfg.channels:
        if channel != "conda-forge":
            echo(f"  {channel}")
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
    if prof.description:
        echo(f"Description: {prof.description}")
    if prof.tags:
        echo(f"Tags: {', '.join(prof.tags)}")
    for req in prof.includes:
        echo(f"  {req}")


def _show_bundle(config: ConfigRoot, name: str) -> None:
    b = config.load_bundle(name)
    echo(f"Bundle: {b.name}")
    if b.description:
        echo(f"Description: {b.description}")
    if b.tags:
        echo(f"Tags: {', '.join(b.tags)}")
    for token in b.includes:
        echo(f"  {token}")
