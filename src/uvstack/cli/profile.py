"""``uvstack profile`` commands."""

from __future__ import annotations

import rich_click as click

from uvstack.cli._render import echo, render_list
from uvstack.config import ConfigRoot


@click.group()
def profile() -> None:
    """Inspect reusable package profiles."""


@profile.command("list")
@click.pass_obj
def profile_list(config: ConfigRoot) -> None:
    """List available profiles."""
    render_list("profiles", config.list_profiles())


@profile.command("show")
@click.argument("name")
@click.pass_obj
def profile_show(config: ConfigRoot, name: str) -> None:
    """Show the requirements in a profile."""
    prof = config.load_profile(name)
    echo(f"Profile: {prof.name}")
    for req in prof.requirements:
        echo(f"  {req}")
