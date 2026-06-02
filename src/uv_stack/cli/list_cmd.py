"""``stack list KIND``: list environments, profiles, or bundles."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import render_list
from uv_stack.config import ConfigRoot

_KINDS = ("env", "profile", "bundle")


@click.command("list")
@click.argument("kind", type=click.Choice(_KINDS))
@click.pass_obj
def list_resources(config: ConfigRoot, kind: str) -> None:
    """List resources of KIND (env, profile, or bundle)."""
    if kind == "env":
        render_list("envs", config.list_envs(), config.envs_dir)
    elif kind == "profile":
        render_list("profiles", config.list_profiles(), config.profiles_dir)
    else:
        render_list("bundles", config.list_bundles(), config.bundles_dir)
