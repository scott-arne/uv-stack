"""``stack list KIND``: list environments, profiles, or bundles."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import render_table
from uv_stack.config import ConfigRoot

_KINDS = ("env", "profile", "bundle")


@click.command("list")
@click.argument("kind", type=click.Choice(_KINDS))
@click.pass_obj
def list_resources(config: ConfigRoot, kind: str) -> None:
    """List resources of KIND (env, profile, or bundle)."""
    # Annotate so mypy widens to a common row type across the branches
    # (env rows are 3-tuples; profile/bundle rows are 2-tuples).
    rows: list[tuple[str, ...]]
    if kind == "env":
        rows = []
        for env in config.list_envs():
            cfg = config.load_env(env)
            rows.append((env, cfg.python, str(len(cfg.stack))))
        render_table(
            "envs",
            [("Env", "left"), ("Python", "left"), ("Stack", "right")],
            rows,
            config.envs_dir,
        )
    elif kind == "profile":
        rows = [
            (name, str(len(config.load_profile(name).requirements)))
            for name in config.list_profiles()
        ]
        render_table(
            "profiles",
            [("Profile", "left"), ("Packages", "right")],
            rows,
            config.profiles_dir,
        )
    else:
        rows = [
            (name, str(len(config.load_bundle(name).tokens)))
            for name in config.list_bundles()
        ]
        render_table(
            "bundles",
            [("Bundle", "left"), ("Tokens", "right")],
            rows,
            config.bundles_dir,
        )
