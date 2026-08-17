"""``stack init``: guided first-run setup for a fresh config root."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo, print_activation_hint, render_warnings
from uv_stack.cli.upgrade import _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.hints import render_positional_arg
from uv_stack.operations.init import init_config_root
from uv_stack.operations.scaffold import write_env_sources, write_starter_profile
from uv_stack.operations.upgrade import UpgradeOptions
from uv_stack.resolver import Resolver


@click.command("init")
@click.option(
    "-y",
    "--yes",
    "yes",
    is_flag=True,
    help="Accept every prompt's default (non-interactive).",
)
@click.pass_obj
def init(config: ConfigRoot, yes: bool) -> None:
    """Guided first-run setup: config tree, starter profile, first env."""
    echo(f"Config root: {config.root}")
    for path in init_config_root(config):
        echo(f"Created {path}")

    if not config.list_profiles() and (
        yes or click.confirm("Seed a starter profile (profiles/starter.yaml)?", default=True)
    ):
        echo(f"Wrote {write_starter_profile(config)}")

    built_env: str | None = None
    scaffolded_env: str | None = None
    if not config.list_envs() and (
        yes or click.confirm("Create your first environment?", default=True)
    ):
        profiles = config.list_profiles()
        default_tokens = "starter" if config.profile_exists("starter") else (
            profiles[0] if profiles else ""
        )
        name = "main" if yes else click.prompt("Environment name", default="main")
        tokens_raw = (
            default_tokens
            if yes
            else click.prompt("Stack tokens (space-separated)", default=default_tokens)
        )
        python = "3.12" if yes else click.prompt("Python version", default="3.12")
        tokens = [token for token in tokens_raw.split() if token]
        if not tokens:
            echo("No tokens given — skipping environment creation.")
        else:
            # Validate before anything durable is written, mirroring
            # 'stack create env': bad tokens or malformed profiles must not
            # leave a half-created env behind.
            resolver = Resolver(config)
            stack = resolver.resolve(tokens)
            resolver.flatten(stack)
            for path in write_env_sources(config, name, tokens, python=python):
                echo(f"Wrote {path}")
            scaffolded_env = name
            if yes or click.confirm("Build it now?", default=True):
                _run_upgrade(config, [name], UpgradeOptions(create=True))
                built_env = name
            else:
                # User declined the build; render warnings once here.
                render_warnings(stack.warnings)

    echo("")
    echo("Next steps:")
    if built_env:
        print_activation_hint(built_env)
    elif scaffolded_env:
        echo(f"Build it with: stack create env {render_positional_arg(scaffolded_env)}")
    echo(f"Config lives in {config.root}")
    echo("See 'stack status' and 'stack list env' for an overview.")
