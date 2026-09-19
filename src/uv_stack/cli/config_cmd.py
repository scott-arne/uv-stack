"""``stack config`` commands (init, portable)."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo
from uv_stack.config import ConfigRoot
from uv_stack.operations.init import init_config_root
from uv_stack.operations.portable import next_steps, write_portable_ignore


@click.group()
def config() -> None:
    """Initialize a config tree and manage its ignore file."""


@config.command("init")
@click.pass_obj
def config_init(config_root: ConfigRoot) -> None:
    """Create missing config directories (profiles/, bundles/, envs/, .locks/)."""
    created = init_config_root(config_root)
    if not created:
        echo("Nothing to do — all config directories already exist.")
        return
    echo(f"Created {len(created)} directories under {config_root.root}:")
    for path in created:
        echo(f"  {path}")


@config.command("portable")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the block and the next steps; write nothing.",
)
@click.pass_obj
def config_portable(config_root: ConfigRoot, dry_run: bool) -> None:
    """Write the managed .gitignore block that keeps a config root portable.

    Generated files — compiled locks, the rendered requirements.in and
    environment.yml, this machine's editor and variable values — are ignored
    so a clone of the root rebuilds them rather than inheriting them.
    """
    result = write_portable_ignore(config_root, dry_run=dry_run)
    suffix = " (dry run; nothing written)" if dry_run else ""
    echo(f"{result.path}: {result.outcome}{suffix}")
    echo("")
    for line in result.block.splitlines():
        echo(f"  {line}")
    echo("")
    echo("Next:")
    for line in next_steps(config_root, result):
        echo(f"  {line}")
