"""``stack refresh``: re-resolve a tracked project against current config."""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from uv_stack.cli._render import console, echo, render_warnings
from uv_stack.config import ConfigRoot
from uv_stack.errors import UvStackError
from uv_stack.operations.project import (
    SKIPPED_REMOVAL_NOTICE,
    RefreshOptions,
    refresh_project,
)
from uv_stack.runner import SubprocessRunner


@click.command("refresh")
@click.option(
    "--python",
    "python",
    default=None,
    help="Override and record the project interpreter spec.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail if an unqualified token falls through to a literal package.",
)
@click.option("--no-sync", is_flag=True, help="Apply dependency changes but do not sync.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the add/remove delta and command plan; change nothing.",
)
@click.pass_obj
def refresh(
    config: ConfigRoot,
    python: str | None,
    strict: bool,
    no_sync: bool,
    dry_run: bool,
) -> None:
    """Re-resolve the tracked project in the current directory against current
    profiles and bundles.
    """
    options = RefreshOptions(python=python, strict=strict, no_sync=no_sync, dry_run=dry_run)
    try:
        result = refresh_project(config, SubprocessRunner(), options, cwd=Path.cwd())
    except UvStackError as error:
        # A failure suppresses the result these would have arrived on, so print
        # them off the error; re-raise so the group-level handler still renders
        # the error panel.
        render_warnings(error.resolution_warnings)
        raise
    render_warnings(result.warnings)
    if result.removed:
        echo(f"Removed ({len(result.removed)}): {', '.join(result.removed)}")
    if result.added:
        echo(f"Added ({len(result.added)}): {', '.join(result.added)}")
    if not result.removed and not result.added:
        echo("Dependencies already match the current stack.")
    for entry in result.skipped_removals:
        echo(SKIPPED_REMOVAL_NOTICE.format(entry=entry))
    if dry_run:
        echo("Planned commands:")
        for command in result.planned:
            echo("  " + " ".join(command.args))
        return
    console.print("[green]Project refreshed.[/green]")
