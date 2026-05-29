"""``uvstack config`` commands (init / doctor)."""

from __future__ import annotations

import rich_click as click

from uvstack.cli._render import console, echo
from uvstack.config import ConfigRoot
from uvstack.operations.doctor import diagnose
from uvstack.operations.init import seed_defaults


@click.group()
def config() -> None:
    """Initialize or diagnose the config tree."""


@config.command("init")
@click.pass_obj
def config_init(config_root: ConfigRoot) -> None:
    """Seed default profiles and bundles (never clobbers existing files)."""
    written = seed_defaults(config_root)
    if not written:
        echo("Nothing to do — all default files already exist.")
        return
    echo(f"Wrote {len(written)} files under {config_root.root}:")
    for path in written:
        echo(f"  {path}")


@config.command("doctor")
@click.pass_obj
def config_doctor(config_root: ConfigRoot) -> None:
    """Detect layout/config problems and print suggested fixes."""
    findings = diagnose(config_root)
    if not findings:
        console.print("[green]No problems detected.[/green]")
        return
    for finding in findings:
        color = "red" if finding.level == "error" else "yellow"
        console.print(f"[{color}]{finding.level.upper()}[/{color}] {finding.message}")
        if finding.fix:
            console.print(f"    [dim]fix:[/dim] {finding.fix}")
