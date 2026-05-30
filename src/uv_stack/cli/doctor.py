"""``stack doctor``: detect layout/config problems and print suggested fixes."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import console
from uv_stack.config import ConfigRoot
from uv_stack.operations.doctor import diagnose


@click.command("doctor")
@click.pass_obj
def doctor(config_root: ConfigRoot) -> None:
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
