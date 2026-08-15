"""``stack doctor``: detect layout/config problems; optionally fix them."""

from __future__ import annotations

import json
import sys

import rich_click as click
from rich.markup import escape

from uv_stack.cli._render import console, echo
from uv_stack.config import ConfigRoot
from uv_stack.operations.doctor import Finding, RepairAction, diagnose, repair


def _finding_dict(finding: Finding) -> dict[str, str | None]:
    return {
        "kind": finding.kind,
        "level": finding.level,
        "message": finding.message,
        "fix": finding.fix,
    }


def _print_findings(findings: list[Finding]) -> None:
    if not findings:
        console.print("[green]No problems detected.[/green]")
        return
    for finding in findings:
        color = "red" if finding.level == "error" else "yellow"
        console.print(
            f"[{color}]{finding.level.upper()}[/{color}] {escape(finding.message)}"
        )
        if finding.fix:
            console.print(f"    [dim]fix:[/dim] {escape(finding.fix)}")


@click.command("doctor")
@click.option("--fix", "fix", is_flag=True, help="Apply the safe fixes and re-run diagnosis.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def doctor(config_root: ConfigRoot, fix: bool, as_json: bool) -> None:
    """Detect layout/config problems and print suggested fixes."""
    findings = diagnose(config_root)
    if not fix:
        if as_json:
            echo(json.dumps([_finding_dict(f) for f in findings], indent=2))
            return
        _print_findings(findings)
        return

    actions: list[RepairAction] = []
    # Repairing can reveal new fixable findings (creating the root exposes
    # the missing subdirectories), so iterate to a fixed point. The bound
    # only guards against a pathological repair that never converges.
    for _ in range(10):
        round_actions = repair(config_root, findings)
        actions.extend(round_actions)
        findings = diagnose(config_root)
        if not findings or not any(action.applied for action in round_actions):
            break
    remaining = findings
    if as_json:
        payload = {
            "actions": [
                {
                    "kind": action.finding.kind,
                    "path": str(action.finding.path) if action.finding.path else None,
                    "applied": action.applied,
                    "description": action.description,
                    "reason": action.reason,
                }
                for action in actions
            ],
            "remaining": [_finding_dict(f) for f in remaining],
        }
        echo(json.dumps(payload, indent=2))
    else:
        for action in actions:
            if action.applied:
                console.print(f"[green]fixed:[/green] {escape(action.description)}")
            else:
                console.print(
                    f"[yellow]skipped:[/yellow] {escape(action.description)} "
                    f"({escape(str(action.reason))})"
                )
        _print_findings(remaining)
    if any(f.level == "error" for f in remaining):
        sys.exit(1)
