"""``stack status``: per-environment build state (drift, lock, existence)."""

from __future__ import annotations

import json

import rich_click as click
from rich.text import Text

from uv_stack.cli._complete import complete_env_names
from uv_stack.cli._render import console, echo, render_table
from uv_stack.config import ConfigRoot
from uv_stack.operations.status import compute_status
from uv_stack.runner import SubprocessRunner


@click.command("status")
@click.argument("names", nargs=-1, shell_complete=complete_env_names)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def status(config: ConfigRoot, names: tuple[str, ...], as_json: bool) -> None:
    """Show each shared environment's build state (drift, lock, existence)."""
    statuses = compute_status(config, SubprocessRunner(), list(names) or None)
    if as_json:
        payload = [
            {
                "name": s.name,
                "python": s.python,
                "created": s.created,
                "lock": s.lock_present,
                "state": s.state,
                "message": s.message,
            }
            for s in statuses
        ]
        echo(json.dumps(payload, indent=2))
        return
    rows = []
    for s in statuses:
        created = "?" if s.created is None else ("yes" if s.created else "no")
        rows.append(
            (
                s.name,
                s.python or "-",
                created,
                "yes" if s.lock_present else "no",
                s.state,
            )
        )
    render_table(
        "status",
        [
            ("Env", "left"),
            ("Python", "left"),
            ("Created", "left"),
            ("Lock", "left"),
            ("State", "left"),
        ],
        rows,
        config.envs_dir,
    )
    for s in statuses:
        if s.message:
            console.print(Text(f"{s.name}: {s.message}", style="dim"))
