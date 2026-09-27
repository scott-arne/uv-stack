"""``stack status``: per-environment build state (drift, lock, existence)."""

from __future__ import annotations

import json

import rich_click as click
from rich.text import Text

from uv_stack.cli._complete import complete_env_names
from uv_stack.cli._render import console, echo, render_table
from uv_stack.config import ConfigRoot
from uv_stack.operations.scaffold import validate_name
from uv_stack.operations.status import compute_status
from uv_stack.runner import SubprocessRunner


@click.command("status")
@click.argument("names", nargs=-1, shell_complete=complete_env_names)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def status(config: ConfigRoot, names: tuple[str, ...], as_json: bool) -> None:
    """Show each shared environment's build state (drift, lock, existence)."""
    for name in names:
        # compute_status joins each NAME onto <root>/envs and reads what it
        # lands on, so the file-stem rule upgrade and sync apply to their
        # own NAMEs holds here too. Refused up front rather than reported as
        # one more "config error" row: a name that cannot name an environment
        # is a bad argument, not a config that failed to load. Discovered names
        # need no check -- list_envs yields single directory components.
        validate_name("environment", name)
    statuses = compute_status(config, SubprocessRunner(), list(names) or None)
    if as_json:
        payload = [
            {
                "name": s.name,
                "python": s.python,
                "actual_python": s.actual_python,
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
        python_cell = s.python or "-"
        if s.actual_python is not None and s.state == "python changed":
            python_cell = f"{s.python} (env {s.actual_python})"
        rows.append(
            (
                s.name,
                python_cell,
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
