"""``stack update``: update existing environments (render → compile → install → check)."""

from __future__ import annotations

import sys

import rich_click as click

from uv_stack.cli._render import console, echo, render_error
from uv_stack.config import ConfigRoot
from uv_stack.errors import UvStackError
from uv_stack.operations.update import UpdateOptions, update_env
from uv_stack.runner import SubprocessRunner


def _run_update(
    config: ConfigRoot,
    names: list[str],
    options: UpdateOptions,
    *,
    stop_on_error: bool = False,
) -> None:
    """Update each environment, continuing past failures by default.

    Each environment's outcome is recorded and printed in a final summary table.
    With ``stop_on_error`` the batch aborts at the first failing environment.
    A non-empty failure set exits the process with status 1.

    :param config: Configuration root.
    :param names: Environment names to update.
    :param options: Update options.
    :param stop_on_error: Abort the batch on the first failure.
    """
    runner = SubprocessRunner()
    failed: list[str] = []

    for name in names:
        console.rule(f"Updating {name}")
        try:
            result = update_env(config, runner, name, options)
        except UvStackError as error:
            render_error(error)
            failed.append(name)
            if stop_on_error:
                break
            continue
        if options.dry_run:
            echo("Planned commands:")
            for command in result.planned:
                echo("  " + " ".join(command.args))

    if options.dry_run:
        return

    _print_summary(names, failed)
    if failed:
        sys.exit(1)


def _print_summary(names: list[str], failed: list[str]) -> None:
    """Print a per-environment ✓/✗ summary of an update batch."""
    failed_set = set(failed)
    console.rule("Summary")
    for name in names:
        if name in failed_set:
            console.print(f"  [red]✗[/red] {name}")
        else:
            console.print(f"  [green]✓[/green] {name}")
    if failed:
        console.print(f"[red]{len(failed)} of {len(names)} environment(s) failed.[/red]")
    else:
        console.print("[green]All requested environments updated.[/green]")


@click.command("update")
@click.argument("names", nargs=-1)
@click.option("-y", "--yes", is_flag=True, help="Confirm bulk update of all envs.")
@click.option("--dry-run", is_flag=True, help="Print the command plan; change nothing.")
@click.option(
    "--stop-on-error",
    is_flag=True,
    help="Abort the batch at the first failing environment.",
)
@click.option("--no-upgrade", is_flag=True, help="Recompile without forcing upgrades.")
@click.option(
    "--upgrade-package",
    "upgrade_packages",
    multiple=True,
    help="Upgrade only this package (repeatable).",
)
@click.pass_obj
def update(
    config: ConfigRoot,
    names: tuple[str, ...],
    yes: bool,
    dry_run: bool,
    stop_on_error: bool,
    no_upgrade: bool,
    upgrade_packages: tuple[str, ...],
) -> None:
    """Render, compile, install, and check one or more existing environments.

    With no NAMES, all discovered environments are updated (with confirmation
    unless -y is given). By default the batch continues past a failing
    environment and reports a ``✓``/``✗`` summary; ``--stop-on-error`` aborts at
    the first failure. To create a missing environment, use ``stack create env``.
    """
    options = UpdateOptions(
        dry_run=dry_run,
        no_upgrade=no_upgrade,
        upgrade_packages=list(upgrade_packages),
    )
    targets = list(names)
    if not targets:
        targets = config.list_envs()
        if not targets:
            console.print("[yellow]No environments discovered.[/yellow]")
            return
        echo("Discovered environments:")
        for name in targets:
            echo(f"  - {name}")
        if not yes and not click.confirm("Update all of these?"):
            echo("Aborted.")
            return
    _run_update(config, targets, options, stop_on_error=stop_on_error)
