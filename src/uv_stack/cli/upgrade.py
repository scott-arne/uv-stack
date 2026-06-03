"""``stack upgrade``: upgrade existing environments (render → compile → install → check)."""

from __future__ import annotations

import sys

import rich_click as click

from uv_stack.cli._render import console, echo, render_error
from uv_stack.config import ConfigRoot
from uv_stack.errors import UvStackError
from uv_stack.operations.upgrade import UpgradeOptions, upgrade_env
from uv_stack.runner import SubprocessRunner


def _run_upgrade(
    config: ConfigRoot,
    names: list[str],
    options: UpgradeOptions,
    *,
    stop_on_error: bool = False,
) -> None:
    """Upgrade each environment, continuing past failures by default.

    Each environment's outcome is recorded and printed in a final summary.
    With ``stop_on_error`` the batch aborts at the first failing environment.
    A non-empty failure set exits the process with status 1.

    :param config: Configuration root.
    :param names: Environment names to upgrade.
    :param options: Upgrade options.
    :param stop_on_error: Abort the batch on the first failure.
    """
    runner = SubprocessRunner()
    failed: list[str] = []

    for name in names:
        console.rule(f"Upgrading {name}")
        try:
            result = upgrade_env(config, runner, name, options)
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
    """Print a per-environment ✓/✗ summary of an upgrade batch."""
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
        console.print("[green]All requested environments upgraded.[/green]")


@click.command("upgrade")
@click.argument("names", nargs=-1)
@click.option(
    "--all",
    "all_envs",
    is_flag=True,
    help="Upgrade all discovered environments (cannot be combined with NAMES).",
)
@click.option("-y", "--yes", is_flag=True, help="Confirm bulk upgrade of all envs.")
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
def upgrade(
    config: ConfigRoot,
    names: tuple[str, ...],
    all_envs: bool,
    yes: bool,
    dry_run: bool,
    stop_on_error: bool,
    no_upgrade: bool,
    upgrade_packages: tuple[str, ...],
) -> None:
    """Render, compile, install, and check one or more existing environments.

    Pass NAMES to upgrade specific environments, or ``--all`` to upgrade every
    discovered environment. With neither, all environments are upgraded as well
    (the implicit form of ``--all``). Upgrading all environments prompts for
    confirmation unless -y is given. By default the batch continues past a
    failing environment and reports a ``✓``/``✗`` summary; ``--stop-on-error``
    aborts at the first failure. To create a missing environment, use
    ``stack create env``.
    """
    if all_envs and names:
        raise click.UsageError("--all cannot be combined with explicit environment NAMES.")
    options = UpgradeOptions(
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
        # The confirmation guards the *implicit* bulk path (``upgrade`` with no
        # arguments). An explicit ``--all`` is itself the confirmation, as is -y.
        if not all_envs and not yes and not click.confirm("Upgrade all of these?"):
            echo("Aborted.")
            return
    _run_upgrade(config, targets, options, stop_on_error=stop_on_error)
