"""``uvstack env`` commands: update, show, create, recreate, list."""

from __future__ import annotations

import rich_click as click

from uvstack.cli._render import console, echo, render_list
from uvstack.config import ConfigRoot
from uvstack.operations.update import UpdateOptions, update_env
from uvstack.render import render_requirements_in
from uvstack.resolver import Resolver
from uvstack.runner import SubprocessRunner


@click.group()
def env() -> None:
    """Manage named micromamba + uv environments."""


def _run_update(config: ConfigRoot, names: list[str], options: UpdateOptions) -> None:
    runner = SubprocessRunner()
    for name in names:
        console.rule(f"Updating {name}")
        result = update_env(config, runner, name, options)
        if options.dry_run:
            echo("Planned commands:")
            for command in result.planned:
                echo("  " + " ".join(command.args))
    if not options.dry_run:
        console.print("[green]All requested environments updated.[/green]")


@env.command("update")
@click.argument("names", nargs=-1)
@click.option("-y", "--yes", is_flag=True, help="Confirm bulk update of all envs.")
@click.option("--create", is_flag=True, help="Create the env if it is missing.")
@click.option("--recreate", is_flag=True, help="Remove and recreate the env first.")
@click.option("--dry-run", is_flag=True, help="Print the command plan; change nothing.")
@click.option("--no-upgrade", is_flag=True, help="Recompile without forcing upgrades.")
@click.option(
    "--upgrade-package",
    "upgrade_packages",
    multiple=True,
    help="Upgrade only this package (repeatable).",
)
@click.pass_obj
def env_update(
    config: ConfigRoot,
    names: tuple[str, ...],
    yes: bool,
    create: bool,
    recreate: bool,
    dry_run: bool,
    no_upgrade: bool,
    upgrade_packages: tuple[str, ...],
) -> None:
    """Render, compile, install, and check one or more environments.

    With no NAMES, all discovered environments are updated (with confirmation
    unless -y is given).
    """
    options = UpdateOptions(
        create=create,
        recreate=recreate,
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
    _run_update(config, targets, options)


@env.command("create")
@click.argument("name")
@click.pass_obj
def env_create(config: ConfigRoot, name: str) -> None:
    """Create the env if missing, then update it."""
    _run_update(config, [name], UpdateOptions(create=True))


@env.command("recreate")
@click.argument("name")
@click.pass_obj
def env_recreate(config: ConfigRoot, name: str) -> None:
    """Remove and recreate the env, then update it."""
    _run_update(config, [name], UpdateOptions(recreate=True))


@env.command("list")
@click.pass_obj
def env_list(config: ConfigRoot) -> None:
    """List discovered environments."""
    render_list("envs", config.list_envs())


@env.command("show")
@click.argument("name", default="main")
@click.pass_obj
def env_show(config: ConfigRoot, name: str) -> None:
    """Show an environment's configuration and resolved stack."""
    cfg = config.load_env(name)
    echo(f"Environment: {cfg.name}")
    echo(f"Python: {cfg.python}")
    echo("Stack:")
    for token in cfg.stack:
        echo(f"  {token}")
    echo("Micromamba packages:")
    for pkg in cfg.micromamba:
        echo(f"  {pkg}")
    stack = Resolver(config).resolve(cfg.stack)
    echo("Resolved profiles:")
    for profile_name in stack.profiles:
        echo(f"  {profile_name}")
    echo("Resolved inline requirements:")
    for req in stack.inline:
        echo(f"  {req}")
    # Touch render to validate it produces text without error.
    render_requirements_in(stack, config, name)
