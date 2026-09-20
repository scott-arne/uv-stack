"""``stack converge``: bring every declared environment to the root's state.

Converge is what a freshly cloned config root needs: create what is missing,
build what is empty, recompile what has drifted, and do it for every
environment the root declares. It adds no pipeline of its own — it runs
``stack upgrade``'s pipeline with ``create=True`` and ``no_upgrade=True``
across the whole root.
"""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._complete import complete_env_names
from uv_stack.cli._render import console
from uv_stack.cli.upgrade import _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.operations.upgrade import UpgradeOptions


@click.command("converge")
@click.argument("names", nargs=-1, shell_complete=complete_env_names)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the command plan and refresh generated files; run no commands.",
)
@click.option(
    "--stop-on-error",
    is_flag=True,
    help="Abort the batch at the first failing environment.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail if an unqualified token falls through to a literal package.",
)
@click.option(
    "--upgrade",
    "upgrade_all",
    is_flag=True,
    help="Force new pins instead of preserving the ones the lock already holds.",
)
@click.pass_obj
def converge(
    config: ConfigRoot,
    names: tuple[str, ...],
    dry_run: bool,
    stop_on_error: bool,
    strict: bool,
    upgrade_all: bool,
) -> None:
    """Create, build, and recompile every environment this config root declares.

    Pass NAMES to converge only those environments. Missing environments are
    created rather than reported as errors, which is what makes this the first
    command to run on a freshly cloned config root.

    Existing pins are preserved wherever the sources still permit them; pass
    '--upgrade' to force new ones. Unlike 'stack upgrade' with no arguments,
    converge never prompts — acting on every environment is the whole point of
    the command, so a confirmation would ask about the thing you just asked for.
    """
    targets = list(names) or config.list_envs()
    if not targets:
        console.print("[yellow]No environments declared in this config root.[/yellow]")
        return
    options = UpgradeOptions(
        create=True,
        no_upgrade=not upgrade_all,
        dry_run=dry_run,
        strict=strict,
    )
    _run_upgrade(
        config,
        targets,
        options,
        stop_on_error=stop_on_error,
        rule_verb="Converging",
        all_succeeded="All requested environments converged.",
    )
