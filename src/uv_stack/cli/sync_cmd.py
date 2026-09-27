"""``stack sync``: bring declared environments and tracked projects into line.

Sync is what a freshly cloned config root needs: create what is missing, build
what is empty, recompile what has drifted, and do it for every environment the
root declares. It adds no pipeline of its own — it runs ``stack upgrade``'s
pipeline with ``create=True`` and ``no_upgrade=True`` across the whole root.

The group takes no positional arguments of its own. A ``nargs=-1`` argument on
an ``invoke_without_command`` group swallows the subcommand name, so
``stack sync env main`` would parse ``env`` as an environment; the bare form is
deliberately the all-environments form and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import rich_click as click

from uv_stack.cli._complete import complete_env_names
from uv_stack.cli._render import console
from uv_stack.cli.upgrade import _checked_names, _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.operations.upgrade import UpgradeOptions

#: Group-level flag destinations mapped to their spelling, for the refusal
#: message below. Click's group and subcommand options are independent, so a
#: flag typed before the subcommand never reaches it.
_GROUP_FLAG_NAMES = {
    "dry_run": "--dry-run",
    "stop_on_error": "--stop-on-error",
    "strict": "--strict",
    "upgrade_all": "--upgrade",
}


def _sync_options[F: Callable[..., Any]](func: F) -> F:
    """Apply the four flags shared by the bare form and ``sync env``.

    Defined once so the two spellings of the same operation cannot drift apart
    as flags are added.

    :param func: The command callback to decorate.
    :returns: The callback with all four options applied.
    """
    options = (
        click.option(
            "--dry-run",
            is_flag=True,
            help=(
                "Print the command plan and refresh generated files; run none "
                "of the planned commands."
            ),
        ),
        click.option(
            "--stop-on-error",
            is_flag=True,
            help="Abort the batch at the first failing environment.",
        ),
        click.option(
            "--strict",
            is_flag=True,
            help="Fail if an unqualified token falls through to a literal package.",
        ),
        click.option(
            "--upgrade",
            "upgrade_all",
            is_flag=True,
            help="Force new pins instead of preserving the ones the lock already holds.",
        ),
    )
    for option in reversed(options):
        func = option(func)
    return func


def _refuse_group_flags(subcommand: str, given: dict[str, bool]) -> None:
    """Reject a shared flag typed before the subcommand.

    Click evaluates group and subcommand options independently: the group
    callback returns as soon as a subcommand is present, so ``--upgrade`` in
    ``stack sync --upgrade env main`` never reaches ``env``, which then runs
    with its own default of ``False``. Silently discarding a flag the user
    typed is the worst available outcome, so it is a usage error instead.

    :param subcommand: The invoked subcommand name, for the hint.
    :param given: Flag destination to value, as the group callback received it.
    :raises click.UsageError: When any flag was given.
    """
    typed = [_GROUP_FLAG_NAMES[dest] for dest, value in given.items() if value]
    if not typed:
        return
    flags = " ".join(typed)
    # Worded so it stays true for every subcommand: 'project' accepts neither
    # --upgrade nor --stop-on-error, so "move it after the subcommand" would be
    # wrong advice there. For the same reason the hint points at the
    # subcommand's help rather than spelling out a corrected command line.
    raise click.UsageError(
        f"{flags} before '{subcommand}' would be ignored: flags before a "
        f"subcommand are not passed to it. See 'stack sync {subcommand} --help' "
        "for the options it accepts."
    )


def _sync_envs(
    config: ConfigRoot,
    names: list[str],
    *,
    dry_run: bool,
    stop_on_error: bool,
    strict: bool,
    upgrade_all: bool,
) -> None:
    """Run the upgrade pipeline in create-and-preserve mode over ``names``.

    :param config: Configuration root.
    :param names: Validated environment names; empty means every declared one.
    :param dry_run: Plan only.
    :param stop_on_error: Abort the batch at the first failure.
    :param strict: Fail when a bare token falls through to a literal package.
    :param upgrade_all: Force new pins instead of preserving existing ones.
    """
    targets = names or config.list_envs()
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
        rule_verb="Syncing",
        all_succeeded="All requested environments synced.",
    )


@click.group("sync", invoke_without_command=True)
@_sync_options
@click.pass_context
def sync(
    ctx: click.Context,
    dry_run: bool,
    stop_on_error: bool,
    strict: bool,
    upgrade_all: bool,
) -> None:
    """Bring environments or a tracked project into line with this config root.

    With no subcommand, creates, builds, and recompiles every environment this
    config root declares. Missing environments are created rather than reported
    as errors, which is what makes this the first command to run on a freshly
    cloned config root.

    Existing pins are preserved wherever the sources still permit them; pass
    '--upgrade' to force new ones. Unlike 'stack upgrade' with no arguments,
    sync never prompts — acting on every environment is the whole point of the
    command, so a confirmation would ask about the thing you just asked for.
    """
    given = {
        "dry_run": dry_run,
        "stop_on_error": stop_on_error,
        "strict": strict,
        "upgrade_all": upgrade_all,
    }
    if ctx.invoked_subcommand is not None:
        _refuse_group_flags(ctx.invoked_subcommand, given)
        return
    config: ConfigRoot = ctx.obj
    _sync_envs(
        config,
        [],
        dry_run=dry_run,
        stop_on_error=stop_on_error,
        strict=strict,
        upgrade_all=upgrade_all,
    )


@sync.command("env")
@click.argument("names", nargs=-1, shell_complete=complete_env_names)
@_sync_options
@click.pass_obj
def sync_env(
    config: ConfigRoot,
    names: tuple[str, ...],
    dry_run: bool,
    stop_on_error: bool,
    strict: bool,
    upgrade_all: bool,
) -> None:
    """Create, build, and recompile the named environments.

    Missing environments are created rather than reported as errors. Existing
    pins are preserved wherever the sources still permit them; pass '--upgrade'
    to force new ones.
    """
    if not names:
        raise click.UsageError(
            "Give at least one NAME, or use 'stack sync' with no arguments for "
            "every environment."
        )
    _sync_envs(
        config,
        _checked_names(names),
        dry_run=dry_run,
        stop_on_error=stop_on_error,
        strict=strict,
        upgrade_all=upgrade_all,
    )
