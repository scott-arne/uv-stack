"""``stack sync``: bring declared environments and tracked projects into line.

Sync is what a freshly cloned config root needs: create what is missing, build
what is empty, recompile what has drifted, and do it for every environment the
root declares. It adds no pipeline of its own — it runs ``stack upgrade``'s
pipeline with ``create=True`` and ``no_upgrade=True`` across the whole root.
``stack sync remote`` brings another machine into line instead: it exports
from this root and pipes the document to ``stack import -`` on the remote over
ssh, so the remote's own import writes and builds it.

The group takes no positional arguments of its own. A ``nargs=-1`` argument on
an ``invoke_without_command`` group swallows the subcommand name, so
``stack sync env main`` would parse ``env`` as an environment; the bare form is
deliberately the all-environments form and nothing else.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import rich_click as click

from uv_stack.cli._complete import complete_env_names
from uv_stack.cli._render import console, render_error, render_warnings
from uv_stack.cli.refresh_cmd import run_refresh
from uv_stack.cli.transfer_cmd import export_items
from uv_stack.cli.upgrade import _checked_names, _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.operations.export import serialize_document
from uv_stack.operations.project import RefreshOptions
from uv_stack.operations.remote import (
    check_destination,
    explain_exit,
    remote_command,
    resolve_settings,
)
from uv_stack.operations.upgrade import UpgradeOptions
from uv_stack.runner import SubprocessRunner

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

    To bring another machine into line instead, 'stack sync remote' exports
    from this config root and imports on that machine over ssh.
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


@sync.command("project")
@click.argument("tokens", nargs=-1)
@click.option(
    "--python",
    "python",
    default=None,
    help="Override and record the project interpreter spec.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail if an unqualified token falls through to a literal package.",
)
@click.option("--no-sync", is_flag=True, help="Apply dependency changes but do not sync.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the add/remove delta and command plan; change nothing.",
)
@click.pass_obj
def sync_project(
    config: ConfigRoot,
    tokens: tuple[str, ...],
    python: str | None,
    strict: bool,
    no_sync: bool,
    dry_run: bool,
) -> None:
    """Union TOKENS into this project's recorded stack and re-resolve.

    TOKENS are ordinary resolver tokens: '@name' is a bundle, 'profile:name' a
    profile, 'pkg:name' a literal, and a bare name is a profile, then a bundle,
    then a literal package. A token already recorded is not an error — it makes
    this a plain re-resolve.

    The recorded stack is extended, never replaced, so an existing token cannot
    be lost by forgetting to repeat it.
    """
    if not tokens:
        raise click.UsageError(
            "Give at least one TOKEN. To re-resolve the tokens this project "
            "already records, use 'stack refresh'."
        )
    options = RefreshOptions(
        python=python,
        strict=strict,
        no_sync=no_sync,
        dry_run=dry_run,
        union=tokens,
    )
    run_refresh(config, options, cwd=Path.cwd())


@sync.command("remote")
@click.argument("items", nargs=-1, metavar="ITEMS...")
@click.argument("dest", metavar="DEST")
@click.option("--remote-root", metavar="PATH", help="Config root on the remote machine.")
@click.option("--remote-stack", metavar="CMD",
              help="Command that runs uv-stack on the remote machine.")
@click.option("--overwrite", is_flag=True,
              help="Replace files that differ and remove target-only environment files.")
@click.option("--no-build", is_flag=True, help="Install the definitions without building.")
@click.option("--recreate", is_flag=True, help="Wipe and rebuild each environment there.")
@click.option("--dry-run", is_flag=True, help="Report what the remote would change.")
@click.option("--strict", is_flag=True, help="Refuse names that fall through to packages.")
@click.pass_obj
def sync_remote(config: ConfigRoot, items: tuple[str, ...], dest: str, remote_root: str | None,
                remote_stack: str | None, overwrite: bool, no_build: bool, recreate: bool,
                dry_run: bool, strict: bool) -> None:
    """Export ITEMS and import them on DEST over ssh, in one step.

    DEST is an ssh destination, user@host or an ssh-config alias. No ITEMS
    means the whole root. The remote must run uv-stack; its command and root
    come from remotes.yaml, or from --remote-stack and --remote-root. The exit
    status is the remote's.
    """
    if no_build and recreate:
        raise click.UsageError("--recreate rebuilds environments, so it cannot be combined "
                               "with --no-build.")
    if remote_stack is not None and remote_stack == "":
        raise click.UsageError("--remote-stack cannot be empty.")
    if remote_root is not None and remote_root == "":
        raise click.UsageError("--remote-root cannot be empty.")
    check_destination(dest)
    exported = export_items(config, items)
    render_warnings(exported.warnings, styled=False)
    document = serialize_document(exported.document)
    settings = resolve_settings(config, dest, stack_flag=remote_stack, root_flag=remote_root)
    flags = [flag for flag, on in (("--overwrite", overwrite), ("--no-build", no_build),
             ("--recreate", recreate), ("--dry-run", dry_run), ("--strict", strict)) if on]
    command = remote_command(dest, settings.stack, settings.root, flags)
    code, tail = SubprocessRunner().run_with_input(command, document)
    explanation = explain_exit(code, tail, dest, settings.stack)
    if explanation is not None:
        render_error(explanation)
    sys.exit(code)
