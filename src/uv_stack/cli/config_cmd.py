"""``stack config`` commands (init, portable, remote)."""

from __future__ import annotations

import json

import rich_click as click

from uv_stack.cli._complete import complete_remote_hosts
from uv_stack.cli._render import echo, render_warnings
from uv_stack.config import ConfigRoot
from uv_stack.hints import escape_controls
from uv_stack.operations.init import init_config_root
from uv_stack.operations.portable import (
    next_steps,
    write_directory_keepers,
    write_portable_ignore,
)
from uv_stack.operations.remote import entry_lines, load_remotes, remove_remote, set_remote


@click.group()
def config() -> None:
    """Initialize a config tree, manage its ignore file, and set per-host remote settings."""


@config.command("init")
@click.pass_obj
def config_init(config_root: ConfigRoot) -> None:
    """Create missing config directories (profiles/, bundles/, envs/, .locks/)."""
    link = config_root.dangling_link()
    created = init_config_root(config_root)
    if link is not None:
        echo(f"Created {link.target} (the missing target of {link.link})")
        created = [path for path in created if path != link.target]
    if not created:
        echo("Nothing to do — all config directories already exist.")
        return
    echo(f"Created {len(created)} directories under {config_root.root}:")
    for path in created:
        echo(f"  {path}")


@config.command("portable")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the block and the next steps; write nothing.",
)
@click.pass_obj
def config_portable(config_root: ConfigRoot, dry_run: bool) -> None:
    """Write the managed .gitignore block that keeps a config root portable.

    Generated files — compiled locks, the rendered requirements.in and
    environment.yml, this machine's editor and variable values — are ignored
    so a clone of the root rebuilds them rather than inheriting them.
    """
    result = write_portable_ignore(config_root, dry_run=dry_run)
    keepers = write_directory_keepers(config_root, dry_run=dry_run)
    suffix = " (dry run; nothing written)" if dry_run else ""
    echo(f"{result.path}: {result.outcome}{suffix}")
    echo("")
    for line in result.block.splitlines():
        echo(f"  {line}")
    echo("")
    if keepers:
        # Named rather than left silent because the untracking sequence below
        # stages .gitignore alone: in that branch these are files the user has
        # to add themselves, and this is where they find out they exist.
        echo(f"Placeholders so empty directories survive a clone{suffix}:")
        for keeper in keepers:
            echo(f"  {keeper}")
        echo("")
    echo("Next:")
    if dry_run:
        # The steps are printed under --dry-run because seeing the whole
        # workflow is the point of asking. They are not pasteable yet: the
        # untracking step would run, the add would fail on an ignore file that
        # was never written, and the bootstrap sequence's 'add .' would stage
        # the generated files the unwritten block exists to exclude.
        echo("  Re-run without --dry-run first; these assume the block is written.")
    for line in next_steps(config_root, result):
        echo(f"  {line}")


@config.group("remote")
def config_remote() -> None:
    """List, set, and remove the per-host settings 'stack sync remote' reads."""


@config_remote.command("list")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def config_remote_list(config_root: ConfigRoot, as_json: bool) -> None:
    """Show each host's settings, with the default shown for an unset field.

    --json gives the stored values instead, with null for an unset field.
    """
    warnings: list[str] = []
    remotes = load_remotes(config_root, warnings=warnings)
    render_warnings(warnings, styled=not as_json)
    if as_json:
        # Stored values rather than effective ones, so a script can tell an
        # unset field from one set to the default. JSON's own escaping covers
        # control characters.
        echo(json.dumps({host: s.model_dump() for host, s in remotes.items()}, indent=2))
        return
    if not remotes:
        echo(f"No remotes in {escape_controls(str(config_root.remotes_path()))}.")
        return
    for host, settings in remotes.items():
        for line in entry_lines(host, settings):
            echo(line)


@config_remote.command("set")
@click.argument("host", shell_complete=complete_remote_hosts)
@click.option("--stack", default=None, help="Command that runs stack on the remote.")
@click.option("--root", default=None, help="Config root on the remote.")
@click.pass_obj
def config_remote_set(
    config_root: ConfigRoot, host: str, stack: str | None, root: str | None
) -> None:
    """Set HOST's stack command, root, or both, adding HOST if it is new.

    A field not given keeps its current value.
    """
    # Before set_remote, so it wins over a bad HOST: the invocation is
    # incomplete whatever the host.
    if stack is None and root is None:
        raise click.UsageError("'config remote set' needs --stack, --root, or both.")
    warnings: list[str] = []
    settings = set_remote(config_root, host, stack=stack, root=root, warnings=warnings)
    render_warnings(warnings)
    for line in entry_lines(host, settings):
        echo(line)


@config_remote.command("remove")
@click.argument("host", shell_complete=complete_remote_hosts)
@click.argument("fields", nargs=-1, type=click.Choice(["stack", "root"]))
@click.pass_obj
def config_remote_remove(config_root: ConfigRoot, host: str, fields: tuple[str, ...]) -> None:
    """Remove HOST's entry, or with FIELDS, only those of its fields.

    An entry left with no fields stays; it means the same as no entry.
    """
    warnings: list[str] = []
    removal = remove_remote(config_root, host, fields, warnings=warnings)
    render_warnings(warnings)
    shown = escape_controls(host)
    if removal.entry is None:
        echo(f"Removed {shown} from {escape_controls(str(config_root.remotes_path()))}.")
        return
    for field in removal.not_set:
        echo(f"{field} is not set for {shown}.")
    for line in entry_lines(host, removal.entry):
        echo(line)
