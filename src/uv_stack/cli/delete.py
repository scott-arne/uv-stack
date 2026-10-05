"""``stack delete``: delete a shared environment, project, profile, or bundle.

The inverse of ``stack create``. Every subcommand asks before the irreversible
step unless ``-y`` is given, and asks only once the operation has found
something to delete: a refused delete never prompts, and the warnings a forced
delete prints are in view before the question. That ordering is what the
operations' ``confirm`` callbacks exist for.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import rich_click as click

from uv_stack.cli._complete import complete_bundle_names, complete_env_names, complete_profile_names
from uv_stack.cli._render import console, echo, render_warnings
from uv_stack.config import ConfigRoot
from uv_stack.hints import escape_controls
from uv_stack.operations.delete import (
    DeleteResult,
    EnvDeleteResult,
    WithdrawResult,
    delete_bundle,
    delete_env,
    delete_profile,
    withdraw_project,
)
from uv_stack.operations.project import SKIPPED_REMOVAL_NOTICE
from uv_stack.runner import SubprocessRunner

_YES_HELP = "Do not ask for confirmation."
_FORCE_HELP = (
    "Delete even if an environment's stack.txt or another bundle still refers to it, "
    "or a source could not be checked."
)


def _ask(question: str, *, yes: bool) -> bool:
    """Put the question unless ``-y`` answered it; a declined prompt aborts.

    ``abort=True`` is what makes a missing answer safe: a script that forgot
    ``-y`` reads EOF, which click treats as a decline, and exits 1 with nothing
    deleted rather than hanging or proceeding.
    """
    if yes:
        return True
    click.confirm(escape_controls(question), default=False, abort=True)
    return True


def _confirm_file(kind: str, name: str, *, yes: bool) -> Callable[[DeleteResult], bool]:
    """The ``confirm`` callback for a profile or bundle delete."""

    def confirm(result: DeleteResult) -> bool:
        # Before the question, so a forced delete's consequences are read first.
        render_warnings(result.warnings)
        return _ask(f"Delete {kind} '{name}' ({result.path})?", yes=yes)

    return confirm


@click.group("delete")
def delete() -> None:
    """Delete a shared environment, project, profile, or bundle."""


@delete.command("env")
@click.argument("name", shell_complete=complete_env_names)
@click.option("-y", "--yes", "yes", is_flag=True, help=_YES_HELP)
@click.pass_obj
def delete_env_cmd(config: ConfigRoot, name: str, yes: bool) -> None:
    """Delete environment NAME: its micromamba environment, then envs/NAME/.

    Only an environment with a stack.txt under the config root is deleted; a
    micromamba environment uv-stack does not manage is left alone. The
    micromamba environment goes first, so a failure there leaves the sources
    in place for a retry.
    """

    def confirm(result: EnvDeleteResult) -> bool:
        what = f"{result.directory}"
        if result.removed_micromamba:
            what += " and its micromamba environment"
        return _ask(f"Delete environment '{name}' ({what})?", yes=yes)

    result = delete_env(config, SubprocessRunner(), name, confirm=confirm)
    assert result is not None  # a declined prompt aborted inside confirm
    if result.removed_micromamba:
        echo(f"Removed micromamba environment {name}")
    echo(f"Deleted {result.directory}")


@delete.command("project")
@click.option("--no-sync", is_flag=True, help="Remove the dependencies but do not sync.")
@click.option("-y", "--yes", "yes", is_flag=True, help=_YES_HELP)
@click.pass_obj
def delete_project_cmd(config: ConfigRoot, no_sync: bool, yes: bool) -> None:
    r"""Withdraw uv-stack from the project in the current directory.

    Runs 'uv remove' for the packages uv-stack applied (the \[tool.uv-stack]
    applied list), then drops the table. Dependencies you added yourself,
    pyproject.toml, uv.lock and .venv all stay.
    """
    cwd = Path.cwd()

    def confirm(result: WithdrawResult) -> bool:
        if result.removed:
            echo(f"Removing ({len(result.removed)}): {', '.join(result.removed)}")
        else:
            echo("Nothing to remove: no uv-stack package is still in [project.dependencies].")
        for entry in result.skipped_removals:
            echo(SKIPPED_REMOVAL_NOTICE.format(entry=entry))
        echo(f"Dropping [tool.uv-stack] from {cwd / 'pyproject.toml'}")
        return _ask("Proceed?", yes=yes)

    result = withdraw_project(config, SubprocessRunner(), cwd=cwd, no_sync=no_sync, confirm=confirm)
    assert result is not None  # a declined prompt aborted inside confirm
    console.print("[green]uv-stack withdrawn from the project.[/green]")


@delete.command("profile")
@click.argument("name", shell_complete=complete_profile_names)
@click.option("--force", is_flag=True, help=_FORCE_HELP)
@click.option("-y", "--yes", "yes", is_flag=True, help=_YES_HELP)
@click.pass_obj
def delete_profile_cmd(config: ConfigRoot, name: str, force: bool, yes: bool) -> None:
    """Delete profiles/NAME.yaml.

    Refused while an environment's stack.txt or a bundle still refers to NAME,
    because a bare token would silently start meaning the pip package of that
    name. '--force' deletes anyway and warns about each reference.
    """
    confirm = _confirm_file("profile", name, yes=yes)
    result = delete_profile(config, name, force=force, confirm=confirm)
    assert result is not None  # a declined prompt aborted inside confirm
    echo(f"Deleted {result.path}")


@delete.command("bundle")
@click.argument("name", shell_complete=complete_bundle_names)
@click.option("--force", is_flag=True, help=_FORCE_HELP)
@click.option("-y", "--yes", "yes", is_flag=True, help=_YES_HELP)
@click.pass_obj
def delete_bundle_cmd(config: ConfigRoot, name: str, force: bool, yes: bool) -> None:
    """Delete bundles/NAME.yaml.

    Refused while an environment's stack.txt or another bundle still refers to
    NAME, for the reason 'delete profile' gives. '--force' deletes anyway and
    warns about each reference.
    """
    confirm = _confirm_file("bundle", name, yes=yes)
    result = delete_bundle(config, name, force=force, confirm=confirm)
    assert result is not None  # a declined prompt aborted inside confirm
    echo(f"Deleted {result.path}")
