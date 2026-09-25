"""``stack show KIND [NAME]``: show a shared environment, project, profile, or bundle."""

from __future__ import annotations

import json
from pathlib import Path

import rich_click as click

from uv_stack.cli._complete import KIND_CHOICES, complete_show_names
from uv_stack.cli._render import echo, render_warnings
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.hints import render_positional_arg
from uv_stack.operations.create import env_interpreter
from uv_stack.operations.pyproject import read_tracking
from uv_stack.operations.scaffold import validate_name
from uv_stack.render import render_requirements_in
from uv_stack.resolver import Resolver
from uv_stack.runner import SubprocessRunner


def _probe_interpreter(config: ConfigRoot, name: str) -> str | None:
    """Probe the env's interpreter via the operations layer."""
    return env_interpreter(config, SubprocessRunner(), name)


@click.command("show")
@click.argument("kind", type=click.Choice(KIND_CHOICES))
@click.argument("name", required=False, shell_complete=complete_show_names)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def show(config: ConfigRoot, kind: str, name: str | None, as_json: bool) -> None:
    """Show details of KIND NAME. NAME defaults to 'main' for 'env'; 'project'
    reads the current directory.
    """
    if kind == "env":
        env_name = name or "main"
        # Each branch below joins NAME onto a config directory and reads what
        # it lands on, so the file-stem rule the create and edit commands apply
        # holds here too: without it a '..'-bearing NAME describes a file
        # outside the root as though the root declared it. "environment"
        # matches the wording `stack create env` uses.
        validate_name("environment", env_name)
        _show_env(config, env_name, as_json)
    elif kind == "project":
        if name is not None:
            raise click.UsageError(
                "'show project' takes no NAME; it reads the project in the "
                "current directory."
            )
        _show_project(as_json)
    elif name is None:
        raise click.UsageError(f"'show {kind}' requires a NAME.")
    elif kind == "profile":
        validate_name(kind, name)
        _show_profile(config, name, as_json)
    else:
        validate_name(kind, name)
        _show_bundle(config, name, as_json)


def _show_env(config: ConfigRoot, name: str, as_json: bool) -> None:
    cfg = config.load_env(name)
    stack = Resolver(config).resolve(cfg.stack)
    render_warnings(stack.warnings, styled=not as_json)
    channels = ["conda-forge"] + [c for c in cfg.channels if c != "conda-forge"]
    if as_json:
        payload = {
            "name": cfg.name,
            "python": cfg.python,
            "config_dir": str(config.env_dir(name)),
            "stack": cfg.stack,
            "channels": channels,
            "micromamba": cfg.micromamba,
            "profiles": stack.profiles,
            "inline": stack.inline,
        }
        echo(json.dumps(payload, indent=2))
        return
    echo(f"Environment: {cfg.name}")
    echo(f"Python: {cfg.python}")
    echo(f"Config: {config.env_dir(name)}")
    interpreter = _probe_interpreter(config, name)
    if interpreter:
        echo(f"Interpreter: {interpreter}")
    else:
        echo(f"Interpreter: not created (run 'stack create env {render_positional_arg(name)}')")
    echo("Stack:")
    for token in cfg.stack:
        echo(f"  {token}")
    echo("Channels:")
    for channel in channels:
        echo(f"  {channel}")
    echo("Micromamba packages:")
    for pkg in cfg.micromamba:
        echo(f"  {pkg}")
    echo("Resolved profiles:")
    for profile_name in stack.profiles:
        echo(f"  {profile_name}")
    echo("Resolved inline requirements:")
    for req in stack.inline:
        echo(f"  {req}")
    # The render runs only to surface a config that cannot produce a
    # requirements.in -- an unresolvable variable, most often. It reports
    # rather than aborts: every line above is already printed and correct, so
    # failing here condemned a description that succeeded. It also failed in
    # this branch alone, since --json returns above and never reaches the
    # check, which made the same root fatal in one output mode and fine in the
    # other. stack status sets the precedent, turning the identical failure
    # into a "config error" row rather than an exit status.
    try:
        render_requirements_in(stack, config, name, config.load_variables())
    except UvStackError as error:
        render_warnings([f"Cannot render requirements.in: {error.message}"])
    except OSError as error:
        # An unreadable variables.txt is a regular file, so require_regular_file
        # passes it and no layer converts the PermissionError. Left to the group
        # handler it would exit 1 and reinstate the very asymmetry above. There
        # is no .message to reuse, so this borrows render_os_error's shape:
        # strerror leads, and the file is named separately because the errno
        # text alone does not say which of the root's files could not be read.
        reason = error.strerror or str(error)
        if error.filename:
            reason = f"{reason}: {error.filename}"
        render_warnings([f"Cannot render requirements.in: {reason}"])


def _show_project(as_json: bool) -> None:
    """Print the current directory's ``[tool.uv-stack]`` tracking table.

    A pure read: no token resolution, no config-root access. That is what
    makes this usable when a referenced profile is missing or malformed.

    :param as_json: Emit the machine-readable payload instead of prose.
    :raises ConfigError: When the current directory holds no tracked project.
    """
    cwd = Path.cwd()
    tracking = read_tracking(cwd / "pyproject.toml")
    if tracking is None:
        raise ConfigError(
            f"No tracked project in {cwd}.",
            hint=(
                "Run 'stack create project TOKENS...' here, or cd into a "
                "tracked project."
            ),
        )
    if as_json:
        payload = {
            "path": str(cwd),
            "version": tracking.version,
            "stack": tracking.stack,
            "python": tracking.python,
            "applied": tracking.applied,
            "pending": tracking.pending,
        }
        echo(json.dumps(payload, indent=2))
        return
    echo(f"Project: {cwd}")
    echo(f"Python: {tracking.python or '(not recorded)'}")
    echo("Stack:")
    for token in tracking.stack:
        echo(f"  {token}")
    echo("Applied packages:")
    for package in tracking.applied:
        echo(f"  {package}")
    if tracking.pending is not None:
        # An empty list is a distinct state from no record at all: it still
        # marks an interrupted run. Say so rather than leaving a bare heading.
        echo("Pending (interrupted run — the next successful 'stack refresh' clears it):")
        if tracking.pending:
            for package in tracking.pending:
                echo(f"  {package}")
        else:
            echo("  (none — the interrupted run had no packages to apply)")


def _show_profile(config: ConfigRoot, name: str, as_json: bool) -> None:
    prof = config.load_profile(name)
    if as_json:
        payload = {
            "name": prof.name,
            "description": prof.description,
            "tags": prof.tags,
            "includes": prof.includes,
        }
        echo(json.dumps(payload, indent=2))
        return
    echo(f"Profile: {prof.name}")
    if prof.description:
        echo(f"Description: {prof.description}")
    if prof.tags:
        echo(f"Tags: {', '.join(prof.tags)}")
    for req in prof.includes:
        echo(f"  {req}")


def _show_bundle(config: ConfigRoot, name: str, as_json: bool) -> None:
    b = config.load_bundle(name)
    if as_json:
        payload = {
            "name": b.name,
            "description": b.description,
            "tags": b.tags,
            "includes": b.includes,
        }
        echo(json.dumps(payload, indent=2))
        return
    echo(f"Bundle: {b.name}")
    if b.description:
        echo(f"Description: {b.description}")
    if b.tags:
        echo(f"Tags: {', '.join(b.tags)}")
    for token in b.includes:
        echo(f"  {token}")
