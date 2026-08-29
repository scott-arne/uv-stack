"""``stack edit KIND [NAME]``: open a config file in the user's editor.

Owns every interaction the feature has: which file to open, how to launch the
editor, and how to report what validation found. The loop here is the sole
renderer of a validation failure — re-raising would have ``UvStackGroup``
print the same panel a second time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import rich_click as click

from uv_stack.cli._complete import complete_show_names
from uv_stack.cli._render import echo, render_error, render_warnings
from uv_stack.config import ConfigRoot
from uv_stack.editor import EditorCommand, editor_argv, resolve_editor
from uv_stack.errors import (
    ConfigError,
    NewerSchemaError,
    ResolutionError,
    ToolError,
)
from uv_stack.hints import render_positional_arg
from uv_stack.operations.edit import missing_project_error, validate
from uv_stack.operations.pyproject import read_tracking
from uv_stack.operations.scaffold import validate_name
from uv_stack.runner import Command, InteractiveRunner, SubprocessRunner

_KINDS = ("env", "profile", "bundle", "project")
_FILES = ("stack", "python", "micromamba", "channels", "local")


def _stdin_is_tty() -> bool:
    """Whether stdin is an interactive terminal.

    A module-level indirection so tests can substitute it. Click's
    ``CliRunner`` layers supplied input over a ``BytesIO``, whose ``isatty()``
    is False even when input is provided, so an inline ``sys.stdin.isatty()``
    would take the non-interactive branch in every CLI test.
    """
    return sys.stdin.isatty()


def _env_target(config: ConfigRoot, name: str, file: str) -> Path:
    """Map a ``--file`` choice to the env source file it names."""
    paths = {
        "stack": config.env_stack_path,
        "python": config.env_python_path,
        "micromamba": config.env_micromamba_path,
        "channels": config.env_channels_path,
        "local": config.env_local_path,
    }
    return paths[file](name)


def _require_regular_file(path: Path) -> None:
    """Refuse a target that exists but is not a regular file.

    ``Path.is_file()`` cannot tell a directory named ``channels.txt`` from an
    absent optional file, so without this the editor is handed a path it can
    never write. For the same reason this must run *before* any
    ``is_file()``-based existence test on the same path: that test cannot
    distinguish "absent" from "present but unusable", so it would claim the
    resource is missing and send the user to ``stack create``, which sees the
    entry and refuses. On a genuinely absent path this is a no-op.

    :raises ConfigError: When something other than a regular file is there.
    """
    if path.is_symlink() and not path.exists():
        raise ConfigError(
            f"Broken symlink: {path}",
            hint="Point it at a real file, or remove it.",
        )
    if path.exists() and not path.is_file():
        raise ConfigError(
            f"Not a regular file: {path}",
            hint="Remove or rename whatever is at that path.",
        )


def _resolve_target(config: ConfigRoot, kind: str, name: str, file: str) -> Path:
    """Decide which file to open, refusing to create a resource by accident.

    ``edit`` never scaffolds: an absent profile, bundle, or env is an error
    with the ``create`` command as its hint. Absent *optional* env sources are
    the exception — ``channels.txt`` not existing yet is the normal reason to
    open it.

    :raises ConfigError: When the resource does not exist, or the target is
        not a regular file.
    """
    if kind == "profile":
        target = config.profile_path(name)
        _require_regular_file(target)
        if not target.is_file():
            raise ConfigError(
                f"Missing profile: {target}",
                hint=(
                    "Create it first: stack create profile "
                    f"{render_positional_arg(name)} PACKAGES..."
                ),
            )
    elif kind == "bundle":
        target = config.bundle_path(name)
        _require_regular_file(target)
        if not target.is_file():
            raise ConfigError(
                f"Missing bundle: {target}",
                hint=(
                    "Create it first: stack create bundle "
                    f"{render_positional_arg(name)} TOKENS..."
                ),
            )
    elif kind == "env":
        # require_env tests stack.txt with is_file(), so a dangling link reads
        # as an absent file and its hint sends the user to `stack create`,
        # which refuses the same path because its O_EXCL open sees the link.
        _require_regular_file(config.env_stack_path(name))
        config.require_env(name)
        target = _env_target(config, name, file)
        _require_regular_file(target)
    else:
        target = Path.cwd() / "pyproject.toml"
        _require_regular_file(target)
        if not target.is_file():
            # Shared with the post-edit re-check so both moments give the same
            # hint; see operations/edit.missing_project_error.
            raise missing_project_error(Path.cwd())
    return target


def _argv(editor: EditorCommand, target: Path) -> list[str]:
    """Build the launch argv, reporting a bad ``--editor`` as a usage error.

    A malformed stored setting is a config problem; a malformed flag is a
    mistake in this invocation, and click's own exit code 2 says so.
    """
    try:
        return editor_argv(editor, target)
    except ConfigError as error:
        if editor.from_flag:
            # UsageError renders a bare string, with no hint slot of its own,
            # so the actionable half would be dropped on the floor. The layout
            # mirrors render_error's panel, which is what every other error in
            # the CLI looks like.
            detail = error.message
            if error.hint:
                detail = f"{detail}\n\nHint: {error.hint}"
            raise click.UsageError(detail) from error
        raise


def _next_step(kind: str, name: str, target: Path) -> None:
    """Print the command that makes the edit take effect."""
    if kind == "env":
        echo(f"Apply it with: stack upgrade {render_positional_arg(name)}")
    elif kind == "project":
        # An untracked pyproject validates with a warning, but refresh refuses
        # it outright, so naming refresh here would contradict that warning.
        if read_tracking(target) is not None:
            echo("Apply it with: stack refresh")
    else:
        echo("Applies on the next 'stack upgrade' or 'stack refresh'.")


def _edit_loop(
    config: ConfigRoot,
    kind: str,
    name: str,
    target: Path,
    argv: list[str],
    runner: InteractiveRunner,
) -> None:
    """Launch the editor, validate what it wrote, and re-offer on failure.

    Modelled on ``visudo``: a rejected file is not reverted, and the user is
    put straight back into the editor with the error in view.

    :raises ToolError: When the editor cannot start or exits non-zero.
    :raises ConfigError: When the editor left something other than a regular
        file at the target. Deliberately outside the re-offer arm: the hint
        tells the user to remove or rename what is there, which is work for
        the shell, not for another editor session.
    :raises NewerSchemaError: Straight through; re-editing cannot resolve a
        forward-schema refusal, so offering the editor again would loop the
        user through an edit that can never satisfy the check.
    """
    cwd = Path.cwd()
    while True:
        status = runner.run_interactive(Command(args=argv))
        if status != 0:
            raise ToolError(
                f"Editor exited with status {status}; {target} was not validated.",
                command=argv,
                returncode=status,
                hint="Exit the editor normally to have the file checked.",
            )
        # The pre-launch guard cannot speak for what the editor did. Optional
        # env sources are read through an is_file() test that calls a directory
        # or a dangling symlink absent, so validation would pass on a target
        # the next `stack edit` refuses.
        _require_regular_file(target)
        try:
            warnings = validate(config, kind, name, cwd)
        except NewerSchemaError:
            raise
        except (ConfigError, ResolutionError) as error:
            # ResolutionError is a sibling of ConfigError, not a subclass, so
            # both arms are needed. Widening to UvStackError would swallow the
            # ToolError above.
            render_error(error)
            if not _stdin_is_tty() or not click.confirm(
                "Re-open the editor?", default=True
            ):
                sys.exit(1)
            continue
        shown = Path(os.path.realpath(target)) if target.is_symlink() else target
        echo(f"Validated {shown}")
        render_warnings(warnings)
        _next_step(kind, name, target)
        return


@click.command("edit")
@click.argument("kind", type=click.Choice(_KINDS))
@click.argument("name", required=False, shell_complete=complete_show_names)
@click.option(
    "--file",
    "file",
    type=click.Choice(_FILES),
    default=None,
    help="Which env source file to open (env only; defaults to 'stack').",
)
@click.option(
    "--editor",
    "editor_flag",
    default=None,
    help="Editor command to use, overriding $UV_STACK_EDITOR, editor.txt, $VISUAL, and $EDITOR.",
)
@click.pass_obj
def edit(
    config: ConfigRoot,
    kind: str,
    name: str | None,
    file: str | None,
    editor_flag: str | None,
) -> None:
    """Open a config file in your editor.

    The file is validated when the editor exits. If it does not validate, the
    error is shown and the editor is offered again, with your changes left in
    place.

    NAME defaults to 'main' for 'env'. 'project' takes no NAME: it edits the
    pyproject.toml in the current directory.
    """
    # --file defaults to None rather than "stack" so that "not supplied" stays
    # distinguishable from an explicit value; a click default is otherwise
    # indistinguishable from one the user typed, and the env-only rule below
    # would then reject every plain `stack edit profile NAME`.
    if kind != "env" and file is not None:
        raise click.UsageError(f"'edit {kind}' takes no --file; the option is env-only.")
    if editor_flag is not None and not editor_flag.strip():
        raise click.UsageError("--editor needs a command.")

    if kind == "project":
        if name is not None:
            raise click.UsageError(
                "'edit project' takes no NAME; it edits the project in the "
                "current directory."
            )
        resource = ""
    elif kind == "env":
        resource = "main" if name is None else name
        # "environment" matches the wording `stack create env` uses.
        validate_name("environment", resource)
    elif name is None:
        raise click.UsageError(f"'edit {kind}' requires a NAME.")
    else:
        resource = name
        validate_name(kind, resource)

    target = _resolve_target(config, kind, resource, file or "stack")
    chosen = resolve_editor(config, editor_flag)
    _edit_loop(config, kind, resource, target, _argv(chosen, target), SubprocessRunner())
