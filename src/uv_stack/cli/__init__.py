"""uv-stack command-line interface (rich-click).

Defines the root group, the shared ``--root`` option (stored on the Click
context), version output, the error wrapper in ``UvStackGroup.invoke`` that
renders :class:`UvStackError` and any bare :class:`OSError` as a panel and
exits non-zero, and the shutdown guard in ``main()`` that flushes buffered
output. A broken pipe met while running a command is caught by the wrapper; one
met while emitting help, or on any output still buffered at exit, is caught by
the guard. Either way the process exits quietly with the shell's conventional
signal status.
"""

from __future__ import annotations

import os
import signal
import sys
from contextlib import suppress

import rich_click as click
from rich_click.rich_help_formatter import RichHelpFormatter
from rich_click.rich_panel import RichCommandPanel

from uv_stack import __version__
from uv_stack.cli._render import render_error, render_os_error
from uv_stack.config import ConfigRoot
from uv_stack.errors import UvStackError

# Width of the command-name column in every command-group panel. rich-click
# renders the Options and Commands panels as independent tables, so their
# description columns only align if the command-name column is pinned to a
# fixed width (the built-in ratio option scales with terminal width and cannot
# match the content-sized Options panel). 15 aligns the command descriptions
# with the Options descriptions for the current option set; widen it if a
# longer option name or metavar pushes the Options column further right.
_COMMAND_NAME_COLUMN_WIDTH = 15


def _redirect_streams_to_devnull() -> None:
    """Redirect stdout and stderr to devnull so shutdown flush succeeds.

    A downstream reader closed the pipe, which is ordinary shell usage. Without
    this redirect the interpreter's shutdown flush meets the dead pipe and
    CPython prints "Exception ignored on flushing sys.stdout" to stderr, exit
    status 120. Best effort: a caller that replaced either stream with a
    non-file object has no descriptor to redirect, and no shutdown flush of a
    real pipe to protect either.
    """
    for stream in (sys.stdout, sys.stderr):
        # CPython sets the attribute to None outright when the descriptor was
        # already closed at startup, which is not a stream with nothing to
        # redirect but no stream at all.
        if stream is not None:
            with suppress(OSError, ValueError):
                os.dup2(os.open(os.devnull, os.O_WRONLY), stream.fileno())


def _sigpipe_status() -> int:
    """Return the exit status for a broken-pipe condition.

    POSIX shells report a process killed by a signal as ``128 + signal_number``.
    ``signal.SIGPIPE`` is POSIX-only, so it is read with ``getattr`` —
    dereferencing it unconditionally would replace a clean exit with an
    ``AttributeError`` on the platforms the fallback exists for — non-POSIX
    ones, Windows above all.

    :returns: ``128 + SIGPIPE`` where available, else ``1``.
    """
    sigpipe = getattr(signal, "SIGPIPE", None)
    return 1 if sigpipe is None else 128 + int(sigpipe)


class _AlignedCommandPanel(RichCommandPanel):
    """Command-group panel with a fixed-width name column.

    Pinning the first column to a constant width aligns the description column
    both across command groups and with the Options panel above them.
    """

    def get_table(self, command, ctx, formatter):  # type: ignore[no-untyped-def]
        table = super().get_table(command, ctx, formatter)
        if len(table.columns) >= 2:
            name_column, help_column = table.columns[0], table.columns[1]
            name_column.width = _COMMAND_NAME_COLUMN_WIDTH
            name_column.ratio = None
            # The table expands to the panel width, so give the help column the
            # flexible ratio; otherwise the name column absorbs the slack in
            # groups whose rows do not wrap and the descriptions misalign.
            help_column.ratio = 1
        return table


RichHelpFormatter.command_panel_class = _AlignedCommandPanel

click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.COMMAND_GROUPS = {
    "stack": [
        # 'create' is cross-cutting — it makes environments, projects,
        # profiles, and bundles — so it gets its own panel rather than being
        # filed under a destination it only partly serves.
        {"name": "Create", "commands": ["create"]},
        {"name": "Environments", "commands": ["upgrade"]},
        {"name": "Projects", "commands": ["refresh"]},
        {"name": "Inspection", "commands": ["list", "show", "resolve", "status"]},
        {"name": "Maintenance", "commands": ["init", "doctor", "config", "completion"]},
    ]
}


class UvStackGroup(click.RichGroup):
    """Custom group that renders UvStackError and bare OSError as panels."""

    def invoke(self, ctx: click.Context) -> None:
        try:
            super().invoke(ctx)
        except UvStackError as error:
            render_error(error)
            sys.exit(1)
        except BrokenPipeError:
            # A downstream reader closed the pipe (`stack list | head`), which
            # is ordinary shell usage and not an error. Redirect both streams to
            # devnull so the interpreter's shutdown flush has somewhere to go,
            # then exit with the shell's conventional status for the signal.
            _redirect_streams_to_devnull()
            sys.exit(_sigpipe_status())
        except OSError as error:
            render_os_error(error)
            sys.exit(1)


@click.group(cls=UvStackGroup)
@click.version_option(__version__, prog_name="stack")
@click.option(
    "--root",
    "root",
    default=None,
    help=(
        "Config root. Defaults to $UV_STACK_ROOT (or legacy $UV_ENV_ROOT) or "
        "~/.config/python-envs."
    ),
)
@click.pass_context
def cli(ctx: click.Context, root: str | None) -> None:
    """stack — reusable package sets for shared environments and uv projects."""
    ctx.obj = ConfigRoot.discover(root)


def _register() -> None:
    from uv_stack.cli import (
        completion_cmd,
        config_cmd,
        create,
        doctor,
        init_cmd,
        list_cmd,
        refresh_cmd,
        resolve,
        show,
        status_cmd,
        upgrade,
    )

    cli.add_command(upgrade.upgrade)
    cli.add_command(create.create)
    cli.add_command(refresh_cmd.refresh)
    cli.add_command(list_cmd.list_resources)
    cli.add_command(show.show)
    cli.add_command(resolve.resolve)
    cli.add_command(init_cmd.init)
    cli.add_command(doctor.doctor)
    cli.add_command(config_cmd.config)
    cli.add_command(status_cmd.status)
    cli.add_command(completion_cmd.completion)


_register()


def main() -> None:
    """Console-script entry point."""
    try:
        cli()
    finally:
        # Click emits --help from an eager parameter callback, before
        # Group.invoke ever runs — so the BrokenPipeError arm there never sees
        # it. The rendered help text is still sitting in stdout's buffer at
        # interpreter shutdown, where CPython's final flush meets a dead pipe,
        # prints "Exception ignored on flushing sys.stdout", and sets status 120.
        # Flush now: a broken pipe raises here, where the guard below can catch
        # it, instead of at shutdown, where it cannot.
        try:
            for stream in (sys.stdout, sys.stderr):
                if stream is not None:
                    stream.flush()
        except BrokenPipeError:
            # What reaches this arm is anything that left bytes in stdout's
            # buffer and never flushed them — help output above all. Output
            # written through click.echo does not, because echo flushes, so the
            # break surfaces at the write instead: inside a command the invoke
            # arm above catches it and exits 141, and from Click's own eager
            # callbacks — --version — Click's EPIPE handling swaps in a
            # pacifying wrapper and exits 1. Neither reaches here: the invoke
            # arm redirected both streams to devnull and Click's wrapper
            # swallows the flush, so this arm is a no-op on both. Redirect and
            # exit with the same signal status the invoke arm uses. Raising
            # SystemExit here deliberately
            # replaces the in-flight SystemExit from cli(); that is the intended
            # behavior on the broken-pipe path only. Never `return` here: a bare
            # return from finally silently swallows the in-flight SystemExit,
            # turning every sys.exit(1) error path into status 0.
            _redirect_streams_to_devnull()
            raise SystemExit(_sigpipe_status()) from None
        except (OSError, ValueError):
            # A flush failure that is not a pipe break — EBADF (bad file
            # descriptor), ENOSPC (no space), or a ValueError on a closed stream.
            # Swallow it silently, which restores the pre-guard behavior exactly
            # in both shapes: for the OSErrors the interpreter's own shutdown
            # flush meets the same failure, reports it, and sets status 120, and
            # for a closed stream CPython skips it at shutdown, so it was silent
            # before this guard existed and stays silent now. Letting either
            # escape the finally would replace the in-flight status with a
            # traceback.
            pass
