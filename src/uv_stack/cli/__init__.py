"""uv-stack command-line interface (rich-click).

Defines the root group, the shared ``--root`` option (stored on the Click
context), version output, and the error wrapper that renders
:class:`UvStackError` as a panel and exits non-zero.
"""

from __future__ import annotations

import sys

import rich_click as click
from rich_click.rich_help_formatter import RichHelpFormatter
from rich_click.rich_panel import RichCommandPanel

from uv_stack import __version__
from uv_stack.cli._render import render_error
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
        {"name": "Environments", "commands": ["update", "create"]},
        {"name": "Inspection", "commands": ["list", "show", "resolve"]},
        {"name": "Maintenance", "commands": ["doctor", "config"]},
    ]
}


class UvStackGroup(click.RichGroup):
    """Custom group that catches UvStackError and renders it as a panel."""

    def invoke(self, ctx: click.Context) -> None:
        try:
            super().invoke(ctx)
        except UvStackError as error:
            render_error(error)
            sys.exit(1)


@click.group(cls=UvStackGroup)
@click.version_option(__version__, prog_name="stack")
@click.option(
    "--root",
    "root",
    default=None,
    help="Config root. Defaults to $UV_ENV_ROOT or ~/.config/python-envs.",
)
@click.pass_context
def cli(ctx: click.Context, root: str | None) -> None:
    """stack — formalized uv + micromamba environment management."""
    ctx.obj = ConfigRoot.discover(root)


def _register() -> None:
    from uv_stack.cli import (
        config_cmd,
        create,
        doctor,
        list_cmd,
        resolve,
        show,
        update,
    )

    cli.add_command(update.update)
    cli.add_command(create.create)
    cli.add_command(list_cmd.list_resources)
    cli.add_command(show.show)
    cli.add_command(resolve.resolve)
    cli.add_command(doctor.doctor)
    cli.add_command(config_cmd.config)


_register()


def main() -> None:
    """Console-script entry point."""
    cli()
