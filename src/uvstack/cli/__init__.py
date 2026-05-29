"""uvstack command-line interface (rich-click).

Defines the root group, the shared ``--root`` option (stored on the Click
context), version output, and the error wrapper that renders
:class:`UvstackError` as a panel and exits non-zero.
"""

from __future__ import annotations

import sys

import rich_click as click

from uvstack import __version__
from uvstack.cli._render import render_error
from uvstack.config import ConfigRoot
from uvstack.errors import UvstackError

click.rich_click.TEXT_MARKUP = "rich"
click.rich_click.SHOW_ARGUMENTS = True


class UvstackGroup(click.RichGroup):
    """Custom group that catches UvstackError and renders it as a panel."""

    def invoke(self, ctx: click.Context) -> None:
        try:
            super().invoke(ctx)
        except UvstackError as error:
            render_error(error)
            sys.exit(1)


@click.group(cls=UvstackGroup)
@click.version_option(__version__, prog_name="uvstack")
@click.option(
    "--root",
    "root",
    default=None,
    help="Config root. Defaults to $UV_ENV_ROOT or ~/.config/python-envs.",
)
@click.pass_context
def cli(ctx: click.Context, root: str | None) -> None:
    """uvstack — formalized uv + micromamba environment management."""
    ctx.obj = ConfigRoot.discover(root)


def _register() -> None:
    from uvstack.cli import bundle, config_cmd, env, profile, resolve

    cli.add_command(env.env)
    cli.add_command(profile.profile)
    cli.add_command(bundle.bundle)
    cli.add_command(resolve.resolve)
    cli.add_command(config_cmd.config)
    from uvstack.cli import project

    cli.add_command(project.project)


_register()


def main() -> None:
    """Console-script entry point."""
    cli()
