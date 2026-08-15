"""Shared rich rendering helpers for the CLI."""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import rich_click as click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uv_stack.errors import UvStackError

console = Console()
error_console = Console(stderr=True)


def render_error(error: UvStackError) -> None:
    """Print a :class:`UvStackError` as a red panel with an optional hint.

    The message and hint are literal text, not markup. They routinely carry
    bracketed TOML table names such as ``[tool.uv-stack]``, which rich would
    otherwise parse as a style tag and render as nothing — deleting the
    subject of the sentence. Assembling a :class:`~rich.text.Text` keeps them
    literal while still styling the ``Hint:`` label.
    """
    body = Text(error.message)
    if error.hint:
        body.append("\n\n")
        body.append("Hint:", style="dim")
        body.append(f" {error.hint}")
    error_console.print(Panel(body, title="uv-stack error", border_style="red"))


def render_warnings(warnings: Iterable[str], *, styled: bool = True) -> None:
    """Print resolution warnings, one per line, on stderr.

    :param styled: Rich yellow markup when true; plain text when false
        (JSON output modes must keep stderr unstyled).
    """
    for warning in warnings:
        if styled:
            error_console.print(f"[yellow]warning:[/yellow] {escape(warning)}")
        else:
            click.echo(f"warning: {warning}", err=True)


def render_table(
    title: str,
    columns: Iterable[tuple[str, Literal["default", "left", "center", "right", "full"]]],
    rows: Iterable[tuple[str, ...]],
    directory: Path | None = None,
) -> None:
    """Print a multi-column table, optionally headed by its directory.

    :param title: Table title (also used in the directory header line). Must be
        a literal string, not user-controlled text, as it is parsed as rich
        markup.
    :param columns: ``(header, justify)`` pairs, where ``justify`` is a
        :mod:`rich` justification such as ``"left"`` or ``"right"``.
    :param rows: Row tuples, already stringified, one value per column.
    :param directory: When given, a dim ``"{title} in {directory}"`` line is
        printed above the table. User-controlled (derived from ``--root`` or
        ``UV_STACK_ROOT``), so rendered as :class:`Text` rather than markup.
    """
    if directory is not None:
        # A full-width line, not a table caption: captions wrap to the
        # content-sized table width and mangle long absolute paths.
        console.print(Text(f"{title} in {directory}", style="dim"))
    table = Table(title=title)
    for header, justify in columns:
        table.add_column(header, justify=justify)
    for row in rows:
        # Cells carry user text (profile descriptions, tags, paths), so they
        # are wrapped as Text rather than parsed as markup.
        table.add_row(*(Text(cell) for cell in row))
    console.print(table)


def echo(message: str) -> None:
    """Print a plain line to stdout.

    Unlike the console renderers above, this never parses markup, so callers
    printing user data through it need no escaping.
    """
    click.echo(message)


def print_activation_hint(name: str) -> None:
    """Print how to enter or use a freshly built environment."""
    quoted = shlex.quote(name)
    echo("Activate it:")
    echo(f"  micromamba activate {quoted}")
    echo("Or run one-offs without activating:")
    echo(f"  micromamba run -n {quoted} python")
