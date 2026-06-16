"""Shared rich rendering helpers for the CLI."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from uv_stack.errors import UvStackError

console = Console()
error_console = Console(stderr=True)


def render_error(error: UvStackError) -> None:
    """Print a :class:`UvStackError` as a red panel with an optional hint."""
    body = error.message
    if error.hint:
        body += f"\n\n[dim]Hint:[/dim] {error.hint}"
    error_console.print(Panel(body, title="uv-stack error", border_style="red"))


def render_table(
    title: str,
    columns: Iterable[tuple[str, Literal["default", "left", "center", "right", "full"]]],
    rows: Iterable[tuple[str, ...]],
    directory: Path | None = None,
) -> None:
    """Print a multi-column table, optionally headed by its directory.

    :param title: Table title (also used in the directory header line).
    :param columns: ``(header, justify)`` pairs, where ``justify`` is a
        :mod:`rich` justification such as ``"left"`` or ``"right"``.
    :param rows: Row tuples, already stringified, one value per column.
    :param directory: When given, a dim ``"{title} in {directory}"`` line is
        printed above the table.
    """
    if directory is not None:
        # A full-width line, not a table caption: captions wrap to the
        # content-sized table width and mangle long absolute paths.
        console.print(f"[dim]{title} in {directory}[/dim]")
    table = Table(title=title)
    for header, justify in columns:
        table.add_column(header, justify=justify)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def echo(message: str) -> None:
    """Print a plain line to stdout (kept here so commands avoid importing rich)."""
    click.echo(message)
