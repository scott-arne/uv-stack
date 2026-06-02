"""Shared rich rendering helpers for the CLI."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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


def render_list(title: str, items: Iterable[str], directory: Path | None = None) -> None:
    """Print a simple single-column table, optionally headed by its directory."""
    if directory is not None:
        # A full-width line, not a table caption: captions wrap to the
        # content-sized table width and mangle long absolute paths.
        console.print(f"[dim]{title} in {directory}[/dim]")
    table = Table(title=title)
    table.add_column(title.rstrip("s").capitalize())
    for item in items:
        table.add_row(item)
    console.print(table)


def echo(message: str) -> None:
    """Print a plain line to stdout (kept here so commands avoid importing rich)."""
    click.echo(message)
