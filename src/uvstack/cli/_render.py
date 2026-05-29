"""Shared rich rendering helpers for the CLI."""

from __future__ import annotations

from collections.abc import Iterable

import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from uvstack.errors import UvstackError

console = Console()
error_console = Console(stderr=True)


def render_error(error: UvstackError) -> None:
    """Print a :class:`UvstackError` as a red panel with an optional hint."""
    body = error.message
    if error.hint:
        body += f"\n\n[dim]Hint:[/dim] {error.hint}"
    console.print(Panel(body, title="uvstack error", border_style="red"))


def render_list(title: str, items: Iterable[str]) -> None:
    """Print a simple single-column table."""
    table = Table(title=title)
    table.add_column(title.rstrip("s").capitalize())
    for item in items:
        table.add_row(item)
    console.print(table)


def echo(message: str) -> None:
    """Print a plain line to stdout (kept here so commands avoid importing rich)."""
    click.echo(message)
