"""``stack export`` and ``stack import``: move definitions between machines."""

from __future__ import annotations

import sys
from pathlib import Path

import rich_click as click

from uv_stack.cli._render import render_error, render_warnings
from uv_stack.config import ConfigRoot
from uv_stack.fsutil import atomic_write
from uv_stack.operations.export import (
    AmbiguousItemError,
    ExportResult,
    build_document,
    serialize_document,
)


def export_items(config: ConfigRoot, items: tuple[str, ...] | list[str]) -> ExportResult:
    """Build the export document, turning an ambiguous item into exit 2.

    The panel goes through :func:`render_error`, which renders user text
    literally; a Click usage error would pass a bracketed name through markup.
    """
    try:
        return build_document(config, items)
    except AmbiguousItemError as error:
        render_error(error)
        sys.exit(2)


@click.command("export")
@click.argument("items", nargs=-1, metavar="ITEMS...")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
    help="Write the document to this file instead of standard output.",
)
@click.pass_obj
def export_cmd(config: ConfigRoot, items: tuple[str, ...], output: Path | None) -> None:
    """Export environments, profiles, and bundles as one portable document.

    ITEMS are env:NAME, profile:NAME, bundle:NAME, @NAME, or a bare NAME
    that names exactly one item. With no ITEMS the whole root is exported.
    Everything the items reach is included, with each environment's lock as
    a seed. Import it elsewhere with 'stack import FILE', or pull straight
    from another machine with 'ssh HOST stack export ITEMS | stack import -'.
    """
    result = export_items(config, items)
    render_warnings(result.warnings, styled=False)
    text = serialize_document(result.document)
    if output is None:
        click.echo(text, nl=False)
    else:
        atomic_write(output, text)
