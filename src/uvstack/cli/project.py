"""``uvstack project`` commands."""

from __future__ import annotations

import rich_click as click


@click.group()
def project() -> None:
    """Create local uv projects from a stack."""
