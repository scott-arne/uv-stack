"""``stack diff``: compare two environments across the layers uv-stack records.

Rendering note: the two source labels are user-supplied paths that may contain
brackets, and :func:`render_table` passes column headers to rich unescaped
(``_render.py:126``). So the labels go in an :func:`echo` legend — plain click
output, which never parses markup — and the table headers stay the literal
``A`` and ``B``.
"""

from __future__ import annotations

import json
import sys
from itertools import zip_longest
from typing import Any

import rich_click as click

from uv_stack.cli._render import echo, render_table
from uv_stack.config import ConfigRoot
from uv_stack.operations.diff import (
    VERDICT_DIFFERENT,
    EnvironmentDiff,
    diff_environments,
    load_source,
)

#: Shown for the side of a pin table row where the package is absent.
_ABSENT = "absent"
#: Shown as the version of an editable or unnamed requirement, which has none.
_NO_VERSION = "(no version)"


def _payload(result: EnvironmentDiff) -> dict[str, Any]:
    """Build the ``--json`` document.

    Every key is always present; a layer a bare lock file hid is ``null``
    rather than absent, so a consumer never has to tell "missing" apart from
    "matching". Difference collections are already sorted by identity;
    ``channels`` is not sorted, because its order *is* the resolution priority
    the comparison just reported on.

    :param result: The comparison.
    :returns: A JSON-serializable document.
    """
    return {
        "a": result.a,
        "b": result.b,
        "verdict": result.verdict,
        "python": None
        if result.python is None
        else {"a": result.python[0], "b": result.python[1]},
        "micromamba": None
        if result.micromamba is None
        else {
            "only_in_a": result.micromamba.only_in_a,
            "only_in_b": result.micromamba.only_in_b,
        },
        "channels": None
        if result.channels is None
        else {"a": result.channels[0], "b": result.channels[1]},
        "pins": {
            "only_in_a": [
                {"name": entry.name, "version": entry.version}
                for entry in result.pins.only_in_a
            ],
            "only_in_b": [
                {"name": entry.name, "version": entry.version}
                for entry in result.pins.only_in_b
            ],
            "version_differs": [
                {"name": change.name, "a": change.a, "b": change.b}
                for change in result.pins.version_differs
            ],
        },
    }


def _render(result: EnvironmentDiff) -> None:
    """Print the comparison as one table per non-empty layer.

    A layer that matches gets a single summary line rather than an empty
    table, and a layer no source could supply says so.

    :param result: The comparison.
    """
    echo(f"A = {result.a}")
    echo(f"B = {result.b}")

    if result.python is None:
        echo(
            "Interpreter, micromamba packages and channels not compared: a bare "
            "lock file records none of them."
        )
    else:
        left, right = result.python
        if left == right:
            echo(f"python: identical ({left})")
        else:
            render_table(
                "python",
                [("Source", "left"), ("Version", "left")],
                [("A", left), ("B", right)],
            )

    if result.micromamba is not None:
        if result.micromamba.is_empty():
            echo("micromamba packages: identical")
        else:
            rows: list[tuple[str, ...]] = [
                (entry, "A") for entry in result.micromamba.only_in_a
            ]
            rows += [(entry, "B") for entry in result.micromamba.only_in_b]
            render_table(
                "micromamba packages", [("Package", "left"), ("Only in", "left")], rows
            )

    if result.channels is not None:
        left_channels, right_channels = result.channels
        if left_channels == right_channels:
            echo("channels: identical")
        else:
            render_table(
                "channels",
                [("A", "left"), ("B", "left")],
                [
                    (one or "", two or "")
                    for one, two in zip_longest(left_channels, right_channels)
                ],
            )

    if result.pins.is_empty():
        echo("pins: identical")
    else:
        pin_rows: list[tuple[str, ...]] = [
            (entry.name, entry.version or _NO_VERSION, _ABSENT)
            for entry in result.pins.only_in_a
        ]
        pin_rows += [
            (entry.name, _ABSENT, entry.version or _NO_VERSION)
            for entry in result.pins.only_in_b
        ]
        pin_rows += [
            (change.name, change.a, change.b) for change in result.pins.version_differs
        ]
        render_table("pins", [("Package", "left"), ("A", "left"), ("B", "left")], pin_rows)

    echo(f"Verdict: {result.verdict}")


@click.command("diff")
@click.argument("sources", nargs=2)
@click.option("--json", "as_json", is_flag=True, help="Emit the comparison as JSON.")
@click.option(
    "--exit-code",
    is_flag=True,
    help="Exit 1 when the two sources differ, like 'git diff --exit-code'.",
)
@click.pass_obj
def diff(config: ConfigRoot, sources: tuple[str, str], as_json: bool, exit_code: bool) -> None:
    """Compare two environments: interpreter, conda packages, channels, and pins.

    Each SOURCE is an environment in this config root, a path to a copied
    envs/<name>/ directory, or a path to a compiled lock file. A bare lock file
    carries pins only; the other three layers are reported as not compared
    rather than as matching.

    Exits 0 even when the sources differ — a difference is a finding, not a
    failure. Pass '--exit-code' to exit 1 on a difference instead.
    """
    left, right = sources
    result = diff_environments(load_source(config, left), load_source(config, right))
    if as_json:
        echo(json.dumps(_payload(result), indent=2))
    else:
        _render(result)
    if exit_code and result.verdict == VERDICT_DIFFERENT:
        sys.exit(1)
