"""Rendering utilities for shell-safe command hints.

Hints that end in a command the user is meant to paste must render every
interpolated value so the pasted text means what it displays. This module is a
pure leaf with no ``uv_stack`` imports so all three layers can reach it: the
obvious alternative homes cannot serve, since ``render`` imports ``config``
(so ``config`` importing ``render`` would cycle) and ``commands`` imports
``runner``, which is the operations layer.
"""

from __future__ import annotations

import shlex


def render_positional_arg(value: str) -> str:
    """Render a value as a positional argument in a pasteable shell command.

    Shell quoting alone is insufficient: ``shlex.quote("--recreate")`` returns
    ``--recreate`` unchanged, because it contains no shell metacharacters. The
    shell then hands that to the CLI, which parses it as an option rather than
    as the positional argument the hint displays.

    Prefix ``-- `` when the value starts with ``-`` so the shell's argument
    separator restores the intended reading. Emit the separator only when
    needed, so ordinary names keep reading cleanly.

    :param value: The argument value to render.
    :returns: The shell-safe positional argument form.
    """
    quoted = shlex.quote(value)
    if value.startswith("-"):
        return f"-- {quoted}"
    return quoted
