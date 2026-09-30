"""Rendering utilities for shell-safe command hints.

Hints that end in a command the user is meant to paste must render every
interpolated value so the pasted text means what it displays. It also holds
the control-character helpers that terminal output needs, since a name read
from disk or from a document can carry an escape sequence. This module is a
pure leaf with no ``uv_stack`` imports so all three layers can reach it: the
obvious alternative homes cannot serve, since ``render`` imports ``config``
(so ``config`` importing ``render`` would cycle) and ``commands`` imports
``runner``, which is the operations layer.
"""

from __future__ import annotations

import shlex
import unicodedata


def has_control(text: str) -> bool:
    """Return whether text holds a Unicode control character (category ``Cc``).

    That is U+0000-U+001F, U+007F, and U+0080-U+009F: the characters a
    terminal can read as the start of an escape sequence.

    :param text: The text to check.
    :returns: ``True`` when any character is a control character.
    """
    return any(unicodedata.category(char) == "Cc" for char in text)


def escape_controls(text: str) -> str:
    """Spell each control character in text as its Python escape.

    A message that must name a value holding a control character can then
    print it without the terminal acting on it. Every other character is
    kept, so text without a control character comes back unchanged.

    :param text: The text to render.
    :returns: The text with, for example, ESC as ``\\x1b`` and LF as ``\\n``.
    """
    return "".join(
        char.encode("unicode_escape").decode("ascii")
        if unicodedata.category(char) == "Cc"
        else char
        for char in text
    )


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
