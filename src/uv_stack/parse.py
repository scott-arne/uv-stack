"""Shared parsing helpers for the env ``.txt`` config files.

Config files use a simple line-oriented grammar: ``#`` starts a comment,
surrounding whitespace is insignificant, and blank lines are ignored.
"""

from __future__ import annotations

from pathlib import Path


def clean_line(line: str) -> str:
    """Strip a trailing ``#`` comment and surrounding whitespace.

    :param line: A raw line from a config file.
    :returns: The cleaned content, or an empty string if the line is blank or
        a pure comment.
    """
    text, _, _ = line.partition("#")
    return text.strip()


def read_clean_lines(path: Path) -> list[str]:
    """Read a file and return its non-empty, comment-stripped lines.

    :param path: File to read.
    :returns: Cleaned lines in order; an empty list if the file does not exist.
    """
    if not path.is_file():
        return []
    cleaned = (clean_line(raw) for raw in path.read_text().splitlines())
    return [line for line in cleaned if line]


def first_clean_line(path: Path, default: str = "") -> str:
    """Return the first non-empty cleaned line of a file.

    :param path: File to read.
    :param default: Value returned when the file is missing or has no content.
    :returns: The first cleaned line, or ``default``.
    """
    lines = read_clean_lines(path)
    return lines[0] if lines else default
