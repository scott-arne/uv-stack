"""Shared parsing helpers for the env ``.txt`` config files.

Config files use a simple line-oriented grammar: ``#`` starts a comment,
surrounding whitespace is insignificant, and blank lines are ignored.
"""

from __future__ import annotations

import re
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


#: Characters that terminate a distribution name inside a requirement string.
_NAME_TERMINATORS = "[<>=!~;@ \t"

_CANONICAL_RE = re.compile(r"[-_.]+")


def canonical_name(name: str) -> str:
    """PEP 503-normalized distribution name (lowercase, runs of ``-_.`` → ``-``).

    :param name: A distribution name as written.
    :returns: The canonical form used for ownership comparisons.
    """
    return _CANONICAL_RE.sub("-", name).lower()


def requirement_name(requirement: str) -> str | None:
    """Return the distribution name of a plain requirement string.

    Entries that are not plain names — flags/editables (leading ``-``) and
    paths/archives/direct references (containing ``/`` or ``\\``) — return
    ``None``; callers must never auto-remove those.

    :param requirement: A requirement string such as ``pkg[extra]>=1``.
    :returns: The leading distribution name, or ``None``.
    """
    req = requirement.strip()
    if not req or req.startswith("-") or "/" in req or "\\" in req:
        return None
    for index, char in enumerate(req):
        if char in _NAME_TERMINATORS:
            return req[:index].strip() or None
    return req


def ownership_name(entry: str) -> str | None:
    """Extract the distribution name for ownership tracking.

    Handles PEP 508 direct references (``pkg @ url``) by extracting the name
    before the ``@``. VCS entries (``git+https://...``), editables (``-e``),
    and paths (containing ``/`` or ``\\``) return ``None``.

    :param entry: A ledger entry or requirement string.
    :returns: The distribution name for ownership comparison, or ``None``.
    """
    entry = entry.strip()
    if "@" in entry:
        # PEP 508 direct reference: "name @ url" or "name[extras] @ url"
        head = entry.split("@", 1)[0].strip()
        return requirement_name(head)
    return requirement_name(entry)
