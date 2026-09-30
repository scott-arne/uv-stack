"""Shared parsing helpers for the env ``.txt`` config files.

Config files use a simple line-oriented grammar: ``#`` starts a comment,
surrounding whitespace is insignificant, and blank lines are ignored.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from uv_stack.fsutil import read_text_utf8


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
    :raises ConfigError: When the file is not valid UTF-8.
    """
    if not path.is_file():
        return []
    cleaned = (clean_line(raw) for raw in read_text_utf8(path).splitlines())
    return [line for line in cleaned if line]


def first_clean_line(path: Path, default: str = "") -> str:
    """Return the first non-empty cleaned line of a file.

    :param path: File to read.
    :param default: Value returned when the file is missing or has no content.
    :returns: The first cleaned line, or ``default``.
    :raises ConfigError: When the file is not valid UTF-8.
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
    paths/archives/direct references (containing ``/`` or ``\\`` in the
    requirement head) — return ``None``; callers must never auto-remove those.
    An environment marker is not part of the name and must not make a plain
    requirement look like a path.

    :param requirement: A requirement string such as ``pkg[extra]>=1``.
    :returns: The leading distribution name, or ``None``.
    """
    req = requirement.strip()
    if not req or req.startswith("-"):
        return None
    req = req.split(";", 1)[0].strip()
    if not req or "/" in req or "\\" in req:
        return None
    for index, char in enumerate(req):
        if char in _NAME_TERMINATORS:
            return req[:index].strip() or None
    return req


def ownership_name(entry: str) -> str | None:
    """Extract the distribution name for ownership tracking.

    Handles PEP 508 direct references (``pkg @ url``) by extracting the name
    before the ``@``. VCS entries (``git+https://...``), editables (``-e``),
    and paths (containing ``/`` or ``\\`` in the requirement head) return
    ``None``.

    :param entry: A ledger entry or requirement string.
    :returns: The distribution name for ownership comparison, or ``None``.
    """
    entry = entry.strip()
    if "@" in entry:
        # PEP 508 direct reference: "name @ url" or "name[extras] @ url"
        head = entry.split("@", 1)[0].strip()
        return requirement_name(head)
    return requirement_name(entry)


def editable_target(entry: str) -> str | None:
    """The local path an editable entry installs from, if it has one.

    An entry counts as a local editable when its first whitespace-separated
    token is ``-e`` or ``--editable``, alone or with the operand attached by
    ``=``, and the value is either a plain path or a local ``file:`` URL: one
    with an empty or ``localhost`` authority and an absolute path, such as
    ``file:///src/pkg`` or ``file:/src/pkg``. A local file URL yields the path
    it names, percent-decoded, without its query or fragment. Any other URL,
    including a file URL naming another host or a relative path, is a remote
    install with no path to check.

    The operand is extracted verbatim from the original entry to preserve
    interior whitespace exactly as written — uv reads ``-e ./my  pkg`` as the
    single path ``my  pkg`` with two spaces, and ``split()`` would discard the
    run length. In both the attached (``-e=PATH``) and separated (``-e PATH``)
    forms, the operand runs to the end of the entry, except that it stops at
    the first whitespace run preceding a token that begins with ``-`` or ``#``.
    A requirements file treats ``#`` as a comment marker at the start of a line
    or after whitespace, so ``-e ./pkg # note`` installs from ``./pkg``, while
    a ``#`` inside a token is ordinary text and ``-e ./pkg#1`` installs from
    ``./pkg#1``.

    A trailing PEP 508 extras suffix is dropped from the operand; in a file
    URL that happens before decoding, so ``%5Bdev%5D`` stays in the path.

    :param entry: One expanded requirement entry.
    :returns: The path operand, or ``None``.
    """
    stripped = entry.strip()
    if not stripped:
        return None
    parts = stripped.split()
    # uv is what consumes these entries, so its parser sets the boundary: it
    # accepts '-e=PATH' and '--editable=PATH' as readily as the separated
    # forms, but refuses '-ePATH' with "Expected '=' or whitespace". Reading a
    # path out of the glued form would report a missing checkout for an entry
    # that cannot install for an entirely different reason.
    flag, attached, operand_start = parts[0].partition("=")
    if flag not in ("-e", "--editable"):
        return None
    # An attached '=' with nothing after it is still a separator to uv, which
    # reads '-e= PATH' exactly as '-e PATH'. Treating the empty operand as the
    # value would drop a checkout doctor is supposed to be watching.
    if attached and operand_start:
        # The attached form: everything after the '=' in the original entry.
        # The operand_start from partition is only what sat in the first
        # token, so we slice the stripped entry to get the whole remainder.
        remainder = stripped[len(flag) + 1 :]
    elif len(parts) >= 2:
        # The separated form: everything after the flag and its trailing
        # whitespace. We slice from the original entry rather than rejoining
        # split() to preserve interior whitespace exactly as written.
        flag_text = parts[0]
        remainder = stripped[len(flag_text) :].lstrip()
        # When the remainder begins with a comment, the operand is empty. The
        # lstrip() ensures a leading '#' here genuinely followed whitespace, so
        # it is a comment marker per the requirements file line semantics rather
        # than ordinary text inside a token. In '-e=#note' the '#' is glued to
        # the '=' with no space, so the attached branch keeps it as a path. uv
        # sees a bare '-e' when the separated operand is only a comment, which
        # is a malformed entry, not a checkout.
        if remainder.startswith("#"):
            return None
    else:
        return None
    # uv reads everything after the flag as one path, but '-e ./my pkg --opt'
    # stops the path at the option, and '-e ./pkg # note' stops the path at the
    # comment. We scan the remainder for the first whitespace run followed by
    # a token starting with '-' or '#', and cut there. A requirements file
    # treats '#' as a comment marker at the start of a line or after whitespace,
    # not inside a token, so './pkg#1' keeps its '#'. An option or comment
    # cannot be part of a path uv would accept here, and a path that genuinely
    # begins with '-' is the first token, so this rule applies only to later
    # tokens.
    target = remainder
    for index, char in enumerate(remainder):
        if char.isspace():
            after = index
            while after < len(remainder) and remainder[after].isspace():
                after += 1
            # The cut lands before the whitespace run rather than before the
            # '-' or '#': the run separates the two tokens and belongs to
            # neither, so keeping it would leave the path with a trailing space.
            if after < len(remainder) and remainder[after] in ("-", "#"):
                target = remainder[:index]
                break
    # A local file URL names a checkout on this machine exactly as a path does,
    # so passing over it as remote would let a missing one reach uv unchecked.
    # Only 'file:' is read this way; another scheme without '//' still falls
    # through as a literal path, as it always has.
    if target[:5].lower() == "file:":
        # urlsplit raises ValueError on a malformed authority such as an
        # unclosed '['. uv cannot install from that either, so it is left to
        # uv's own error rather than raised out of doctor or the pre-flight.
        try:
            url = urlsplit(target)
        except ValueError:
            return None
        if url.netloc.lower() not in ("", "localhost") or not url.path.startswith("/"):
            return None
        # The extras suffix is stripped before decoding, and not again after:
        # a URL spells a bracket that belongs to the directory name as '%5B',
        # so a bracket that appears only once decoded is part of the path.
        path = url.path
        if path.endswith("]") and "[" in path:
            path = path[: path.rindex("[")]
        return unquote(path)
    elif "://" in target or target.startswith("git+"):
        return None
    # pip reads '-e ./pkg[dev]' as the path './pkg' carrying extras, so probing
    # the whole token would report a checkout that is present as missing. The
    # suffix is stripped rather than parsed: the operand is a path here, and
    # nothing downstream has any use for the extras names.
    if target.endswith("]") and "[" in target:
        target = target[: target.rindex("[")]
    return target or None
