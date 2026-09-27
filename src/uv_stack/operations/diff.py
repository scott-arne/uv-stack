"""``stack diff``: compare two environments across the layers uv-stack records.

A *source* is an environment in this config root, a copied ``envs/<name>/``
directory, or a bare lock file. The first two carry all four layers; a bare
lock carries pins only and reports the rest as unavailable rather than as
matching, so the output never claims more than it checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from uv_stack.errors import ConfigError
from uv_stack.fsutil import read_text_utf8
from uv_stack.parse import canonical_name

#: All four comparisons ran and all four were empty.
VERDICT_IDENTICAL = "identical"
#: Every comparison that ran was empty, but a bare lock hid at least one layer.
VERDICT_WHERE_COMPARABLE = "identical-where-comparable"
#: At least one comparison that ran was non-empty.
VERDICT_DIFFERENT = "different"

#: A resolved pin. ``\S+`` for the version so a local version segment or an
#: epoch survives verbatim; uv-stack never rewrites what uv compiled.
_PIN_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>\S+)$")
#: A PEP 508 direct reference as ``uv pip compile`` writes it, spaced.
_DIRECT_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)\s+@\s+(?P<url>\S.*)$")
#: pip's comment rule: a ``#`` at line start or after whitespace. Deliberately
#: not ``parse.clean_line``, which partitions at the first ``#`` and would
#: truncate a ``#sha256=`` fragment inside a direct reference URL.
_COMMENT_RE = re.compile(r"(^|\s)#.*$")
#: Editable spellings, longest first so ``--editable=`` is not shadowed.
_EDITABLE_PREFIXES = ("--editable=", "--editable ", "-e ")
#: Entries the interpreter layer already reports; printing them in the
#: micromamba layer would report one difference twice.
_PRESEEDED_CONDA_KEYS = frozenset({"python", "pip"})


@dataclass(frozen=True)
class PinEntry:
    """One pin present in exactly one source.

    :param name: The canonical distribution name, or an ``-e <target>``
        identity for an editable.
    :param version: The version, the direct-reference URL, or ``None`` for an
        editable.
    """

    name: str
    version: str | None


@dataclass(frozen=True)
class PinChange:
    """One identity present in both sources at differing versions.

    Both sides are always strings: the only line form with a ``None`` version
    is an editable, whose identity embeds its target, so two editables that
    share an identity also share a version.

    :param name: The shared canonical identity.
    :param a: The version (or URL) in source A.
    :param b: The version (or URL) in source B.
    """

    name: str
    a: str
    b: str


@dataclass(frozen=True)
class PinDiff:
    """The pin layer's three difference groups.

    :param only_in_a: Pins present only in source A.
    :param only_in_b: Pins present only in source B.
    :param version_differs: Identities present in both at differing versions.
    """

    only_in_a: list[PinEntry]
    only_in_b: list[PinEntry]
    version_differs: list[PinChange]

    def is_empty(self) -> bool:
        """:returns: True when no pin differs."""
        return not (self.only_in_a or self.only_in_b or self.version_differs)


@dataclass(frozen=True)
class PackageDiff:
    """An unordered layer's two difference groups.

    :param only_in_a: Entries present only in source A.
    :param only_in_b: Entries present only in source B.
    """

    only_in_a: list[str]
    only_in_b: list[str]

    def is_empty(self) -> bool:
        """:returns: True when no entry differs."""
        return not (self.only_in_a or self.only_in_b)


@dataclass(frozen=True)
class DiffSource:
    """One side of a comparison.

    ``python``, ``micromamba`` and ``channels`` are all ``None`` together:
    either the source is an environment and carries all three, or it is a bare
    lock file and carries none of them.

    :param label: The argument as the user typed it, echoed in the output.
    :param pins: Canonical identity to version, URL, or ``None``.
    :param python: The declared interpreter spec, or ``None``.
    :param micromamba: Effective conda packages, or ``None``.
    :param channels: Effective channels in priority order, or ``None``.
    """

    label: str
    pins: dict[str, str | None]
    python: str | None
    micromamba: list[str] | None
    channels: list[str] | None


@dataclass(frozen=True)
class EnvironmentDiff:
    """The four-layer comparison of two sources.

    :param a: Source A's label.
    :param b: Source B's label.
    :param verdict: One of the three ``VERDICT_*`` constants.
    :param python: Both declared interpreters, or ``None`` when unavailable.
    :param micromamba: The conda package difference, or ``None``.
    :param channels: Both effective channel lists, or ``None``.
    :param pins: The pin difference; always available.
    """

    a: str
    b: str
    verdict: str
    python: tuple[str, str] | None
    micromamba: PackageDiff | None
    channels: tuple[list[str], list[str]] | None
    pins: PinDiff


def _strip_comment(line: str) -> str:
    """:returns: ``line`` with any pip-style comment and surrounding space removed."""
    return _COMMENT_RE.sub("", line).strip()


def _editable_target(line: str) -> str | None:
    """:returns: The target of an editable line, or ``None`` if it is not one."""
    for prefix in _EDITABLE_PREFIXES:
        if line.startswith(prefix):
            target = line[len(prefix) :].strip()
            return target or None
    return None


def _parse_lock_line(raw: str, path: Path, number: int) -> tuple[str, str | None] | None:
    """Classify one lock line into exactly one grammar row.

    Rows are tried in a fixed order, and the editable row before the generic
    ``--`` row: ``--editable /src/tool`` matches both, so without an order a
    valid editable would silently vanish from the comparison.

    :param raw: The line as read, without its newline.
    :param path: The file, for error messages.
    :param number: The 1-based line number, for error messages.
    :returns: ``(identity, version)``, or ``None`` for a discarded line.
    :raises ConfigError: On a continued line or a line matching no row.
    """
    if raw.rstrip().endswith("\\"):
        raise ConfigError(
            f"{path}, line {number}: line continuations are not supported.",
            hint=(
                "This looks like 'uv pip compile --generate-hashes' output. "
                "uv-stack never generates hashed locks and cannot compare one."
            ),
        )
    line = _strip_comment(raw)
    if not line:
        return None
    target = _editable_target(line)
    if target is not None:
        return f"-e {target}", None
    if line.startswith("--"):
        return None
    pin = _PIN_RE.match(line)
    if pin is not None:
        return canonical_name(pin.group("name")), pin.group("version")
    direct = _DIRECT_RE.match(line)
    if direct is not None:
        return canonical_name(direct.group("name")), direct.group("url")
    raise ConfigError(
        f"{path}, line {number}: cannot parse '{line}' as a requirement.",
        hint=(
            "Expected 'name==version', 'name @ url', '-e target', an option "
            "line, or a comment."
        ),
    )


def parse_lock(path: Path) -> dict[str, str | None]:
    """Read a compiled lock file into identity-to-version pairs.

    :param path: The lock file.
    :returns: Canonical identity to version, URL, or ``None`` for an editable.
    :raises ConfigError: On an unreadable file, an unparseable line, or a
        duplicate identity — which a compiled lock never contains, and whose
        silent last-wins resolution would hide that the file is not one.
    """
    try:
        text = read_text_utf8(path)
    except OSError as error:
        # read_text_utf8 converts only a decode failure. A permission error
        # would otherwise reach the CLI edge as a traceback for what is really
        # "this file cannot be read".
        raise ConfigError(
            f"Cannot read {path}: {error.strerror or error}.",
            hint="Check the file's permissions, or pass a different lock file.",
        ) from error
    pins: dict[str, str | None] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        parsed = _parse_lock_line(raw, path, number)
        if parsed is None:
            continue
        identity, version = parsed
        if identity in pins:
            raise ConfigError(
                f"{path}, line {number}: '{identity}' appears more than once.",
                hint="A compiled lock file contains each requirement exactly once.",
            )
        pins[identity] = version
    return pins
