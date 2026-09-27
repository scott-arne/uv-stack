"""``stack diff``: compare two environments across the layers uv-stack records.

A *source* is an environment in this config root, a copied ``envs/<name>/``
directory, or a bare lock file. The first two carry all four layers; a bare
lock carries pins only and reports the rest as unavailable rather than as
matching, so the output never claims more than it checked.
"""

from __future__ import annotations

import errno
import os
import re
import stat as stat_module
from dataclasses import dataclass
from pathlib import Path

from uv_stack.config import ConfigRoot, load_env_from_dir
from uv_stack.errors import ConfigError
from uv_stack.hints import render_positional_arg
from uv_stack.operations.scaffold import validate_name
from uv_stack.parse import canonical_name
from uv_stack.render import conda_name_key, effective_conda_inputs

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


def _describe(result: os.stat_result) -> str:
    """:returns: A human name for a non-regular file's kind."""
    mode = result.st_mode
    if stat_module.S_ISDIR(mode):
        return "a directory"
    if stat_module.S_ISFIFO(mode):
        return "a named pipe"
    if stat_module.S_ISSOCK(mode):
        return "a socket"
    return "not a regular file"


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
    # Unnamed requirement: a bare path or URL that uv pip compile passes through.
    if not line.startswith("-") and not any(ch.isspace() for ch in line) and "/" in line:
        return line, None
    raise ConfigError(
        f"{path}, line {number}: cannot parse '{line}' as a requirement.",
        hint=(
            "Expected 'name==version', 'name @ url', '-e target', a path or URL, "
            "an option line, or a comment."
        ),
    )


def parse_lock(path: Path) -> dict[str, str | None]:
    """Read a compiled lock file into identity-to-version pairs.

    :param path: The lock file.
    :returns: Canonical identity to version, URL, or ``None`` for an editable
        or unnamed requirement.
    :raises ConfigError: On an unreadable file, a non-regular file, an
        unparseable line, or a duplicate identity — which a compiled lock never
        contains, and whose silent last-wins resolution would hide that the
        file is not one.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as error:
        raise ConfigError(
            f"Cannot read {path}: {error.strerror or error}.",
            hint="Check the file's permissions, or pass a different lock file.",
        ) from error

    try:
        result = os.fstat(fd)
        if not stat_module.S_ISREG(result.st_mode):
            raise ConfigError(
                f"{path} is {_describe(result)}, not a lock file.",
                hint="Replace it with a compiled lock file.",
            )
        handle = os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise

    with handle:
        try:
            raw_bytes = handle.read()
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ConfigError(
                f"Cannot read {path}: not valid UTF-8.",
                hint="Re-save the file as UTF-8 text.",
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


def _stat_or_none(path: Path) -> os.stat_result | None:
    """Stat ``path``, telling absence apart from an unreadable parent.

    ``Path.is_file`` cannot make this distinction: on an unsearchable parent it
    propagates ``PermissionError`` on 3.13 and answers ``False`` elsewhere, and
    ``try/except OSError`` cannot recover the ``False`` case because there is
    no exception to catch. A permission problem on ``envs/`` is a real error
    for a command that has to read there.

    :param path: The path to stat.
    :returns: The stat result, or ``None`` when the path does not exist.
    :raises ConfigError: On any other stat failure.
    """
    try:
        return os.stat(path)
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ENOTDIR):
            return None
        raise ConfigError(
            f"Cannot read {path}: {error.strerror or error}.",
            hint="Check the permissions on the directories leading to it.",
        ) from error


def _is_regular(result: os.stat_result | None) -> bool:
    """:returns: True when ``result`` describes an existing regular file."""
    return result is not None and stat_module.S_ISREG(result.st_mode)


def _environment_name(config: ConfigRoot, argument: str) -> str | None:
    """:returns: ``argument`` when it names an environment in ``config``, else None.

    The name check runs first: ``ConfigRoot.env_dir`` is an unchecked join, so
    an absolute or ``..``-bearing argument would leave ``envs/`` before
    anything looked at it. The probe is then keyed on ``stack.txt`` rather than
    on directory existence, so a stray ``envs/junk/`` is not an environment
    here either — matching ``env_exists``, ``list_envs`` and ``stack status``.
    """
    try:
        validate_name("environment", argument)
    except ConfigError:
        return None
    if not _is_regular(_stat_or_none(config.env_dir(argument) / "stack.txt")):
        return None
    return argument


def _directory_source(directory: Path, name: str, label: str, *, named: bool) -> DiffSource:
    """Build a source from an environment directory.

    :param directory: The directory holding the four sources and the lock.
    :param name: The name to record on the loaded :class:`EnvConfig`.
    :param label: The argument as typed, echoed in the output.
    :param named: True for an environment in this root, whose missing lock has
        a local remedy; False for a copied directory, whose remedy is on the
        machine it came from.
    :returns: A fully populated source.
    :raises ConfigError: When the environment was never built, or when the lock
        is not a regular file.
    :raises OSError: When one of the four source files is unreadable.
    """
    env = load_env_from_dir(directory, name)
    lock = directory / "requirements.lock.txt"
    if _stat_or_none(lock) is None:
        if named:
            raise ConfigError(
                f"Environment '{name}' has no lock at {lock}.",
                hint=f"Build it first: stack sync env {render_positional_arg(name)}",
            )
        raise ConfigError(
            f"No lock at {lock}.",
            hint=(
                "The directory was copied before the environment was built. "
                "Build it on the machine it came from and copy it again."
            ),
        )
    channels, dependencies = effective_conda_inputs(env)
    packages = [
        spec for spec in dependencies if conda_name_key(spec) not in _PRESEEDED_CONDA_KEYS
    ]
    return DiffSource(
        label=label,
        pins=parse_lock(lock),
        python=env.python,
        micromamba=packages,
        channels=channels,
    )


def load_source(config: ConfigRoot, argument: str) -> DiffSource:
    """Classify one ``stack diff`` argument and read it.

    Tried in order: an environment in this config root, a directory containing
    ``stack.txt``, a bare lock file.

    :param config: Configuration root.
    :param argument: The argument as typed.
    :returns: The source, with unavailable layers left as ``None``.
    :raises ConfigError: When the argument is none of the three, when it is an
        existing path that is not a regular file, or when reading it fails.
    """
    name = _environment_name(config, argument)
    if name is not None:
        return _directory_source(config.env_dir(name), name, argument, named=True)

    path = Path(argument)
    if _is_regular(_stat_or_none(path / "stack.txt")):
        return _directory_source(path, path.name or argument, argument, named=False)

    result = _stat_or_none(path)
    if result is None:
        raise ConfigError(
            f"No source named {argument}.",
            hint=(
                "Tried it as an environment in this config root, as a "
                "directory containing stack.txt, and as a lock file path; "
                "none of the three exists."
            ),
        )
    if not stat_module.S_ISREG(result.st_mode):
        # Reached before any read: a directory here would fail with
        # IsADirectoryError, and a FIFO would block forever on a read that
        # never returns.
        raise ConfigError(
            f"{path} is {_describe(result)}, not a lock file.",
            hint=(
                "Pass an environment name, a directory containing stack.txt, "
                "or a path to a compiled lock file."
            ),
        )
    return DiffSource(
        label=argument, pins=parse_lock(path), python=None, micromamba=None, channels=None
    )


def _diff_pins(a: dict[str, str | None], b: dict[str, str | None]) -> PinDiff:
    """Compare two identity-to-version maps.

    :param a: Source A's pins.
    :param b: Source B's pins.
    :returns: The three difference groups, each sorted by identity.
    """
    only_in_a = [PinEntry(name, a[name]) for name in sorted(a.keys() - b.keys())]
    only_in_b = [PinEntry(name, b[name]) for name in sorted(b.keys() - a.keys())]
    version_differs: list[PinChange] = []
    for name in sorted(a.keys() & b.keys()):
        left, right = a[name], b[name]
        if left == right:
            continue
        # Unreachable for an editable: its identity embeds its target, so two
        # editables sharing an identity also share a None version. The guard
        # keeps the dataclass's str contract honest rather than asserting it.
        if left is None or right is None:
            continue
        version_differs.append(PinChange(name, left, right))
    return PinDiff(only_in_a=only_in_a, only_in_b=only_in_b, version_differs=version_differs)


def diff_environments(a: DiffSource, b: DiffSource) -> EnvironmentDiff:
    """Compare two sources across every layer both of them carry.

    :param a: Source A.
    :param b: Source B.
    :returns: The four-layer comparison and its verdict.
    """
    pins = _diff_pins(a.pins, b.pins)
    # Tested field by field rather than through a local flag so that mypy
    # narrows all six attributes for the comparison below the return.
    if (
        a.python is None
        or b.python is None
        or a.micromamba is None
        or b.micromamba is None
        or a.channels is None
        or b.channels is None
    ):
        verdict = VERDICT_DIFFERENT if not pins.is_empty() else VERDICT_WHERE_COMPARABLE
        return EnvironmentDiff(
            a=a.label,
            b=b.label,
            verdict=verdict,
            python=None,
            micromamba=None,
            channels=None,
            pins=pins,
        )

    micromamba = PackageDiff(
        only_in_a=sorted(set(a.micromamba) - set(b.micromamba)),
        only_in_b=sorted(set(b.micromamba) - set(a.micromamba)),
    )
    differs = (
        not pins.is_empty()
        or not micromamba.is_empty()
        or a.python != b.python
        or a.channels != b.channels
    )
    return EnvironmentDiff(
        a=a.label,
        b=b.label,
        verdict=VERDICT_DIFFERENT if differs else VERDICT_IDENTICAL,
        python=(a.python, b.python),
        micromamba=micromamba,
        channels=(a.channels, b.channels),
        pins=pins,
    )
