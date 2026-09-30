"""Build a portable export document from a config root.

The document ships config files verbatim, keyed by their path under the
root, plus each exported environment's lock as a seed. Import re-derives
everything else (token meaning, variable names) from these files, so the
document carries no derived fields that could disagree with them.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from uv_stack import __version__
from uv_stack.config import ConfigRoot, _parse_declarations
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.fsutil import read_text_utf8, require_regular_file
from uv_stack.hints import escape_controls
from uv_stack.models import EXPORT_FORMAT, EXPORT_VERSION, ExportDocument
from uv_stack.operations.diff import parse_lock_text, read_lock_text
from uv_stack.operations.scaffold import validate_name
from uv_stack.parse import editable_target, read_clean_lines
from uv_stack.resolver import Resolver
from uv_stack.variables import UNDECLARED_HINT, referenced_names

ITEM_KINDS = ("env", "profile", "bundle")
ITEM_NOUNS = {"env": "environment", "profile": "profile", "bundle": "bundle"}
ENV_FILES = ("stack.txt", "python.txt", "micromamba.txt", "channels.txt")
OPTIONAL_ENV_FILES = ENV_FILES[1:]

_KEY_RE = re.compile(
    r"(?P<dir>profiles|bundles)/(?P<stem>[^/]+)\.yaml"
    r"|envs/(?P<env>[^/]+)/(?P<file>stack|python|micromamba|channels)\.txt"
)
_MISSING_HINT = "Run 'stack list' to see what this root has."


class FileKey(NamedTuple):
    """A parsed document file key."""

    kind: str
    name: str
    filename: str | None


class AmbiguousItemError(ConfigError):
    """A bare ITEM names more than one kind; the CLI exits 2 on it."""


def parse_file_key(key: str) -> FileKey | None:
    """Parse a document file key, or return ``None`` if it is not one.

    Keys are matched whole and each name passes :func:`validate_name`, so a
    ``../`` segment, an absolute path, an unlisted file, or a control
    character (NUL among them) never maps to a path under the root.
    ``fullmatch`` rather than a ``$`` anchor, because ``$`` also matches before
    a trailing newline that the key would still carry into the path.
    """
    match = _KEY_RE.fullmatch(key)
    if match is None:
        return None
    if match["dir"] is not None:
        kind, name, filename = match["dir"][:-1], match["stem"], None
    else:
        kind, name, filename = "env", match["env"], f"{match['file']}.txt"
    try:
        validate_name(kind, name)
    except UvStackError:
        return None
    return FileKey(kind, name, filename)


def file_key(kind: str, name: str, filename: str | None = None) -> str:
    """Return the document key for a profile, bundle, or env file."""
    if kind == "env":
        return f"envs/{name}/{filename or 'stack.txt'}"
    return f"{kind}s/{name}.yaml"


def _item_path(config: ConfigRoot, kind: str, name: str) -> Path:
    if kind == "env":
        return config.env_stack_path(name)
    if kind == "profile":
        return config.profile_path(name)
    return config.bundle_path(name)


def _item_exists(config: ConfigRoot, kind: str, name: str) -> bool:
    path = _item_path(config, kind, name)
    require_regular_file(path)
    return path.is_file()


def normalize_items(config: ConfigRoot, raw_items: Iterable[str]) -> list[str]:
    """Qualify, check, sort, and deduplicate the requested items.

    :param config: Source root.
    :param raw_items: ITEM arguments; empty means every item in the root.
    :returns: ``kind:name`` strings, sorted.
    :raises AmbiguousItemError: When a bare name matches more than one kind.
    :raises ConfigError: When an item is missing or malformed.
    """
    raw = list(raw_items)
    if not raw:
        # list_* return every stem on disk, unvalidated; a stray profiles/-x.yaml
        # must refuse here rather than ship a key import would reject.
        whole = [("env", n) for n in config.list_envs()]
        whole += [("profile", n) for n in config.list_profiles()]
        whole += [("bundle", n) for n in config.list_bundles()]
        for kind, name in whole:
            validate_name(kind, name)
        return sorted(f"{kind}:{name}" for kind, name in whole)
    items: set[str] = set()
    for token in raw:
        if token.startswith("@"):
            kind, name = "bundle", token[1:]
        elif ":" in token:
            kind, _, name = token.partition(":")
            if kind not in ITEM_KINDS:
                raise ConfigError(
                    f"Unknown item kind '{kind}' in '{token}'.",
                    hint="Use env:NAME, profile:NAME, bundle:NAME, or @NAME.",
                )
        else:
            validate_name("item", token)
            found = [k for k in ITEM_KINDS if _item_exists(config, k, token)]
            if not found:
                raise ConfigError(
                    f"No environment, profile, or bundle named '{token}'.", hint=_MISSING_HINT
                )
            if len(found) > 1:
                spelled = [f"{k}:{token}" for k in found]
                raise AmbiguousItemError(
                    f"'{token}' names more than one item: {', '.join(spelled)}.",
                    hint=f"Qualify it: {' or '.join(spelled)}.",
                )
            items.add(f"{found[0]}:{token}")
            continue
        validate_name(kind, name)
        if not _item_exists(config, kind, name):
            raise ConfigError(
                f"No {ITEM_NOUNS[kind]} named '{name}': {_item_path(config, kind, name)}",
                hint=_MISSING_HINT,
            )
        items.add(f"{kind}:{name}")
    return sorted(items)


def reference_key(kind: str, name: str) -> str:
    """Return the file key for a profile or bundle reference.

    :param kind: Either ``"profile"`` or ``"bundle"``.
    :param name: The reference name.
    :returns: The file key if valid.
    :raises ConfigError: When the reference does not map to a valid file key.
    """
    key = file_key(kind, name)
    if parse_file_key(key) is None:
        noun = kind.capitalize()
        raise ConfigError(
            f"{noun} reference '{escape_controls(name)}' does not map to a valid file key.",
            hint=f"A {kind} reference must be a plain name, not a path.",
        )
    return key


def closure(config: ConfigRoot, items: Iterable[str]) -> set[str]:
    """Return the file keys the normalized items reach.

    One resolve over every item's tokens gives the profiles and bundles
    reached; package literals bring no file. The resolver's missing-reference
    error for a qualified reference propagates.

    :raises ConfigError: When a profile or bundle reference escapes the root.
    """
    keys: set[str] = set()
    tokens: list[str] = []
    for item in items:
        kind, _, name = item.partition(":")
        if kind == "env":
            env = config.load_env(name)
            keys.add(file_key("env", name))
            for filename in OPTIONAL_ENV_FILES:
                if (config.env_dir(name) / filename).is_file():
                    keys.add(file_key("env", name, filename))
            tokens.extend(env.stack)
        else:
            tokens.append(item)
    resolved = Resolver(config).resolve(tokens)
    for name in resolved.profiles:
        keys.add(reference_key("profile", name))
    for name in resolved.bundles:
        keys.add(reference_key("bundle", name))
    return keys


def _spelled_as_stored(config: ConfigRoot, key: FileKey) -> bool:
    # listdir returns names as the directory stores them, so a spelling that
    # differs in letter case or Unicode normalization is not among them.
    if key.kind == "env":
        return key.name in os.listdir(config.envs_dir)
    path = _item_path(config, key.kind, key.name)
    return path.name in os.listdir(path.parent)


def _refuse_folded_aliases(config: ConfigRoot, keys: Iterable[str]) -> None:
    """Refuse two keys that reach one file when one is not the file's own name.

    A filesystem that folds letter case or Unicode normalization opens
    ``profiles/Foo.yaml`` for a reference spelled ``foo``, which would ship one
    file under two keys: a folding target refuses the document as aliased, and
    a case-sensitive one gets two definitions where this root has one. Two
    stored names for one file, a hard link or a symlink, are not refused. A
    lone key spelled unlike its file is not either, since its reference finds
    it on the target under that spelling.

    :raises ConfigError: When two keys name one file and either is not spelled
        as the directory entry it reaches.
    """
    by_file: dict[tuple[int, int], list[tuple[str, FileKey]]] = {}
    for key in sorted(keys):
        parsed = parse_file_key(key)
        assert parsed is not None  # closure only emits well-formed keys
        if parsed.kind == "env":
            path = config.env_dir(parsed.name) / (parsed.filename or "stack.txt")
        else:
            path = _item_path(config, parsed.kind, parsed.name)
        try:
            info = os.stat(path)
        except FileNotFoundError:
            continue
        by_file.setdefault((info.st_dev, info.st_ino), []).append((key, parsed))
    for group in by_file.values():
        if len(group) > 1 and not all(_spelled_as_stored(config, p) for _, p in group):
            raise ConfigError(
                f"{group[0][0]} and {group[1][0]} name the same file on this machine.",
                hint="This filesystem ignores letter case or Unicode normalization in names. "
                "Spell each reference exactly as its file is named.",
            )


def file_entries(config: ConfigRoot, key: FileKey) -> list[str]:
    """Return the requirement entries of a profile, bundle, or ``stack.txt``."""
    if key.kind == "profile":
        return list(config.load_profile(key.name).includes)
    if key.kind == "bundle":
        return list(config.load_bundle(key.name).includes)
    if key.filename == "stack.txt":
        return read_clean_lines(config.env_stack_path(key.name))
    return []


def read_seed(config: ConfigRoot, name: str) -> str | None:
    """Return an environment's lock text, or ``None`` when it has none.

    :raises ConfigError: When the lock is not a regular file or does not parse.
    """
    lock = config.env_requirements_lock(name)
    if not os.path.lexists(lock):
        return None
    text = read_lock_text(lock)
    parse_lock_text(text, str(lock))
    return text


def _check_declared(config: ConfigRoot, key: str, entries: list[str], declared: set[str]) -> None:
    missing = sorted({n for e in entries for n in referenced_names(e)} - declared)
    if missing:
        raise ConfigError(
            f"{key} references {len(missing)} name(s) that {config.variables_path()} "
            f"does not declare: {', '.join(missing)}.",
            hint=UNDECLARED_HINT,
        )


_FILE_URL = re.compile(r"(?:^|[\s=@])file://(\S*)")


def _file_url_path(entry: str) -> str | None:
    """Return what follows ``file://`` when an entry installs from a file URL.

    That is a bare URL, an editable's operand, or a ``name @ URL`` direct
    reference. Other options are skipped, as their plain-path forms are.
    """
    words = entry.split()
    if not words:
        return None
    if words[0].startswith("-") and words[0].partition("=")[0] not in ("-e", "--editable"):
        return None
    code = re.split(r"\s#", entry, maxsplit=1)[0]
    match = _FILE_URL.search(code)
    return None if match is None else match[1]


def _is_absolute_path(entry: str) -> bool:
    # A file URL always names an absolute path, but editable_target reads only
    # a local editable one, and the '://' test below passes over the rest as
    # a remote location.
    url_path = _file_url_path(entry)
    if url_path is not None:
        return "${" not in url_path
    target = editable_target(entry)
    if target is None:
        first = entry.split()[0] if entry.split() else ""
        if not first or first.startswith("-") or "://" in first:
            return False
        target = first
    if "${" in target:
        return False
    try:
        return os.path.isabs(os.path.expanduser(target))
    except ValueError:
        return False


def absolute_path_warnings(config: ConfigRoot, keys: Iterable[str]) -> list[str]:
    """Warn once per file that holds absolute editable or local paths."""
    warnings = []
    for key in sorted(keys):
        parsed = parse_file_key(key)
        if parsed is None:
            continue
        count = sum(_is_absolute_path(e) for e in file_entries(config, parsed))
        if count:
            warnings.append(
                f"{key}: {count} absolute editable or local path(s); these build only where "
                "the same paths exist. Use ${NAME} to make them portable."
            )
    return warnings


def source_platform() -> str:
    """Return this machine's platform tag, e.g. ``darwin-arm64``."""
    return f"{sys.platform}-{platform.machine()}"


@dataclass(frozen=True)
class ExportResult:
    """An export document and the warnings raised while building it."""

    document: ExportDocument
    warnings: list[str]


def build_document(config: ConfigRoot, raw_items: Iterable[str]) -> ExportResult:
    """Build the export document for the requested items.

    :raises ConfigError: For a missing or ambiguous item, a non-regular file,
        an undeclared variable reference, a malformed seed lock, or two keys
        that name one file only because this filesystem folds a spelling.
    """
    items = normalize_items(config, raw_items)
    keys = closure(config, items)
    # Checked here rather than in closure, which import also runs over the
    # document's files: there a reference that folds onto a shipped key is
    # already refused as reaching a file the document does not ship.
    _refuse_folded_aliases(config, keys)
    # Declarations only: the document ships no values, so a malformed
    # variables.local.txt or environment override must not stop an export.
    declared = set(_parse_declarations(config.variables_path()))
    files: dict[str, str] = {}
    for key in sorted(keys):
        path = config.root / key
        require_regular_file(path)
        files[key] = read_text_utf8(path, exact_newlines=True)
        parsed = parse_file_key(key)
        assert parsed is not None  # closure only emits well-formed keys
        _check_declared(config, key, file_entries(config, parsed), declared)
    seeds: dict[str, str] = {}
    for item in items:
        kind, _, name = item.partition(":")
        if kind == "env" and (seed := read_seed(config, name)) is not None:
            seeds[name] = seed
    document = ExportDocument(
        format=EXPORT_FORMAT, version=EXPORT_VERSION, created_by=f"uv-stack {__version__}",
        source_platform=source_platform(), items=items, files=files, seeds=seeds,
    )
    return ExportResult(document, absolute_path_warnings(config, keys))


def serialize_document(document: ExportDocument) -> str:
    """Serialize a document deterministically: sorted keys, two-space indent."""
    return json.dumps(document.model_dump(), sort_keys=True, indent=2) + "\n"
