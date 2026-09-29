"""Import an export document into a config root.

The phases follow the spec: (1) read and check the document on its own;
(2) stage the target with the document overlaid and validate the result;
(3) classify every file against the target; (4) refuse any token whose
meaning would change; (5) pre-flight the build; (6) write; (7) build.
Nothing is written before phase 6, so every refusal leaves the target as
it was.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.models import EXPORT_FORMAT, EXPORT_VERSION, ExportDocument
from uv_stack.operations.diff import parse_lock_text
from uv_stack.operations.export import (
    ITEM_KINDS,
    closure,
    file_entries,
    file_key,
    parse_file_key,
    reference_key,
)
from uv_stack.resolver import Resolver
from uv_stack.variables import referenced_names

_REEXPORT_HINT = "Re-create the document with 'stack export' on the source machine."
_ALIAS_HINT = (
    "Two shipped names differ only in letter case or Unicode normalization, "
    "which this filesystem treats as one name; rename one on the source machine."
)


def read_document(source: str, stdin: BinaryIO) -> str:
    """Read the document from a file path, or from ``stdin`` when ``-``.

    :raises ConfigError: When the source cannot be read or is not UTF-8.
    """
    label = "standard input" if source == "-" else source
    try:
        data = stdin.read() if source == "-" else Path(source).read_bytes()
    except OSError as error:
        raise ConfigError(f"Cannot read {label}: {error.strerror or error}") from error
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigError(f"{label} is not UTF-8 text: {error}", hint=_REEXPORT_HINT) from error


def load_document(text: str) -> ExportDocument:
    """Parse and check a document without looking at any config root.

    The format and version are checked before schema validation so a newer
    writer's document gets the upgrade advice rather than an unknown-key error.

    :raises ConfigError: For any malformed, foreign, or inconsistent document.
    """
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as error:
        # ValueError covers JSONDecodeError and huge integers; RecursionError covers deep nesting.
        raise ConfigError(
            f"The document is not valid JSON: {error}", hint=_REEXPORT_HINT
        ) from error
    # A JSON escape can spell a lone surrogate that UTF-8 input cannot carry.
    try:
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as error:
        raise ConfigError(
            f"The document holds text that is not valid Unicode: {error}", hint=_REEXPORT_HINT
        ) from error
    except RecursionError:
        # No valid export document is this deep.
        raise ConfigError("The document is not valid JSON.", hint=_REEXPORT_HINT) from None
    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected a JSON object, got {type(data).__name__}.", hint=_REEXPORT_HINT
        )
    if data.get("format") != EXPORT_FORMAT:
        raise ConfigError(
            f"Not a uv-stack export document: format is {data.get('format')!r}, "
            f"expected '{EXPORT_FORMAT}'.",
            hint=_REEXPORT_HINT,
        )
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigError("The document's version is not an integer.", hint=_REEXPORT_HINT)
    if version > EXPORT_VERSION:
        raise ConfigError(
            f"The document is format version {version}, written by "
            f"{data.get('created_by', 'an unknown writer')}; this uv-stack reads "
            f"version {EXPORT_VERSION}.",
            hint="Upgrade uv-stack on this machine: 'uv tool upgrade uv-stack'.",
        )
    if version < EXPORT_VERSION:
        raise ConfigError(
            f"The document is format version {version}, which this uv-stack does not read.",
            hint=_REEXPORT_HINT,
        )
    try:
        document = ExportDocument.model_validate(data)
    except ValidationError as error:
        raise ConfigError(f"Invalid export document: {error}", hint=_REEXPORT_HINT) from error
    _check_invariants(document)
    for name, seed in document.seeds.items():
        parse_lock_text(seed, f"seeds/{name}")
    return document


def _check_invariants(document: ExportDocument) -> None:
    def refuse(message: str) -> ConfigError:
        return ConfigError(message, hint=_REEXPORT_HINT)

    for key in document.files:
        if parse_file_key(key) is None:
            raise refuse(f"The document ships an invalid file key: {key!r}.")
    env_items: set[str] = set()
    for item in document.items:
        kind, sep, name = item.partition(":")
        if not sep or kind not in ITEM_KINDS or parse_file_key(file_key(kind, name)) is None:
            raise refuse(f"The document lists an invalid item: {item!r}.")
        if file_key(kind, name) not in document.files:
            raise refuse(f"The document lists {item} but does not ship {file_key(kind, name)}.")
        if kind == "env":
            env_items.add(name)
    if document.items != sorted(set(document.items)):
        raise refuse("The document's items are not sorted and unique.")
    for key in document.files:
        parsed = parse_file_key(key)
        if (parsed is not None and parsed.kind == "env"
                and file_key("env", parsed.name) not in document.files):
            raise refuse(f"The document ships {key} without envs/{parsed.name}/stack.txt.")
    orphans = sorted(set(document.seeds) - env_items)
    if orphans:
        raise refuse(f"The document ships seeds for {', '.join(orphans)}, not listed as env items.")


def _write_plain(path: Path, text: str) -> None:
    """Write text byte-for-byte into a scratch root (no newline translation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")


def _write_exclusive(path: Path, text: str, key: str) -> None:
    """Write a file exclusively; refuse if it already exists (case aliasing).

    :param path: Target file path.
    :param text: Content to write.
    :param key: Document key (for error messages).
    :raises ConfigError: When the file already exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(text)
    except FileExistsError as error:
        raise ConfigError(
            f"The document ships {key}, which names the same file on this machine "
            "as another shipped key.",
            hint=_ALIAS_HINT,
        ) from error


class _DocumentOnlyRoot(ConfigRoot):
    """Answer 'no such profile/bundle' without a stat for invalid names.

    The document ships only valid keys, so answering without a stat keeps bare
    traversal tokens (e.g. ../../../x) from probing outside the root.
    """

    def profile_exists(self, name: str) -> bool:
        """Return whether a profile exists, without statting invalid names."""
        if parse_file_key(file_key("profile", name)) is None:
            return False
        return super().profile_exists(name)

    def bundle_exists(self, name: str) -> bool:
        """Return whether a bundle exists, without statting invalid names."""
        if parse_file_key(file_key("bundle", name)) is None:
            return False
        return super().bundle_exists(name)


@contextmanager
def document_root(document: ExportDocument) -> Iterator[ConfigRoot]:
    """Materialize the document's files as a temporary, document-only root.

    The directory is resolved so a message quoting a resolved path (macOS
    ``/private/var``) still matches the prefix :func:`_relabeled` strips.
    """
    directory = Path(os.path.realpath(tempfile.mkdtemp(prefix="uv-stack-document-")))
    try:
        for key, text in document.files.items():
            _write_exclusive(directory / key, text, key)
        yield _DocumentOnlyRoot(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _relabel_text(text: str, prefix: str, replacement: str) -> str:
    return text.replace(prefix, replacement)


@contextmanager
def _relabeled(root: Path, replacement: str) -> Iterator[None]:
    """Rewrite a scratch root's path in any error raised inside the block.

    A refusal must name the file the user knows, the target's path or the
    document key, never the temporary directory it was detected in.
    """
    prefix = str(root) + os.sep
    try:
        yield
    except UvStackError as error:
        error.message = _relabel_text(error.message, prefix, replacement)
        error.args = (error.message,)
        if error.hint is not None:
            error.hint = _relabel_text(error.hint, prefix, replacement)
        error.resolution_warnings = [
            _relabel_text(w, prefix, replacement) for w in error.resolution_warnings
        ]
        if isinstance(error, ConfigError) and error.path is not None:
            error.path = Path(_relabel_text(str(error.path), prefix, replacement))
        raise


def referenced_variables(root: ConfigRoot, keys: Iterable[str]) -> list[str]:
    """Return the variable names the given files reference, sorted.

    A file that does not load is skipped: phase 2's validators report it
    against the target path, which is the more useful refusal.
    """
    names: set[str] = set()
    for key in keys:
        parsed = parse_file_key(key)
        if parsed is None:
            continue
        try:
            entries = file_entries(root, parsed)
        except UvStackError:
            continue
        names.update(n for entry in entries for n in referenced_names(entry))
    return sorted(names)


def _refuse_escaping_references(document: ExportDocument, doc_root: ConfigRoot) -> None:
    """Check that bundle includes and env stacks contain no escaping references.

    This runs before closure so an escaping reference (e.g. bundle:../../../x)
    is refused before resolving would read outside the scratch root.
    """
    for key in document.files:
        parsed = parse_file_key(key)
        if parsed is None:
            continue
        # Skip profiles: their includes are packages, not references
        if parsed.kind not in ("bundle", "env"):
            continue
        if parsed.kind == "env" and parsed.filename != "stack.txt":
            continue
        try:
            entries = file_entries(doc_root, parsed)
        except UvStackError:
            # If loading raises, skip: closure or the unreachable-file check reports it
            continue
        classified = Resolver(doc_root).classify(entries)
        for entry in classified.entries:
            kind, sep, name = entry.partition(":")
            if sep and kind in ("profile", "bundle"):
                reference_key(kind, name)


def check_document(document: ExportDocument, doc_root: ConfigRoot) -> list[str]:
    """Check that the document's items reach exactly the files it ships.

    :returns: The variable names the shipped files reference.
    :raises ConfigError: When a shipped file is unreachable, a reference escapes,
        or the bundle chain is too deep to resolve.
    :raises UvStackError: The resolver's missing-reference error, relabeled.
    """
    try:
        with _relabeled(doc_root.root, ""):
            _refuse_escaping_references(document, doc_root)
            reached = closure(doc_root, document.items)
    except (RecursionError, ConfigError) as error:
        if isinstance(error, RecursionError) or "RecursionError" in str(error):
            raise ConfigError(
                "The document's bundles nest too deeply to resolve.", hint=_REEXPORT_HINT
            ) from None
        raise
    shipped = set(document.files)
    extra = sorted(shipped - reached)
    if extra:
        raise ConfigError(
            f"The document ships {len(extra)} unreachable file(s) its items do not reach: "
            f"{', '.join(extra)}.",
            hint=_REEXPORT_HINT,
        )
    unshipped = sorted(reached - shipped)
    if unshipped:
        raise ConfigError(
            f"The document's items reach {len(unshipped)} file(s) it does not ship: "
            f"{', '.join(unshipped)}.",
            hint="A reference that differs from a shipped name only in letter case or Unicode "
            "normalization names that file on this filesystem; "
            "fix the reference on the source machine.",
        )
    return referenced_variables(doc_root, document.files)
