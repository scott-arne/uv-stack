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
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.fsutil import read_text_utf8, require_regular_file
from uv_stack.models import EXPORT_FORMAT, EXPORT_VERSION, ExportDocument
from uv_stack.operations.diff import parse_lock_text
from uv_stack.operations.edit import validate_bundle, validate_env, validate_profile
from uv_stack.operations.export import (
    ENV_FILES,
    ITEM_KINDS,
    OPTIONAL_ENV_FILES,
    closure,
    file_entries,
    file_key,
    parse_file_key,
    reference_key,
)
from uv_stack.operations.scaffold import _SHADOW_HINT
from uv_stack.resolver import Resolver
from uv_stack.variables import referenced_names

_REEXPORT_HINT = "Re-create the document with 'stack export' on the source machine."
_ALIAS_HINT = (
    "Two shipped names differ only in letter case or Unicode normalization, "
    "which this filesystem treats as one name; rename one on the source machine."
)
_FOLD_HINT = (
    "Where a filesystem treats the two as one name, the bundle's bare name opens "
    "the profile; rename one on the source machine."
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
    :raises ConfigError: When the file already exists or cannot be created.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(text)
    except FileExistsError as error:
        raise ConfigError(
            f"The document ships {key}, which names the same file on this machine "
            "as another shipped key.",
            hint=_ALIAS_HINT,
        ) from error
    except OSError as error:
        raise ConfigError(
            f"The document ships {key}, which this machine cannot store: "
            f"{error.strerror or type(error).__name__}.",
            hint="Shorten the name on the source machine, then re-create the document "
            "with 'stack export'.",
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
        try:
            # Every shipped file was materialized, so a name that cannot be statted
            # is not a shipped file.
            return super().profile_exists(name)
        except OSError:
            return False

    def bundle_exists(self, name: str) -> bool:
        """Return whether a bundle exists, without statting invalid names."""
        if parse_file_key(file_key("bundle", name)) is None:
            return False
        try:
            # Every shipped file was materialized, so a name that cannot be statted
            # is not a shipped file.
            return super().bundle_exists(name)
        except OSError:
            return False


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
        # ConfigError is inspected because the YAML loader wraps a RecursionError
        # hit while loading a bundle, which is where the limit is normally crossed.
        if isinstance(error, RecursionError) or isinstance(error.__cause__, RecursionError):
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


@dataclass(frozen=True)
class ImportOptions:
    """Options for ``stack import``.

    :param overwrite: Replace different files and remove target-only ones.
    :param strict: Validate shipped files as ``stack resolve --strict`` would.
    """

    overwrite: bool = False
    strict: bool = False


@dataclass(frozen=True)
class Staged:
    """The staged target view and the variables the import would declare."""

    root: ConfigRoot
    missing_variables: list[str]
    variables_text: str | None


def _shipped(document: ExportDocument, kind: str) -> list[str]:
    return sorted({k.name for k in map(parse_file_key, document.files) if k and k.kind == kind})


def _caseless(name: str) -> str:
    # Canonical caseless matching, computed rather than probed: a pair not yet
    # on the target cannot be statted there, and whether the two names collide
    # depends on the filesystem that will hold them, not on the one checking.
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def refuse_shadowing(config: ConfigRoot, document: ExportDocument) -> None:
    """Refuse a profile or bundle that would shadow the other kind here.

    A same-name profile and bundle arriving together are allowed: the source
    root already had them side by side. A shipped pair whose names differ only
    in letter case or Unicode normalization is refused, because a folding
    filesystem would open the profile for the bundle's bare name.

    :raises ConfigError: When a shipped pair folds together, or a shipped
        profile or bundle would shadow an existing one of the other kind.
    """
    shipped_profiles, shipped_bundles = _shipped(document, "profile"), _shipped(document, "bundle")
    for bundle in shipped_bundles:
        for profile in shipped_profiles:
            if profile != bundle and _caseless(profile) == _caseless(bundle):
                raise ConfigError(
                    f"Profile '{profile}' would shadow the shipped bundle '{bundle}': "
                    "the names differ only in letter case or Unicode normalization.",
                    hint=_FOLD_HINT,
                )
    profiles, bundles = set(shipped_profiles), set(shipped_bundles)
    for name in sorted(profiles - bundles):
        if config.bundle_exists(name):
            raise ConfigError(
                f"Profile '{name}' would shadow the existing bundle: {config.bundle_path(name)}",
                hint=_SHADOW_HINT,
            )
    for name in sorted(bundles - profiles):
        if config.profile_exists(name):
            raise ConfigError(
                f"Bundle '{name}' would be shadowed by the existing profile: "
                f"{config.profile_path(name)}",
                hint=_SHADOW_HINT,
            )


def _copy_source(source: Path, destination: Path) -> None:
    require_regular_file(source)
    if source.is_file():
        _write_plain(destination, read_text_utf8(source, exact_newlines=True))


def _appended_variables(current: str, missing: list[str]) -> str:
    if current and not current.endswith("\n"):
        current += "\n"
    return current + "".join(f"{name}\n" for name in missing)


def _refuse_non_directory_blockers(config: ConfigRoot, document: ExportDocument) -> None:
    """Refuse when a file exists where the import needs a directory.

    :raises ConfigError: When a path that must be a directory exists but is not.
    """
    # list_* report a non-directory container as empty, so the staged copy
    # would build a directory the real write cannot create.
    dirs = {config.root / parent for key in document.files for parent in Path(key).parents}
    for path in sorted(dirs):
        if os.path.lexists(path) and not path.is_dir():
            raise ConfigError(
                f"Not a directory: {path}",
                hint="Remove or rename whatever is at that path.",
            )


def _physical_identity(path: Path) -> tuple[int, int, tuple[str, ...]]:
    # Identifies a path by the deepest part of it that exists, so two keys the
    # target's own directory links route to one place compare equal before
    # either file is written.
    tail: list[str] = []
    while not os.path.exists(path):
        tail.append(path.name)
        path = path.parent
    info = os.stat(path)
    return info.st_dev, info.st_ino, tuple(reversed(tail))


def _refuse_linked_keys(config: ConfigRoot, document: ExportDocument) -> None:
    """Refuse a shipped key that the target's directory links make another key's file.

    The other key may be shipped too, or be one of the target's own definition
    files, which the staged view would model apart from the write that changes
    it. An existing key that differs only in letter case or Unicode
    normalization names the same item on a folding filesystem, and is
    overwritten like any existing file.

    :raises ConfigError: When a shipped key would be written to another key's file.
    """
    keys = [file_key("profile", name) for name in config.list_profiles()]
    keys += [file_key("bundle", name) for name in config.list_bundles()]
    keys += [file_key("env", name, f) for name in config.list_envs() for f in ENV_FILES]
    existing: dict[tuple[int, int, tuple[str, ...]], list[str]] = {}
    for key in keys:
        existing.setdefault(_physical_identity(config.root / key), []).append(key)
    seen: dict[tuple[int, int, tuple[str, ...]], str] = {}
    for key in sorted(document.files):
        identity = _physical_identity(config.root / key)
        if identity in seen:
            raise ConfigError(
                f"The document ships {seen[identity]} and {key}, "
                "which are the same file on this machine.",
                hint=(
                    "The target root's directory links make them one file; remove "
                    "the link or leave one of them out of the export."
                ),
            )
        others = [
            other for other in existing.get(identity, []) if _caseless(other) != _caseless(key)
        ]
        if others:
            raise ConfigError(
                f"The document ships {key}, which on this machine is the same file "
                f"as {others[0]}.",
                hint=(
                    "The target root's directory links make them one file, so writing "
                    "one would change the other; remove the link or leave the item out "
                    "of the export."
                ),
            )
        seen[identity] = key


@contextmanager
def staged_root(
    config: ConfigRoot, document: ExportDocument, referenced: list[str]
) -> Iterator[Staged]:
    """Stage the target with the document overlaid, in a temporary root.

    The copy holds every profile, bundle, and environment definition file of
    the target, then the shipped files on top, then drops a shipped
    environment's optional files the document does not ship, so the staged
    view is exactly what a successful ``--overwrite`` import would leave.
    """
    _refuse_non_directory_blockers(config, document)
    _refuse_linked_keys(config, document)
    directory = Path(os.path.realpath(tempfile.mkdtemp(prefix="uv-stack-staged-")))
    staged = ConfigRoot(directory)
    try:
        for name in config.list_profiles():
            _copy_source(config.profile_path(name), staged.profile_path(name))
        for name in config.list_bundles():
            _copy_source(config.bundle_path(name), staged.bundle_path(name))
        for name in config.list_envs():
            for filename in ENV_FILES:
                _copy_source(config.env_dir(name) / filename, staged.env_dir(name) / filename)
        _copy_source(config.variables_path(), staged.variables_path())
        for key, content in document.files.items():
            _write_plain(directory / key, content)
        for name in _shipped(document, "env"):
            for filename in OPTIONAL_ENV_FILES:
                if file_key("env", name, filename) not in document.files:
                    (staged.env_dir(name) / filename).unlink(missing_ok=True)
        with _relabeled(directory, str(config.root) + os.sep):
            # Declarations first, local values second: the target's
            # variables.local.txt may already hold a value for a name that only
            # this import declares, and loading it earlier would refuse it.
            declared = set(staged.load_variables().declared)
            missing = [name for name in referenced if name not in declared]
            text: str | None = None
            if missing:
                current = ""
                if staged.variables_path().is_file():
                    current = read_text_utf8(staged.variables_path(), exact_newlines=True)
                text = _appended_variables(current, missing)
                _write_plain(staged.variables_path(), text)
            _copy_source(config.variables_local_path(), staged.variables_local_path())
            staged.load_variables()
        yield Staged(staged, missing, text)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def validate_staged(
    config: ConfigRoot, staged: Staged, document: ExportDocument, options: ImportOptions
) -> list[str]:
    """Run ``stack edit``'s validators over each shipped item in the staged view.

    :returns: Their warnings, relabeled to the target root, first occurrence kept.
    """
    prefix, replacement = str(staged.root.root) + os.sep, str(config.root) + os.sep
    warnings: list[str] = []
    with _relabeled(staged.root.root, replacement):
        for name in _shipped(document, "profile"):
            warnings += validate_profile(staged.root, name, strict=options.strict).warnings
        for name in _shipped(document, "bundle"):
            warnings += validate_bundle(staged.root, name, strict=options.strict).warnings
        for name in _shipped(document, "env"):
            warnings += validate_env(staged.root, name, strict=options.strict).warnings
    return list(dict.fromkeys(_relabel_text(w, prefix, replacement) for w in warnings))
