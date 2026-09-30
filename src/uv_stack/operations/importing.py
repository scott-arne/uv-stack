"""Import an export document into a config root.

The phases follow the spec: (1) read and check the document on its own;
(2) stage the target with the document overlaid and validate the result;
(3) classify every file against the target; (4) refuse any token whose
meaning would change; (5) pre-flight the build; (6) write; (7) build.
Nothing is written before phase 6, so every refusal leaves the target as
it was.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from uv_stack.commands import micromamba_python_info
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.fsutil import atomic_write, name_lock, read_text_utf8, require_regular_file
from uv_stack.models import EXPORT_FORMAT, EXPORT_VERSION, ExportDocument
from uv_stack.operations.diff import PinEntry, _diff_pins, parse_lock, parse_lock_text
from uv_stack.operations.edit import validate_bundle, validate_env, validate_profile
from uv_stack.operations.export import (
    ENV_FILES,
    ITEM_KINDS,
    OPTIONAL_ENV_FILES,
    FileKey,
    closure,
    file_entries,
    file_key,
    parse_file_key,
    reference_key,
)
from uv_stack.operations.scaffold import _SHADOW_HINT
from uv_stack.parse import editable_target
from uv_stack.pyversion import is_comparable, parse_python_info, satisfies
from uv_stack.render import render_requirements_in
from uv_stack.resolver import Resolver
from uv_stack.runner import Runner
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
    except OSError as error:
        # A bare OSError (a probe of an overlong name, say) carries its path in
        # filename, which both str() and the CLI's panel print.
        for attribute in ("filename", "filename2"):
            value = getattr(error, attribute)
            if isinstance(value, str):
                setattr(error, attribute, _relabel_text(value, prefix, replacement))
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


_Identity = tuple[int, int, tuple[str, ...]]


def _physical_identity(path: Path) -> _Identity:
    # Identifies a path by the deepest part of it that exists, so two keys the
    # target's own directory links route to one place compare equal before
    # either file is written.
    tail: list[str] = []
    while not os.path.exists(path):
        tail.append(path.name)
        path = path.parent
    info = os.stat(path)
    return info.st_dev, info.st_ino, tuple(reversed(tail))


def _by_identity(config: ConfigRoot, keys: Iterable[str]) -> dict[_Identity, list[str]]:
    """Group keys by the file each names under the root, keeping their order."""
    grouped: dict[_Identity, list[str]] = {}
    for key in keys:
        grouped.setdefault(_physical_identity(config.root / key), []).append(key)
    return grouped


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
    existing = _by_identity(config, keys)
    seen: dict[_Identity, str] = {}
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
    # Staged outside the config root, which nothing touches before phase 6.
    # The cost: checks of whether two names are one file observe this
    # directory's filesystem, so they are exact only when it folds letter case
    # and Unicode normalization the same way as the config root's.
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


STATUS_LABELS = {"new": "new", "identical": "identical", "different": "replace",
                 "target-only": "remove"}


@dataclass(frozen=True)
class FileChange:
    """One file's state: shipped text against this machine's copy."""

    key: str
    status: str
    current: str | None
    incoming: str | None


class ConflictError(ConfigError):
    """Different or target-only files met an import without ``--overwrite``.

    :param conflicts: The different and target-only files.
    :param found: The target environments that use each conflicting key.
    :param warnings: The staged validation and "used by" warnings, carried as
        ``resolution_warnings`` so they print with the conflict report: an
        environment that could not be read leaves ``found`` incomplete.
    """

    def __init__(self, conflicts: list[FileChange], found: dict[str, list[str]],
                 warnings: list[str]) -> None:
        super().__init__(
            f"{len(conflicts)} file(s) differ from this machine's copy; nothing was written.",
            hint="Review the differences above, then re-run with --overwrite to replace them.",
        )
        self.conflicts = conflicts
        self.used_by = found
        self.resolution_warnings = warnings


def _read_target(path: Path) -> str | None:
    if not os.path.lexists(path):
        return None
    require_regular_file(path)
    return read_text_utf8(path, exact_newlines=True)


def classify_changes(config: ConfigRoot, document: ExportDocument) -> list[FileChange]:
    """Classify each shipped file, and each target-only optional env file.

    Optional files are checked for every shipped environment, including an
    orphan ``envs/<name>/`` without ``stack.txt`` left by an interrupted create.
    """
    changes = []
    for key, incoming in document.files.items():
        current = _read_target(config.root / key)
        status = "new" if current is None else "identical" if current == incoming else "different"
        changes.append(FileChange(key, status, current, incoming))
    for name in _shipped(document, "env"):
        for filename in OPTIONAL_ENV_FILES:
            key = file_key("env", name, filename)
            if key not in document.files:
                current = _read_target(config.root / key)
                if current is not None:
                    changes.append(FileChange(key, "target-only", current, None))
    return sorted(changes, key=lambda change: change.key)


def _diff_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n\\ No newline at end of file\n"
    return lines


def change_diff(change: FileChange) -> str:
    """Return a unified diff of this machine's copy against the incoming text."""
    return "".join(difflib.unified_diff(
        _diff_lines(change.current or ""), _diff_lines(change.incoming or ""),
        fromfile=f"{change.key} (this machine)", tofile=f"{change.key} (incoming)",
    ))


def _os_reason(error: OSError) -> str:
    # render_os_error's shape: the errno text alone does not say which file.
    reason = error.strerror or str(error)
    return f"{reason}: {error.filename}" if error.filename else reason


def used_by(config: ConfigRoot, keys: Iterable[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Map each profile or bundle key to the target environments that reach it.

    Keys match by the file they name here, not by spelling: on a filesystem
    that folds letter case, a shipped ``profiles/Foo.yaml`` replaces
    ``profiles/foo.yaml``, so whatever reaches ``foo``, under any spelling, is
    reported under the shipped key.

    :param config: The unmodified target root.
    :param keys: Document file keys.
    :returns: Non-empty entries only, and a warning per environment that
        does not resolve or cannot be read (its users cannot be known, which
        the user should see).
    """
    wanted = _by_identity(config, keys)
    identities: dict[str, _Identity] = {}
    found: dict[str, list[str]] = {}
    warnings: list[str] = []
    for env in config.list_envs():
        try:
            resolved = Resolver(config).resolve(config.load_env(env).stack)
        except UvStackError as error:
            warnings.append(f"Cannot tell what environment '{env}' uses: {error.message}")
            continue
        except OSError as error:
            warnings.append(f"Cannot tell what environment '{env}' uses: {_os_reason(error)}")
            continue
        reached = [file_key("profile", p) for p in resolved.profiles]
        reached += [file_key("bundle", b) for b in resolved.bundles]
        hits: set[str] = set()
        for key in reached:
            if key not in identities:
                identities[key] = _physical_identity(config.root / key)
            hits.update(wanted.get(identities[key], []))
        for key in sorted(hits):
            found.setdefault(key, []).append(env)
    return found, warnings


_QUALIFIED_PREFIXES = ("@", "bundle:", "profile:", "package:", "pkg:")
_FIX = {"bundle": "@{}", "profile": "profile:{}", "package": "pkg:{}"}


def _meaning(root: ConfigRoot, token: str) -> str:
    return Resolver(root).classify([token]).entries[0]


def _describe(entry: str) -> str:
    kind, _, name = entry.partition(":")
    return f"the package '{name}'" if kind == "package" else f"{kind} '{name}'"


def _bare_tokens(root: ConfigRoot, key: FileKey) -> list[str]:
    if not (key.kind == "bundle" or key.filename == "stack.txt"):
        return []
    # The bundle model accepts a blank include, and classify() yields no entry
    # for one, so an unstripped blank would leave _meaning nothing to index.
    stripped = (entry.strip() for entry in file_entries(root, key))
    return [t for t in stripped if t and not t.startswith(_QUALIFIED_PREFIXES)]


def _refuse_change(token: str, path: Path, before: str, after: str, *, incoming: bool) -> None:
    if before == after:
        return
    where = ("in the incoming document but would mean {} on this machine" if incoming
             else "on this machine but would mean {} after the import")
    fix = _FIX[before.partition(":")[0]].format(before.partition(":")[2])
    raise ConfigError(
        f"'{token}' in {path} means {_describe(before)} " + where.format(_describe(after)) + ".",
        hint=f"Write {fix} in that file to keep its meaning; --overwrite cannot change "
        "what a token means.",
    )


def check_meanings(
    config: ConfigRoot, document: ExportDocument, doc_root: ConfigRoot, staged: Staged
) -> None:
    """Refuse an import that changes what an unqualified token means.

    Direction 1 checks the shipped stack and bundle files: a token must mean
    here what it meant in the document. Direction 2 checks the target's files
    the import does not replace: a token must mean after the import what it
    means now. A replaced environment's old stack is gone, so it is not checked.

    :raises ConfigError: When a token's meaning would change, or a target
        file cannot be read.
    :raises OSError: When a name cannot be probed; a staged path in it is
        relabeled to the target root.
    """
    with _relabeled(staged.root.root, str(config.root) + os.sep):
        for key in sorted(document.files):
            parsed = parse_file_key(key)
            assert parsed is not None  # phase 1 refused every other key
            for token in _bare_tokens(doc_root, parsed):
                _refuse_change(token, config.root / key, _meaning(doc_root, token),
                               _meaning(staged.root, token), incoming=True)
        # Compared by filesystem identity, not spelling: on a target that folds
        # letter case, envs/Main/stack.txt is the file a shipped
        # envs/main/stack.txt replaces, and its old tokens are gone after the
        # import.
        shipped = {_physical_identity(config.root / key) for key in document.files}
        # Built from the listed names directly, not through parse_file_key: a
        # target bundle whose stem validate_name rejects is still reachable by
        # '@stem', so its tokens need the check as much as any other file's.
        target = [FileKey("env", n, "stack.txt") for n in config.list_envs()]
        target += [FileKey("bundle", n, None) for n in config.list_bundles()]
        for parsed in target:
            key = file_key(parsed.kind, parsed.name, parsed.filename)
            if _physical_identity(config.root / key) in shipped:
                continue
            # A target file this check cannot read propagates its ConfigError:
            # the import cannot show that file's tokens keep their meaning, so
            # it stops.
            for token in _bare_tokens(config, parsed):
                _refuse_change(token, config.root / key, _meaning(config, token),
                               _meaning(staged.root, token), incoming=False)


def _recaptured_bundles(
    config: ConfigRoot, changes: list[FileChange], staged: ConfigRoot
) -> list[str]:
    """Return the identical shipped bundles whose bare tokens change meaning here.

    Direction 1 judges a shipped bundle against the document alone, and
    direction 2 skips it as shipped, so a bare token that a newly imported
    profile captures in an unchanged bundle passes both. The environments
    that reach such a bundle change meaning without a refusal.
    """
    keys = []
    for change in changes:
        parsed = parse_file_key(change.key)
        if change.status != "identical" or parsed is None or parsed.kind != "bundle":
            continue
        tokens = _bare_tokens(config, parsed)
        if any(_meaning(config, token) != _meaning(staged, token) for token in tokens):
            keys.append(change.key)
    return keys


_NO_BUILD = "Or pass --no-build to install the definitions without building."


@dataclass(frozen=True)
class BuildRequest:
    """How phase 5 probes, and whether the build recreates environments."""

    runner: Runner
    recreate: bool = False


@dataclass(frozen=True)
class BuildStep:
    """One environment phase 7 builds, and how."""

    name: str
    action: str


def _missing_editables(config: ConfigRoot, requirements: str) -> list[str]:
    """Editable targets in a rendered ``requirements.in`` absent on this machine.

    Resolved as doctor's missing-checkout finding does: ``~`` expanded, a
    relative path against the target root, where the build runs. Other
    path-bearing entries are left to uv, which reports them itself.
    """
    missing = []
    for line in requirements.splitlines():
        target = editable_target(line)
        if target is None:
            continue
        # An embedded NUL in the '~user' form makes expanduser raise
        # ValueError; keep the text as written, as doctor does, so the entry is
        # reported missing rather than silently passed to the build.
        try:
            path = Path(os.path.expanduser(target))
        except ValueError:
            path = Path(target)
        if not path.is_absolute():
            path = config.root / path
        if not os.path.exists(path):
            missing.append(target)
    return missing


def _python_action(name: str, python: str, env_python_path: Path, build: BuildRequest) -> str:
    """Decide how phase 7 builds one environment, refusing what cannot work.

    Mirrors ``upgrade_env``'s two guards so they fire before any write: a
    recreate needs a plain version, and without one the probe and
    ``satisfies`` test of the drift guard, failing open exactly as it does.
    """
    if build.recreate:
        if not is_comparable(python):
            raise ConfigError(
                f"Cannot recreate env '{name}': python.txt requests "
                f"'{python}', which is not a plain version. Recreating "
                "resolves the lock against the target version before "
                "rebuilding, and only a plain version can be resolved "
                "against.",
                hint=(
                    "Set a plain version such as 3.14 in "
                    f"{env_python_path}, or upgrade "
                    "without --recreate to keep the current interpreter."
                ),
            )
        return "recreate"
    try:
        result = build.runner.run(micromamba_python_info(name), capture=True, check=False)
    except Exception:
        # Probe failure (e.g., micromamba not installed): fail open, as the
        # drift guard does, and let the build report what is really wrong.
        return "create"
    if result.returncode != 0 or not result.stdout.strip():
        return "create"
    _, actual = parse_python_info(result.stdout)
    if actual and is_comparable(python) and not satisfies(python, actual):
        raise ConfigError(
            f"Environment '{name}' runs Python {actual}, but the incoming python.txt "
            f"requests {python}.",
            hint="The interpreter is only rebuilt when the environment is recreated. "
            "Re-run the import with --recreate to rebuild it from the shipped pins "
            "(this wipes and reinstalls the environment).",
        )
    return "sync"


def preflight(
    config: ConfigRoot, staged: ConfigRoot, document: ExportDocument, build: BuildRequest
) -> list[BuildStep]:
    """Refuse, before any write, a build that is certain to fail.

    Checks each shipped environment's variable values and editable paths,
    then its interpreter. Every refusal's hint also names ``--no-build``.

    :returns: One step per environment in ``items``, in order.
    :raises UvStackError: On the first refusal, relabeled to the target root.
    """
    steps = []
    try:
        with _relabeled(staged.root, str(config.root) + os.sep):
            variables = staged.load_variables()
            for name in _env_items(document):
                env = staged.load_env(name)
                stack = Resolver(staged).resolve(env.stack)
                requirements = render_requirements_in(stack, staged, name, variables)
                missing = _missing_editables(config, requirements)
                if missing:
                    raise ConfigError(
                        f"Environment '{name}' installs {len(missing)} editable "
                        f"checkout(s) missing on this machine: {', '.join(missing)}.",
                        hint="Check out each path, or set the variable that locates it in "
                        "variables.local.txt.",
                    )
                action = _python_action(name, env.python, staged.env_python_path(name), build)
                steps.append(BuildStep(name, action))
    except UvStackError as error:
        error.hint = f"{error.hint.rstrip()} {_NO_BUILD}" if error.hint else _NO_BUILD
        raise
    return steps


@dataclass
class ImportPlan:
    """Everything an import decided before writing."""

    document: ExportDocument
    changes: list[FileChange]
    missing_variables: list[str]
    variables_text: str | None
    warnings: list[str]
    used_by: dict[str, list[str]]
    dependents: list[str]
    builds: list[BuildStep] = field(default_factory=list)


def _env_items(document: ExportDocument) -> list[str]:
    return [i.partition(":")[2] for i in document.items if i.startswith("env:")]


def plan_import(
    config: ConfigRoot, document: ExportDocument, doc_root: ConfigRoot,
    referenced: list[str], options: ImportOptions, *, build: BuildRequest | None = None,
) -> ImportPlan:
    """Run phases 2-4 in spec order and decide the writes; raise before any write.

    The conflict refusal (phase 3) comes before the meaning check (phase 4) so
    a document with both reports the diffs and the ``--overwrite`` hint first.
    The dependents also count users of an identical bundle whose bare token
    now means something else; they change meaning though no file of theirs
    is written, so ``used_by`` and the conflict report leave them out.
    """
    refuse_shadowing(config, document)
    with staged_root(config, document, referenced) as staged:
        warnings = validate_staged(config, staged, document, options)
        changes = classify_changes(config, document)
        touched = [c.key for c in changes if c.status != "identical"]
        found, used_warnings = used_by(config, touched)
        conflicts = [c for c in changes if c.status in ("different", "target-only")]
        if conflicts and not options.overwrite:
            raise ConflictError(conflicts, found, warnings + used_warnings)
        check_meanings(config, document, doc_root, staged)
        recaptured = _recaptured_bundles(config, changes, staged.root)
        extra, extra_warnings = used_by(config, recaptured) if recaptured else ({}, [])
        builds = preflight(config, staged.root, document, build) if build is not None else []
    users = {env for envs in (*found.values(), *extra.values()) for env in envs}
    return ImportPlan(
        document=document, changes=changes, missing_variables=staged.missing_variables,
        variables_text=staged.variables_text,
        # An environment both calls cannot read would otherwise warn twice.
        warnings=list(dict.fromkeys(warnings + used_warnings + extra_warnings)),
        used_by=found, dependents=sorted(users - set(_env_items(document))), builds=builds,
    )


@contextmanager
def import_locks(config: ConfigRoot, document: ExportDocument) -> Iterator[None]:
    """Hold the root import lock, then stem locks, then env locks, in sorted order.

    One fixed order across every import keeps two imports from deadlocking;
    a profile and a bundle of one name share one stem lock, taken once.
    """
    stems = sorted(set(_shipped(document, "profile")) | set(_shipped(document, "bundle")))
    with ExitStack() as stack:
        stack.enter_context(
            name_lock(config.import_lock_path(), str(config.root), action="importing into")
        )
        for stem in stems:
            stack.enter_context(name_lock(config.stem_lock_path(stem), stem, action="importing"))
        for env in _shipped(document, "env"):
            stack.enter_context(name_lock(config.env_lock_path(env), env, action="importing"))
        yield


def _write_order(change: FileChange) -> tuple[int, str, int, str]:
    parsed = parse_file_key(change.key)
    assert parsed is not None
    rank = {"profile": 0, "bundle": 1, "env": 3}[parsed.kind]
    return rank, parsed.name, int(parsed.filename == "stack.txt"), change.key


def write_plan(config: ConfigRoot, plan: ImportPlan) -> int:
    """Write the plan: profiles, bundles, variables, then each env with stack.txt last.

    ``stack.txt`` defines whether an environment exists, so writing it last
    means an interrupted import never leaves a half-written environment that
    looks complete. A re-run reports finished files as identical.

    :returns: The number of files written or removed.
    :raises ConfigError: When a write fails part-way.
    """
    steps = sorted((c for c in plan.changes if c.status != "identical"), key=_write_order)
    definitions = [c for c in steps if _write_order(c)[0] < 3]
    env_steps = [c for c in steps if _write_order(c)[0] == 3]
    total = len(steps) + (plan.variables_text is not None)
    done = 0
    try:
        for change in definitions:
            _apply(config, change)
            done += 1
        if plan.variables_text is not None:
            _write_variables(config, plan.variables_text)
            done += 1
        for change in env_steps:
            _apply(config, change)
            done += 1
    except (OSError, UvStackError) as error:
        cause = error.message if isinstance(error, UvStackError) else str(error)
        raise ConfigError(
            f"Import interrupted after writing {done} of {total} file(s): {cause}",
            hint="Fix the cause and run the same import again: files already written are "
            "reported as identical and the rest are completed.",
        ) from error
    return done


def _apply(config: ConfigRoot, change: FileChange) -> None:
    path = config.root / change.key
    if change.incoming is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, change.incoming)


def _write_variables(config: ConfigRoot, text: str) -> None:
    config.variables_path().parent.mkdir(parents=True, exist_ok=True)
    atomic_write(config.variables_path(), text)


@contextmanager
def prepared_import(
    config: ConfigRoot, document: ExportDocument, options: ImportOptions, *,
    dry_run: bool, build: BuildRequest | None = None,
) -> Iterator[ImportPlan]:
    """Plan an import and, unless ``dry_run``, write it under the import locks.

    The locks stay held until the ``with`` block exits, so the caller's build
    runs under them and a second import of the same root waits, then refuses.
    A dry run takes no locks and writes nothing.
    """
    with document_root(document) as doc_root:
        referenced = check_document(document, doc_root)
        if dry_run:
            yield plan_import(config, document, doc_root, referenced, options, build=build)
            return
        with import_locks(config, document):
            plan = plan_import(config, document, doc_root, referenced, options, build=build)
            write_plan(config, plan)
            yield plan


def pin_report(name: str, seed_text: str, lock: Path) -> list[str]:
    """Compare the shipped pins with the lock the build produced here."""
    before = parse_lock_text(seed_text, f"seeds/{name}")
    after = parse_lock(lock)
    diff = _diff_pins(before, after)
    kept = sum(1 for n, v in before.items() if v is not None and after.get(n) == v)
    lines = [f"{name}: {kept} pins kept, {len(diff.version_differs)} changed, "
             f"{len(diff.only_in_a)} dropped, {len(diff.only_in_b)} added"]
    lines += [f"  changed: {c.name} {c.a} -> {c.b}" for c in diff.version_differs]
    lines += [f"  dropped: {_pin(p)}" for p in diff.only_in_a]
    lines += [f"  added: {_pin(p)}" for p in diff.only_in_b]
    return lines


def _pin(entry: PinEntry) -> str:
    # An editable's identity already embeds its target and has no version.
    return entry.name if entry.version is None else f"{entry.name} {entry.version}"
