"""Read and write the ``[tool.uv-stack]`` project tracking table.

uv-stack owns exactly one table in ``pyproject.toml``. Reads go through
:mod:`tomllib`; writes regenerate the owned table's text and splice it into
the file, validating that the WHOLE result still parses before publishing —
the no-corruption guarantee that lets us avoid a TOML-writer dependency.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from pydantic import ValidationError

from uv_stack.errors import ConfigError
from uv_stack.fsutil import atomic_write
from uv_stack.models import ProjectTracking
from uv_stack.parse import ownership_name

_HEADER = "[tool.uv-stack]"
_UV_STACK_PATH = ("tool", "uv-stack")

#: Shared newer-schema refusal text — read_tracking raises it centrally and
#: the init/refresh guards reuse it (defense in depth).
NEWER_SCHEMA_MESSAGE = "This project was tracked by a newer uv-stack (schema {version})."
NEWER_SCHEMA_HINT = "Upgrade uv-stack, or edit [tool.uv-stack] manually."


def _read_exact(pyproject: Path) -> str:
    """Read without newline translation so CRLF content outside the owned
    span survives a splice byte-for-byte."""
    with pyproject.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def _header_path(line: str) -> tuple[str, ...] | None:
    """Dotted-key path of a table-header line, or None if not a header.

    A standalone ``[a.b]`` line is complete TOML, so parsing it with
    :mod:`tomllib` yields the exact key path under every quoting and escape
    form the format allows (trailing comments included).
    """
    stripped = line.strip()
    if not stripped.startswith("[") or stripped.startswith("[["):
        return None
    try:
        parsed = tomllib.loads(stripped)
    except tomllib.TOMLDecodeError:
        return None
    path: list[str] = []
    node: object = parsed
    while isinstance(node, dict) and len(node) == 1:
        key, node = next(iter(node.items()))
        path.append(key)
    return tuple(path)


def _subtable_keys(text: str) -> set[str]:
    """Collect first-segment keys of actual [tool.uv-stack.<key>...] headers."""
    keys: set[str] = set()
    for line in text.split("\n"):
        path = _header_path(line)
        if path is not None and len(path) > 2 and path[:2] == _UV_STACK_PATH:
            keys.add(path[2])
    return keys


def read_tracking(pyproject: Path) -> ProjectTracking | None:
    """Load the tracking table, or ``None`` when file or table is absent.

    Foreign subtables (``[tool.uv-stack.*]`` — dict values under our table)
    are filtered out before validation: tolerated on read exactly as
    :func:`write_tracking` preserves them on write.

    :param pyproject: Path to ``pyproject.toml``.
    :raises ConfigError: On TOML syntax errors or schema-invalid tables.
    """
    if not pyproject.is_file():
        return None
    text = pyproject.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Invalid TOML in {pyproject}: {exc}", hint="Fix the TOML syntax."
        ) from exc
    table = data.get("tool", {}).get("uv-stack")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ConfigError(
            f"[tool.uv-stack] in {pyproject} must be a table.",
            hint="Replace the scalar/array value with a [tool.uv-stack] table.",
        )
    # PRE-loop: raise the newer-schema error immediately when 'version' is a
    # genuine scalar int > 1, preempting any shape errors the loop would hit
    # on future fields (inline tables, etc.). Acting only on strict scalar ints
    # preserves foreign [tool.uv-stack.version] subtable tolerance (dict values
    # fall through to the loop).
    raw_version = table.get("version")
    if isinstance(raw_version, int) and not isinstance(raw_version, bool):
        if raw_version > 1:
            raise ConfigError(
                NEWER_SCHEMA_MESSAGE.format(version=raw_version),
                hint=NEWER_SCHEMA_HINT,
            )
    subtable_keys = _subtable_keys(text)
    scalars = {}
    for key, value in table.items():
        if isinstance(value, dict):
            if key in subtable_keys:
                # Foreign subtable: tolerated and preserved on write.
                continue
            else:
                # Inline table: schema violation.
                raise ConfigError(
                    f"Invalid [tool.uv-stack] table in {pyproject}: "
                    f"'{key}' must not be an inline table.",
                    hint="Check the fields against the schema (version, stack, python, applied).",
                )
        scalars[key] = value
    if not scalars:
        # Header-only, or an implicit parent created solely by foreign
        # subtables (e.g. after remove_tracking left [tool.uv-stack.extra]
        # behind): no tracking data means no tracked project.
        return None
    # POST-loop: strict typing on the filtered scalar view. Reject bool or
    # non-int version values (pydantic would lax-coerce them). The > 1 guard
    # already ran pre-loop for genuine scalar ints, so this only validates type.
    if "version" in scalars:
        raw_version = scalars["version"]
        if isinstance(raw_version, bool) or not isinstance(raw_version, int):
            raise ConfigError(
                f"Invalid [tool.uv-stack] table in {pyproject}: 'version' must be an integer.",
                hint="Check the fields against the schema (version, stack, python, applied).",
            )
    try:
        return ProjectTracking.model_validate(scalars)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid [tool.uv-stack] table in {pyproject}: {exc}",
            hint="Check the fields against the schema (version, stack, python, applied).",
        ) from exc


def _string_list_lines(name: str, values: list[str]) -> list[str]:
    if not values:
        return [f"{name} = []"]
    lines = [f"{name} = ["]
    lines += [f"    {json.dumps(value)}," for value in values]
    lines.append("]")
    return lines


def render_tracking(tracking: ProjectTracking) -> str:
    """Render the full owned table as deterministic TOML text.

    JSON string escaping is valid TOML basic-string escaping, so
    :func:`json.dumps` emits the values safely.
    """
    lines = [_HEADER, f"version = {tracking.version}"]
    lines += _string_list_lines("stack", tracking.stack)
    if tracking.python is not None:
        lines.append(f"python = {json.dumps(tracking.python)}")
    lines += _string_list_lines("applied", tracking.applied)
    if tracking.pending is not None:
        lines += _string_list_lines("pending", tracking.pending)
    return "\n".join(lines) + "\n"


def _find_span(lines: list[str]) -> tuple[int, int] | None:
    """Locate the owned table: header line to the next header-shaped line.

    ANY line whose stripped text starts with ``[`` terminates the span —
    ``[x]``, ``[[x]]``, and ``[tool.uv-stack.sub]`` alike; our table never
    contains header-shaped lines, so anything that looks like one belongs
    to someone else and must be preserved.
    """
    start = None
    for index, line in enumerate(lines):
        if _header_path(line) == _UV_STACK_PATH:
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].strip().startswith("["):
            end = index
            break
    return start, end


def _splice(text: str, table_text: str | None) -> str:
    """Replace, insert, or (with ``None``) delete the owned table's span."""
    lines = text.split("\n")
    span = _find_span(lines)
    table_lines = table_text.rstrip("\n").split("\n") if table_text is not None else []
    if span is None:
        if table_text is None:
            return text
        if not text:
            return table_text
        # Preserve the existing content byte-for-byte; add only the
        # separator the current tail is missing.
        if text.endswith("\n\n"):
            separator = ""
        elif text.endswith("\n"):
            separator = "\n"
        else:
            separator = "\n\n"
        return text + separator + table_text
    start, end = span
    # Append separator when content follows OR when we're at EOF and the
    # original ended with a newline (preserving the trailing-newline phantom).
    if table_lines and (end < len(lines) or (end == len(lines) and text.endswith("\n"))):
        table_lines = [*table_lines, ""]
    return "\n".join([*lines[:start], *table_lines, *lines[end:]])


def _validate_result(text: str, pyproject: Path) -> None:
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Refusing to write {pyproject}: result would not parse.",
            hint="The existing pyproject.toml may be malformed; fix it and retry.",
        ) from exc


def _spliced_result(pyproject: Path, tracking: ProjectTracking) -> str:
    """Read, render, splice, and validate — the one write composition.

    Shared by :func:`validate_tracking_write` (pre-flight, result discarded)
    and :func:`write_tracking` (result published) so the two can never
    drift.

    :raises ConfigError: If the spliced result would not parse.
    """
    text = _read_exact(pyproject) if pyproject.is_file() else ""
    new_text = _splice(text, render_tracking(tracking))
    _validate_result(new_text, pyproject)
    return new_text


def validate_tracking_write(pyproject: Path, tracking: ProjectTracking) -> None:
    """Pre-flight validation for tracking writes without side effects.

    Renders and splices the tracking table into the current file content and
    validates that the result would parse, but does not write anything. Used
    before mutating dependencies to detect deterministic TOML validation
    failures before any external commands run.

    :param pyproject: Path to pyproject.toml.
    :param tracking: The tracking table to validate.
    :raises ConfigError: If the spliced result would not parse.
    """
    _spliced_result(pyproject, tracking)


def write_tracking(pyproject: Path, tracking: ProjectTracking) -> None:
    """Publish the tracking table, preserving everything outside its span.

    :raises ConfigError: If the spliced result would not parse (nothing is
        written in that case).
    """
    atomic_write(pyproject, _spliced_result(pyproject, tracking))


def remove_tracking(pyproject: Path) -> bool:
    """Delete the owned table; ``False`` when no table (or file) existed."""
    if not pyproject.is_file():
        return False
    text = _read_exact(pyproject)
    if _find_span(text.split("\n")) is None:
        return False
    new_text = _splice(text, None)
    _validate_result(new_text, pyproject)
    atomic_write(pyproject, new_text)
    return True


def read_project_dependency_names(pyproject: Path) -> set[str]:
    """Distribution names currently in ``[project.dependencies]``.

    Used by refresh to determine ownership: names returned here are considered
    user-owned and must not be auto-removed or adopted into the applied ledger.
    Includes names from PEP 508 direct references (``pkg @ https://...``).
    VCS-style entries (``git+https://...``) remain excluded.

    Unreadable/missing files yield the empty set (refresh's own tracking
    read reports real errors).
    """
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return set()
    dependencies = data.get("project", {}).get("dependencies", [])
    names: set[str] = set()
    for dependency in dependencies:
        if isinstance(dependency, str):
            name = ownership_name(dependency)
            if name:
                names.add(name)
    return names
