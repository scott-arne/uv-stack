from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.errors import ConfigError
from uv_stack.models import ProjectTracking
from uv_stack.operations.pyproject import (
    read_project_dependency_names,
    read_tracking,
    remove_tracking,
    render_tracking,
    validate_tracking_write,
    write_tracking,
)

_BASE = (
    '[project]\nname = "demo"\nversion = "0.1.0"\n'
    'dependencies = ["numpy>=2", "httpx"]\n'
)


def _tracking() -> ProjectTracking:
    return ProjectTracking(
        stack=["@standard", "pkg:httpx"],
        python="main",
        applied=["numpy", "pandas", "httpx"],
    )


def test_round_trip(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE)
    write_tracking(pyproject, _tracking())
    loaded = read_tracking(pyproject)
    assert loaded == _tracking()
    # Surrounding content is byte-identical.
    assert pyproject.read_text().startswith(_BASE)


def test_read_absent_table_returns_none(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE)
    assert read_tracking(pyproject) is None


def test_read_missing_file_returns_none(tmp_path: Path):
    assert read_tracking(tmp_path / "pyproject.toml") is None


def test_replace_preserves_following_tables(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    following = '[tool.ruff]\nline-length = 100\n\n[[tool.arr]]\nx = 1\n'
    pyproject.write_text(
        _BASE + "\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n\n" + following
    )
    write_tracking(pyproject, _tracking())
    text = pyproject.read_text()
    assert text.startswith(_BASE)
    assert '[tool.ruff]\nline-length = 100' in text
    assert '[[tool.arr]]\nx = 1' in text
    assert text.count("[tool.uv-stack]") == 1
    assert read_tracking(pyproject) == _tracking()


def test_foreign_subtable_tolerated_and_preserved(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE
        + "\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n"
        + "\n[tool.uv-stack.extra]\ncustom = true\n"
    )
    loaded = read_tracking(pyproject)  # dict-valued 'extra' filtered pre-validation
    assert loaded is not None and loaded.stack == []
    write_tracking(pyproject, _tracking())
    text = pyproject.read_text()
    assert "[tool.uv-stack.extra]" in text and "custom = true" in text
    assert read_tracking(pyproject) == _tracking()


def test_unknown_scalar_key_is_config_error(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE + "\n[tool.uv-stack]\nstack = []\nbogus = 3\n")
    with pytest.raises(ConfigError):
        read_tracking(pyproject)


def test_toml_syntax_error_is_config_error(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("not = [valid\n")
    with pytest.raises(ConfigError):
        read_tracking(pyproject)


def test_write_refuses_unparseable_result(tmp_path: Path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE)
    monkeypatch.setattr(
        "uv_stack.operations.pyproject.render_tracking",
        lambda tracking: "[tool.uv-stack\nbroken",
    )
    with pytest.raises(ConfigError) as excinfo:
        write_tracking(pyproject, _tracking())
    assert "would not parse" in str(excinfo.value)
    assert pyproject.read_text() == _BASE  # untouched


def test_append_when_absent(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE)
    write_tracking(pyproject, _tracking())
    assert "\n[tool.uv-stack]\n" in pyproject.read_text()


def test_remove_tracking(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE)
    write_tracking(pyproject, _tracking())
    assert remove_tracking(pyproject) is True
    assert "[tool.uv-stack]" not in pyproject.read_text()
    assert pyproject.read_text().startswith(_BASE)
    assert remove_tracking(pyproject) is False


def test_python_omitted_when_none(tmp_path: Path):
    tracking = ProjectTracking(stack=["ds"], applied=["numpy"])
    text = render_tracking(tracking)
    assert "python" not in text
    assert 'version = 1' in text


def test_read_project_dependency_names(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE)
    assert read_project_dependency_names(pyproject) == {"numpy", "httpx"}
    assert read_project_dependency_names(tmp_path / "nope.toml") == set()


def test_read_project_dependency_names_includes_direct_references(tmp_path: Path):
    """Direct reference names are included; VCS entries excluded."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = ["pkg @ https://h/x.whl", "numpy>=2", "git+https://h/r.git@v1"]\n'
    )
    names = read_project_dependency_names(pyproject)
    assert "pkg" in names
    assert "numpy" in names
    # VCS entry still returns None from requirement_name → excluded
    assert len(names) == 2


def test_read_scalar_uv_stack_is_config_error(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE + '\n[tool]\nuv-stack = "bad"\n')
    with pytest.raises(ConfigError):
        read_tracking(pyproject)


def test_remove_tracking_leaves_subtable_and_reads_none(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE
        + "\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n"
        + "\n[tool.uv-stack.extra]\ncustom = true\n"
    )
    assert remove_tracking(pyproject) is True
    assert "[tool.uv-stack.extra]" in pyproject.read_text()
    assert read_tracking(pyproject) is None  # implicit parent = not tracked


def test_append_preserves_trailing_newlines(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE + "\n\n")
    write_tracking(pyproject, _tracking())
    assert pyproject.read_text().startswith(_BASE + "\n\n")


def test_quoted_header_with_comment(tmp_path: Path):
    """[tool."uv-stack"] with trailing comment matches and updates correctly."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE
        + '\n[tool."uv-stack"]  # tracked by uv-stack\n'
        + "version = 1\nstack = []\napplied = []\n"
    )
    loaded = read_tracking(pyproject)
    assert loaded is not None
    write_tracking(pyproject, _tracking())
    text = pyproject.read_text()
    assert text.count("[tool.") == 1  # Only one [tool.*] table present (the one we wrote)
    assert read_tracking(pyproject) == _tracking()


def test_quoted_header_replaces_in_place(tmp_path: Path):
    """[tool."uv-stack"] is replaced, not duplicated."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE + '\n[tool."uv-stack"]\nversion = 1\nstack = []\napplied = []\n')
    write_tracking(pyproject, _tracking())
    text = pyproject.read_text()
    assert text.count("uv-stack") == 1  # Only one occurrence in the canonical header
    assert "[tool.uv-stack]" in text


def test_header_with_comment_removes(tmp_path: Path):
    """[tool.uv-stack]   # comment can be removed."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE
        + "\n[tool.uv-stack]   # tracked by uv-stack\n"
        + "version = 1\nstack = []\napplied = []\n"
    )
    assert remove_tracking(pyproject) is True
    assert "[tool.uv-stack]" not in pyproject.read_text()


def test_inline_table_python_is_config_error(tmp_path: Path):
    """python = { value = "3.12" } (inline table) is rejected."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE + '\n[tool.uv-stack]\nstack = []\npython = { value = "3.12" }\n')
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert "inline table" in str(excinfo.value)


def test_inline_table_bogus_is_config_error(tmp_path: Path):
    """bogus = { x = 1 } (inline table) is rejected."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE + "\n[tool.uv-stack]\nstack = []\nbogus = { x = 1 }\n")
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert "inline table" in str(excinfo.value)


def test_quoted_dotted_header_read_write_remove(tmp_path: Path):
    """["tool"."uv-stack"] header: read, write replaces in place, remove works."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE + '\n["tool"."uv-stack"]\nversion = 1\nstack = []\napplied = []\n'
    )
    loaded = read_tracking(pyproject)
    assert loaded is not None and loaded.version == 1
    write_tracking(pyproject, _tracking())
    text = pyproject.read_text()
    assert text.count("[tool.uv-stack]") == 1  # Canonical form, no duplicate
    assert '["tool"."uv-stack"]' not in text  # Replaced, not preserved
    assert read_tracking(pyproject) == _tracking()
    assert remove_tracking(pyproject) is True
    assert "[tool.uv-stack]" not in pyproject.read_text()


def test_quoted_dotted_subtable_preserved(tmp_path: Path):
    """["tool"."uv-stack"."extra"] subtable tolerated and preserved."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE
        + "\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n"
        + '\n["tool"."uv-stack"."extra"]\ncustom = true\n'
    )
    loaded = read_tracking(pyproject)
    assert loaded is not None and loaded.stack == []
    write_tracking(pyproject, _tracking())
    text = pyproject.read_text()
    assert '["tool"."uv-stack"."extra"]' in text and "custom = true" in text
    assert read_tracking(pyproject) == _tracking()


def test_crlf_content_outside_span_preserved(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    crlf_base = _BASE.replace("\n", "\r\n")
    pyproject.write_bytes(crlf_base.encode())
    write_tracking(pyproject, _tracking())
    raw = pyproject.read_bytes()
    assert raw.startswith(crlf_base.encode())  # untouched CRLF prefix
    assert read_tracking(pyproject) == _tracking()
    assert remove_tracking(pyproject) is True
    assert pyproject.read_bytes().startswith(crlf_base.encode())


def test_crlf_replace_in_place(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    content = (
        _BASE + "\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n"
        + "\n[tool.ruff]\nline-length = 100\n"
    ).replace("\n", "\r\n")
    pyproject.write_bytes(content.encode())
    write_tracking(pyproject, _tracking())
    raw = pyproject.read_bytes()
    assert b"[tool.ruff]\r\nline-length = 100" in raw  # following table intact
    assert read_tracking(pyproject) == _tracking()


def test_replace_at_eof_preserves_trailing_newline(tmp_path: Path):
    """When [tool.uv-stack] is the LAST table, replacing it preserves the final newline."""
    pyproject = tmp_path / "pyproject.toml"
    # create-shaped file: table last, file ends with newline
    pyproject.write_text(_BASE + "\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n")
    original_bytes = pyproject.read_bytes()
    assert original_bytes.endswith(b"\n")  # precondition: original has trailing newline
    # An identical table at EOF must round-trip the whole file byte-for-byte,
    # including the trailing newline.
    write_tracking(pyproject, ProjectTracking(version=1, stack=[], applied=[]))
    assert pyproject.read_bytes() == original_bytes
    # A *changed* table is where the splice can lose or double the final
    # newline, so pin the entire resulting file rather than just its suffix.
    write_tracking(pyproject, _tracking())
    assert pyproject.read_text() == _BASE + (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = [\n    "@standard",\n    "pkg:httpx",\n]\n'
        'python = "main"\n'
        'applied = [\n    "numpy",\n    "pandas",\n    "httpx",\n]\n'
    )


def test_validate_tracking_write_accepts_valid_result(tmp_path: Path):
    """validate_tracking_write does not raise when the result would parse."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE)
    validate_tracking_write(pyproject, _tracking())  # should not raise
    assert pyproject.read_text() == _BASE  # untouched


def test_validate_tracking_write_rejects_unparseable_result(tmp_path: Path, monkeypatch):
    """validate_tracking_write raises ConfigError when result would not parse."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_BASE)
    monkeypatch.setattr(
        "uv_stack.operations.pyproject.render_tracking",
        lambda tracking: "[tool.uv-stack\nbroken",
    )
    with pytest.raises(ConfigError) as excinfo:
        validate_tracking_write(pyproject, _tracking())
    assert "would not parse" in str(excinfo.value)
    assert pyproject.read_text() == _BASE  # untouched


def test_render_tracking_omits_pending_when_none(tmp_path: Path):
    tracking = ProjectTracking(stack=["ds"], applied=["numpy"])
    text = render_tracking(tracking)
    assert "pending" not in text


def test_render_tracking_emits_pending_when_set(tmp_path: Path):
    tracking = ProjectTracking(stack=["ds"], applied=["numpy"], pending=["numpy", "chemprop"])
    text = render_tracking(tracking)
    assert 'pending = [\n    "numpy",\n    "chemprop",\n]' in text


def test_tracking_pending_round_trips(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    write_tracking(
        pyproject,
        ProjectTracking(stack=["ds"], applied=["numpy"], pending=["chemprop"]),
    )
    loaded = read_tracking(pyproject)
    assert loaded is not None
    assert loaded.pending == ["chemprop"]
    write_tracking(pyproject, ProjectTracking(stack=["ds"], applied=["numpy"]))
    reloaded = read_tracking(pyproject)
    assert reloaded is not None
    assert reloaded.pending is None
    assert "pending" not in pyproject.read_text()


def test_read_tracking_newer_schema_message_beats_shape_errors(tmp_path: Path):
    # A future table with unknown keys must get the actionable newer-schema
    # error, not the generic invalid-table error from extra="forbid".
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n'
        "[tool.uv-stack]\nversion = 2\n"
        'stack = ["ds"]\n'
        'applied = []\n'
        'future_key = "whatever"\n'
    )
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert "newer uv-stack (schema 2)" in str(excinfo.value)


@pytest.mark.parametrize(
    "version_value",
    ['"1"', "1.0", "true", '"2"'],
    ids=["string-1", "float-1.0", "bool-true", "string-2"],
)
def test_read_tracking_strict_version_typing(tmp_path: Path, version_value: str):
    """version field must be a strict integer; lax coercion is rejected."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n'
        f"[tool.uv-stack]\nversion = {version_value}\n"
        'stack = ["ds"]\n'
        'applied = []\n'
    )
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert "'version' must be an integer" in str(excinfo.value)


def test_read_tracking_version_2_int_still_raises_newer_schema(tmp_path: Path):
    """version = 2 (real int) raises NewerSchemaError, not a plain ConfigError.

    The type, not the message text, is what init's ``--force`` guard keys on
    to tell a forward-compatibility refusal apart from corrupt tracking.
    """
    from uv_stack.errors import NewerSchemaError

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n'
        "[tool.uv-stack]\nversion = 2\n"
        'stack = ["ds"]\n'
        'applied = []\n'
    )
    with pytest.raises(NewerSchemaError) as excinfo:
        read_tracking(pyproject)
    assert "newer uv-stack (schema 2)" in str(excinfo.value)


@pytest.mark.parametrize("field", ["version", "python", "pending"])
def test_reserved_name_subtable_refused_on_read(tmp_path: Path, field: str):
    """A subtable named after one of our fields is refused where it is diagnosable.

    Supersedes the former `test_foreign_version_subtable_tolerated_on_read`.
    Tolerating it deferred the failure past the point of diagnosis, and where
    it landed depended on the field. `version` is always rendered, so the next
    write died on `Refusing to write ...: result would not parse`. `python` and
    `pending` are rendered only when set, so with them unset — as they are in
    this document — the write succeeded and the subtable went on masking the
    field indefinitely. Neither outcome names the subtable.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE
        + "\n[tool.uv-stack]\nstack = []\napplied = []\n"
        + f"\n[tool.uv-stack.{field}]\ncustom = 42\n"
    )
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert f"[tool.uv-stack.{field}] shadows the '{field}' field" in str(excinfo.value)


def test_reserved_name_subtable_refused_stack_and_applied(tmp_path: Path):
    """stack and applied subtables refused when scalar is absent.

    tomllib itself preempts with "Cannot declare ... twice" when the scalar
    is present (stack = [], applied = []), so the only shape our guard reaches
    is the one where the subtable exists alone. Pin both to detect the mutation
    that narrows _RESERVED_KEYS to only the three fields the parametrized test
    covers.
    """
    pyproject = tmp_path / "pyproject.toml"
    # The stack scalar is omitted: present, it would collide in tomllib first.
    pyproject.write_text(
        _BASE + "\n[tool.uv-stack]\napplied = []\n\n[tool.uv-stack.stack]\nx = 1\n"
    )
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert "[tool.uv-stack.stack] shadows the 'stack' field" in str(excinfo.value)

    # Likewise for applied; the other scalar stays so the table is otherwise valid.
    pyproject.write_text(
        _BASE + "\n[tool.uv-stack]\nstack = []\n\n[tool.uv-stack.applied]\nx = 1\n"
    )
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert "[tool.uv-stack.applied] shadows the 'applied' field" in str(excinfo.value)


def test_subtable_only_document_reserved_name_refused(tmp_path: Path):
    """Reserved-name subtable alone (implicit parent) is refused; non-reserved reads None.

    Before this task, both read as None. The reserved-name refusal is correct:
    returning None would let stack init write an owned table that read_tracking
    then refuses. The paired negative case pins that the guard does not refuse
    ALL subtable-only documents — only reserved names.
    """
    pyproject = tmp_path / "pyproject.toml"
    # Reserved-name subtable alone -> refused
    pyproject.write_text(_BASE + "\n[tool.uv-stack.python]\nx = 1\n")
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert "[tool.uv-stack.python] shadows the 'python' field" in str(excinfo.value)

    # Non-reserved subtable alone -> None (not tracked)
    pyproject.write_text(_BASE + "\n[tool.uv-stack.extra]\nx = 1\n")
    assert read_tracking(pyproject) is None


def test_non_reserved_subtable_still_tolerated_on_read(tmp_path: Path):
    """The refusal is scoped to our own field names; foreign subtables are ours to keep."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE
        + "\n[tool.uv-stack]\nstack = []\napplied = []\n"
        + "\n[tool.uv-stack.mytool]\ncustom = 42\n"
    )
    loaded = read_tracking(pyproject)
    assert loaded is not None and loaded.version == 1 and loaded.stack == []


def test_version_2_with_inline_table_gives_newer_schema_error(tmp_path: Path):
    """version = 2 plus an inline table field raises newer-schema, not inline-table error.

    The pre-loop newer-schema guard (acting on scalar int version > 1) must fire before
    the filtering loop reaches the inline-table rejection, ensuring the friendly message
    preempts shape errors for genuinely newer records.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n'
        "[tool.uv-stack]\nversion = 2\n"
        'stack = ["ds"]\n'
        'applied = []\n'
        'future = { x = 1 }\n'
    )
    with pytest.raises(ConfigError) as excinfo:
        read_tracking(pyproject)
    assert "newer uv-stack (schema 2)" in str(excinfo.value)
    # Ensure we got the newer-schema error, not the inline-table error
    assert "inline table" not in str(excinfo.value)


_TRAP = (
    '[project]\nname = "demo"\nversion = "0.1.0"\n'
    'dependencies = ["numpy>=2", "httpx"]\n'
    'description = """\n[tool.uv-stack]\nstack = ["evil"]\n"""\n'
)


def test_header_inside_multiline_string_is_not_the_owned_table(tmp_path: Path):
    """A header-shaped line inside a string is string content, not our table."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_TRAP + '\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n')

    write_tracking(pyproject, _tracking())

    # startswith, not 'in': a span start off by even one line still leaves the
    # decoy's inner text present, so containment would not detect the miss.
    assert pyproject.read_text().startswith(_TRAP)
    assert read_tracking(pyproject) == _tracking()


def test_owned_table_is_written_below_a_trap_docstring(tmp_path: Path):
    """With no real table present, the decoy must not be mistaken for one."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_TRAP)

    write_tracking(pyproject, _tracking())

    assert pyproject.read_text().startswith(_TRAP)
    assert read_tracking(pyproject) == _tracking()


def test_remove_tracking_ignores_a_trap_docstring(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_TRAP + '\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n')

    assert remove_tracking(pyproject) is True

    # Removing the real table restores the decoy file exactly.
    assert pyproject.read_text() == _TRAP


def test_subtable_keys_ignores_headers_inside_strings(tmp_path: Path):
    from uv_stack.operations.pyproject import _subtable_keys

    text = (
        '[project]\nnotes = """\n[tool.uv-stack.decoy]\n"""\n'
        "\n[tool.uv-stack]\nstack = []\napplied = []\n"
        "\n[tool.uv-stack.extra]\ncustom = true\n"
    )
    assert _subtable_keys(text) == {"extra"}


def test_array_of_tables_terminates_the_owned_span(tmp_path: Path):
    """An [[array of tables]] immediately after the owned table ends its span.

    _table_header_lines maps [[...]] to None specifically so it terminates a
    span without ever matching the owned path. Drop those entries and the span
    runs on through this table, deleting it — and the result still parses, so
    _validate_result cannot catch the loss. test_replace_preserves_following_
    tables does not cover this: its [[tool.arr]] sits behind a [tool.ruff]
    header that had already ended the span.
    """
    pyproject = tmp_path / "pyproject.toml"
    following = '[[tool.mypy.overrides]]\nmodule = "demo.*"\nignore_missing_imports = true\n'
    pyproject.write_text("[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n\n" + following)

    write_tracking(pyproject, _tracking())

    text = pyproject.read_text()
    assert text.endswith(following)
    assert read_tracking(pyproject) == _tracking()


_MALFORMED = '[project]\nname = "demo"\n\n[tool.uv-stack\nversion = 1\n'


def test_write_tracking_refuses_a_malformed_file(tmp_path: Path):
    """A pyproject.toml that does not parse is refused, not spliced."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_MALFORMED)

    with pytest.raises(ConfigError) as excinfo:
        validate_tracking_write(pyproject, _tracking())
    assert "not valid TOML" in str(excinfo.value)

    with pytest.raises(ConfigError) as excinfo:
        write_tracking(pyproject, _tracking())
    assert "not valid TOML" in str(excinfo.value)
    assert pyproject.read_text() == _MALFORMED  # untouched


def test_remove_tracking_refuses_a_malformed_file(tmp_path: Path):
    """Removal refuses a malformed file rather than reporting 'no table'."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(_MALFORMED)

    with pytest.raises(ConfigError) as excinfo:
        remove_tracking(pyproject)
    assert "not valid TOML" in str(excinfo.value)
    assert pyproject.read_text() == _MALFORMED  # untouched


# Parses under universal newlines (read_tracking) but not raw (the write path):
# the stray carriage return mid-line is what tomllib rejects.
_STRAY_CR = (
    b'[project]\n'
    b'name = "demo"\n'
    b'description = """\n'
    b'Add this to pyproject.toml:\n'
    b'\n'
    b'[tool.uv-stack]\n'
    b'stack = ["ds"]\n'
    b'"""\n'
    b'keep_a = "one"\rkeep_b = "two"\n'
    b'keep_c = """\n'
    b'[tool.ruff]\n'
    b'line-length = 100\n'
    b'"""\n'
)


def test_stray_carriage_return_refuses_instead_of_deleting_content(tmp_path: Path):
    """Regression: the write path must not guess a span in unparseable text.

    read_tracking translates the stray \\r away and reports "untracked", so the
    command happily proceeds to a write path where tomllib rejects the same
    bytes. Guessing the span there deleted keep_a/keep_b/keep_c and both
    docstring interiors, and because the deletion re-paired the \"\"\"
    delimiters the wreckage parsed — _validate_result never fired.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_bytes(_STRAY_CR)
    assert read_tracking(pyproject) is None  # the \r is invisible to this read

    with pytest.raises(ConfigError) as excinfo:
        write_tracking(pyproject, _tracking())
    assert "not valid TOML" in str(excinfo.value)
    with pytest.raises(ConfigError):
        remove_tracking(pyproject)

    assert pyproject.read_bytes() == _STRAY_CR
    text = pyproject.read_text()
    assert "keep_a" in text and "keep_b" in text and "keep_c" in text


def test_find_span_survives_crlf(tmp_path: Path):
    """CRLF content outside the owned span is preserved byte-for-byte."""
    pyproject = tmp_path / "pyproject.toml"
    body = _BASE + "\n[tool.uv-stack]\nversion = 1\nstack = []\napplied = []\n"
    pyproject.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))

    write_tracking(pyproject, _tracking())

    raw = pyproject.read_bytes()
    assert b'name = "demo"\r\n' in raw
    assert read_tracking(pyproject) == _tracking()
