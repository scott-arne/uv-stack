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
    # write_tracking with an identical table should be byte-identical
    write_tracking(pyproject, ProjectTracking(version=1, stack=[], applied=[]))
    # Identical table at EOF must round-trip the whole file byte-for-byte,
    # including the trailing newline.
    assert pyproject.read_bytes() == original_bytes
    # write_tracking with a changed table should also end with exactly newline
    write_tracking(pyproject, _tracking())
    result = pyproject.read_text()
    assert result.endswith("\n"), "trailing newline lost on changed replace"
    assert not result.endswith("\n\n"), "extra newline added"


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


def test_foreign_version_subtable_tolerated_on_read(tmp_path: Path):
    """[tool.uv-stack.version] subtable (dict with real header) is tolerated on read.

    The 'version' dict value is filtered out before validation (it's a real subtable,
    not a scalar), so version defaults to 1 from the model. This verifies the fix:
    version checks now run on the filtered scalar view, not the raw table.

    Note: TOML semantics prevent both a 'version' scalar and a '[tool.uv-stack.version]'
    subtable from coexisting, so unlike [tool.uv-stack.extra], this specific foreign
    subtable cannot be preserved on write.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        _BASE
        + "\n[tool.uv-stack]\nstack = []\napplied = []\n"
        + "\n[tool.uv-stack.version]\ncustom = 42\n"
    )
    loaded = read_tracking(pyproject)
    # 'version' key filtered out (it's a dict with a real subtable header),
    # so version defaults to 1 from the model.
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
