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
