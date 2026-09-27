"""Tests for stack diff: source classification, lock grammar, comparison."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from uv_stack.errors import ConfigError

_IS_ROOT = os.geteuid() == 0


def _lock(tmp_path: Path, text: str, name: str = "a.lock.txt") -> Path:
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    return target


def test_parse_lock_accepts_pins_editables_and_direct_references(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(
        tmp_path,
        "# via nothing\n"
        "\n"
        "numpy==2.1.0\n"
        "rich==13.7.0  # via textual\n"
        "-e /src/tool\n"
        "mypkg @ https://example.invalid/mypkg.whl#sha256=abc\n"
        "--index-url https://example.invalid/simple\n",
    )

    assert parse_lock(path) == {
        "numpy": "2.1.0",
        "rich": "13.7.0",
        "-e /src/tool": None,
        "mypkg": "https://example.invalid/mypkg.whl#sha256=abc",
    }


def test_parse_lock_normalizes_distribution_names(tmp_path):
    from uv_stack.operations.diff import parse_lock

    assert parse_lock(_lock(tmp_path, "Foo_Bar==1.0\n")) == {"foo-bar": "1.0"}


@pytest.mark.parametrize(
    "line",
    ["-e /src/tool\n", "--editable /src/tool\n", "--editable=/src/tool\n"],
)
def test_parse_lock_normalizes_every_editable_spelling(tmp_path, line):
    from uv_stack.operations.diff import parse_lock

    assert parse_lock(_lock(tmp_path, line)) == {"-e /src/tool": None}


def test_parse_lock_discards_a_genuine_option_line(tmp_path):
    from uv_stack.operations.diff import parse_lock

    assert parse_lock(_lock(tmp_path, "--find-links /wheels\n")) == {}


def test_parse_lock_discards_indented_via_continuations(tmp_path):
    from uv_stack.operations.diff import parse_lock

    assert parse_lock(_lock(tmp_path, "numpy==2.1.0\n    #   via pandas\n")) == {
        "numpy": "2.1.0"
    }


def test_parse_lock_rejects_an_unparseable_line(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(tmp_path, "numpy==2.1.0\nthis is not a requirement\n")
    with pytest.raises(ConfigError) as excinfo:
        parse_lock(path)
    assert str(path) in str(excinfo.value)
    assert "line 2" in str(excinfo.value)


def test_parse_lock_rejects_a_hashed_continuation(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(tmp_path, "numpy==2.1.0 \\\n    --hash=sha256:abc\n")
    with pytest.raises(ConfigError) as excinfo:
        parse_lock(path)
    assert "line 1" in str(excinfo.value)
    # str() of a UvStackError is the message alone; the remedy is the hint.
    assert "hash" in (excinfo.value.hint or "")


def test_parse_lock_rejects_a_duplicate_identity(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(tmp_path, "numpy==2.1.0\nNumPy==2.2.0\n")
    with pytest.raises(ConfigError) as excinfo:
        parse_lock(path)
    assert "numpy" in str(excinfo.value)
    assert "line 2" in str(excinfo.value)


def test_parse_lock_reports_an_unreadable_file_as_a_config_error(tmp_path):
    from uv_stack.operations.diff import parse_lock

    if _IS_ROOT:
        pytest.skip("root reads regardless of mode")
    path = _lock(tmp_path, "numpy==2.1.0\n")
    path.chmod(0o000)
    try:
        with pytest.raises(ConfigError):
            parse_lock(path)
    finally:
        path.chmod(0o644)
