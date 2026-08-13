from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from uv_stack.fsutil import atomic_write, atomic_write_new


def test_atomic_write_creates_file(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write(target, "hello\n")
    assert target.read_text() == "hello\n"


def test_atomic_write_replaces_existing(tmp_path: Path):
    target = tmp_path / "out.txt"
    target.write_text("old")
    atomic_write(target, "new")
    assert target.read_text() == "new"


def test_atomic_write_leaves_no_temp_files(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write(target, "x")
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_atomic_write_uses_conventional_mode(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write(target, "x")
    mode = stat.S_IMODE(target.stat().st_mode)
    expected = 0o666 & ~_current_umask()
    assert mode == expected


def _current_umask() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return umask


def test_atomic_write_new_creates_file(tmp_path: Path):
    target = tmp_path / "new.txt"
    stat_result = atomic_write_new(target, "content\n")
    assert target.read_text() == "content\n"
    # Verify the returned stat matches the published file's identity.
    actual_stat = target.stat()
    assert isinstance(stat_result, os.stat_result)
    assert (stat_result.st_dev, stat_result.st_ino) == (actual_stat.st_dev, actual_stat.st_ino)


def test_atomic_write_new_refuses_existing_file(tmp_path: Path):
    target = tmp_path / "existing.txt"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        atomic_write_new(target, "replacement")
    assert target.read_text() == "original"


def test_atomic_write_new_leaves_no_temp_files_on_success(tmp_path: Path):
    target = tmp_path / "clean.txt"
    stat_result = atomic_write_new(target, "x")
    assert isinstance(stat_result, os.stat_result)
    assert [p.name for p in tmp_path.iterdir()] == ["clean.txt"]


def test_atomic_write_new_leaves_no_temp_files_on_failure(tmp_path: Path):
    target = tmp_path / "clash.txt"
    target.write_text("first")
    with pytest.raises(FileExistsError):
        atomic_write_new(target, "second")
    assert [p.name for p in tmp_path.iterdir()] == ["clash.txt"]


def test_atomic_write_skips_identical_content(tmp_path: Path):
    target = tmp_path / "gen.txt"
    atomic_write(target, "same\n")
    old = target.stat().st_mtime
    os.utime(target, (old - 100, old - 100))
    stamped = target.stat().st_mtime
    atomic_write(target, "same\n")
    assert target.stat().st_mtime == stamped  # untouched
    atomic_write(target, "different\n")
    assert target.read_text() == "different\n"
