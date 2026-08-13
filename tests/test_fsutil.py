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


def test_atomic_write_crlf_exactness(tmp_path: Path):
    target = tmp_path / "crlf.txt"
    target.write_bytes(b"same\r\n")
    old_mtime = target.stat().st_mtime
    os.utime(target, (old_mtime - 100, old_mtime - 100))
    stamped = target.stat().st_mtime
    atomic_write(target, "same\r\n")
    assert target.stat().st_mtime == stamped  # skip: CRLF match
    assert target.read_bytes() == b"same\r\n"
    atomic_write(target, "same\n")
    assert target.stat().st_mtime > stamped  # rewritten
    assert target.read_bytes() == b"same\n"


def test_atomic_write_replaces_symlink(tmp_path: Path):
    external = tmp_path / "target.txt"
    external.write_text("content")
    link = tmp_path / "link.txt"
    link.symlink_to(external)
    assert link.is_symlink()
    atomic_write(link, "content")
    assert not link.is_symlink()
    assert link.is_file()
    assert link.read_text() == "content"
    assert external.read_text() == "content"  # unchanged


def test_atomic_write_isolates_hardlink(tmp_path: Path):
    original = tmp_path / "a.txt"
    original.write_text("old")
    sibling = tmp_path / "b.txt"
    os.link(original, sibling)
    assert original.stat().st_nlink == 2
    atomic_write(original, "old")
    assert original.stat().st_nlink == 1  # rewritten, isolated
    assert sibling.read_text() == "old"  # keeps old inode


def test_atomic_write_repairs_corrupt_file(tmp_path: Path):
    target = tmp_path / "corrupt.txt"
    target.write_bytes(b"\xff\xfe")
    atomic_write(target, "x\n")
    assert target.read_text() == "x\n"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform lacks mkfifo")
def test_atomic_write_replaces_fifo(tmp_path: Path):
    fifo = tmp_path / "pipe.txt"
    os.mkfifo(fifo)
    assert stat.S_ISFIFO(fifo.stat().st_mode)
    atomic_write(fifo, "x\n")
    assert stat.S_ISREG(fifo.stat().st_mode)
    assert fifo.read_text() == "x\n"


def test_atomic_write_detects_concurrent_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "swapped.txt"
    atomic_write(target, "same\n")
    old_mtime = target.stat().st_mtime
    os.utime(target, (old_mtime - 100, old_mtime - 100))
    stamped = target.stat().st_mtime

    # Simulate concurrent swap: intercept lstat only during atomic_write
    original_lstat = os.lstat
    intercept_active = [True]

    def fake_lstat(path, *args, **kwargs):
        result = original_lstat(path, *args, **kwargs)
        path_str = str(path) if isinstance(path, Path) else path
        if intercept_active[0] and path_str == str(target):
            # First lstat call during atomic_write is the identity recheck
            # Return a modified stat_result with different inode to simulate swap
            values = list(result)
            values[1] = result.st_ino + 999  # st_ino at index 1
            return os.stat_result(tuple(values))
        return result

    monkeypatch.setattr(os, "lstat", fake_lstat)
    atomic_write(target, "same\n")
    intercept_active[0] = False  # disable interception for assertions
    # Should rewrite due to identity mismatch, mtime changes
    assert target.stat().st_mtime > stamped
    assert target.read_text() == "same\n"
