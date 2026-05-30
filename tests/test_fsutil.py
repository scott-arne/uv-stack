from __future__ import annotations

import os
import stat
from pathlib import Path

from uv_stack.fsutil import atomic_write


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
