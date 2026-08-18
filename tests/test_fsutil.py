from __future__ import annotations

import errno
import os
import stat
import time
from pathlib import Path

import pytest

from tests.conftest import _lock_held_by_another_process
from uv_stack.errors import ConfigError
from uv_stack.fsutil import (
    _LOCK_AVAILABLE,
    _O_NOFOLLOW,
    atomic_write,
    atomic_write_new,
    name_lock,
)


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


def test_atomic_write_forgoes_the_fast_path_without_its_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Without both open flags the identical-rewrite skip is not taken.

    Supersedes test_atomic_write_degrades_without_nofollow_nonblock, which
    deleted os.O_NOFOLLOW/os.O_NONBLOCK and asserted the skip still happened —
    the behavior this replaces. Deleting the attributes also no longer reaches
    the decision: availability is now computed once at import time from the
    real os module, so an os-level patch during the test would leave the test
    green while exercising nothing. Patch the module constant instead.

    The assertion is on the inode, not the mtime: the slow path publishes via
    os.replace, so a changed inode is exact evidence that it ran, whereas mtime
    resolution is a property of the filesystem under tmp_path.
    """
    from uv_stack import fsutil

    monkeypatch.setattr(fsutil, "_FASTPATH_AVAILABLE", False)

    target = tmp_path / "file.txt"
    fsutil.atomic_write(target, "content\n")
    before = target.stat().st_ino
    fsutil.atomic_write(target, "content\n")
    assert target.stat().st_ino != before
    assert target.read_text() == "content\n"


def test_atomic_write_new_falls_back_without_hardlinks(tmp_path, monkeypatch):
    import errno
    import os as _os

    from uv_stack.fsutil import atomic_write_new

    def _no_link(src, dst, **kwargs):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(_os, "link", _no_link)
    target = tmp_path / "made.txt"
    stat_result = atomic_write_new(target, "content\n")
    assert target.read_text() == "content\n"
    assert (stat_result.st_dev, stat_result.st_ino) == (
        target.stat().st_dev,
        target.stat().st_ino,
    )
    # Returned stat reflects post-write content: size matches actual file.
    expected_size = len(b"content\n")
    assert stat_result.st_size == expected_size
    assert stat_result.st_size == target.stat().st_size
    # Exclusive-create semantics preserved on the fallback path.
    with pytest.raises(FileExistsError):
        atomic_write_new(target, "other\n")


def test_atomic_write_new_fallback_cleans_partial_on_failure(tmp_path, monkeypatch):
    import errno
    import os as _os

    from uv_stack.fsutil import atomic_write_new

    def _no_link(src, dst, **kwargs):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(_os, "link", _no_link)

    real_fdopen = _os.fdopen

    class _ExplodingWriter:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            return False

        def write(self, text):
            raise OSError("disk full")

    calls = {"count": 0}

    def _fdopen(fd, *args, **kwargs):
        handle = real_fdopen(fd, *args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 2:  # 1st fdopen = temp file; 2nd = the fallback
            return _ExplodingWriter(handle)
        return handle

    monkeypatch.setattr(_os, "fdopen", _fdopen)
    target = tmp_path / "partial.txt"
    with pytest.raises(OSError):
        atomic_write_new(target, "content\n")
    assert not target.exists()  # partial target withdrawn


def test_atomic_write_publishes_exact_bytes_for_crlf_text(tmp_path):
    from uv_stack.fsutil import atomic_write, atomic_write_new

    target = tmp_path / "crlf.txt"
    atomic_write(target, "a\r\nb\n")
    assert target.read_bytes() == b"a\r\nb\n"
    target2 = tmp_path / "crlf_new.txt"
    atomic_write_new(target2, "a\r\nb\n")
    assert target2.read_bytes() == b"a\r\nb\n"


def test_atomic_write_new_fallback_publishes_exact_crlf_bytes(tmp_path, monkeypatch):
    import errno
    import os as _os

    from uv_stack.fsutil import atomic_write_new

    def _no_link(src, dst, **kwargs):
        raise OSError(errno.EPERM, "hard links not supported")

    monkeypatch.setattr(_os, "link", _no_link)
    target = tmp_path / "crlf_fallback.txt"
    atomic_write_new(target, "a\r\nb\n")
    assert target.read_bytes() == b"a\r\nb\n"


def test_atomic_write_non_ascii_utf8_byte_exactness(tmp_path):
    from uv_stack.fsutil import atomic_write

    content = "# Généré — ünïcode\n"
    expected_bytes = content.encode("utf-8")
    target = tmp_path / "unicode.txt"
    atomic_write(target, content)
    assert target.read_bytes() == expected_bytes
    # Verify identical-skip guard preserves mtime for non-ASCII content.
    old_mtime = target.stat().st_mtime
    os.utime(target, (old_mtime - 100, old_mtime - 100))
    stamped = target.stat().st_mtime
    atomic_write(target, content)
    assert target.stat().st_mtime == stamped  # skip: UTF-8 match


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_excludes_another_process(tmp_path):
    """A second process cannot enter while the first holds the lock.

    Deterministic rather than timing-based: the exclusion is proved by the
    timeout firing, which is guaranteed while the child holds the lock.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)

    with _lock_held_by_another_process(lock_path):
        with pytest.raises(ConfigError) as excinfo:
            with name_lock(lock_path, "x", timeout=0.2):
                pytest.fail("entered the lock while another process held it")

    assert "another stack process" in str(excinfo.value)
    assert "x" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_is_acquirable_once_the_holder_exits(tmp_path):
    """The kernel releases the lock when the holder dies, so no name wedges."""
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)

    with _lock_held_by_another_process(lock_path):
        pass  # the child exits here, without ever unlocking explicitly

    with name_lock(lock_path, "x", timeout=0.2):
        pass  # acquires; a wedged lock would raise


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_releases_on_exception(tmp_path):
    """An exception in the body still unlocks — the next caller is not blocked."""
    lock_path = tmp_path / ".locks" / "stem-x.lock"

    with pytest.raises(RuntimeError):
        with name_lock(lock_path, "x", timeout=0.2):
            raise RuntimeError("boom")

    with name_lock(lock_path, "x", timeout=0.2):
        pass


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_creates_the_directory_and_keeps_the_file(tmp_path):
    """The lock file is created on demand and deliberately never removed."""
    lock_path = tmp_path / ".locks" / "stem-x.lock"

    with name_lock(lock_path, "x"):
        assert lock_path.is_file()
    # Unlinking a file another process may hold open would let a third take a
    # lock that excludes nobody, so the empty file stays.
    assert lock_path.is_file()


def test_name_lock_is_a_noop_without_fcntl(tmp_path, monkeypatch):
    """Where fcntl is absent the block still runs and no lock file is made."""
    monkeypatch.setattr("uv_stack.fsutil._LOCK_AVAILABLE", False)
    lock_path = tmp_path / ".locks" / "stem-x.lock"

    entered = False
    with name_lock(lock_path, "x"):
        entered = True

    assert entered
    assert not lock_path.exists()


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_degrades_when_the_filesystem_cannot_lock(tmp_path, monkeypatch):
    """ENOLCK from a network mount degrades like a missing fcntl, and does not stall.

    Before this was discriminated, every create on such a filesystem waited out
    the full timeout and then blamed a competing process that did not exist.
    """
    import fcntl

    def refuse(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", refuse)
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 5.0)
    lock_path = tmp_path / ".locks" / "stem-x.lock"

    started = time.monotonic()
    entered = False
    with name_lock(lock_path, "x"):
        entered = True
    elapsed = time.monotonic() - started

    assert entered
    assert elapsed < 1.0, f"degraded path waited {elapsed:.2f}s instead of proceeding"
    assert lock_path.is_file()


@pytest.mark.skipif(not _O_NOFOLLOW, reason="requires O_NOFOLLOW")
@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_refuses_a_symlink_at_the_lock_path(tmp_path):
    """The lock's open refuses to follow a symlink planted at the lock path.

    Without O_NOFOLLOW the open follows the link, so anyone able to write to
    the config root can redirect it at a file of their choosing — and O_CREAT
    through a dangling link would create that file. Exclusion still works in
    that case, which is why nothing else in the suite catches a missing flag.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    external = tmp_path / "target"
    external.write_text("content")
    lock_path.symlink_to(external)

    with pytest.raises(OSError):
        with name_lock(lock_path, "x"):
            pytest.fail("entered with a symlink at the lock path")


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_treats_eacces_as_contention(tmp_path, monkeypatch):
    """EACCES from fcntl.flock is treated as contention, not a degrade trigger.

    CPython's fcntl.flock emulates with fcntl(F_SETLK) where the build lacks
    HAVE_FLOCK, and POSIX permits F_SETLK to report a conflicting lock as
    either EACCES or EAGAIN. Before EACCES was added to the contended set,
    it fell through to the degrade branch and name_lock entered its body
    holding nothing.
    """
    import fcntl

    def refuse_with_eacces(fd, op):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(fcntl, "flock", refuse_with_eacces)
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    lock_path = tmp_path / ".locks" / "stem-x.lock"

    with pytest.raises(ConfigError) as excinfo:
        with name_lock(lock_path, "x"):
            pytest.fail("entered despite EACCES contention")

    assert "another stack process" in str(excinfo.value)
