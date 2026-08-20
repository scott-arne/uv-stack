from __future__ import annotations

import contextlib
import errno
import os
import signal
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
    through a *dangling* link creates that file outright. Exclusion still
    works in that case, so nothing else in the suite would catch the missing
    flag; the target's non-creation is the property worth pinning. The errno
    is not asserted: it is ELOOP here, EMLINK on FreeBSD, EFTYPE on NetBSD.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    target = tmp_path / "planted-target"
    lock_path.symlink_to(target)
    assert not target.exists()

    with pytest.raises(OSError):
        with name_lock(lock_path, "x"):
            pytest.fail("entered with a symlink at the lock path")

    assert not target.exists(), "O_CREAT followed the link and created the target"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_refuses_a_fifo_at_the_lock_path(tmp_path):
    """A non-regular file at the lock path is refused, not silently tolerated.

    flock on a FIFO fails with an errno outside the contended set (ENOTSUP on
    macOS), which without this check takes the degrade branch and makes the
    lock a silent no-op for that name — verified by having a second process
    acquire the same path while the first believed it held the lock.

    The mode is asserted afterwards because it also pins the order of the two
    steps on this side of the open: the type check runs before the 0666
    re-apply, so a plant is refused rather than widened. Reversing them leaves
    the suite green otherwise, and hands anyone who plants a FIFO in a shared
    root a mode change on it for free.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    os.mkfifo(lock_path, 0o644)
    os.chmod(lock_path, 0o644)  # mkfifo's mode is subject to the umask; this is not

    with pytest.raises(ConfigError) as excinfo:
        with name_lock(lock_path, "x"):
            pytest.fail("entered with a FIFO at the lock path")

    assert "not a regular file" in str(excinfo.value)
    assert stat.S_IMODE(os.lstat(lock_path).st_mode) == 0o644, "the refused plant was chmod'd"


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


#: Root bypasses the mode bits these tests rely on, so their premise cannot
#: hold there. Checked with getattr because the expression is evaluated at
#: collection time, on every platform, before the fcntl skip applies.
_IS_ROOT = getattr(os, "geteuid", lambda: -1)() == 0


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
def test_name_lock_locks_a_lock_file_it_may_not_write(tmp_path):
    """A lock file created by another user in a shared config root still locks.

    The create mode is 0o666 & ~umask, so under a typical 022 umask the file
    is 0644 and a second user sharing the root cannot open it O_RDWR. Failing
    there would break every create in a root the rest of stack writes fine —
    os.replace and os.link need the directory, not the file. A 0444 file
    reproduces that refusal for the owner. Exclusion has to survive the
    read-only fd, not just the open, so a real second process is checked
    against it first.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.touch()
    lock_path.chmod(0o444)
    with pytest.raises(PermissionError):
        os.close(os.open(lock_path, os.O_RDWR))  # the premise, not the behavior

    with _lock_held_by_another_process(lock_path):
        with pytest.raises(ConfigError) as excinfo:
            with name_lock(lock_path, "x", timeout=0.2):
                pytest.fail("entered while another process held the lock")
    assert "another stack process" in str(excinfo.value)

    with name_lock(lock_path, "x", timeout=0.2):
        pass  # acquires read-only; without the fallback the open raised EACCES


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_name_lock_degrades_when_the_lock_file_cannot_be_had(tmp_path):
    """A .locks directory this user may not write degrades, it does not fail.

    Two shapes, both from a shared root: the directory exists but belongs to
    another user (no new lock file can be created in it), and the directory is
    absent under a root that is not ours to write (the mkdir itself is
    refused). Either way there is no lock to take — but the data directories
    may well still be writable, since os.replace and os.link need those, not
    this one, so failing here would break a create that would otherwise
    succeed. Degrading matches the unlockable-filesystem branch: the three
    stem writers keep their post-publish checks, and write_env_sources, which
    has none, is left exactly where it stood before the lock existed.
    """
    read_only_dir = tmp_path / "occupied" / ".locks"
    read_only_dir.mkdir(parents=True)
    read_only_dir.chmod(0o555)
    try:
        with name_lock(read_only_dir / "stem-x.lock", "x", timeout=0.2):
            pass  # degraded; a raise here is the regression
    finally:
        read_only_dir.chmod(0o755)

    read_only_root = tmp_path / "unwritable"
    read_only_root.mkdir()
    read_only_root.chmod(0o555)
    try:
        with name_lock(read_only_root / ".locks" / "stem-x.lock", "x", timeout=0.2):
            pass
        assert not (read_only_root / ".locks").exists()
    finally:
        read_only_root.chmod(0o755)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.parametrize(
    ("where", "kind"),
    [
        ("locks", "file"),
        ("locks", "dangling symlink"),
        ("root", "file"),
        ("root", "dangling symlink"),
    ],
)
def test_name_lock_refuses_a_non_directory_where_it_needs_one(tmp_path, where, kind):
    """A non-directory where the lock directory goes is refused, and named.

    Both errnos mkdir(exist_ok=True) raises for this are covered, and they do
    not split by location: anything at .locks gives FileExistsError, since
    mkdir meets it as a component to create, and so does a dangling symlink at
    the root, which mkdir retries as a component once the first attempt gives
    ENOENT; a file at the root gives NotADirectoryError, because there mkdir
    traverses it rather than creating it. Neither is a permission
    error and neither is an flock errno, so they reach neither degrade.
    Degrading would cost every name in the root its lock at once, for nothing —
    the same trade the FIFO check above refuses for a single name.

    parents=True means the offender may be an ancestor rather than .locks
    itself, so the reported path is asserted exactly: a message naming a .locks
    that does not exist sends the reader looking for nothing, which is worse
    than the bare traceback it replaced.
    """
    root = tmp_path / "root"
    if where == "root":
        offender = root
    else:
        root.mkdir()
        offender = root / ".locks"
    if kind == "file":
        offender.write_text("")
    else:
        offender.symlink_to(tmp_path / "missing")

    with pytest.raises(ConfigError) as excinfo:
        with name_lock(root / ".locks" / "stem-x.lock", "x", timeout=0.2):
            pytest.fail("entered with a non-directory where the lock goes")

    assert str(excinfo.value) == f"Not a directory: {offender}"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_name_lock_degrading_does_not_chain_onto_the_callers_exception(tmp_path):
    """A degraded lock leaves nothing of its own above the caller's failure.

    Yielding from inside the except block chained everything the body raised
    onto an unrelated lock-file error, so the user's traceback opened with a
    lock file they had never heard of and reached the file they actually asked
    for three exceptions later.
    """
    locks = tmp_path / ".locks"
    locks.mkdir()
    locks.chmod(0o555)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            with name_lock(locks / "stem-x.lock", "x", timeout=0.2):
                raise RuntimeError("the caller's own failure")
    finally:
        locks.chmod(0o755)

    # Without this the test passes vacuously wherever 0555 stops forcing the
    # degrade: an undegraded lock has nothing to chain in the first place.
    assert not (locks / "stem-x.lock").exists(), "the lock was taken, so nothing degraded"
    assert excinfo.value.__context__ is None
    assert excinfo.value.__cause__ is None


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_surfaces_a_non_permission_error_from_the_read_only_retry(
    tmp_path, monkeypatch
):
    """The read-only retry degrades on permission errnos only.

    An object swapped in at the lock path between the two opens — a symlink,
    giving ELOOP — is not "no lock file to be had": degrading on it would cost
    exclusion silently. Only a real TOCTOU produces that pairing, so it is
    injected.
    """
    calls = []

    def swap_after_the_first(path, flags, *args):
        calls.append(flags)
        if len(calls) == 1:
            raise PermissionError(errno.EACCES, "Permission denied")
        raise OSError(errno.ELOOP, "Too many levels of symbolic links")

    monkeypatch.setattr(os, "open", swap_after_the_first)
    with pytest.raises(OSError) as excinfo:
        with name_lock(tmp_path / ".locks" / "stem-x.lock", "x", timeout=0.2):
            pytest.fail("degraded on an errno that is not a permission problem")

    assert excinfo.value.errno == errno.ELOOP
    assert len(calls) == 2, "the retry never ran, so nothing was exercised"


@contextlib.contextmanager
def _deadline(seconds):
    """Fail rather than hang if the body blocks.

    The defect the two tests below cover does not make them fail — it makes
    them wait forever inside ``os.open``, which stalls the whole suite with no
    output and no failing test to point at. SIGALRM turns that into an ordinary
    assertion failure. The handler raises, so it interrupts the blocked syscall
    instead of letting PEP 475 retry it.
    """

    def _fire(signum, frame):
        raise AssertionError(f"blocked for more than {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform lacks mkfifo")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
def test_name_lock_refuses_an_unwritable_fifo_without_waiting(tmp_path):
    """A FIFO the caller may read but not write is refused, not waited on.

    The FIFO test above plants one this user owns, so the create opens it
    O_RDWR and the regular-file check refuses it. Deny write and the create
    fails EACCES first, which sends it to the read-only retry — and a read-only
    open of a FIFO waits for a writer. Nothing supplies one, so without
    O_NONBLOCK on that open the timeout is never consulted and every writer of
    this name parks forever: a shape the lock is supposed to refuse turned into
    a hang, planted by anyone who owns a path in a shared root.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    os.mkfifo(lock_path, 0o444)

    with _deadline(5.0):
        with pytest.raises(ConfigError) as excinfo:
            with name_lock(lock_path, "x", timeout=0.2):
                pytest.fail("entered with a FIFO at the lock path")

    assert "not a regular file" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
@pytest.mark.parametrize("mode", [0o000, 0o200])
def test_name_lock_refuses_a_regular_lock_file_it_cannot_open(tmp_path, mode):
    """A descriptor is the mechanism, so no descriptor has to mean no entry.

    A regular file at the lock path that neither open can obtain is the one
    unopenable shape that is not obviously a plant: a lock another user created
    under a restrictive umask looks exactly like this. Degrading for it used to
    seem like the conservative choice — it keeps a shared root working — but
    what it actually does is hand every process that meets the file a lock that
    excludes nobody, silently and permanently, which is the whole failure this
    context manager exists to prevent. Refuse, and say so.

    0200 is the second mode because it denies the two opens separately rather
    than at a stroke: the O_RDWR create needs read as well as write, and the
    read-only retry needs the read it does not have. 0000 alone would leave a
    create widened to O_WRONLY passing this test.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    os.chmod(lock_path, mode)

    with pytest.raises(ConfigError) as excinfo:
        with name_lock(lock_path, "x", timeout=0.2):
            pytest.fail("entered without a descriptor, so without any exclusion")

    assert "Cannot open the lock file" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
def test_name_lock_widens_its_own_lock_file_past_a_restrictive_umask(tmp_path):
    """What keeps the refusal above off stack's own locks.

    The create takes ``0o666 & ~umask``, so whichever user reaches a name first
    fixes its lock file's mode for everyone after them — and a 077 umask leaves
    one that no other user can open. That is not a plant and not a
    misconfiguration; it is the default outcome of a common umask, and under the
    refusal above it would turn a shared root into a hard error for every user
    but one. Re-applying the mode on each acquisition keeps the file openable
    however it was created.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    previous = os.umask(0o077)
    try:
        with name_lock(lock_path, "x", timeout=0.2):
            pass
    finally:
        os.umask(previous)

    mode = stat.S_IMODE(lock_path.stat().st_mode)
    assert mode == 0o666, f"a 077 umask left the lock file {oct(mode)}, unopenable by others"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_still_locks_when_the_mode_re_apply_is_refused(tmp_path, monkeypatch):
    """The widening above is a courtesy to the next user, never this one's business.

    It can be refused. POSIX reserves ``chmod`` to the file's owner and to root,
    so another user's lock file is one way; a file this user owns is another,
    since the owner can be refused too — macOS does it for a file flagged
    ``uchg``, measured here on a descriptor this process had just opened
    read-only. By then the descriptor is in hand, so letting the error out would
    fail a create whose lock was there for the taking. The refusal is injected
    rather than planted because what produces it is platform-specific.

    A live competitor is checked first: it is what says the lock was really
    taken, rather than the acquisition having quietly degraded past the flock.
    """

    def _refuse(fd, mode):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", _refuse)
    lock_path = tmp_path / ".locks" / "stem-x.lock"

    with _lock_held_by_another_process(lock_path):
        with pytest.raises(ConfigError) as excinfo:
            with name_lock(lock_path, "x", timeout=0.2):
                pytest.fail("entered while another process held the lock")
    assert "another stack process" in str(excinfo.value)

    entered = False
    with name_lock(lock_path, "x", timeout=0.2):
        entered = True
    assert entered, "a refused mode re-apply was allowed to fail the acquisition"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.parametrize("swap", ["rename", "symlink"])
def test_name_lock_widens_the_descriptor_it_holds_not_the_path(tmp_path, monkeypatch, swap):
    """The mode goes to the open file, so a swap after the open cannot redirect it.

    Who can replace the name depends on the umask that created ``.locks`` — it is
    ``0o777 & ~umask``, measured 0o755 under a 022 and 0o775 under a 002 — but its
    owner always can, and a rename is atomic. Against the path the widening would
    reach whatever answers to the name by then and chmod that 0666; against the
    descriptor it reaches the file that was actually opened. ``O_NOFOLLOW`` does
    not cover this: it refuses a symlink standing there at open time, not an
    object put there a moment later.

    Both replacements are kept because a path-based widening harms them
    differently: a renamed-in regular file is left 0666 even by a rewrite that
    declines to follow symlinks, while a symlink swap costs only the link's own
    mode. So the rename is the shape the assertion on the name catches alone; the
    symlink is caught only by the hard link.

    That hard link, taken from the open file just before the swap, is what says
    the widening happened at all — the assertion on the swapped-in file is an
    absence, which a widening that does nothing satisfies just as well. The umask
    is forced so the create cannot already leave the mode the widening would set,
    and the swap is driven from the ``fstat`` immediately before the widening, the
    only hook between it and the open.

    The name is read at the instant the widening returns rather than after the
    block, because the same swap now also fails the post-lock identity recheck:
    the acquisition releases and reopens, and the rename it then meets is a
    regular file it legitimately widens on the second pass. Reading afterwards
    would see that second widening and call it a leak. The symlink meets
    ``O_NOFOLLOW`` on the way back in instead, so that half raises — which is
    the documented refusal of a symlink at the lock path, reached a moment
    later than usual.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    held = tmp_path / "held"
    victim = tmp_path / "victim"
    victim.write_text("")
    os.chmod(victim, 0o600)

    real_fstat = os.fstat
    real_fchmod = os.fchmod
    swapped = []
    name_after_widening = []

    def _swap_the_name(fd, **kwargs):
        found = real_fstat(fd, **kwargs)
        if not swapped and lock_path.is_file() and os.path.samestat(found, os.lstat(lock_path)):
            os.link(lock_path, held)
            if swap == "rename":
                os.rename(victim, lock_path)
            else:
                lock_path.unlink()
                lock_path.symlink_to(victim)
            swapped.append(True)
        return found

    def _record_the_name(fd, mode):
        real_fchmod(fd, mode)
        if not name_after_widening:
            # Resolves to the renamed-in file directly, and to the symlink's
            # target otherwise, so one reading covers both swaps.
            name_after_widening.append(stat.S_IMODE(os.stat(lock_path).st_mode))

    monkeypatch.setattr(os, "fstat", _swap_the_name)
    monkeypatch.setattr(os, "fchmod", _record_the_name)

    reopen = pytest.raises(OSError) if swap == "symlink" else contextlib.nullcontext()
    previous = os.umask(0o077)
    try:
        with reopen:
            with name_lock(lock_path, "x", timeout=0.2):
                pass
    finally:
        os.umask(previous)

    assert swapped, "the swap never happened, so nothing was exercised"
    opened = stat.S_IMODE(held.stat().st_mode)
    assert opened == 0o666, f"the widening never reached the file it held open: {oct(opened)}"
    assert name_after_widening == [0o600], (
        f"the widening followed the name and left {name_after_widening}"
    )


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
def test_name_lock_degrades_where_it_never_learned_the_file_was_unopenable(tmp_path, monkeypatch):
    """The refusal is for what was tried and failed, not for what was skipped.

    Without the open guards the read-only retry is not attempted, so the EACCES
    that reached here came from the ``O_RDWR`` create alone — and that says
    nothing about whether the file could have been locked, since a 0444 file
    fails the create and satisfies the retry. Refusing on that evidence would
    fail every readable lock file on such a platform. Degrade instead: this is
    the same "we never found out" the missing-``fcntl`` path already takes.
    """
    from uv_stack import fsutil

    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    os.chmod(lock_path, 0o000)
    monkeypatch.setattr(fsutil, "_FASTPATH_AVAILABLE", False)

    entered = False
    with name_lock(lock_path, "x", timeout=0.2):
        entered = True

    assert entered, "refused on evidence the platform never gathered"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
def test_name_lock_skips_the_retry_it_cannot_guard(tmp_path, monkeypatch):
    """Without both guard flags the read-only retry is not attempted at all.

    The retry is the one open in name_lock that can be declined — declining
    costs this name its lock, which the degrade below already accepts, while
    taking it unguarded risks following a planted symlink or waiting on a
    planted FIFO. So a platform missing either constant skips it.

    A regular file this user may read but not write is the target that tells
    skipping apart from opening: it is the only shape the retry would go on to
    lock rather than refuse, so whether ``flock`` is reached says which
    happened. A planted shape cannot distinguish them — it is refused on both
    paths. Taking the lock with the flags present is the control: it proves the
    file is lockable, so the silence in the second half is the skip and not some
    unrelated failure to lock.
    """
    import fcntl

    from uv_stack import fsutil

    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    os.chmod(lock_path, 0o444)

    locked = []
    real_flock = fcntl.flock

    def _spy(fd, operation):
        locked.append(operation)
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", _spy)

    with name_lock(lock_path, "x", timeout=0.2):
        pass
    assert locked, "a readable regular file should have been locked via the retry"

    locked.clear()
    # The control half took the lock, and taking it re-applies 0666 so the next
    # user of a shared root is never locked out. Narrow it again, or the second
    # half's create would simply succeed and never reach the retry at all.
    os.chmod(lock_path, 0o444)
    monkeypatch.setattr(fsutil, "_FASTPATH_AVAILABLE", False)

    with name_lock(lock_path, "x", timeout=0.2):
        pass
    assert not locked, "the retry was taken despite the guard flags being absent"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform lacks mkfifo")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
@pytest.mark.parametrize("mode", [0o200, 0o000])
def test_name_lock_refuses_a_plant_it_cannot_open(tmp_path, mode):
    """A plant whose permission bits deny every open is refused, not degraded.

    Both opens fail EACCES, which is what a lock file owned by another user on
    a shared root also looks like — and that degrades to a no-op. Reading the
    two the same way costs this name its exclusion for as long as the plant
    sits there, silently, which is the one failure this lock must never have.
    ``lstat`` tells them apart: it needs no permission on the file itself, only
    search on ``.locks``.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    os.mkfifo(lock_path, mode)

    with _deadline(5.0):
        with pytest.raises(ConfigError) as excinfo:
            with name_lock(lock_path, "x", timeout=0.2):
                pytest.fail("entered with a FIFO at the lock path")

    assert "not a regular file" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
@pytest.mark.parametrize(
    "shape", [stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK], ids=["socket", "chardev", "blockdev"]
)
def test_name_lock_refuses_an_unopenable_plant_of_any_shape(tmp_path, monkeypatch, shape):
    """The plant check asks for a regular file, not for the shapes we can plant.

    A socket is what separates those two predicates. Where the kernel checks
    permission before type — Linux does — a socket whose mode denies both opens
    reaches this check, and a predicate narrowed to FIFOs would degrade instead
    of refusing it: exactly the silent loss of exclusion the check exists to
    prevent. This platform declines a socket by type at every mode, so the
    shape that discriminates cannot be planted here. The type is fabricated
    instead, over a real file whose bits genuinely deny both opens, so
    everything up to the ``lstat`` is the ordinary EACCES route.

    The device nodes are the same argument one step further out: they need
    privilege to create, so no test can plant them either, and a predicate
    written as a list of the shapes a test happens to use would let them
    through. Only the parametrization makes ``not S_ISREG`` the thing under
    test rather than the two shapes this suite can produce.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    os.chmod(lock_path, 0o000)

    real_lstat = os.lstat

    def _report_the_shape(target, **kwargs):
        found = real_lstat(target, **kwargs)
        if os.fspath(target) == str(lock_path):
            return os.stat_result((shape | 0o000, *tuple(found)[1:]))
        return found

    monkeypatch.setattr(os, "lstat", _report_the_shape)

    with pytest.raises(ConfigError) as excinfo:
        with name_lock(lock_path, "x", timeout=0.2):
            pytest.fail("entered with a non-regular file at the lock path")

    assert "not a regular file" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.parametrize(
    "shape", [stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK], ids=["socket", "chardev", "blockdev"]
)
def test_name_lock_refuses_an_openable_plant_of_any_shape(tmp_path, monkeypatch, shape):
    """The same generality argument, for the check on the far side of the open.

    A plant this user can open never reaches the ``lstat`` above: it yields a
    descriptor, and only the ``fstat`` refuses it. The two checks are separate
    predicates in separate branches, and only a FIFO had ever reached this one
    — so narrowing it to ``S_ISFIFO`` left the entire suite green while
    ``name_lock`` would have gone on to ``flock`` an openable device node and
    take the degrade its errno lands in. The shape is fabricated for the reason
    the case above gives: the openable non-regular shapes need privilege to
    create.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    planted = os.stat(lock_path).st_ino

    real_fstat = os.fstat

    def _report_the_shape(fd, **kwargs):
        found = real_fstat(fd, **kwargs)
        if found.st_ino == planted:
            return os.stat_result((shape | 0o666, *tuple(found)[1:]))
        return found

    monkeypatch.setattr(os, "fstat", _report_the_shape)

    with pytest.raises(ConfigError) as excinfo:
        with name_lock(lock_path, "x", timeout=0.2):
            pytest.fail("entered with a non-regular file at the lock path")

    assert "not a regular file" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.skipif(not os.path.exists("/dev/null"), reason="requires /dev/null")
def test_name_lock_refuses_a_real_openable_non_regular_file():
    """One plant that reaches the descriptor check with nothing fabricated.

    Every other case here either plants the one shape this suite can create or
    fabricates the type, so the refusal has only ever been watched through a
    monkeypatch. ``/dev/null`` is an openable character device needing no
    privilege to reach, and it drives the create, the ``fstat`` and the refusal
    exactly as a planted device node would.
    """
    with pytest.raises(ConfigError) as excinfo:
        with name_lock(Path("/dev/null"), "x", timeout=0.2):
            pytest.fail("entered with a character device at the lock path")

    assert "not a regular file" in str(excinfo.value)


def _still_locked_against_a_fresh_open(path: Path) -> bool:
    """Whether a second open of ``path`` is refused the lock this process holds.

    ``flock`` conflicts between two open file descriptions of the same file even
    within one process, so this answers "is the object standing at ``path``
    right now the one the caller has locked" without a second process.

    :param path: Lock file to test.
    :returns: ``True`` if a fresh non-blocking acquisition is refused.
    """
    import fcntl

    # Not inside the try: an implementation that left the lock on a file since
    # unlinked would otherwise fail its caller with a bare FileNotFoundError
    # from here rather than the caller's own message.
    assert path.exists(), f"nothing stands at {path} to test the lock against"
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
@pytest.mark.parametrize("gone", ["renamed over", "unlinked"])
def test_name_lock_holds_the_file_at_the_name_not_the_one_it_opened(tmp_path, monkeypatch, gone):
    """A lock is only handed back on the object the name currently resolves to.

    ``flock`` binds to the open file description, so a lock taken on a file that
    has since been moved out from under the name excludes nobody: the next
    process opens whatever answers to the name and locks that instead, and both
    run the critical section at once. The assertion is made from inside the
    block, where a fresh open of the name must meet the lock this process is
    holding — an assertion after the block, or on the swap alone, would pass
    just as well against a lock on the file that was swapped away.

    Both disappearances are kept because they reach the identity check by
    different routes: a rename leaves it two stats to compare, while an unlink
    leaves it nothing to stat at all, and treating that second case as a match
    would hand back a lock on a file no other process can even open.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    real_fchmod = os.fchmod
    swaps: list[bool] = []

    def _lose_the_file(fd, mode):
        real_fchmod(fd, mode)
        if swaps:
            return
        if gone == "renamed over":
            fresh = lock_path.parent / "fresh"
            fresh.write_text("")
            os.rename(fresh, lock_path)
        else:
            lock_path.unlink()
        swaps.append(True)

    monkeypatch.setattr(os, "fchmod", _lose_the_file)

    with name_lock(lock_path, "x", timeout=2.0):
        assert _still_locked_against_a_fresh_open(lock_path), (
            "the lock is on the file that was swapped away, not the one at the name"
        )

    assert swaps, "the swap never happened, so nothing was exercised"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_waits_for_the_holder_of_the_file_that_replaced_it(tmp_path, monkeypatch):
    """The replacement is contended, so the reopen must block rather than enter.

    The swapped-in file is one another process already holds. Taking the lock on
    the file that was opened first would sail past that holder — which is the
    concurrency failure itself, not merely an identity mismatch — so this pins
    the outcome rather than the mechanism.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    replacement = tmp_path / ".locks" / "replacement"
    real_fchmod = os.fchmod
    swaps: list[bool] = []

    def _swap_in_the_held_file(fd, mode):
        real_fchmod(fd, mode)
        if not swaps:
            # The holder keeps its descriptor across the rename, so the file
            # arrives at the lock path already locked.
            os.rename(replacement, lock_path)
            swaps.append(True)

    with _lock_held_by_another_process(replacement):
        monkeypatch.setattr(os, "fchmod", _swap_in_the_held_file)
        with pytest.raises(ConfigError) as excinfo:
            with name_lock(lock_path, "x", timeout=0.2):
                pytest.fail("entered while another process held the file at the lock path")

    assert swaps, "the swap never happened, so nothing was exercised"
    assert "another stack process" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_gives_up_when_the_lock_file_keeps_being_replaced(tmp_path, monkeypatch):
    """The reopen shares the caller's timeout instead of retrying forever.

    A replacement on every acquisition is the shape that would spin: each pass
    takes a lock, finds the name has moved on, and reopens. The deadline is
    computed once for the whole acquisition, so the loop ends on it — and says
    what actually happened rather than blaming a competing ``stack`` process
    that was never there.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    real_fchmod = os.fchmod
    swaps: list[bool] = []

    def _swap_every_time(fd, mode):
        real_fchmod(fd, mode)
        fresh = lock_path.parent / "fresh"
        fresh.write_text("")
        os.rename(fresh, lock_path)
        swaps.append(True)

    monkeypatch.setattr(os, "fchmod", _swap_every_time)

    started = time.monotonic()
    with pytest.raises(ConfigError) as excinfo:
        with name_lock(lock_path, "x", timeout=0.2):
            pytest.fail("entered on a file the name no longer resolves to")
    elapsed = time.monotonic() - started

    assert len(swaps) > 1, "the acquisition never reopened, so no retry was exercised"
    assert "keeps being replaced" in str(excinfo.value)
    # Generous against a loaded CI box while still failing an unbounded retry.
    assert elapsed < 5.0, f"the retry outlived the timeout it was given: {elapsed:.1f}s"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_name_lock_keeps_a_lock_whose_name_cannot_be_examined(tmp_path, monkeypatch):
    """A name that cannot be stat-ed is not evidence of a swap, so the lock stands.

    Only a name that is gone means the next process will open something else.
    Every other ``lstat`` failure — a parent that stopped being searchable, a
    mount answering ESTALE — leaves the question unanswered, and answering it
    "swapped" throws away a lock the kernel granted for nothing in return.
    Under the shape here that costs the whole timeout; under an unsearchable
    parent it costs the lock outright, which is the one outcome this context
    manager exists to prevent.

    Refusing on every call rather than once is what makes this discriminate.
    Treating the failure as a swap then reopens, hits it again, and burns the
    whole timeout — so the mistake shows up as a raise, not as a lock silently
    downgraded somewhere the assertions cannot see it.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    real_lstat = Path.lstat
    refusals: list[bool] = []

    def _refuse(self: Path, *args, **kwargs):
        if self == lock_path:
            refusals.append(True)
            raise PermissionError("Permission denied")
        return real_lstat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", _refuse)

    with name_lock(lock_path, "x", timeout=0.2):
        assert refusals, "the post-grant identity check never ran"
        assert _still_locked_against_a_fresh_open(lock_path), (
            "a name that could not be examined threw away a lock that was held"
        )


def test_nofollow_read_flags_composes_both_guards():
    """Both guard flags plus O_RDONLY, so one function is the single decision point."""
    from uv_stack import fsutil

    flags = fsutil.nofollow_read_flags()
    assert flags is not None
    assert flags == os.O_RDONLY | fsutil._O_NOFOLLOW | fsutil._O_NONBLOCK


def test_nofollow_read_flags_declines_without_both_guards(monkeypatch):
    """A platform missing either flag gets None, never a weakened flag set.

    The caller's contract is "decline the read rather than make an unsafe
    one", so returning O_RDONLY alone would silently convert every guarded
    read into an unguarded one.
    """
    from uv_stack import fsutil

    monkeypatch.setattr(fsutil, "_FASTPATH_AVAILABLE", False)
    assert fsutil.nofollow_read_flags() is None


def test_link_or_copy_no_replace_reports_link(tmp_path):
    """The normal path publishes src's own inode and says so."""
    from uv_stack.fsutil import link_or_copy_no_replace

    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("content\n")
    published = link_or_copy_no_replace(src, dst)
    assert published.kind == "link"
    assert (published.stat.st_dev, published.stat.st_ino) == (
        dst.stat().st_dev,
        dst.stat().st_ino,
    )
    assert dst.read_text() == "content\n"


def test_link_or_copy_no_replace_reports_link_even_when_src_was_replaced(tmp_path, monkeypatch):
    """A pre-link replacement of src does not turn a link into a reported copy.

    This is the misclassification the returned ``kind`` exists to prevent: the
    two identities differ, but os.link still published a second name for an
    inode this call did not create, so the caller must keep the link path's
    weaker provenance rules.
    """
    import os as _os

    from uv_stack.fsutil import link_or_copy_no_replace

    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("original\n")
    real_link = _os.link

    def racing_link(a, b, **kwargs):
        src.unlink()
        src.write_text("replacement\n")
        real_link(a, b, **kwargs)

    monkeypatch.setattr(_os, "link", racing_link)
    published = link_or_copy_no_replace(src, dst)
    assert published.kind == "link"
    # The stat is what src named when this call read it, NOT what got published.
    assert (published.stat.st_dev, published.stat.st_ino) != (
        dst.stat().st_dev,
        dst.stat().st_ino,
    )
    assert dst.read_text() == "replacement\n"


def test_link_or_copy_no_replace_reports_copy_on_a_link_less_filesystem(tmp_path, monkeypatch):
    """Without hard links the bytes are copied into a fresh, provably-ours inode."""
    import errno as _errno
    import os as _os

    from uv_stack.fsutil import link_or_copy_no_replace

    def _no_link(a, b, **kwargs):
        raise OSError(_errno.EOPNOTSUPP, "hard links not supported")

    monkeypatch.setattr(_os, "link", _no_link)
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("content\n")
    published = link_or_copy_no_replace(src, dst)
    assert published.kind == "copy"
    assert dst.read_text() == "content\n"
    assert (dst.stat().st_dev, dst.stat().st_ino) != (src.stat().st_dev, src.stat().st_ino)
    assert (published.stat.st_dev, published.stat.st_ino) == (
        dst.stat().st_dev,
        dst.stat().st_ino,
    )
    assert published.stat.st_size == len(b"content\n")


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
def test_link_or_copy_no_replace_copy_preserves_source_mode(tmp_path, monkeypatch):
    """The copy is a publication of an existing file, so it keeps that file's bits."""
    import errno as _errno
    import os as _os
    import stat as _stat

    from uv_stack.fsutil import link_or_copy_no_replace

    def _no_link(a, b, **kwargs):
        raise OSError(_errno.EOPNOTSUPP, "hard links not supported")

    monkeypatch.setattr(_os, "link", _no_link)
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("content\n")
    _os.chmod(src, 0o640)
    link_or_copy_no_replace(src, dst)
    assert _stat.S_IMODE(dst.stat().st_mode) == 0o640


def test_link_or_copy_no_replace_refuses_an_existing_destination_on_both_paths(
    tmp_path, monkeypatch
):
    """FileExistsError semantics are identical whichever mechanism would run."""
    import errno as _errno
    import os as _os

    from uv_stack.fsutil import link_or_copy_no_replace

    src = tmp_path / "src.txt"
    src.write_text("content\n")
    linked_dst = tmp_path / "linked.txt"
    linked_dst.write_text("occupied\n")
    with pytest.raises(FileExistsError):
        link_or_copy_no_replace(src, linked_dst)
    assert linked_dst.read_text() == "occupied\n"

    def _no_link(a, b, **kwargs):
        raise OSError(_errno.EOPNOTSUPP, "hard links not supported")

    monkeypatch.setattr(_os, "link", _no_link)
    copied_dst = tmp_path / "copied.txt"
    copied_dst.write_text("occupied\n")
    with pytest.raises(FileExistsError):
        link_or_copy_no_replace(src, copied_dst)
    assert copied_dst.read_text() == "occupied\n"


def test_link_or_copy_no_replace_propagates_an_unlisted_link_errno(tmp_path, monkeypatch):
    """ENOSPC is not "this filesystem cannot hard-link", so it must not fall back."""
    import errno as _errno
    import os as _os

    from uv_stack.fsutil import link_or_copy_no_replace

    def _full(a, b, **kwargs):
        raise OSError(_errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(_os, "link", _full)
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("content\n")
    with pytest.raises(OSError) as excinfo:
        link_or_copy_no_replace(src, dst)
    assert excinfo.value.errno == _errno.ENOSPC
    assert not dst.exists()


def test_link_or_copy_no_replace_withdraws_a_partial_copy(tmp_path, monkeypatch):
    """A mid-copy failure must not leave a truncated file at a no-clobber target."""
    import errno as _errno
    import os as _os

    from uv_stack.fsutil import link_or_copy_no_replace

    def _no_link(a, b, **kwargs):
        raise OSError(_errno.EOPNOTSUPP, "hard links not supported")

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

        def write(self, data):
            raise OSError("disk full")

    def _fdopen(fd, *args, **kwargs):
        return _ExplodingWriter(real_fdopen(fd, *args, **kwargs))

    monkeypatch.setattr(_os, "fdopen", _fdopen)
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("content\n")
    with pytest.raises(OSError):
        link_or_copy_no_replace(src, dst)
    assert not dst.exists()
