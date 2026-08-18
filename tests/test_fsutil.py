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
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    os.mkfifo(lock_path)

    with pytest.raises(ConfigError) as excinfo:
        with name_lock(lock_path, "x"):
            pytest.fail("entered with a FIFO at the lock path")

    assert "not a regular file" in str(excinfo.value)


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
def test_name_lock_refuses_an_unopenable_plant_of_any_shape(tmp_path, monkeypatch):
    """The plant check asks for a regular file, not for the shapes we can plant.

    A socket is what separates those two predicates. Where the kernel checks
    permission before type — Linux does — a socket whose mode denies both opens
    reaches this check, and a predicate narrowed to FIFOs would degrade instead
    of refusing it: exactly the silent loss of exclusion the check exists to
    prevent. This platform declines a socket by type at every mode, so the
    shape that discriminates cannot be planted here. The type is fabricated
    instead, over a real file whose bits genuinely deny both opens, so
    everything up to the ``lstat`` is the ordinary EACCES route.
    """
    lock_path = tmp_path / ".locks" / "stem-x.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("")
    os.chmod(lock_path, 0o000)

    real_lstat = os.lstat

    def _report_a_socket(target, **kwargs):
        found = real_lstat(target, **kwargs)
        if os.fspath(target) == str(lock_path):
            return os.stat_result((stat.S_IFSOCK | 0o000, *tuple(found)[1:]))
        return found

    monkeypatch.setattr(os, "lstat", _report_a_socket)

    with pytest.raises(ConfigError) as excinfo:
        with name_lock(lock_path, "x", timeout=0.2):
            pytest.fail("entered with a socket at the lock path")

    assert "not a regular file" in str(excinfo.value)
