"""Filesystem helpers shared by operations."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from uv_stack.errors import ConfigError

#: os.link failures that mean "this filesystem cannot hard-link" — fall back
#: to exclusive create. EEXIST is deliberately absent: that is the no-clobber
#: contract, never a fallback trigger.
_LINK_FALLBACK_ERRNOS = frozenset(
    {errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK}
    | ({getattr(errno, "ENOTSUP")} if hasattr(errno, "ENOTSUP") else set())  # noqa: B009
)

#: True only where BOTH guard flags exist. Opening a target whose type is not
#: known in advance needs both: O_NOFOLLOW so a symlinked target is never
#: opened, O_NONBLOCK so a FIFO target never blocks the open. Where either is
#: absent getattr yields 0, which does not weaken the guard — it removes it,
#: and an identity recheck afterwards comes too late to make up for it: it can
#: reject what the open returned, not stop the open from following a symlink
#: or hanging on a FIFO. Both callers that can decline the open therefore skip
#: it entirely rather than take it unguarded; what each gives up by skipping
#: differs, and is stated at the call site. name_lock is the one caller that
#: cannot decline — there is no lock without the open — so it takes O_NOFOLLOW
#: alone and rejects a non-regular target after the fact.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_FASTPATH_AVAILABLE = bool(_O_NOFOLLOW) and bool(_O_NONBLOCK)


def _lock_supported() -> bool:
    """Whether this platform has the POSIX advisory locking ``name_lock`` needs."""
    try:
        import fcntl  # noqa: F401
    except ImportError:  # pragma: no cover - non-Unix platforms
        return False
    return True


#: True only where fcntl.flock exists. Where it does not, name_lock is a no-op
#: and the writers fall back to their post-publish collision checks alone: the
#: both-processes-survive race stays closed, while the killed-mid-window case
#: and the adopter residual do not. Same degrade-silently posture as
#: _FASTPATH_AVAILABLE above and _PTY_AVAILABLE in runner.py.
_LOCK_AVAILABLE = _lock_supported()

#: Read at call time, not bound as a default argument, so a test can shorten it.
_LOCK_TIMEOUT = 5.0

#: How long to wait between flock attempts. Small enough that an uncontended
#: handoff is imperceptible, large enough not to spin.
_LOCK_POLL = 0.01

#: Errnos treated as contention — another process holds the lock. EACCES is
#: here because CPython's fcntl.flock falls back to fcntl(F_SETLK) where the
#: build lacks HAVE_FLOCK, and POSIX permits F_SETLK to report a conflicting
#: lock as either EACCES or EAGAIN. Any other errno is treated as "this
#: filesystem cannot lock" and degrades to the no-fcntl no-op: some NFS and
#: FUSE mounts return ENOLCK or EOPNOTSUPP, and the degrade is chosen so an
#: unlockable filesystem keeps working rather than failing every create. The
#: trade-off is that a transient ENOLCK (e.g., kernel out of memory for lock
#: records on Linux) also degrades, where a retry loop might succeed.
_LOCK_CONTENDED_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    The content is written to a temporary file in the same directory and then
    moved into place with :func:`os.replace`, so a crash mid-write never leaves
    a partially-written target.

    Identical content is not rewritten (the mtime is preserved) when the target is
    a regular, un-hardlinked file and the exact bytes match. That check needs
    ``O_NOFOLLOW`` and ``O_NONBLOCK`` to be safe, so on a platform missing either
    constant it is not made and every write is a real write.

    Writes are byte-exact: UTF-8, no newline translation.

    :param path: Destination file.
    :param text: Content to write.
    """
    # Skip identical rewrites: generated files keep their mtime, so
    # mtime-based staleness checks (stack status) see no phantom drift
    # after a dry-run re-render. Only a regular, un-hardlinked file may
    # be skipped — symlinks, extra links, and special files must still
    # be replaced — and the comparison is on exact bytes so newline
    # differences count as changes. The skip is bound to one file descriptor
    # and path identity is rechecked so a concurrent swap falls through to a
    # real write. The whole path runs only when _FASTPATH_AVAILABLE: without
    # both open flags the open can block on a FIFO or land on a symlink's
    # target, and the recheck comes too late to prevent either. Declining the
    # skip costs a rewrite of unchanged content — a new inode and a fresh
    # mtime, which is exactly the phantom drift described above.
    if _FASTPATH_AVAILABLE:
        try:
            flags = os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK
            fd = os.open(path, flags)
            try:
                st_fd = os.fstat(fd)
                if stat.S_ISREG(st_fd.st_mode) and st_fd.st_nlink == 1:
                    with os.fdopen(fd, "rb") as handle:
                        fd = -1  # ownership transferred to file object
                        current_bytes = handle.read()
                    if current_bytes == text.encode("utf-8"):
                        # Re-verify path identity: same inode as the fd we read from?
                        st_path = os.lstat(path)
                        if (st_fd.st_dev, st_fd.st_ino) == (st_path.st_dev, st_path.st_ino):
                            return  # identical content, verified no swap
            finally:
                if fd != -1:
                    os.close(fd)
        except OSError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        # mkstemp creates the file 0600; relax it to the conventional file mode
        # (honoring the process umask) so generated config files are readable
        # like the hand-authored sources alongside them.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def atomic_write_new(path: Path, text: str) -> os.stat_result:
    """Write ``text`` to ``path`` atomically, failing if ``path`` exists.

    The content is written to a temporary file and published with
    :func:`os.link`, which refuses to replace an existing target — the
    no-clobber counterpart of :func:`atomic_write` for user-authored files.
    Falls back to an exclusive O_CREAT|O_EXCL create on filesystems without
    hard links; FileExistsError semantics are identical on both paths.

    Writes are byte-exact: UTF-8, no newline translation.

    :param path: Destination file (must not exist).
    :param text: Content to write.
    :returns: The stat of the published inode, captured race-free from the temporary file.
    :raises FileExistsError: If ``path`` already exists at publication time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        # Capture the identity before linking: the temp file IS the published inode
        # once linked — os.link creates a second name for the same inode.
        identity = os.stat(tmp_name)
        try:
            os.link(tmp_name, path)
        except FileExistsError:
            raise
        except OSError as error:
            if error.errno not in _LINK_FALLBACK_ERRNOS:
                raise
            # Link-less filesystem (FAT/exFAT, some network mounts): fall
            # back to exclusive create — still no-clobber, losing only the
            # write-then-publish atomicity of the content.
            fallback_fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            created = os.fstat(fallback_fd)
            try:
                with os.fdopen(fallback_fd, "w", encoding="utf-8", newline="") as handle:
                    handle.write(text)
                    handle.flush()
                    # Return post-write stat: same inode as 'created', but with
                    # correct size/mtime after content flush.
                    return os.fstat(handle.fileno())
            except BaseException:
                # Never leave a partial no-clobber target behind: withdraw
                # only while the path still names the inode we created.
                try:
                    current = path.lstat()
                    if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                        path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
                raise
        return identity
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


@contextmanager
def name_lock(path: Path, name: str, *, timeout: float | None = None) -> Iterator[None]:
    """Hold an exclusive interprocess lock at ``path`` for the duration of the block.

    Serializes whole create-a-name operations across processes, so a pre-check,
    the publish it guards, and the post-check that confirms it cannot be
    interleaved with another process doing the same. The kernel releases the
    lock when the holder exits or dies, so a killed holder never wedges the
    name.

    The lock file is created on first use and **never removed**. Unlinking a
    file another process may already hold open would let a third process create
    a fresh file at the same path and take a lock that excludes nobody, so the
    empty file is left behind deliberately.

    Where ``fcntl`` does not exist, or where the filesystem refuses to lock
    (any errno outside the contended set — some NFS and FUSE mounts return
    ENOLCK or EOPNOTSUPP), this is a no-op that yields immediately; what that
    gives up is stated at each call site.

    :param path: Lock file. Parent directories are created if absent.
    :param name: The profile/bundle/environment name being created, for the
        timeout message.
    :param timeout: Seconds to wait. ``None`` reads the module default at call
        time.
    :raises ConfigError: If the lock is still held when the timeout expires,
        or if something other than a regular file sits at ``path``.
    :raises OSError: If the lock file cannot be created or opened — a bad
        config root, an over-long name, or a symlink planted at ``path``
        surfaces here rather than at the write it guards.
    """
    if not _LOCK_AVAILABLE:
        yield
        return

    import fcntl

    limit = _LOCK_TIMEOUT if timeout is None else timeout
    path.parent.mkdir(parents=True, exist_ok=True)
    # 0o666 & ~umask, matching atomic_write_new: a lock file created by one
    # user must stay openable by another sharing the config root, which 0o600
    # would break. O_NOFOLLOW refuses a planted symlink; unlike the two skips
    # below this open cannot be declined when the flags are missing, since
    # there is no lock without it, so it takes O_NOFOLLOW alone rather than
    # both guards. O_NONBLOCK is not needed to keep the open from waiting —
    # O_RDWR on a FIFO returns immediately on Linux and the BSDs — so the
    # regular-file check below, not the open flags, is what handles a planted
    # FIFO. Where the constant is absent _O_NOFOLLOW is 0 and only the symlink
    # refusal is lost; the locking itself is unaffected.
    fd = os.open(path, os.O_CREAT | os.O_RDWR | _O_NOFOLLOW, 0o666)
    try:
        # flock on a FIFO or device node fails with an errno outside the
        # contended set, which would take the degrade branch below and make
        # this a silent no-op for the name — the one failure this lock must
        # never have. Anyone who can write to a shared config root can plant
        # one, so refuse loudly instead.
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ConfigError(
                f"Lock file is not a regular file: {path}",
                hint="Remove it and retry; stack only ever creates a plain file here.",
            )
        deadline = time.monotonic() + limit
        locked = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError as exc:
                if exc.errno not in _LOCK_CONTENDED_ERRNOS:
                    # This filesystem cannot lock. Degrade to the no-op the
                    # missing-fcntl path takes rather than stalling out the
                    # deadline and blaming a competitor that does not exist.
                    break
                if time.monotonic() >= deadline:
                    raise ConfigError(
                        "Timed out waiting for another stack process to finish "
                        f"creating '{name}'",
                        hint=(
                            "Another stack process may be stuck; retry, or remove "
                            f"{path} if none is running."
                        ),
                    ) from None
                time.sleep(_LOCK_POLL)
        try:
            yield
        finally:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
