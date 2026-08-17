"""Filesystem helpers shared by operations."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
from pathlib import Path

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
#: and the identity recheck each caller runs afterwards comes too late to make
#: up for it: it can reject what the open returned, not stop the open from
#: following a symlink or hanging on a FIFO. Callers therefore skip the guarded
#: open entirely rather than take it unguarded; what each gives up by skipping
#: differs, and is stated at the call site.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_FASTPATH_AVAILABLE = bool(_O_NOFOLLOW) and bool(_O_NONBLOCK)


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
