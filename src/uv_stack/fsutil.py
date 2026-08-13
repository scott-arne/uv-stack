"""Filesystem helpers shared by operations."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    The content is written to a temporary file in the same directory and then
    moved into place with :func:`os.replace`, so a crash mid-write never leaves
    a partially-written target.

    Identical content is not rewritten (the mtime is preserved) when the target is
    a regular, un-hardlinked file and the exact bytes match.

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
    # real write.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            st_fd = os.fstat(fd)
            if stat.S_ISREG(st_fd.st_mode) and st_fd.st_nlink == 1:
                with os.fdopen(fd, "rb") as handle:
                    fd = -1  # ownership transferred to file object
                    current_bytes = handle.read()
                if current_bytes == text.encode():
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
        with os.fdopen(fd, "w") as handle:
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

    :param path: Destination file (must not exist).
    :param text: Content to write.
    :returns: The stat of the published inode, captured race-free from the temporary file.
    :raises FileExistsError: If ``path`` already exists at publication time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        # Capture the identity before linking: the temp file IS the published inode
        # once linked — os.link creates a second name for the same inode.
        identity = os.stat(tmp_name)
        os.link(tmp_name, path)
        return identity
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
