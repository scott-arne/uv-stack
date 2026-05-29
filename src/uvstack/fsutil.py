"""Filesystem helpers shared by operations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    The content is written to a temporary file in the same directory and then
    moved into place with :func:`os.replace`, so a crash mid-write never leaves
    a partially-written target.

    :param path: Destination file.
    :param text: Content to write.
    """
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
