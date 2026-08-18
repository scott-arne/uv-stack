from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot


@pytest.fixture
def config_tree(tmp_path: Path) -> ConfigRoot:
    """A minimal but realistic config tree under tmp_path.

    profiles: ds, chem, utils
    bundles:  standard (ds chem utils), qsar (standard + umap-learn)
    envs:     main (stack: @standard, python 3.12, micromamba: graphviz,
              channels: bioconda)
    """
    root = tmp_path / "python-envs"
    (root / "profiles").mkdir(parents=True)
    (root / "bundles").mkdir(parents=True)
    (root / "envs" / "main").mkdir(parents=True)

    (root / "profiles" / "ds.yaml").write_text(
        "description: Core data-science stack\n"
        "tags: [data, core]\n"
        "includes:\n  - numpy\n  - pandas\n"
    )
    (root / "profiles" / "chem.yaml").write_text(
        "description: Cheminformatics\nincludes:\n  - rdkit\n"
    )
    (root / "profiles" / "utils.yaml").write_text("includes:\n  - rich\n")

    (root / "bundles" / "standard.yaml").write_text(
        "description: Everything for daily work\n"
        "includes:\n  - ds\n  - chem\n  - utils\n"
    )
    (root / "bundles" / "qsar.yaml").write_text(
        "includes:\n  - standard\n  - umap-learn\n"
    )

    env = root / "envs" / "main"
    env.joinpath("python.txt").write_text("3.12\n")
    env.joinpath("stack.txt").write_text("@standard\n")
    env.joinpath("micromamba.txt").write_text("graphviz\n")
    env.joinpath("channels.txt").write_text("bioconda\n")
    env.joinpath("requirements.local.in").write_text("some-local-pkg\n")

    return ConfigRoot(root)


_HOLD_LOCK = """\
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o666)
fcntl.flock(fd, fcntl.LOCK_EX)
sys.stdout.write("ready\\n")
sys.stdout.flush()
sys.stdin.readline()
"""


@contextmanager
def _lock_held_by_another_process(lock_path: Path) -> Iterator[None]:
    """Run a child process holding an exclusive flock on ``lock_path``.

    Yields once the child confirms it has the lock, so the test body never
    races the child's startup. The child releases by exiting when its stdin
    closes.
    """
    import subprocess

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline() == "ready\n"
        yield
    finally:
        try:
            assert proc.stdin is not None
            proc.stdin.close()
            proc.wait(timeout=10)
        finally:
            # Unconditional: a failed assert or a child that outlives the
            # timeout must not leave a process holding the lock for the rest
            # of the session. kill() no-ops once the child has been reaped,
            # and the wait() reaps it when it has not.
            proc.kill()
            proc.wait()
            proc.stdout.close()
