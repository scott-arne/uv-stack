"""Pure builders for the ``uv`` and ``micromamba`` commands uv-stack runs.

Keeping command construction in one place (free of side effects) lets both the
real execution path and the ``--dry-run`` plan share identical argv lists, and
makes the command shapes trivially unit-testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from uv_stack.runner import Command

_PYTHON_PATH_SNIPPET = "import sys; print(sys.executable)"


def uv_pip_compile(
    python: str,
    requirements_in: Path,
    output: Path,
    *,
    upgrade: bool = False,
    upgrade_packages: Sequence[str] = (),
) -> Command:
    """Build ``uv pip compile`` for a named environment's requirements."""
    args = ["uv", "pip", "compile", "--python", python, str(requirements_in), "-o", str(output)]
    if upgrade:
        args.append("--upgrade")
    for pkg in upgrade_packages:
        args += ["--upgrade-package", pkg]
    return Command(args)


def uv_pip_sync(python: str, lock: Path, *, editable_mode: str = "compat") -> Command:
    """Build ``uv pip sync <lock>`` against a specific interpreter.

    ``sync`` makes the environment match the lock *exactly* — installing what is
    missing and uninstalling anything not in the lock. This is the correct
    semantics for a fully-locked, spec-driven environment: a plain
    ``uv pip install -r`` is additive and leaves orphaned packages behind, so a
    later upgrade can bump a shared dependency past an orphan's version ceiling
    and leave the env internally inconsistent (caught by ``uv pip check``).

    Passes ``-C editable_mode=<mode>`` to control how setuptools records ``-e``
    packages. ``compat`` (the default) writes a plain-path ``.pth`` pointing at
    the live source tree: it survives recompiled C-extension ``.so`` files and
    resolves cleanly in static analyzers such as Pylance/Pyright. ``strict``
    writes an import-hook finder that snapshots the package's files at install
    time — it breaks when an extension module is rebuilt and resolves poorly in
    Pyright — so it is wrong for the C++ extension packages in this stack. The
    flag is inert for non-editable wheel installs in the lock.

    :param python: Path to the target interpreter.
    :param lock: Path to the compiled lock file to sync against.
    :param editable_mode: setuptools editable layout, ``compat`` or ``strict``.
    :returns: The ``uv pip sync`` command.
    """
    return Command(
        [
            "uv", "pip", "sync", "--python", python,
            "-C", f"editable_mode={editable_mode}",
            str(lock),
        ]
    )


def uv_pip_check(python: str) -> Command:
    """Build ``uv pip check`` against a specific interpreter."""
    return Command(["uv", "pip", "check", "--python", python])


def micromamba_create(environment_yml: Path) -> Command:
    """Build ``micromamba create -f <env.yml> -y``."""
    return Command(["micromamba", "create", "-f", str(environment_yml), "-y"])


def micromamba_remove(env_name: str) -> Command:
    """Build ``micromamba remove -n <env> --all -y``."""
    return Command(["micromamba", "remove", "-n", env_name, "--all", "-y"])


def micromamba_python_path(env_name: str) -> Command:
    """Build a command that prints the env's Python executable path."""
    return Command(
        ["micromamba", "run", "-n", env_name, "python", "-c", _PYTHON_PATH_SNIPPET]
    )


def uv_init(python: str, name: str | None = None) -> Command:
    """Build ``uv init --bare`` for a new project."""
    args = ["uv", "init", "--bare"]
    if name:
        args += ["--name", name]
    args += ["--python", python]
    return Command(args)


def uv_add(requirements_file: Path) -> Command:
    """Build ``uv add --no-sync -r <file>``."""
    return Command(["uv", "add", "--no-sync", "-r", str(requirements_file)])


def uv_sync(python: str | None = None) -> Command:
    """Build ``uv sync``, optionally pinning the project interpreter.

    :param python: Interpreter passed to ``--python``. When omitted, ``uv sync``
        re-resolves the interpreter itself; pass an explicit path to keep the
        project environment deterministic.
    """
    args = ["uv", "sync"]
    if python:
        args += ["--python", python]
    return Command(args)
