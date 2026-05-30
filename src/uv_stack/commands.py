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


def uv_pip_install(python: str, lock: Path) -> Command:
    """Build ``uv pip install -r <lock>`` against a specific interpreter."""
    return Command(["uv", "pip", "install", "--python", python, "-r", str(lock)])


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


def uv_sync() -> Command:
    """Build ``uv sync``."""
    return Command(["uv", "sync"])
