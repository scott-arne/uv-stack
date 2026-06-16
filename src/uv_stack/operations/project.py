"""The ``project init`` operation: scaffold a uv project from a stack.

The resolved stack is flattened (profiles expanded inline) so the project's
dependencies do not reference files under the config root. Dependencies are
written to a temp requirements file and added via ``uv add``.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from uv_stack.commands import micromamba_python_path, uv_add, uv_init, uv_sync
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, EnvError
from uv_stack.render import render_requirements_flat
from uv_stack.resolver import Resolver
from uv_stack.runner import Command, Runner

#: Environment variable consulted between the ``--python`` flag and the
#: config-root default when selecting the project interpreter.
PROJECT_PYTHON_ENV = "UV_STACK_PROJECT_PYTHON"

#: Fallback interpreter spec when nothing else is configured.
DEFAULT_PROJECT_PYTHON = "3.12"

_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")
_IMPLEMENTATION_RE = re.compile(r"^(cpython|pypy|graalpy)[-@]")


@dataclass
class ProjectOptions:
    """Options for ``project init``.

    :param python: Interpreter spec — a Python version, a micromamba env name,
        or ``None`` to fall back to ``UV_STACK_PROJECT_PYTHON``, then the
        config-root default, then ``"3.12"``.
    :param name: Optional project name passed to ``uv init``.
    :param no_sync: Add dependencies but skip ``uv sync``.
    :param force: Allow adding to an existing ``pyproject.toml``.
    """

    python: str | None = None
    name: str | None = None
    no_sync: bool = False
    force: bool = False


def init_project(
    config: ConfigRoot,
    runner: Runner,
    tokens: list[str],
    options: ProjectOptions,
    *,
    cwd: Path,
) -> None:
    """Initialize a uv project in ``cwd`` from resolved stack ``tokens``.

    :param config: Configuration root.
    :param runner: Command runner.
    :param tokens: Stack tokens (profiles, bundles, packages).
    :param options: Project options.
    :param cwd: Directory in which to create the project.
    :raises ConfigError: If a ``pyproject.toml`` exists and ``force`` is False.
    """
    pyproject = cwd / "pyproject.toml"
    if pyproject.is_file() and not options.force:
        raise ConfigError(
            "pyproject.toml already exists.",
            hint="Use --force to add to the existing project.",
        )

    # Resolve the interpreter up front so a bad env name fails before any
    # scaffolding runs, and so both uv init and uv sync receive the same value.
    python = resolve_project_python(config, runner, options.python)

    stack = Resolver(config).resolve(tokens)
    flat = render_requirements_flat(stack, config)

    fd, tmp_name = tempfile.mkstemp(prefix="uv-stack-stack.", suffix=".txt")
    tmp_req = Path(tmp_name)
    try:
        with open(fd, "w") as handle:
            handle.write(flat)

        if not pyproject.is_file():
            runner.run(_with_cwd(uv_init(python, options.name), cwd))
        runner.run(_with_cwd(uv_add(tmp_req), cwd))
        if not options.no_sync:
            runner.run(_with_cwd(uv_sync(python), cwd))
    finally:
        if tmp_req.exists():
            tmp_req.unlink()


def select_project_python(config: ConfigRoot, flag: str | None) -> str:
    """Resolve the interpreter spec by precedence (no env-name resolution yet).

    Precedence: the ``--python`` flag, then ``UV_STACK_PROJECT_PYTHON``, then the
    config-root default, then :data:`DEFAULT_PROJECT_PYTHON`. The returned value
    may be a version, a path, or a micromamba env name; resolving an env name to
    an interpreter path is :func:`resolve_project_python`'s job.

    :param config: Configuration root (for the file-backed default).
    :param flag: The raw ``--python`` value, or ``None`` when unset.
    :returns: The selected interpreter spec.
    """
    return (
        flag
        or os.environ.get(PROJECT_PYTHON_ENV)
        or config.default_project_python()
        or DEFAULT_PROJECT_PYTHON
    )


def _is_python_passthrough(spec: str) -> bool:
    """Return whether ``spec`` is a uv interpreter spec rather than an env name.

    Versions, explicit paths, and uv implementation forms pass straight through
    to uv; anything else is treated as a micromamba environment name.

    :param spec: The interpreter spec to classify.
    """
    if "/" in spec or "\\" in spec:
        return True
    if "@" in spec:
        return True
    if _VERSION_RE.match(spec):
        return True
    return bool(_IMPLEMENTATION_RE.match(spec))


def resolve_project_python(
    config: ConfigRoot, runner: Runner, flag: str | None
) -> str:
    """Select and fully resolve the project interpreter.

    Applies :func:`select_project_python`, then resolves a micromamba env name to
    that env's interpreter path (versions and paths are returned unchanged).

    :param config: Configuration root.
    :param runner: Command runner used to probe the micromamba env.
    :param flag: The raw ``--python`` value, or ``None`` when unset.
    :returns: A version, path, or resolved interpreter path suitable for uv.
    :raises EnvError: If the spec names a micromamba env that cannot be probed.
    """
    spec = select_project_python(config, flag)
    if _is_python_passthrough(spec):
        return spec

    result = runner.run(micromamba_python_path(spec), capture=True, check=False)
    path = result.stdout.strip()
    if result.returncode != 0 or not path:
        raise EnvError(
            f"Could not resolve micromamba environment '{spec}' to an interpreter.",
            hint=(
                f"Ensure the env exists ('stack create env {spec}') and that "
                "MAMBA_ROOT_PREFIX is set, or pass --python <version>."
            ),
        )
    return path


def _with_cwd(command: Command, cwd: Path) -> Command:
    return Command(command.args, cwd=cwd)
