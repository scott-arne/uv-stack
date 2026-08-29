"""Post-edit validation for ``stack edit``.

Each kind is checked with the same loaders and renderers the command that
consumes it already runs, so ``stack edit`` accepts exactly what the next
``stack upgrade`` or ``stack refresh`` would accept. Anything shallower would
report success on a file the very next command rejects, which is the failure
the validate-on-exit loop exists to prevent.

Undecodable input needs no handling here. Every reader these functions reach
converts its own ``UnicodeDecodeError`` in the frame that still holds the
path, so the error already names the file. A net at this level could only say
"a file under <directory>", which is the one thing the user cannot act on.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.fsutil import read_text_utf8
from uv_stack.operations.pyproject import (
    read_project_dependency_names,
    read_tracking,
    validate_tracking_write,
)
from uv_stack.render import render_environment_yml, render_requirements_in
from uv_stack.resolver import Resolver, bundle_self_references


class Validation(NamedTuple):
    """What validation found, and what the caller would otherwise re-derive.

    :param warnings: Non-fatal findings to report alongside success.
    :param tracked: Whether the project carries a ``[tool.uv-stack]`` table;
        ``None`` for every other kind, where the question does not arise.
        Reported here because :func:`validate_project` parsed the file to
        answer it already, and the file cannot change between that parse and
        the caller acting on it.
    """

    warnings: list[str]
    tracked: bool | None = None


def missing_project_error(cwd: Path) -> ConfigError:
    """The refusal for a directory with no ``pyproject.toml``.

    Built here, and imported by ``cli/edit.py``, so the pre-launch check and
    the post-edit re-check raise the *same* error. They are the same condition
    seen at two moments — the user whose editor deleted the file needs the same
    way forward as the user who never had one — and two hand-written hints
    would drift.

    :param cwd: The directory that has no ``pyproject.toml``.
    :returns: The error to raise.
    """
    return ConfigError(
        f"No pyproject.toml in {cwd}.",
        hint="Run 'stack create project TOKENS...' here, or cd into a project.",
    )


def validate_profile(config: ConfigRoot, name: str) -> Validation:
    """Validate an edited profile.

    :param config: Config root.
    :param name: Profile name.
    :returns: No warnings; a profile references nothing, so it produces no
        resolution warnings.
    :raises ConfigError: When the YAML is unreadable or fails the schema.
    """
    config.load_profile(name)
    return Validation([])


def validate_bundle(config: ConfigRoot, name: str) -> Validation:
    """Validate an edited bundle.

    :param config: Config root.
    :param name: Bundle name.
    :returns: Resolution warnings.
    :raises ConfigError: When the bundle's own YAML — or that of a profile or
        bundle it includes — is unreadable or fails the schema, or when the
        bundle includes itself.
    :raises ResolutionError: When an include cannot be resolved.
    """
    bundle = config.load_bundle(name)
    self_refs = bundle_self_references(name, bundle.includes)
    if self_refs:
        raise ConfigError(
            f"Bundle '{name}' cannot include itself: {self_refs[0]}",
            hint=(
                f"A self-reference resolves to nothing. Use pkg:{name} for the "
                "literal package, or drop the include."
            ),
        )
    resolver = Resolver(config)
    stack = resolver.resolve(bundle.includes)
    resolver.flatten(stack)
    return Validation(list(stack.warnings))


def validate_env(config: ConfigRoot, name: str) -> Validation:
    """Validate every source file of an edited environment.

    The whole env is validated regardless of which file was opened: the source
    files are interdependent, and a ``python.txt`` edit can only be judged
    against the rest.

    :param config: Config root.
    :param name: Environment name.
    :returns: Resolution warnings.
    :raises ConfigError: When a source file is unreadable or renders no output.
    :raises ResolutionError: When a stack token cannot be resolved.
    """
    env = config.load_env(name)
    resolver = Resolver(config)
    stack = resolver.resolve(env.stack)
    resolver.flatten(stack)
    render_requirements_in(stack, config, name)
    render_environment_yml(env)
    local = config.env_local_path(name)
    if local.is_file():
        # render_requirements_in emits '-r <path>' for this file without ever
        # opening it, so an explicit read is the only thing that sees a decode
        # failure in what the user may have just edited.
        read_text_utf8(local)
    return Validation(list(stack.warnings))


def validate_project(config: ConfigRoot, cwd: Path) -> Validation:
    """Validate an edited project ``pyproject.toml``.

    :param config: Config root, for resolving the tracked stack.
    :param cwd: The directory holding ``pyproject.toml``.
    :returns: Warnings; a single advisory when the file carries no
        ``[tool.uv-stack]`` table.
    :raises ConfigError: When the file was deleted, is unreadable, its
        tracking table fails the schema, ``[project.dependencies]`` is
        malformed, or the file would be refused by the write preflight.
    :raises NewerSchemaError: When the table declares a newer schema.
    :raises ResolutionError: When a tracked stack token cannot be resolved.
    """
    pyproject = cwd / "pyproject.toml"
    if not pyproject.is_file():
        # read_tracking returns None for a missing file and for an absent
        # table alike, so without this the deleted case reads as "untracked"
        # and exits successfully.
        raise missing_project_error(cwd)
    tracking = read_tracking(pyproject)
    if tracking is None:
        return Validation(
            [
                f"{pyproject} has no [tool.uv-stack] table; 'stack refresh' "
                "will not manage this project."
            ],
            tracked=False,
        )
    # Both of these are things `stack refresh` does before it mutates anything,
    # and neither has a side effect. Skipping them lets the re-offer loop call
    # a file valid that the next refresh refuses.
    read_project_dependency_names(pyproject)
    validate_tracking_write(pyproject, tracking)
    resolver = Resolver(config)
    stack = resolver.resolve(tracking.stack)
    resolver.flatten(stack)
    return Validation(list(stack.warnings), tracked=True)


def validate(config: ConfigRoot, kind: str, name: str, cwd: Path) -> Validation:
    """Validate the sources for ``kind``/``name`` after an edit.

    :param config: Config root.
    :param kind: One of ``profile``, ``bundle``, ``env``, ``project``.
    :param name: The resource name; ignored when ``kind`` is ``project``.
    :param cwd: The directory holding ``pyproject.toml``; ignored otherwise.
    :returns: Non-fatal warnings to report alongside success.
    :raises ConfigError: When the edited file is invalid.
    :raises ResolutionError: When a stack token cannot be resolved.
    """
    if kind == "profile":
        return validate_profile(config, name)
    if kind == "bundle":
        return validate_bundle(config, name)
    if kind == "env":
        return validate_env(config, name)
    return validate_project(config, cwd)
