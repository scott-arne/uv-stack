"""Post-edit validation for ``stack edit``.

Each kind is checked with the same loaders and renderers the command that
consumes it already runs, so ``stack edit`` accepts exactly what the next
``stack upgrade`` or ``stack refresh`` would accept. Anything shallower would
report success on a file the very next command rejects, which is the failure
the validate-on-exit loop exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.operations.pyproject import read_tracking
from uv_stack.render import render_environment_yml, render_requirements_in
from uv_stack.resolver import Resolver, bundle_self_references


@contextmanager
def _decoding(subject: str) -> Iterator[None]:
    """Convert a UTF-8 decode failure into a ``ConfigError``.

    ``UnicodeDecodeError`` is a ``ValueError``, so it is neither a
    ``UvStackError`` nor an ``OSError`` and would escape the CLI edge as a
    traceback — for what is really just another "fix it and re-open" state.

    :param subject: What could not be read, for the message.
    """
    try:
        yield
    except UnicodeDecodeError as error:
        raise ConfigError(
            f"Cannot read {subject}: not valid UTF-8.",
            hint="Re-save the file as UTF-8 text.",
        ) from error


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


def validate_profile(config: ConfigRoot, name: str) -> list[str]:
    """Validate an edited profile.

    :param config: Config root.
    :param name: Profile name.
    :returns: An empty list; a profile references nothing, so it produces no
        resolution warnings.
    :raises ConfigError: When the YAML is unreadable or fails the schema.
    """
    with _decoding(str(config.profile_path(name))):
        config.load_profile(name)
    return []


def validate_bundle(config: ConfigRoot, name: str) -> list[str]:
    """Validate an edited bundle.

    :param config: Config root.
    :param name: Bundle name.
    :returns: Resolution warnings.
    :raises ConfigError: When the YAML fails the schema, or the bundle
        includes itself.
    :raises ResolutionError: When an include cannot be resolved.
    """
    with _decoding(str(config.bundle_path(name))):
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
    with _decoding(f"a file under {config.root}"):
        resolver = Resolver(config)
        stack = resolver.resolve(bundle.includes)
        resolver.flatten(stack)
        return list(stack.warnings)


def validate_env(config: ConfigRoot, name: str) -> list[str]:
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
    with _decoding(f"a file under {config.env_dir(name)}"):
        env = config.load_env(name)
    with _decoding(f"a file under {config.root}"):
        resolver = Resolver(config)
        stack = resolver.resolve(env.stack)
        resolver.flatten(stack)
        render_requirements_in(stack, config, name)
        render_environment_yml(env)
    with _decoding(f"a file under {config.env_dir(name)}"):
        local = config.env_local_path(name)
        if local.is_file():
            # render_requirements_in emits '-r <path>' for this file without
            # ever opening it, so an explicit read is the only thing that sees
            # a decode failure in what the user may have just edited.
            local.read_text(encoding="utf-8")
    return list(stack.warnings)


def validate_project(config: ConfigRoot, cwd: Path) -> list[str]:
    """Validate an edited project ``pyproject.toml``.

    :param config: Config root, for resolving the tracked stack.
    :param cwd: The directory holding ``pyproject.toml``.
    :returns: Warnings; a single advisory when the file carries no
        ``[tool.uv-stack]`` table.
    :raises ConfigError: When the file was deleted, is unreadable, or its
        tracking table fails the schema.
    :raises NewerSchemaError: When the table declares a newer schema.
    :raises ResolutionError: When a tracked stack token cannot be resolved.
    """
    pyproject = cwd / "pyproject.toml"
    if not pyproject.is_file():
        # read_tracking returns None for a missing file and for an absent
        # table alike, so without this the deleted case reads as "untracked"
        # and exits successfully.
        raise missing_project_error(cwd)
    with _decoding(str(pyproject)):
        tracking = read_tracking(pyproject)
    if tracking is None:
        return [
            f"{pyproject} has no [tool.uv-stack] table; 'stack refresh' will "
            "not manage this project."
        ]
    with _decoding(f"a file under {config.root}"):
        resolver = Resolver(config)
        stack = resolver.resolve(tracking.stack)
        resolver.flatten(stack)
        return list(stack.warnings)


def validate(config: ConfigRoot, kind: str, name: str, cwd: Path) -> list[str]:
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
