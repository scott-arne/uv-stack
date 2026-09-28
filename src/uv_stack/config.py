"""Filesystem layout and loaders for a uv-stack config root.

A config root contains ``profiles/``, ``bundles/``, and ``envs/`` directories.
``ConfigRoot`` centralizes path construction, existence checks, enumeration, and
loading of :mod:`uv_stack.models` objects. It performs the only config-file I/O
in the pure layer.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import ValidationError

from uv_stack.errors import ConfigError
from uv_stack.fsutil import read_text_utf8, require_regular_file
from uv_stack.hints import render_positional_arg
from uv_stack.models import Bundle, EnvConfig, Profile
from uv_stack.parse import clean_line, first_clean_line, read_clean_lines
from uv_stack.variables import NAME_RE, Variables

_ModelT = TypeVar("_ModelT", Profile, Bundle)

DEFAULT_ROOT = Path.home() / ".config" / "python-envs"


def _numbered_clean_lines(path: Path) -> list[tuple[int, str]]:
    """Clean, non-empty lines of ``path`` paired with their 1-based numbers.

    The variable files report defects by line, which
    :func:`~uv_stack.parse.read_clean_lines` cannot do because it discards
    positions.

    :param path: File to read; a missing file yields no lines.
    :returns: ``(line number, cleaned text)`` pairs in file order.
    :raises ConfigError: When the path is not a regular file, or the file is
        not valid UTF-8.
    """
    # Ahead of is_file(), per require_regular_file's own contract: is_file()
    # answers False for a directory named variables.txt just as it does for an
    # absent one, and "no variables declared" is not a safe reading of "I
    # could not read your declarations".
    require_regular_file(path)
    if not path.is_file():
        return []
    numbered = enumerate(read_text_utf8(path).splitlines(), start=1)
    return [(number, clean_line(raw)) for number, raw in numbered if clean_line(raw)]


def _parse_declarations(path: Path) -> list[str]:
    """Parse ``variables.txt``: one name per line.

    :param path: The declaration file.
    :returns: The declared names in file order.
    :raises ConfigError: On a malformed or duplicated name.
    """
    declared: list[str] = []
    first_seen: dict[str, int] = {}
    for number, text in _numbered_clean_lines(path):
        if not NAME_RE.match(text):
            raise ConfigError(
                f"Invalid variable name on line {number} of {path}: {text!r}",
                hint=(
                    "A name must start with a letter or underscore and hold "
                    "only letters, digits, and underscores. Declare one name "
                    "per line; values belong in variables.local.txt."
                ),
            )
        if text in first_seen:
            raise ConfigError(
                f"Variable {text!r} is declared twice in {path}: "
                f"lines {first_seen[text]} and {number}.",
                hint="Remove the duplicate line.",
            )
        first_seen[text] = number
        declared.append(text)
    return declared


def _normalize_value(name: str, raw: str, where: str) -> str:
    """Strip, expand ``~``, and validate one variable value.

    The whitespace refusal runs *after* ``expanduser`` on purpose: what has to
    be whitespace-free is the text that lands in ``requirements.in``, where a
    space would split one entry into two arguments and could turn an admitted
    ``${DEV}/deps.txt`` into a recursive include.

    :param name: The variable being assigned, for the message.
    :param raw: The value as written.
    :param where: Human-readable origin, for the message.
    :returns: The normalized value.
    :raises ConfigError: When the value is empty, cannot be expanded, or
        contains whitespace.
    """
    value = raw.strip()
    if not value:
        raise ConfigError(
            f"Variable {name!r} has an empty value ({where}).",
            hint="Give it a value, or delete the assignment.",
        )
    try:
        value = os.path.expanduser(value)
    except ValueError as error:
        # expanduser is not total: the '~user' form looks the name up in the
        # password database, and a NUL in it raises ValueError. That is
        # neither a UvStackError nor an OSError, so unconverted it reaches the
        # CLI edge as a traceback — out of doctor, sync, and status alike,
        # for a file the user can fix in one edit. Converting it here is the
        # trade read_text_utf8 makes for UnicodeDecodeError, and for its
        # reason too: this frame is the last one that still knows which
        # variable the text belongs to and where it was read from.
        raise ConfigError(
            f"Variable {name!r} has a value that cannot be expanded "
            f"({where}): {value!r}",
            hint=(
                "A value starting with '~' is expanded before use, and this "
                "one holds a character that expansion rejects — an embedded "
                "NUL. Remove it, or write the path out in full."
            ),
        ) from error
    if any(character.isspace() for character in value):
        raise ConfigError(
            f"Variable {name!r} has a value containing whitespace ({where}): "
            f"{value!r}",
            hint=(
                "A value is substituted into a whitespace-delimited "
                "requirements file, so it may not contain whitespace — not "
                "even after '~' expansion. Move the checkout somewhere without "
                "a space in its path."
            ),
        )
    return value


def _parse_local_values(path: Path, declared: list[str], declared_path: Path) -> dict[str, str]:
    """Parse ``variables.local.txt``: ``NAME=value``, one per line.

    :param path: The value file.
    :param declared: Names the root declares.
    :param declared_path: The declaration file, for the message.
    :returns: The values this machine's file supplies.
    :raises ConfigError: On a line with no ``=``, an undeclared name, a
        duplicate assignment, or a value that fails :func:`_normalize_value`.
    """
    values: dict[str, str] = {}
    first_seen: dict[str, int] = {}
    for number, text in _numbered_clean_lines(path):
        name, separator, raw = text.partition("=")
        name = name.strip()
        if not separator:
            raise ConfigError(
                f"Line {number} of {path} is not an assignment: {text!r}",
                hint="Write 'NAME=value', one per line.",
            )
        if name not in declared:
            raise ConfigError(
                f"Line {number} of {path} assigns {name!r}, which "
                f"{declared_path} does not declare.",
                hint=(
                    f"Add {name!r} to {declared_path} so it travels with the "
                    "root, or delete the line."
                ),
            )
        if name in first_seen:
            raise ConfigError(
                f"Variable {name!r} is assigned twice in {path}: "
                f"lines {first_seen[name]} and {number}.",
                hint="Keep one assignment.",
            )
        first_seen[name] = number
        values[name] = _normalize_value(name, raw, f"line {number} of {path}")
    return values


def load_env_from_dir(directory: Path, name: str) -> EnvConfig:
    """Load an environment's four source files from a directory.

    Split out of :meth:`ConfigRoot.load_env` so a *copied* environment
    directory — the unit that travels between machines — reads through the
    identical contract: the same regular-file checks, the same UTF-8 handling,
    and the same ``3.12`` default. A second reader would lose all three.

    :param directory: The directory holding the four source files.
    :param name: The name to record on the returned config.
    :returns: The environment's declared interpreter, stack entries,
        micromamba packages and channels.
    :raises ConfigError: When any of the four sources is present but is not a
        regular file, or when one of them is not valid UTF-8.
    """
    # Ahead of the reads, per require_regular_file's own contract:
    # read_clean_lines and first_clean_line both test is_file(), which answers
    # False for a directory or a dangling symlink named channels.txt exactly as
    # it does for an absent one. Absence is a supported state for all three --
    # python.txt falls back to 3.12, the other two to no entries -- so the
    # unreadable path does not fail, it silently becomes the default, and sync
    # then builds against the wrong interpreter or generates an
    # environment.yml with no channels while doctor reports the root clean.
    # stack.txt is checked too. For ConfigRoot.load_env that is a no-op, since
    # require_env has already held it to the same rule, but a copied directory
    # has no such caller, and there a directory named stack.txt would read as
    # an empty stack. On a genuinely absent path each call is a no-op, which is
    # what keeps an optional file optional.
    for source in (
        directory / "stack.txt",
        directory / "python.txt",
        directory / "micromamba.txt",
        directory / "channels.txt",
    ):
        require_regular_file(source)
    return EnvConfig(
        name=name,
        python=first_clean_line(directory / "python.txt", default="3.12"),
        stack=read_clean_lines(directory / "stack.txt"),
        micromamba=read_clean_lines(directory / "micromamba.txt"),
        channels=read_clean_lines(directory / "channels.txt"),
    )


class ConfigRoot:
    """Resolves and reads a uv-stack configuration tree.

    :param root: The base directory containing ``profiles/``, ``bundles/``,
        and ``envs/``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @classmethod
    def discover(cls, root: str | Path | None = None) -> ConfigRoot:
        """Resolve the config root using flag, then env var, then default.

        ``UV_STACK_ROOT`` is the preferred variable name; ``UV_ENV_ROOT`` is
        the historical spelling and is still honored when ``UV_STACK_ROOT`` is
        unset or empty.

        :param root: Explicit root from ``--root`` (highest precedence).
        :returns: A configured :class:`ConfigRoot`.
        """
        if root is not None:
            return cls(root)
        env = os.environ.get("UV_STACK_ROOT") or os.environ.get("UV_ENV_ROOT")
        if env:
            return cls(env)
        return cls(DEFAULT_ROOT)

    # -- directories -----------------------------------------------------
    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def bundles_dir(self) -> Path:
        return self.root / "bundles"

    @property
    def envs_dir(self) -> Path:
        return self.root / "envs"

    @property
    def locks_dir(self) -> Path:
        return self.root / ".locks"

    # -- path helpers ----------------------------------------------------
    def project_python_path(self) -> Path:
        """Path to the root-level default for ``create project --python``.

        This is config-root scoped (one per tree), distinct from the per-env
        ``python.txt`` returned by :meth:`env_python_path`.
        """
        return self.root / "project-python.txt"

    def editor_path(self) -> Path:
        """Path to the root-level editor command file."""
        return self.root / "editor.txt"

    def variables_path(self) -> Path:
        """Path to the root's declared-variable list.

        Travels with the root: it is the contract a clone must satisfy.
        """
        return self.root / "variables.txt"

    def variables_local_path(self) -> Path:
        """Path to this machine's variable values.

        Does not travel; ``stack config portable`` adds it to the ignore block.
        """
        return self.root / "variables.local.txt"

    def profile_path(self, name: str) -> Path:
        return self.profiles_dir / f"{name}.yaml"

    def bundle_path(self, name: str) -> Path:
        return self.bundles_dir / f"{name}.yaml"

    def env_dir(self, name: str) -> Path:
        return self.envs_dir / name

    def env_python_path(self, name: str) -> Path:
        return self.env_dir(name) / "python.txt"

    def env_stack_path(self, name: str) -> Path:
        return self.env_dir(name) / "stack.txt"

    def env_micromamba_path(self, name: str) -> Path:
        return self.env_dir(name) / "micromamba.txt"

    def env_channels_path(self, name: str) -> Path:
        return self.env_dir(name) / "channels.txt"

    def env_local_path(self, name: str) -> Path:
        return self.env_dir(name) / "requirements.local.in"

    def env_requirements_in(self, name: str) -> Path:
        return self.env_dir(name) / "requirements.in"

    def env_requirements_lock(self, name: str) -> Path:
        return self.env_dir(name) / "requirements.lock.txt"

    def env_environment_yml(self, name: str) -> Path:
        return self.env_dir(name) / "environment.yml"

    def stem_lock_path(self, name: str) -> Path:
        """Lock covering the shared profile/bundle stem namespace.

        Profiles and bundles collide on a bare stem — that is the collision
        this lock serializes — so both kinds take the same file.
        """
        return self.locks_dir / f"stem-{name}.lock"

    def env_lock_path(self, name: str) -> Path:
        """Lock covering one environment's source files.

        A separate namespace from :meth:`stem_lock_path`: an env named ``x``
        does not collide with a profile named ``x``.
        """
        return self.locks_dir / f"env-{name}.lock"

    def probe_lock_path(self) -> Path:
        """Lock used only to test whether this root supports locking.

        A fixed name in a namespace of its own: the stem and env locks are
        ``stem-<name>.lock`` and ``env-<name>.lock``, so no user-chosen name
        can collide with it.
        """
        return self.locks_dir / "probe.lock"

    def project_lock_path(self, project_dir: Path) -> Path:
        """Lock covering one tracked project's ``[tool.uv-stack]`` ledger.

        Keyed on a digest of the resolved directory because a path is not a
        file name, and resolved so that a project reached through a symlink
        takes the same lock as its target. The ``project-`` prefix keeps it
        out of the stem, env and probe namespaces. Runs against the same
        project under different config roots take different locks.
        """
        digest = hashlib.sha256(str(project_dir.resolve()).encode()).hexdigest()
        return self.locks_dir / f"project-{digest[:16]}.lock"

    # -- existence -------------------------------------------------------
    def profile_exists(self, name: str) -> bool:
        return self.profile_path(name).is_file()

    def bundle_exists(self, name: str) -> bool:
        return self.bundle_path(name).is_file()

    def env_exists(self, name: str) -> bool:
        return self.env_stack_path(name).is_file()

    # -- listing ---------------------------------------------------------
    def list_profiles(self) -> list[str]:
        if not self.profiles_dir.is_dir():
            return []
        return sorted(p.stem for p in self.profiles_dir.glob("*.yaml"))

    def list_bundles(self) -> list[str]:
        if not self.bundles_dir.is_dir():
            return []
        return sorted(p.stem for p in self.bundles_dir.glob("*.yaml"))

    def list_envs(self) -> list[str]:
        if not self.envs_dir.is_dir():
            return []
        return sorted(
            d.name for d in self.envs_dir.iterdir() if (d / "stack.txt").is_file()
        )

    # -- loaders ---------------------------------------------------------
    def default_project_python(self) -> str | None:
        """Return the configured default for ``create project --python``.

        :returns: The first clean line of ``project-python.txt``, or ``None``
            when the file is absent or empty.
        :raises ConfigError: When the path is not a regular file, or the file
            is not valid UTF-8.
        """
        path = self.project_python_path()
        # Ahead of the read, per require_regular_file's own contract:
        # read_clean_lines under first_clean_line tests is_file(), which
        # answers False for a directory named project-python.txt just as it
        # does for an absent one, and "no default is configured" is not a safe
        # reading of "I could not read the default you configured".
        require_regular_file(path)
        line = first_clean_line(path, default="")
        return line or None

    def default_editor(self) -> str | None:
        """Return the editor command configured in ``editor.txt``.

        :returns: The first clean line of ``editor.txt``, or ``None`` when the
            file is absent or holds nothing but blanks and comments.
        :raises ConfigError: When the path is not a regular file, or the file
            is not valid UTF-8. The decoding failure is raised by
            :func:`~uv_stack.fsutil.read_text_utf8` under the read; this one
            matters enough to document because it happens before the editor
            launches, so the post-edit validators cannot cover it.
        """
        path = self.editor_path()
        # Ahead of the read, exactly as default_project_python does it:
        # first_clean_line reads through an is_file() test, which answers
        # False for a directory named editor.txt just as it does for an absent
        # one. Falling through to $VISUAL is the right answer to "no editor is
        # configured" and the wrong one to "there is something at editor.txt I
        # could not read".
        require_regular_file(path)
        return first_clean_line(path, default="") or None

    def load_variables(self) -> Variables:
        """Load the root's declared names and this machine's values.

        The environment overrides the file: a value exported in the shell wins
        over ``variables.local.txt``, which lets a CI job or a one-off shell
        point a root somewhere else without editing it. An unset or
        whitespace-only environment variable is not a value and falls through
        to the file. Only declared names are read from the environment, so an
        unrelated variable that happens to share a name cannot leak in.

        :returns: The declared names and the values available here.
        :raises ConfigError: When either file is malformed. ``ConfigError.path``
            names whichever of the two failed, so a caller that reports instead
            of aborting -- ``stack doctor`` -- does not have to blame the call
            it made. The environment branch below leaves it unset: an override
            belongs to no file, and its message says which variable it was.
        """
        try:
            declared = _parse_declarations(self.variables_path())
        except ConfigError as error:
            error.path = error.path or self.variables_path()
            raise
        try:
            values = _parse_local_values(
                self.variables_local_path(), declared, self.variables_path()
            )
        except ConfigError as error:
            error.path = error.path or self.variables_local_path()
            raise
        for name in declared:
            raw = os.environ.get(name)
            if raw is None or not raw.strip():
                continue
            values[name] = _normalize_value(name, raw, f"environment variable {name}")
        return Variables(declared=tuple(declared), values=values)

    def _load_yaml_model(
        self, path: Path, name: str, model: type[_ModelT]
    ) -> _ModelT:
        """Parse ``path`` as YAML and validate it into ``model``.

        :param path: The YAML file to read.
        :param name: The resource name (file stem), injected into the model.
        :param model: The pydantic model class (:class:`Profile` or
            :class:`Bundle`).
        :returns: The validated model instance.
        :raises ConfigError: If the file is missing, not valid YAML, not a
            mapping, or fails schema validation.
        """
        if not path.is_file():
            raise ConfigError(
                f"Missing {model.__name__.lower()}: {path}",
                hint=f"Create {path} or check the name.",
            )
        # Read outside the guard below. read_text_utf8 reports bad UTF-8 as a
        # ConfigError and a failed read as an OSError, and both already say the
        # right thing; relabelling either one "Invalid YAML" would be wrong.
        text = read_text_utf8(path)
        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            # Deliberately broad, and not a substitute for naming the error.
            # PyYAML documents YAMLError as what a loader raises, but its
            # constructors break that contract in at least three families: a
            # date of 2020-99-99 escapes as ValueError, '!!bool "nope"' as
            # KeyError, '!!timestamp "nope"' as AttributeError. The set is
            # open-ended -- any constructor can have the same bug, including
            # ones added later -- so an enumerated except would be a list that
            # silently stops being complete, and one unlisted document would
            # be a traceback out of a command that is meant to report it.
            raise ConfigError(
                # The type is named because the non-YAMLError families carry
                # bare operands: '!!bool "nope"' stringifies to just "'nope'",
                # which tells a reader nothing about what went wrong.
                f"Invalid YAML in {path}: {type(exc).__name__}: {exc}",
                hint="Fix the YAML syntax.",
            ) from exc
        if not isinstance(data, dict):
            raise ConfigError(
                f"Expected a YAML mapping in {path}, got {type(data).__name__}.",
                hint="The file must define at least 'includes:'.",
            )
        try:
            return model.model_validate({**data, "name": name})
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid {model.__name__.lower()} config in {path}: {exc}",
                hint="Check the fields against the schema (description, tags, includes).",
            ) from exc

    def load_profile(self, name: str) -> Profile:
        return self._load_yaml_model(self.profile_path(name), name, Profile)

    def load_bundle(self, name: str) -> Bundle:
        return self._load_yaml_model(self.bundle_path(name), name, Bundle)

    def require_env(self, name: str) -> None:
        """Raise unless env ``name`` has a stack file.

        Extracted from :meth:`load_env` so callers that must check existence
        without reading the env's other files — ``stack edit``, which is about
        to hand one of them to an editor precisely because it is broken — get
        the identical message.

        :param name: The environment name.
        :raises ConfigError: When ``stack.txt`` is missing, or when something
            that is not a regular file occupies its path.
        """
        # Must precede env_exists, whose is_file() calls a directory here
        # absent and would send the user to `stack create env`, which refuses
        # the same path.
        require_regular_file(self.env_stack_path(name))
        if self.env_exists(name):
            return
        raise ConfigError(
            f"Missing stack file for env '{name}': expected {self.env_stack_path(name)}",
            hint=(
                # The hint is a command the user is meant to paste; an env
                # name is a directory name and may contain shell syntax.
                "Create stack.txt in the env config directory, or pass "
                f"TOKENS: stack create env {render_positional_arg(name)} TOKENS..."
            ),
        )

    def load_env(self, name: str) -> EnvConfig:
        """Load an environment's four source files.

        :param name: The environment name.
        :returns: The environment's declared interpreter, stack entries,
            micromamba packages and channels.
        :raises ConfigError: When the env has no ``stack.txt``, when any of the
            four sources is present but is not a regular file, or when one of
            them is not valid UTF-8.
        """
        self.require_env(name)
        return load_env_from_dir(self.env_dir(name), name)
