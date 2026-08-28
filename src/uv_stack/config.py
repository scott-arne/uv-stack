"""Filesystem layout and loaders for a uv-stack config root.

A config root contains ``profiles/``, ``bundles/``, and ``envs/`` directories.
``ConfigRoot`` centralizes path construction, existence checks, enumeration, and
loading of :mod:`uv_stack.models` objects. It performs the only config-file I/O
in the pure layer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import ValidationError

from uv_stack.errors import ConfigError
from uv_stack.hints import render_positional_arg
from uv_stack.models import Bundle, EnvConfig, Profile
from uv_stack.parse import first_clean_line, read_clean_lines

_ModelT = TypeVar("_ModelT", Profile, Bundle)

DEFAULT_ROOT = Path.home() / ".config" / "python-envs"


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
        """
        line = first_clean_line(self.project_python_path(), default="")
        return line or None

    def default_editor(self) -> str | None:
        """Return the editor command configured in ``editor.txt``.

        :returns: The first clean line of ``editor.txt``, or ``None`` when the
            file is absent or holds nothing but blanks and comments.
        :raises ConfigError: When the file is not valid UTF-8. The underlying
            ``UnicodeDecodeError`` is a ``ValueError``, which the CLI edge does
            not render, and this read happens before the editor launches, so
            the post-edit validators cannot cover it.
        """
        path = self.editor_path()
        try:
            line = first_clean_line(path, default="")
        except UnicodeDecodeError as error:
            raise ConfigError(
                f"Cannot read {path}: it is not valid UTF-8.",
                hint="Re-save editor.txt as UTF-8 text, or delete it.",
            ) from error
        return line or None

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
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Invalid YAML in {path}: {exc}",
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
        :raises ConfigError: When ``stack.txt`` is missing.
        """
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
        self.require_env(name)
        return EnvConfig(
            name=name,
            python=first_clean_line(self.env_python_path(name), default="3.12"),
            stack=read_clean_lines(self.env_stack_path(name)),
            micromamba=read_clean_lines(self.env_micromamba_path(name)),
            channels=read_clean_lines(self.env_channels_path(name)),
        )
