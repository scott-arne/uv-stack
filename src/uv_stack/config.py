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

        :param root: Explicit root from ``--root`` (highest precedence).
        :returns: A configured :class:`ConfigRoot`.
        """
        if root is not None:
            return cls(root)
        env = os.environ.get("UV_ENV_ROOT")
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

    # -- path helpers ----------------------------------------------------
    def project_python_path(self) -> Path:
        """Path to the root-level default for ``create project --python``.

        This is config-root scoped (one per tree), distinct from the per-env
        ``python.txt`` returned by :meth:`env_python_path`.
        """
        return self.root / "project-python.txt"

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

    def env_lock(self, name: str) -> Path:
        return self.env_dir(name) / "requirements.lock.txt"

    def env_environment_yml(self, name: str) -> Path:
        return self.env_dir(name) / "environment.yml"

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
            data = yaml.safe_load(path.read_text())
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

    def load_env(self, name: str) -> EnvConfig:
        if not self.env_exists(name):
            raise ConfigError(
                f"Missing stack file for env '{name}': expected {self.env_stack_path(name)}",
                hint="Create stack.txt in the env config directory.",
            )
        return EnvConfig(
            name=name,
            python=first_clean_line(self.env_python_path(name), default="3.12"),
            stack=read_clean_lines(self.env_stack_path(name)),
            micromamba=read_clean_lines(self.env_micromamba_path(name)),
            channels=read_clean_lines(self.env_channels_path(name)),
        )
