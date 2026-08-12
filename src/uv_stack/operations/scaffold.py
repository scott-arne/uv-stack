"""Scaffold writers for user-authored config files.

These write the *source* files a user would otherwise author by hand
(profile/bundle YAML, an env's ``stack.txt``/``python.txt``). They refuse to
overwrite existing files — editing belongs to the user — and write atomically.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.fsutil import atomic_write

_OVERWRITE_HINT = "Edit the file directly or choose another name."


def _render_yaml(
    description: str | None, tags: list[str], includes: list[str]
) -> str:
    """Render a profile/bundle mapping as YAML, omitting empty optional keys."""
    data: dict[str, object] = {}
    if description:
        data["description"] = description
    if tags:
        data["tags"] = tags
    data["includes"] = includes
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def write_profile(
    config: ConfigRoot,
    name: str,
    packages: list[str],
    *,
    description: str | None = None,
    tags: list[str] | None = None,
) -> Path:
    """Write ``profiles/<name>.yaml``.

    :param config: Configuration root.
    :param name: Profile name (file stem).
    :param packages: Literal package specifications for ``includes``.
    :param description: Optional one-line description.
    :param tags: Optional tags.
    :returns: The path written.
    :raises ConfigError: If the profile already exists.
    """
    path = config.profile_path(name)
    if path.exists():
        raise ConfigError(
            f"Profile '{name}' already exists: {path}", hint=_OVERWRITE_HINT
        )
    atomic_write(path, _render_yaml(description, list(tags or []), packages))
    return path


def write_bundle(
    config: ConfigRoot,
    name: str,
    tokens: list[str],
    *,
    description: str | None = None,
    tags: list[str] | None = None,
) -> Path:
    """Write ``bundles/<name>.yaml``.

    :param config: Configuration root.
    :param name: Bundle name (file stem).
    :param tokens: Stack tokens for ``includes``.
    :param description: Optional one-line description.
    :param tags: Optional tags.
    :returns: The path written.
    :raises ConfigError: If the bundle already exists.
    """
    path = config.bundle_path(name)
    if path.exists():
        raise ConfigError(
            f"Bundle '{name}' already exists: {path}", hint=_OVERWRITE_HINT
        )
    atomic_write(path, _render_yaml(description, list(tags or []), tokens))
    return path


def write_env_sources(
    config: ConfigRoot,
    name: str,
    tokens: list[str],
    *,
    python: str | None = None,
) -> list[Path]:
    """Write a new environment's source files (``stack.txt``, ``python.txt``).

    :param config: Configuration root.
    :param name: Environment name.
    :param tokens: Stack tokens, one per ``stack.txt`` line.
    :param python: When given, also write ``python.txt`` with this version.
    :returns: The paths written, in order.
    :raises ConfigError: If the environment already has a ``stack.txt``.
    """
    stack_path = config.env_stack_path(name)
    if stack_path.exists():
        raise ConfigError(
            f"Environment '{name}' already has a stack.txt.",
            hint="Edit it directly, or omit TOKENS to rebuild the env.",
        )
    written: list[Path] = []
    atomic_write(stack_path, "\n".join(tokens) + "\n")
    written.append(stack_path)
    if python:
        python_path = config.env_python_path(name)
        atomic_write(python_path, python + "\n")
        written.append(python_path)
    return written


_STARTER_PROFILE = """\
# A profile is a reusable, named group of pip packages.
# Reference it from an environment's stack.txt or a bundle by name.
description: Starter profile
tags: [starter]
includes:
  - rich
"""


def write_starter_profile(config: ConfigRoot) -> Path:
    """Write the commented starter template to ``profiles/starter.yaml``.

    :param config: Configuration root.
    :returns: The path written.
    :raises ConfigError: If a starter profile already exists.
    """
    path = config.profile_path("starter")
    if path.exists():
        raise ConfigError(
            f"Profile 'starter' already exists: {path}", hint=_OVERWRITE_HINT
        )
    atomic_write(path, _STARTER_PROFILE)
    return path
