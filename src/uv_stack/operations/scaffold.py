"""Scaffold writers for user-authored config files.

These write the *source* files a user would otherwise author by hand
(profile/bundle YAML, an env's ``stack.txt``/``python.txt``). They refuse to
overwrite existing files — editing belongs to the user — and write atomically.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.fsutil import atomic_write_new

_OVERWRITE_HINT = "Edit the file directly or choose another name."


def _validate_name(kind: str, name: str) -> None:
    """Validate a profile/bundle/environment name.

    :param kind: Type of entity ("profile", "bundle", "environment").
    :param name: Name to validate.
    :raises ConfigError: If name is invalid.
    """
    if not name or name == "." or name == ".." or "/" in name or "\\" in name:
        raise ConfigError(
            f"Invalid {kind} name: '{name}'",
            hint="Names are file stems: no path separators or dot segments.",
        )


def _publish(path: Path, text: str, message: str, hint: str) -> os.stat_result:
    """Atomically publish ``text`` to ``path``, mapping FileExistsError.

    :param path: Destination file.
    :param text: Content to write.
    :param message: ConfigError message if the target exists.
    :param hint: ConfigError hint if the target exists.
    :returns: The stat of the published inode.
    :raises ConfigError: If the target already exists.
    """
    try:
        return atomic_write_new(path, text)
    except FileExistsError as exc:
        raise ConfigError(message, hint=hint) from exc


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
    :raises ConfigError: If the profile already exists or would shadow an existing bundle.
    """
    _validate_name("profile", name)
    if config.bundle_exists(name):
        bundle_path = config.bundle_path(name)
        raise ConfigError(
            f"Profile '{name}' would shadow the existing bundle: {bundle_path}",
            hint="Unqualified tokens prefer profiles over bundles; choose another name.",
        )
    path = config.profile_path(name)
    if path.exists():
        raise ConfigError(
            f"Profile '{name}' already exists: {path}", hint=_OVERWRITE_HINT
        )
    _publish(
        path,
        _render_yaml(description, list(tags or []), packages),
        f"Profile '{name}' already exists: {path}",
        _OVERWRITE_HINT,
    )
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
    :raises ConfigError: If the bundle already exists or would be shadowed by an existing profile.
    """
    _validate_name("bundle", name)
    if config.profile_exists(name):
        profile_path = config.profile_path(name)
        raise ConfigError(
            f"Bundle '{name}' would be shadowed by the existing profile: {profile_path}",
            hint="Unqualified tokens prefer profiles over bundles; choose another name.",
        )
    path = config.bundle_path(name)
    if path.exists():
        raise ConfigError(
            f"Bundle '{name}' already exists: {path}", hint=_OVERWRITE_HINT
        )
    _publish(
        path,
        _render_yaml(description, list(tags or []), tokens),
        f"Bundle '{name}' already exists: {path}",
        _OVERWRITE_HINT,
    )
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
    :raises ConfigError: If the environment already has a ``stack.txt`` or ``python.txt``.
    """
    _validate_name("environment", name)
    stack_path = config.env_stack_path(name)
    python_path = config.env_python_path(name) if python else None

    # Preflight all targets before writing anything.
    if stack_path.exists():
        raise ConfigError(
            f"Environment '{name}' already has a stack.txt.",
            hint="Edit it directly, or omit TOKENS to rebuild the env.",
        )
    if python_path and python_path.exists():
        raise ConfigError(
            f"Environment '{name}' already has a python.txt.",
            hint="Edit it directly, or omit --python.",
        )

    written: list[Path] = []
    stack_text = "\n".join(tokens) + "\n"
    stack_identity: tuple[int, int] | None = None
    stack_stat = _publish(
        stack_path,
        stack_text,
        f"Environment '{name}' already has a stack.txt.",
        "Edit it directly, or omit TOKENS to rebuild the env.",
    )
    # Capture the identity of the file we just published for safe rollback.
    stack_identity = (stack_stat.st_dev, stack_stat.st_ino)
    written.append(stack_path)

    if python:
        try:
            _publish(
                python_path,  # type: ignore[arg-type]
                python + "\n",
                f"Environment '{name}' already has a python.txt.",
                "Edit it directly, or omit --python.",
            )
            written.append(python_path)  # type: ignore[arg-type]
        except BaseException:
            # Rollback is best-effort: only remove the exact file this call created.
            # A concurrent process could have replaced stack.txt between our publish
            # and this rollback; we must not unlink a replacement.
            if stack_identity is not None:
                try:
                    current_stat = stack_path.lstat()
                    if (current_stat.st_dev, current_stat.st_ino) == stack_identity:
                        stack_path.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
            raise

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
    :raises ConfigError: If a starter profile already exists or would shadow an existing bundle.
    """
    if config.bundle_exists("starter"):
        starter_bundle_path = config.bundle_path("starter")
        raise ConfigError(
            f"Profile 'starter' would shadow the existing bundle: {starter_bundle_path}",
            hint="Unqualified tokens prefer profiles over bundles; choose another name.",
        )
    path = config.profile_path("starter")
    if path.exists():
        raise ConfigError(
            f"Profile 'starter' already exists: {path}", hint=_OVERWRITE_HINT
        )
    _publish(
        path,
        _STARTER_PROFILE,
        f"Profile 'starter' already exists: {path}",
        _OVERWRITE_HINT,
    )
    return path
