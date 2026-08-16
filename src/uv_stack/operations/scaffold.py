"""Scaffold writers for user-authored config files.

These write the *source* files a user would otherwise author by hand
(profile/bundle YAML, an env's ``stack.txt``/``python.txt``). They refuse to
overwrite existing files — editing belongs to the user — and write atomically.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import yaml

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.fsutil import atomic_write_new
from uv_stack.resolver import bundle_self_references

_OVERWRITE_HINT = "Edit the file directly or choose another name."
_SHADOW_HINT = "Unqualified tokens prefer profiles over bundles; choose another name."


def _validate_name(kind: str, name: str) -> None:
    """Validate a profile/bundle/environment name.

    :param kind: Type of entity ("profile", "bundle", "environment").
    :param name: Name to validate.
    :raises ConfigError: If name is invalid.
    """
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or ":" in name
        or "@" in name
        or name.startswith("-")
        or any(char.isspace() for char in name)
    ):
        raise ConfigError(
            f"Invalid {kind} name: '{name}'",
            hint=(
                "Names are file stems: no path separators, dot segments, "
                "':', '@', whitespace, or leading '-'."
            ),
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


def _withdraw(path: Path, published: os.stat_result) -> bool:
    """Unlink ``path`` while it is still the inode we published there.

    POSIX has no unlink-by-inode, so the identity check and the unlink cannot
    be one atomic step; the check narrows the window to the point where a
    concurrent replacement is no longer plausibly ours. Inode numbers are
    reusable, so on a filesystem that recycles them promptly a replacement
    created inside that window could still match — the caller reports a
    failed withdrawal rather than assuming either outcome.

    :param path: Path to withdraw.
    :param published: The stat of the inode this process published at ``path``.
    :returns: True when ``path`` no longer holds the published inode.
    """
    try:
        current = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if (current.st_dev, current.st_ino) != (published.st_dev, published.st_ino):
        return True
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _publish_unshadowed(
    path: Path,
    text: str,
    exists_message: str,
    *,
    shadowed: Callable[[], bool],
    shadow_message: str,
) -> None:
    """Publish a profile/bundle, withdrawing it if the opposite kind appeared.

    The shadow pre-check and the exclusive create cannot be a single atomic
    step, so the check is repeated AFTER publishing: a concurrent writer that
    created the other kind inside that window is caught, and our file is
    withdrawn. The withdrawal is matched by inode, so a third writer that
    replaced the path in the meantime keeps its file.

    :param path: Destination file.
    :param text: Content to write.
    :param exists_message: ConfigError message if the target already exists.
    :param shadowed: Predicate re-evaluated after publishing; true means the
        opposite kind now exists under this name.
    :param shadow_message: ConfigError message when ``shadowed`` answers true.
    :raises ConfigError: If the target already exists, or the name became
        shadowed during the publish (the successfully-withdrawn case leaves
        nothing behind; the failed-withdrawal case states so in its message).
    """
    published = _publish(path, text, exists_message, _OVERWRITE_HINT)
    if not shadowed():
        return
    if _withdraw(path, published):
        raise ConfigError(shadow_message, hint=_SHADOW_HINT)
    raise ConfigError(
        f"{shadow_message}; the file just written could not be removed",
        hint=f"Delete {path} by hand, then choose another name.",
    )


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
    shadow_message = (
        f"Profile '{name}' would shadow the existing bundle: {config.bundle_path(name)}"
    )
    if config.bundle_exists(name):
        raise ConfigError(shadow_message, hint=_SHADOW_HINT)
    path = config.profile_path(name)
    _publish_unshadowed(
        path,
        _render_yaml(description, list(tags or []), packages),
        f"Profile '{name}' already exists: {path}",
        shadowed=lambda: config.bundle_exists(name),
        shadow_message=shadow_message,
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
    :raises ConfigError: If the bundle already exists, references itself, or
        would be shadowed by an existing profile.
    """
    _validate_name("bundle", name)
    self_refs = bundle_self_references(name, tokens)
    if self_refs:
        raise ConfigError(
            f"Bundle '{name}' cannot include itself: {self_refs[0]}",
            hint=(
                f"A self-reference resolves to nothing. Use pkg:{name} for the "
                "literal package, or choose another bundle name."
            ),
        )
    shadow_message = (
        f"Bundle '{name}' would be shadowed by the existing profile: "
        f"{config.profile_path(name)}"
    )
    if config.profile_exists(name):
        raise ConfigError(shadow_message, hint=_SHADOW_HINT)
    path = config.bundle_path(name)
    _publish_unshadowed(
        path,
        _render_yaml(description, list(tags or []), tokens),
        f"Bundle '{name}' already exists: {path}",
        shadowed=lambda: config.profile_exists(name),
        shadow_message=shadow_message,
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
    shadow_message = (
        "Profile 'starter' would shadow the existing bundle: "
        f"{config.bundle_path('starter')}"
    )
    if config.bundle_exists("starter"):
        raise ConfigError(shadow_message, hint=_SHADOW_HINT)
    path = config.profile_path("starter")
    _publish_unshadowed(
        path,
        _STARTER_PROFILE,
        f"Profile 'starter' already exists: {path}",
        shadowed=lambda: config.bundle_exists("starter"),
        shadow_message=shadow_message,
    )
    return path
