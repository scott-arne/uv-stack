"""Scaffold writers for user-authored config files.

These write the *source* files a user would otherwise author by hand
(profile/bundle YAML, an env's ``stack.txt``/``python.txt``). They refuse to
overwrite existing files — editing belongs to the user — and write atomically.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

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
    concurrent replacement is no longer plausibly ours. Two residuals survive
    it: a replacement made between the check and the unlink is removed as if
    it were ours, and so is one that lands on a recycled inode number. Neither
    can be closed without unlink-by-inode.

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


def _withdraw_and_raise(
    path: Path, published: os.stat_result, message: str, hint: str
) -> NoReturn:
    """Retract ``path`` and raise, reporting whether the file is really gone.

    :param path: The file this process published and is now retracting.
    :param published: The stat of the inode published at ``path``.
    :param message: Base ConfigError message — used verbatim when the
        retraction succeeded, and as the prefix of the residual message when
        it did not, so it must not itself assert what happened on disk.
    :param hint: ConfigError hint when the retraction succeeded.
    :raises ConfigError: Always — with the residual named when ``path`` could
        not be removed.
    """
    if _withdraw(path, published):
        raise ConfigError(message, hint=hint)
    raise ConfigError(
        f"{message}; {path} was just written and could not be removed",
        hint=f"Delete {path} by hand, then choose another name.",
    )


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
    replaced the path in the meantime keeps its file. A post-publish probe
    that cannot answer is treated as a collision rather than trusted.

    :param path: Destination file.
    :param text: Content to write.
    :param exists_message: ConfigError message if the target already exists.
    :param shadowed: Predicate re-evaluated after publishing; true means the
        opposite kind now exists under this name.
    :param shadow_message: ConfigError message when ``shadowed`` answers true.
    :raises ConfigError: If the target already exists, or the name became
        shadowed during the publish, or the post-publish probe could not
        answer (the successfully-withdrawn case leaves nothing behind; the
        failed-withdrawal case states that the file could not be removed).
    """
    published = _publish(path, text, exists_message, _OVERWRITE_HINT)
    try:
        collided = shadowed()
    except OSError as exc:
        # The post-publish probe is the only thing between a successful write
        # and an undetected cross-kind collision.
        _withdraw_and_raise(
            path,
            published,
            f"Could not check for a conflicting file after writing {path}: {exc}",
            "Check that the config directory is readable, then try again.",
        )
    if collided:
        _withdraw_and_raise(path, published, shadow_message, _SHADOW_HINT)


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
    """Write a new environment's source files (``python.txt``, then ``stack.txt``).

    ``stack.txt`` is what makes an environment exist, so it is published LAST.
    A hard crash between the two writes therefore leaves only an orphan
    ``python.txt``, plus possibly a ``.tmp`` leftover from the interrupted
    atomic write that nothing ever collects (harmless: the retry's temporary
    file always takes a fresh name). The env still reads as absent, and the
    retry adopts that orphan when its bytes match what this call would have
    written. Ordering, not rollback, is what makes the crash case retryable —
    a best-effort rollback cannot run at all when the process is killed, which
    is the case it was there for. Publication is all-or-nothing where ``atomic_write_new``
    publishes by hard link, but on a filesystem without hard-link support
    (FAT/exFAT, some network mounts) its exclusive-create fallback can leave
    a partial file that this function's adoption cannot recognize as ours.

    An *ordinary* failure of the ``stack.txt`` publish is different: the
    process is still alive, and most often the cause is a concurrent writer
    winning that race, which would leave our ``python.txt`` attached to their
    environment. So that path withdraws the ``python.txt`` this call
    published, matched on the inode identity ``_publish`` returned so a third
    party's replacement is left alone (modulo ``_withdraw``'s two residuals).
    A concurrent writer that *adopted* this ``python.txt`` and won the
    ``stack.txt`` race loses its interpreter pin when we withdraw; nothing on
    disk distinguishes that writer from one that never asked for an
    interpreter.

    Adoption is deliberately narrow: it requires a regular file (not a
    symlink, directory, FIFO, socket, or device node — reading any non-regular
    file can block or fail, and none is something this code could have
    written), readable as UTF-8 (a file we cannot prove is our own debris is
    refused like any other foreign file), whose content matches the requested
    version (anything else is a user edit, not our debris), and present before
    the preflight. Every other existing ``python.txt`` is refused — including
    one that appears after the preflight, which the exclusive create rejects
    even when its content matches. A retry with no ``--python`` skips the
    adoption preflight entirely, so it inherits the crashed run's orphan
    ``python.txt`` with no byte comparison. The descriptor-bound probe binds
    the read to a file opened with ``O_NOFOLLOW | O_NONBLOCK``, so no swap can
    turn it into a blocking open or produce bytes other than those compared;
    a swap that lands *after* adoption, however, leaves a ``python.txt`` this
    call neither wrote nor verified.

    When the ``stack.txt`` publish fails for a reason other than ``ConfigError``
    (ENOSPC, EACCES, KeyboardInterrupt), the original exception propagates
    unchanged and a ``python.txt`` that could not be withdrawn is left as an
    orphan.

    :param config: Configuration root.
    :param name: Environment name.
    :param tokens: Stack tokens, one per ``stack.txt`` line.
    :param python: When given, also write ``python.txt`` with this version.
    :returns: The paths written by THIS call, ``stack.txt`` first. An adopted
        ``python.txt`` is absent from the list.
    :raises ConfigError: If ``name`` is not a valid environment name, if the
        environment already has a ``stack.txt``, if it has a ``python.txt``
        that cannot be adopted on the terms above, or if the ``stack.txt``
        publish was itself refused and the ``python.txt`` this call published
        could not then be withdrawn.
    """
    _validate_name("environment", name)
    stack_path = config.env_stack_path(name)
    python_path = config.env_python_path(name)
    python_text = python + "\n" if python is not None else None

    # Preflight both targets before writing anything.
    if stack_path.exists():
        raise ConfigError(
            f"Environment '{name}' already has a stack.txt.",
            hint="Edit it directly, or omit TOKENS to rebuild the env.",
        )
    adopt_python = False
    if python_text is not None and python_path.exists():
        # Adoption may only take a regular file: a non-regular python.txt (symlink,
        # FIFO, directory, socket, device node) is not something this code could
        # have written, and reading one can block indefinitely or fail in ways
        # refusing it does not. Anything other than a regular file falls through to
        # the existing "already has a python.txt" refusal — the pre-change behavior.
        try:
            # O_NOFOLLOW/O_NONBLOCK degrade to 0 when absent; they keep the open
            # itself from following a symlink or blocking on a FIFO, and binding
            # the check to the descriptor means the bytes compared are the ones
            # fstat approved.
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            fd = os.open(python_path, flags)
            try:
                st_fd = os.fstat(fd)
                if stat.S_ISREG(st_fd.st_mode):
                    with os.fdopen(fd, "rb") as handle:
                        fd = -1  # ownership transferred to file object
                        current_bytes = handle.read()
                    adopt_python = current_bytes.decode("utf-8") == python_text
            finally:
                if fd != -1:
                    os.close(fd)
        except (OSError, UnicodeDecodeError):
            # A python.txt we cannot open, read, or decode as UTF-8 is not one we
            # can prove is our own debris, so it is refused like any other foreign file.
            pass
        if not adopt_python:
            raise ConfigError(
                f"Environment '{name}' already has a python.txt.",
                hint="Edit or delete it, or omit --python to inherit it.",
            )

    written: list[Path] = []
    published_python: os.stat_result | None = None
    if python_text is not None and not adopt_python:
        published_python = _publish(
            python_path,
            python_text,
            f"Environment '{name}' already has a python.txt.",
            "Edit or delete it, or omit --python to inherit it.",
        )
        written.append(python_path)
    try:
        _publish(
            stack_path,
            "\n".join(tokens) + "\n",
            f"Environment '{name}' already has a stack.txt.",
            "Edit it directly, or omit TOKENS to rebuild the env.",
        )
    except BaseException as exc:
        # Reached only while the process is alive, so most often a concurrent
        # writer won the stack.txt race: withdraw our python.txt rather than
        # leave this run's interpreter attached to their environment. A kill
        # skips this handler, which is exactly the orphan adoption handles.
        if published_python is not None and not _withdraw(python_path, published_python):
            if isinstance(exc, ConfigError):
                # Residual clause matches _withdraw_and_raise; hints differ intentionally.
                raise ConfigError(
                    f"{exc.message.rstrip('.')}; {python_path} was just written and could "
                    "not be removed",
                    hint=f"Delete {python_path} by hand, then try again.",
                ) from exc
            # Replacing a KeyboardInterrupt or an unexpected error with a
            # ConfigError would hide the real failure. The residual python.txt
            # is then an orphan, which adoption already handles.
        raise
    return [stack_path, *written]


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
