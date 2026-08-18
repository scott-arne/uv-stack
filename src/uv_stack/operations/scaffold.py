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
from uv_stack.fsutil import (
    _FASTPATH_AVAILABLE,
    _O_NOFOLLOW,
    _O_NONBLOCK,
    atomic_write_new,
    name_lock,
)
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
    :raises ConfigError: If the profile already exists, would shadow an existing
        bundle, or another process holds the lock when the timeout expires.
    :raises OSError: If the lock file cannot be created or opened.
    """
    _validate_name("profile", name)
    shadow_message = (
        f"Profile '{name}' would shadow the existing bundle: {config.bundle_path(name)}"
    )
    path = config.profile_path(name)
    # The pre-check, the publish, and _publish_unshadowed's post-check are one
    # operation as far as another create is concerned. Without this the
    # post-check still catches a live competitor, but a competitor killed
    # between its publish and its own post-check leaves the collision on disk.
    # Where the lock is unavailable that is exactly the residual.
    with name_lock(config.stem_lock_path(name), name):
        if config.bundle_exists(name):
            raise ConfigError(shadow_message, hint=_SHADOW_HINT)
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
    :raises ConfigError: If the bundle already exists, references itself,
        would be shadowed by an existing profile, or another process holds
        the lock when the timeout expires.
    :raises OSError: If the lock file cannot be created or opened.
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
    path = config.bundle_path(name)
    # Same stem lock as write_profile — the two kinds share one namespace, so
    # they must contend on one file. See write_profile for what it buys.
    with name_lock(config.stem_lock_path(name), name):
        if config.profile_exists(name):
            raise ConfigError(shadow_message, hint=_SHADOW_HINT)
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
    environment. So the ``python.txt`` this call published is withdrawn on
    either of two conditions: a ``ConfigError``, which ``_publish`` raises only
    from a ``FileExistsError`` and which therefore does prove the ``stack.txt``
    on disk is somebody else's; or no ``stack.txt`` at ``stack_path`` — which
    is not proof that ours never landed, only that none is there now to be left
    without its interpreter pin. Nothing here can tell *our* ``stack.txt`` from
    a third party's, so that second condition claims no more than the existence
    check can see. The withdrawal is matched on the inode identity ``_publish``
    returned so a third party's replacement is left alone (modulo
    ``_withdraw``'s two residuals). A concurrent writer that *adopted* this
    ``python.txt`` and won the ``stack.txt`` race loses its interpreter pin
    when we withdraw; nothing on disk distinguishes that writer from one that
    never asked for an interpreter — but only where that writer did not take
    the lock, such as an older ``stack``, a hand-editing user, or a degraded
    lock. When neither condition holds — the publish
    raised something other than a ``ConfigError`` and a ``stack.txt`` is
    present — no withdrawal happens: ``python.txt`` then either sits beside a
    ``stack.txt`` that did land (a complete environment), or, where that
    ``stack.txt`` is a third party's (whose file may equally have appeared
    before the publish raised), remains as an adopter-residual orphan (strictly
    less harmful than the unretryable state the withdrawal would have created).

    Adoption is deliberately narrow: it requires a regular file (not a
    directory, FIFO, socket, or device node — reading any non-regular file can
    block or fail, and none is something this code could have written; symlinks
    are refused unconditionally — the ``O_NOFOLLOW`` open fails on one where
    the preflight runs, and where it does not run nothing is adopted at all),
    readable as UTF-8 (a file we cannot prove is our own debris is refused like
    any other foreign file), whose content matches the requested version (anything else
    is a user edit, not our debris), present before the preflight, and still
    the same inode after the read. Every other existing ``python.txt`` is
    refused — including one that appears after the preflight, which the
    no-clobber publish rejects even when its content matches. A retry with no
    ``--python`` skips the adoption preflight entirely, so it inherits the
    crashed run's orphan ``python.txt`` with no byte comparison. Reading
    through the descriptor itself is what makes the bytes compared the ones
    ``fstat`` approved; no swap can change that. ``O_NOFOLLOW | O_NONBLOCK``
    keeps the open itself from following a symlink or blocking on a FIFO, and
    both constants must exist for the preflight to run at all: where either is
    missing there is no safe way to open a file whose type is not known in
    advance, so nothing is opened, nothing is adopted, and an existing
    ``python.txt`` is refused instead. That costs the user a refusal they must
    clear by hand — edit or delete the file, or omit ``--python`` — which a
    blocked open would not let them do. The recheck after the read
    verifies that ``python.txt`` still names the inode whose bytes matched,
    narrowing the window from "any time after the open" to "after the
    recheck"; a swap after the recheck still leaves a ``python.txt`` this call
    neither wrote nor verified.

    Withdrawal reaches only a ``python.txt`` this call published — never one
    that was adopted, and never a run without ``--python`` — and for that file
    it is skipped in exactly one case: the exception is not a ``ConfigError``
    and a ``stack.txt`` is present (for the reasons above). Every other failure
    of the ``stack.txt`` publish withdraws it: a ``ConfigError``, or a
    non-``ConfigError`` (ENOSPC, EACCES, KeyboardInterrupt) with no
    ``stack.txt`` on disk. Whichever branch runs, the original exception
    propagates unchanged, with one substitution: a ``ConfigError`` whose
    withdrawal was attempted and failed is re-raised as the residual
    ``ConfigError`` naming the ``python.txt`` left behind. A
    failed withdrawal under any other exception is not reported that way —
    replacing it would hide the real failure, so it propagates as itself and
    the ``python.txt`` remains an orphan.

    :param config: Configuration root.
    :param name: Environment name.
    :param tokens: Stack tokens, one per ``stack.txt`` line.
    :param python: When given, also write ``python.txt`` with this version.
    :returns: The paths written by THIS call, ``stack.txt`` first. An adopted
        ``python.txt`` is absent from the list.
    :raises ConfigError: If ``name`` is not a valid environment name, if the
        environment already has a ``stack.txt``, if it has a ``python.txt``
        that cannot be adopted on the terms above, if the ``stack.txt``
        publish was itself refused and the ``python.txt`` this call published
        could not then be withdrawn, or if another process holds the lock
        when the timeout expires.
    :raises OSError: If the lock file cannot be created or opened.
    """
    _validate_name("environment", name)
    # Preflight, adoption, both publishes and the withdrawal are one operation.
    # Unlocked, a competitor can adopt the python.txt this call published, win
    # the stack.txt race, and then lose its interpreter pin when this call's
    # handler withdraws that inode — the inode is byte-identical either way, so
    # nothing in the POSIX file API can tell the two apart after the fact.
    with name_lock(config.env_lock_path(name), name):
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
            # Adoption may only take a regular file: a non-regular python.txt (FIFO,
            # directory, socket, device node) is not something this code could have
            # written, and reading one can block indefinitely or fail in ways refusing
            # it does not. Anything other than a regular file falls through to the
            # existing "already has a python.txt" refusal — the pre-change behavior.
            # A symlink is refused by the open itself, which carries O_NOFOLLOW.
            # Where either open flag does not exist on the platform, the preflight is
            # not attempted: adoption is impossible and every existing python.txt
            # reaches that same refusal, which costs the user a manual edit or delete
            # but is recoverable, unlike an open that blocks on a FIFO.
            if _FASTPATH_AVAILABLE:
                try:
                    # O_NOFOLLOW and O_NONBLOCK keep the open itself from following a
                    # symlink or blocking on a FIFO, and binding the check to the
                    # descriptor means the bytes compared are the ones fstat approved.
                    # Both flags and the gate above come from fsutil, so one place
                    # decides whether this open is safe to make.
                    flags = os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK
                    fd = os.open(python_path, flags)
                    try:
                        st_fd = os.fstat(fd)
                        if stat.S_ISREG(st_fd.st_mode):
                            with os.fdopen(fd, "rb") as handle:
                                fd = -1  # ownership transferred to file object
                                current_bytes = handle.read()
                            if current_bytes.decode("utf-8") == python_text:
                                # The read proves only what the descriptor's inode held,
                                # while adoption is a claim about the pathname: re-verify
                                # the path still names that inode before skipping the write.
                                st_path = os.lstat(python_path)
                                adopt_python = (
                                    (st_fd.st_dev, st_fd.st_ino) == (st_path.st_dev, st_path.st_ino)
                                )
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
            # Withdraw unless a stack.txt that may be this call's own is sitting
            # there. The two disjuncts are not equally strong: ConfigError comes
            # only from _publish mapping FileExistsError, so it proves the
            # no-clobber publish lost the name and the file is somebody else's,
            # while the exists() check proves nothing about history — only that no
            # stack.txt is there now to be left without its interpreter pin.
            should_withdraw = isinstance(exc, ConfigError) or not stack_path.exists()
            if published_python is not None and should_withdraw:
                if not _withdraw(python_path, published_python):
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
    :raises ConfigError: If a starter profile already exists, would shadow an
        existing bundle, or another process holds the lock when the timeout
        expires.
    :raises OSError: If the lock file cannot be created or opened.
    """
    shadow_message = (
        "Profile 'starter' would shadow the existing bundle: "
        f"{config.bundle_path('starter')}"
    )
    path = config.profile_path("starter")
    # Same lock coverage as write_profile — pre-check, publish, and post-check
    # are one operation. See write_profile for what it buys.
    with name_lock(config.stem_lock_path("starter"), "starter"):
        if config.bundle_exists("starter"):
            raise ConfigError(shadow_message, hint=_SHADOW_HINT)
        _publish_unshadowed(
            path,
            _STARTER_PROFILE,
            f"Profile 'starter' already exists: {path}",
            shadowed=lambda: config.bundle_exists("starter"),
            shadow_message=shadow_message,
        )
    return path
