"""``stack sync remote``: settings, the ssh command, and exit-status hints."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass

import yaml
from pydantic import ValidationError

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.fsutil import read_text_utf8, require_regular_file
from uv_stack.models import RemoteSettings
from uv_stack.runner import Command


def load_remotes(config: ConfigRoot) -> dict[str, RemoteSettings]:
    """Load ``remotes.yaml``; an absent file has no entries.

    :raises ConfigError: When the file is not regular, is not valid UTF-8, or
        does not parse or validate.
    :raises OSError: When the file exists but cannot be read.
    """
    path = config.remotes_path()
    if not os.path.lexists(path):
        return {}
    require_regular_file(path)
    # Read outside the broad except: a decode failure is already a ConfigError
    # naming the file, and an unreadable file is an OSError, not bad YAML.
    text = read_text_utf8(path)
    try:
        data = yaml.safe_load(text)
    except Exception as exc:  # Same breadth as ConfigRoot._load_yaml_model.
        raise ConfigError(f"Invalid YAML in {path}: {type(exc).__name__}: {exc}",
                          hint="Fix the YAML syntax.", path=path) from exc
    if data is None:
        # PyYAML loads an empty document and an explicit null alike. Only the
        # empty one means "no entries": it composes to no node, or to the
        # zero-width scalar a bare ``---`` leaves. ``null``, ``~`` or a
        # ``!!null`` tag is written out, so it has width, and the file must be
        # a mapping.
        node = yaml.compose(text, Loader=yaml.SafeLoader)
        if node is None or node.start_mark.index == node.end_mark.index:
            return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}, got {type(data).__name__}.",
                          path=path)
    remotes: dict[str, RemoteSettings] = {}
    for host, entry in data.items():
        if not isinstance(host, str):
            raise ConfigError(
                f"Host {host!r} in {path} is a YAML {type(host).__name__}, not a name.",
                hint="Quote the host name in remotes.yaml, for example \"yes\":",
                path=path)
        try:
            remotes[host] = RemoteSettings.model_validate({} if entry is None else entry)
        except ValidationError as exc:
            raise ConfigError(f"Invalid remotes config in {path}: {exc}", path=path) from exc
    return remotes


_DEFAULT_STACK = "stack"


def check_destination(dest: str) -> None:
    """Refuse a DEST that ssh would parse as an option."""
    if dest.startswith("-"):
        raise UvStackError(
            f"Refusing destination '{dest}': it begins with '-', which ssh reads as an option.",
            hint="Name the host as user@host or an ssh-config alias.",
        )


@dataclass(frozen=True)
class ResolvedRemote:
    """The remote ``stack`` command and root, after precedence."""

    stack: str
    root: str | None


def resolve_settings(config: ConfigRoot, dest: str, *, stack_flag: str | None,
                     root_flag: str | None) -> ResolvedRemote:
    """Apply flag, then ``remotes.yaml`` entry (DEST exactly as typed), then default."""
    entry = load_remotes(config).get(dest, RemoteSettings())
    return ResolvedRemote(stack_flag or entry.stack or _DEFAULT_STACK, root_flag or entry.root)


def remote_command(dest: str, stack: str, root: str | None, flags: list[str]) -> Command:
    """Build ``ssh DEST <stack> [--root ROOT] import - FLAGS``.

    ``stack`` is inserted as written so ``~``, ``$HOME``, and multi-word
    commands work on the remote; everything after it is quoted.
    """
    tail = (["--root", root] if root else []) + ["import", "-", *flags]
    return Command(["ssh", dest, " ".join([stack, *map(shlex.quote, tail)])])


def explain_exit(code: int, tail: str, dest: str, stack: str) -> UvStackError | None:
    """Explain the exit statuses that are ssh's or the remote shell's, not import's."""
    if code == 255:
        return UvStackError(
            f"Could not connect to {dest}: ssh exited with status 255.",
            hint=f"Check that 'ssh {dest}' works from this shell; this is a connection "
            "failure, not an import failure.",
        )
    if code == 127:
        return UvStackError(
            f"'{stack}' was not found on {dest} (exit status 127).",
            hint="Install uv-stack there with 'uv tool install uv-stack', or add its path "
            f"to remotes.yaml:\n  {dest}:\n    stack: ~/.local/bin/stack",
        )
    if code == 2 and "No such command 'import'" in tail:
        return UvStackError(
            f"The uv-stack on {dest} is too old to have 'import'.",
            hint=f"Upgrade uv-stack on {dest}: 'uv tool upgrade uv-stack'.",
        )
    return None
