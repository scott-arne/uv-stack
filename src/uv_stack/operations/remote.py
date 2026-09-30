"""``stack sync remote``: settings, the ssh command, and exit-status hints."""

from __future__ import annotations

import os
import shlex
import textwrap
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.fsutil import read_text_utf8, require_regular_file
from uv_stack.models import RemoteSettings
from uv_stack.runner import Command


def load_remotes(config: ConfigRoot) -> dict[str, RemoteSettings]:
    """Load ``remotes.yaml``; an absent file has no entries.

    :param config: The local root holding ``remotes.yaml``.
    :returns: Each host name, exactly as written, mapped to its settings.
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
    return parse_remotes(read_text_utf8(path), path)


def parse_remotes(text: str, path: Path) -> dict[str, RemoteSettings]:
    """Parse ``remotes.yaml`` text; an empty document has no entries.

    :param text: The file's text.
    :param path: The file the text came from, named in errors.
    :returns: Each host name, exactly as written, mapped to its settings, in
        file order.
    :raises ConfigError: When the text does not parse or validate.
    """
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
            kind = "null" if host is None else type(host).__name__
            raise ConfigError(
                f"Host {host!r} in {path} is a YAML {kind}, not a name.",
                hint="Quote the host name in remotes.yaml, for example \"yes\":",
                path=path)
        remotes[host] = _settings({} if entry is None else entry, path)
    return remotes


def _settings(entry: object, path: Path) -> RemoteSettings:
    """Validate one host's entry, naming the file on failure."""
    try:
        return RemoteSettings.model_validate(entry)
    except ValidationError as exc:
        raise ConfigError(f"Invalid remotes config in {path}: {exc}", path=path) from exc


#: The line breaks YAML recognizes; a block scalar's header ends at the first.
_BREAKS = "\r\n\x85\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}"


def has_comment(text: str) -> bool:
    """Return whether YAML text holds a comment.

    A comment is a ``#`` that no scanner token covers. PyYAML's token for a
    block scalar (``|`` or ``>``) also spans its header line, where a comment
    may sit, so that span is taken to start at the header's line break. A
    ``#`` in a gap the scanner leaves counts as a comment, so a gap refuses a
    clean file rather than dropping a comment.

    :param text: Text that scans as YAML; call this only after it parsed.
    :returns: True when a comment is present.
    """
    covered = [False] * len(text)
    for token in yaml.scan(text, Loader=yaml.SafeLoader):
        start = token.start_mark.index
        if isinstance(token, yaml.ScalarToken) and token.style in ("|", ">"):
            while start < token.end_mark.index and text[start] not in _BREAKS:
                start += 1
        for index in range(start, token.end_mark.index):
            covered[index] = True
    return any(char == "#" and not covered[index] for index, char in enumerate(text))


_DEFAULT_STACK = "stack"


def check_destination(dest: str) -> None:
    """Refuse a DEST that ssh would parse as an option.

    :param dest: The destination as typed.
    :raises UvStackError: When it begins with ``-``.
    """
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
    """Apply flag, then ``remotes.yaml`` entry (DEST exactly as typed), then default.

    :param config: The local root holding ``remotes.yaml``.
    :param dest: The destination as typed, looked up without normalizing.
    :param stack_flag: The ``--remote-stack`` value, if given.
    :param root_flag: The ``--remote-root`` value, if given.
    :returns: The remote ``stack`` command and root; the root is ``None`` when
        neither source sets one, leaving the remote's default.
    :raises ConfigError: When ``remotes.yaml`` is invalid, as ``load_remotes``.
    :raises OSError: When ``remotes.yaml`` cannot be read.
    """
    entry = load_remotes(config).get(dest, RemoteSettings())
    return ResolvedRemote(stack_flag or entry.stack or _DEFAULT_STACK, root_flag or entry.root)


def remote_command(dest: str, stack: str, root: str | None, flags: list[str]) -> Command:
    """Build ``ssh DEST <stack> [--root ROOT] import - FLAGS``.

    ``stack`` is inserted as written so ``~``, ``$HOME``, and multi-word
    commands work on the remote; everything after it is quoted.

    :param dest: The ssh destination, passed as its own argument.
    :param stack: The remote command that runs uv-stack, inserted verbatim.
    :param root: The remote config root, or ``None`` to leave the remote's
        default.
    :param flags: Import flags forwarded after ``import -``.
    :returns: The local ``ssh`` command; the remote part is one shell string.
    """
    tail = (["--root", root] if root else []) + ["import", "-", *flags]
    return Command(["ssh", dest, " ".join([stack, *map(shlex.quote, tail)])])


# Non-interactive ssh often lacks ~/.local/bin, where uv tool and the
# micromamba installer put their commands. stack is inserted verbatim, so a
# PATH prefix reaches stack itself and the uv and micromamba it runs.
_PATH_STACK = "PATH=$HOME/.local/bin:$PATH stack"


def explain_exit(code: int, tail: str, dest: str, stack: str) -> UvStackError | None:
    """Explain the exit statuses that are ssh's or the remote shell's, not import's.

    :param code: The ssh process's exit status.
    :param tail: The end of ssh's standard error, which carries the remote's,
        searched for the message an ``import``-less uv-stack prints.
    :param dest: The destination, named in the message.
    :param stack: The remote command, named when the remote did not find it.
    :returns: An error to print before exiting with the same status, or
        ``None`` when the status is import's own and its output explains it.
    """
    if code == 255:
        return UvStackError(
            f"ssh to {dest} exited with status 255: the connection failed or dropped.",
            hint=f"Check that 'ssh {dest}' works from this shell. Status 255 is ssh's own "
            "failure, not an import refusal. If the connection dropped during the import, "
            "re-running the same command is safe: files already written are reported as "
            "identical and the rest are completed.",
        )
    if code == 127:
        # Dumped rather than formatted, so a DEST that YAML would read as a
        # bool, number, or null comes out quoted and the entry loads as shown.
        entry = yaml.safe_dump({dest: {"stack": _PATH_STACK}}, default_flow_style=False,
                               allow_unicode=True)
        return UvStackError(
            f"'{stack}' was not found on {dest} (exit status 127).",
            hint="Install uv-stack there with 'uv tool install uv-stack'. If it is installed, "
            "the non-interactive ssh shell may not have it on PATH; a remotes.yaml entry can "
            "put the directories holding stack, uv and micromamba on PATH:\n"
            + textwrap.indent(entry.rstrip("\n"), "  "),
        )
    if code == 2 and "No such command 'import'" in tail:
        return UvStackError(
            f"The uv-stack on {dest} is too old to have 'import'.",
            hint=f"Upgrade uv-stack on {dest}: 'uv tool upgrade uv-stack'.",
        )
    return None
