"""``stack sync remote``: settings, the ssh command, and exit-status hints."""

from __future__ import annotations

import os
import shlex
import stat
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.fsutil import atomic_write, name_lock, read_text_utf8, require_regular_file
from uv_stack.hints import escape_controls, has_control
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
    _refuse_repeated_hosts(text, path)
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


def _refuse_repeated_hosts(text: str, path: Path) -> None:
    """Refuse a host written twice, which YAML silently collapses to the last.

    :param text: Text that loads as a YAML mapping.
    :param path: The file the text came from, named in errors.
    :raises ConfigError: When a host appears twice.
    """
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    assert isinstance(root, yaml.MappingNode)
    first_lines: dict[str, int] = {}
    for key, _ in root.value:
        # Only string keys can repeat a host: a merge key follows YAML's rule
        # that an explicit key overrides it, and any other key is refused as a
        # host by the caller.
        if key.tag != "tag:yaml.org,2002:str":
            continue
        line = key.start_mark.line + 1
        if key.value in first_lines:
            raise ConfigError(
                f"Host {key.value!r} is listed twice in {path}: "
                f"lines {first_lines[key.value]} and {line}.",
                hint="Remove or merge one of them; YAML would keep only the last.",
                path=path)
        first_lines[key.value] = line


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


_HOST_HINT = "Name the host as user@host or an ssh-config alias."
_DEFAULT_STACK = "stack"


@dataclass(frozen=True)
class Removal:
    """What :func:`remove_remote` did.

    :param entry: HOST's entry after the change, or ``None`` when the whole
        entry was removed.
    :param not_set: The named fields HOST did not have set, in the order given.
    """

    entry: RemoteSettings | None
    not_set: tuple[str, ...]


def _check_new_host(host: str) -> None:
    """Refuse a HOST that ``set`` must not create.

    ``stack sync remote`` can never use an empty or ``-``-leading name, and a
    control character makes a name no one can type as DEST.

    :param host: The host name as typed.
    :raises ConfigError: When it is empty or holds a control character.
    :raises UvStackError: When it begins with ``-``, as :func:`check_destination`.
    """
    if not host:
        raise ConfigError("Refusing an empty host name.", hint=_HOST_HINT)
    check_destination(host)
    if has_control(host):
        raise ConfigError(f"Refusing host '{host}': it contains a control character.",
                          hint=_HOST_HINT)


def _read_target(path: Path) -> tuple[Path, str | None]:
    """Resolve ``path`` once and read the file it names.

    Every symlink on the way is resolved, not only a final one, so a
    retargeted ancestor such as a symlinked config root shows up as a
    different target.

    :param path: The ``remotes.yaml`` path; it or any ancestor may be a symlink.
    :returns: The file to write, with every symlink resolved, and its exact
        text, or ``None`` when nothing is at ``path``.
    :raises ConfigError: When something other than a regular file is there,
        or the file is not valid UTF-8.
    :raises OSError: When the file cannot be read.
    """
    if not os.path.lexists(path):
        return Path(os.path.realpath(path)), None
    require_regular_file(path)
    target = Path(os.path.realpath(path))
    # Exact newlines, so the re-read before publishing sees a line-ending
    # change as the change it is.
    return target, read_text_utf8(target, exact_newlines=True)


def _serialize_remotes(remotes: dict[str, RemoteSettings]) -> str:
    """Render ``remotes`` as ``remotes.yaml`` text, keeping host order."""
    return yaml.safe_dump(
        {host: settings.model_dump(exclude_none=True) for host, settings in remotes.items()},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )


def _update_remotes(
    config: ConfigRoot, change: Callable[[dict[str, RemoteSettings]], bool]
) -> dict[str, RemoteSettings]:
    """Apply ``change`` to ``remotes.yaml`` and write the result.

    The file is resolved and read once, under the root's remotes lock, and the
    re-read before publishing compares against that one read, so content read
    from one file is never published to another.

    :param config: The local root holding ``remotes.yaml``.
    :param change: Edits the parsed mapping in place and returns False when
        there is nothing to write. It may raise to refuse.
    :returns: The mapping after ``change``.
    :raises ConfigError: When the file is not regular or is invalid, holds
        comments, would not read back as the intended settings, or changed
        while it was being updated; or when the lock times out.
    :raises OSError: When the file cannot be read or written.
    """
    path = config.remotes_path()
    # Resolve the root once so the lock and the target both resolve through the
    # same tree; a retargeted root link must not leave the writer holding one
    # tree's lock while it updates another tree's file.
    locked_root = ConfigRoot(os.path.realpath(config.root))
    # Where locking is unsupported, name_lock degrades to a no-op as it does
    # for every caller; the re-read before publishing is then the only guard.
    with name_lock(locked_root.remotes_lock_path(), "remotes.yaml", action="updating"):
        target, text = _read_target(path)
        # Refuse if the target does not resolve through the locked root.
        if target != Path(os.path.realpath(locked_root.remotes_path())):
            raise ConfigError(f"{path} changed while it was being updated; nothing was written.",
                              hint="Run the command again.",
                              path=path)
        remotes = {} if text is None else parse_remotes(text, path)
        if text is not None and has_comment(text):
            raise ConfigError(f"{path} contains comments, which rewriting would drop.",
                              hint="Edit it with 'stack edit remotes', or remove the comments "
                                   "first.",
                              path=path)
        if not change(remotes):
            return remotes
        new_text = _serialize_remotes(remotes)
        if list(parse_remotes(new_text, path).items()) != list(remotes.items()):
            # Reachable from a typed argument: safe_dump writes U+0085 as a
            # line break, which loads back as a space.
            raise ConfigError(
                f"{path} was not rewritten: its new contents would not read back as the "
                "intended settings.",
                hint="A value may hold a character the YAML writer changes, such as U+0085 "
                     "(NEL); edit the file with 'stack edit remotes' instead.",
                path=path)
        # Catches a save from 'stack edit remotes' (which takes no lock), any
        # writer while the lock is degraded, a writer from another root sharing
        # this file through a symlink (the lock is per root), and a retargeted
        # link. A change landing between here and the rename is still lost:
        # POSIX has no compare-and-swap rename.
        if _read_target(path) != (target, text):
            raise ConfigError(f"{path} changed while it was being updated; nothing was written.",
                              hint="Run the command again.", path=path)
        mode = None if text is None else stat.S_IMODE(os.stat(target).st_mode)
        atomic_write(target, new_text, mode=mode)
    return remotes


def set_remote(config: ConfigRoot, host: str, *, stack: str | None = None,
               root: str | None = None) -> RemoteSettings:
    """Merge the given fields into HOST's entry, creating the entry if absent.

    :param config: The local root holding ``remotes.yaml``.
    :param host: The host name, stored exactly as given.
    :param stack: The remote ``stack`` command; ``None`` leaves it as it is.
    :param root: The remote's config root; ``None`` leaves it as it is.
    :returns: HOST's entry as written.
    :raises UvStackError: When HOST begins with ``-``.
    :raises ConfigError: When HOST is empty or holds a control character, a
        value is invalid, or the file cannot be updated.
    :raises OSError: When the file cannot be read or written.
    """
    _check_new_host(host)
    path = config.remotes_path()
    # Validated before the lock, like HOST, so a typo is refused without
    # waiting on another writer.
    given = _settings(
        {key: value for key, value in (("stack", stack), ("root", root)) if value is not None},
        path,
    ).model_dump(exclude_none=True)

    def change(remotes: dict[str, RemoteSettings]) -> bool:
        current = remotes.get(host, RemoteSettings()).model_dump(exclude_none=True)
        remotes[host] = _settings({**current, **given}, path)
        return True

    return _update_remotes(config, change)[host]


def remove_remote(config: ConfigRoot, host: str, fields: Sequence[str] = ()) -> Removal:
    """Remove HOST's entry, or only the named fields of it.

    HOST gets no shape checks beyond existence, so a hand-written entry that
    :func:`set_remote` would refuse can still be removed.

    :param config: The local root holding ``remotes.yaml``.
    :param host: The host name, matched exactly.
    :param fields: Fields to clear (``stack``, ``root``); none removes the
        entry. An entry left with no fields stays, meaning the same as none.
    :returns: HOST's resulting entry, and the named fields it did not have.
        Nothing is written when every named field was unset.
    :raises ConfigError: When HOST is not in the file, or the file cannot be
        updated.
    :raises OSError: When the file cannot be read or written.
    """
    path = config.remotes_path()
    named = tuple(dict.fromkeys(fields))
    not_set: tuple[str, ...] = ()

    def change(remotes: dict[str, RemoteSettings]) -> bool:
        nonlocal not_set
        if host not in remotes:
            raise ConfigError(f"No remote named '{host}' in {path}.",
                              hint="Run 'stack config remote list' to see the configured hosts.",
                              path=path)
        if not named:
            del remotes[host]
            return True
        stored = remotes[host].model_dump(exclude_none=True)
        not_set = tuple(field for field in named if field not in stored)
        if len(not_set) == len(named):
            return False
        for field in named:
            stored.pop(field, None)
        remotes[host] = _settings(stored, path)
        return True

    remotes = _update_remotes(config, change)
    return Removal(remotes[host] if named else None, not_set)


def check_destination(dest: str) -> None:
    """Refuse a DEST that ssh would parse as an option.

    :param dest: The destination as typed.
    :raises UvStackError: When it begins with ``-``.
    """
    if dest.startswith("-"):
        raise UvStackError(
            f"Refusing destination '{dest}': it begins with '-', which ssh reads as an option.",
            hint=_HOST_HINT,
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


def entry_lines(host: str, settings: RemoteSettings) -> list[str]:
    """Render one host's entry as ``stack config remote list`` prints it.

    An unset field shows its effective value. Hosts and values are escaped
    because a hand-written one may hold a control character.

    :param host: The host name.
    :param settings: Its stored settings.
    :returns: The host line and one indented line per field.
    """
    stack = settings.stack or f"{_DEFAULT_STACK} (default)"
    root = settings.root or "(remote's default)"
    return [escape_controls(host), f"  stack: {escape_controls(stack)}",
            f"  root: {escape_controls(root)}"]


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
