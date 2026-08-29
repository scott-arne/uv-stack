"""Editor selection for ``stack edit``.

Pure: decides *which* editor command to run and *how* to spell its argv. It
never launches anything, so the precedence chain and the string-to-argv rule
are testable without a terminal or a subprocess.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError

#: Every rung of the precedence chain, in the order the hint should list them.
_CHAIN_HINT = (
    "Set one with --editor, $UV_STACK_EDITOR, <config-root>/editor.txt, "
    "$VISUAL, or $EDITOR."
)


@dataclass(frozen=True)
class EditorCommand:
    """A resolved editor command and where it came from.

    :param command: The command string, exactly as configured.
    :param source: Human-readable origin, used in error messages.
    :param from_flag: True when ``--editor`` supplied it. The caller reports a
        malformed value as a usage error in that case and as a config error
        otherwise, because only one of the two is a mistake in this
        invocation.
    """

    command: str
    source: str
    from_flag: bool


def resolve_editor(config: ConfigRoot, flag: str | None) -> EditorCommand:
    """Pick the editor command from the precedence chain.

    Order: ``--editor``, ``$UV_STACK_EDITOR``, ``<config-root>/editor.txt``,
    ``$VISUAL``, ``$EDITOR``. A rung holding only whitespace is skipped rather
    than winning, since an exported-but-empty ``VISUAL`` is common and should
    not shadow a usable ``EDITOR``.

    :param config: Config root, for ``editor.txt``.
    :param flag: The ``--editor`` value, or ``None`` when it was not supplied.
    :returns: The winning command and its source.
    :raises ConfigError: When no rung yields a non-blank value, or when
        ``editor.txt`` cannot be decoded.
    """
    if flag is not None and flag.strip():
        return EditorCommand(flag, "--editor", from_flag=True)
    from_env = os.environ.get("UV_STACK_EDITOR")
    if from_env and from_env.strip():
        return EditorCommand(from_env, "$UV_STACK_EDITOR", from_flag=False)
    from_file = config.default_editor()
    if from_file and from_file.strip():
        return EditorCommand(from_file, str(config.editor_path()), from_flag=False)
    for var in ("VISUAL", "EDITOR"):
        value = os.environ.get(var)
        if value and value.strip():
            return EditorCommand(value, f"${var}", from_flag=False)
    raise ConfigError("No editor configured.", hint=_CHAIN_HINT)


def _is_path_like(command: str) -> bool:
    """Whether a command string is spelled as a path, not as a program name.

    ``Path.is_file()`` resolves a bare name against the current directory, so
    on its own it lets an unrelated file decide how the command is split: a
    file named ``code -w`` in the cwd would make that whole string ``argv[0]``
    and hand ``execvp`` a program that does not exist.

    :param command: The configured command string.
    :returns: True when the string is absolute, ``~``-prefixed, or contains a
        path separator.
    """
    if command.startswith("~") or Path(command).is_absolute():
        return True
    return os.sep in command or (os.altsep is not None and os.altsep in command)


def _expand_tilde_if_present(path: str) -> str:
    """Expand a leading ``~`` in a path, or return the path unchanged.

    When the path starts with ``~``, attempt to expand it. The original is
    returned unchanged both when no home directory can be determined and when
    there is nothing to expand — ``~nosuchuser`` names no user, so
    ``expanduser`` leaves it alone without raising. Callers cannot tell those
    two apart from the return value and should not try; either way the result
    is a path that either names a file or does not. Non-``~`` paths are
    returned as-is to preserve forms like ``./editor``, since
    ``str(Path('./editor'))`` would drop the ``./`` and turn a
    directory-relative command into a ``$PATH`` lookup.

    :param path: A path string that may start with ``~``.
    :returns: The expanded path, or the original when it did not expand.
    """
    if not path.startswith("~"):
        return path
    try:
        return str(Path(path).expanduser())
    except RuntimeError:
        # No home directory to expand against. Returning the original path
        # degrades to a launch failure, which reports the command that was
        # tried. Letting the RuntimeError out would be a traceback, since
        # UvStackGroup.invoke handles only UvStackError and OSError.
        return path


def _launchable_path(command: str) -> str | None:
    """The command as a single launchable path, or ``None`` when it is not one.

    A ``~`` prefix is expanded here rather than left for the runner: argv goes
    to ``execvp``, with no shell behind it to expand anything, so a literal
    ``~/bin/editor`` would name nothing. Only that form is rewritten. Every
    other spelling already resolves as written, and putting it through ``Path``
    would drop a leading ``./`` and so turn a directory-relative command into a
    ``$PATH`` lookup.

    :param command: The configured command string.
    :returns: The path to launch, or ``None`` when the command is not spelled
        as a path or names no existing file.
    """
    if not _is_path_like(command):
        return None
    # An unexpanded ``~`` needs no special case here. It survives expansion
    # only when it names nothing to expand, and the existence test below is
    # what decides whether it is launchable either way.
    expanded = _expand_tilde_if_present(command)
    if not Path(expanded).is_file():
        return None
    return expanded


def editor_argv(editor: EditorCommand, target: Path) -> list[str]:
    """Build the argv that opens ``target`` in ``editor``.

    A command string that is spelled as a path *and* names an existing file is
    used as one argument: such a path may legitimately contain spaces, and
    ``shlex`` would split it into arguments that do not exist. Anything else is
    a small shell-like command line and is split, so ``code -w`` and
    ``emacsclient -nw`` work no matter what the current directory happens to
    contain.

    A leading ``~`` on ``argv[0]`` is expanded in both cases, because nothing
    downstream expands it: the runner hands argv to ``execvp``, with no shell
    behind it.

    :param editor: The resolved command and its source.
    :param target: The file to open, appended as the final argument.
    :returns: The full argv list.
    :raises ConfigError: When the command cannot be split because its quoting
        is unbalanced, or splits to nothing at all.
    """
    launchable = _launchable_path(editor.command)
    if launchable is not None:
        return [launchable, str(target)]
    try:
        parts = shlex.split(editor.command)
    except ValueError as error:
        raise ConfigError(
            # shlex's own messages ("No closing quotation") carry no user data,
            # and `source` is a fixed label or a config path, so this message
            # never interpolates the editor string itself.
            f"Cannot parse the editor command from {editor.source}: {error}",
            hint="Check the quoting.",
        ) from error
    if not parts or not parts[0]:
        raise ConfigError("No editor configured.", hint=_CHAIN_HINT)
    # Expand ~ in argv[0] to match the single-argument branch behavior.
    parts[0] = _expand_tilde_if_present(parts[0])
    return [*parts, str(target)]
