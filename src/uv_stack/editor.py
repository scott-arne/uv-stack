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


def editor_argv(editor: EditorCommand, target: Path) -> list[str]:
    """Build the argv that opens ``target`` in ``editor``.

    A command string that names an existing file is used verbatim: an absolute
    path may legitimately contain spaces, and ``shlex`` would split it into
    arguments that do not exist. Anything else is a small shell-like command
    line and is split, so ``code -w`` and ``emacsclient -nw`` work.

    :param editor: The resolved command and its source.
    :param target: The file to open, appended as the final argument.
    :returns: The full argv list.
    :raises ConfigError: When the command cannot be split because its quoting
        is unbalanced, or splits to nothing at all.
    """
    if Path(editor.command).is_file():
        return [editor.command, str(target)]
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
    # Filter out empty strings (shlex.split("''") returns [''])
    parts = [p for p in parts if p]
    if not parts:
        raise ConfigError("No editor configured.", hint=_CHAIN_HINT)
    return [*parts, str(target)]
