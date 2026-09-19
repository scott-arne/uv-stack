"""The managed ``.gitignore`` block that makes a config root committable.

A config root holds two kinds of file: the ones that describe what you want
(profiles, bundles, stack tokens, declared variable names) and the ones a
machine generates from them (compiled locks, generated ``requirements.in`` and
``environment.yml``, this machine's editor and variable values). Only the first
kind should travel, so committing a root means ignoring the second kind.

``stack config portable`` writes that ignore list into a delimited block it
owns, leaving everything outside the markers untouched. It runs no git itself —
it prints the commands, because untracking files that were committed before
they were ignored is a history-touching operation that belongs to the user.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.fsutil import atomic_write, read_text_utf8, require_regular_file

BEGIN_MARKER = "# BEGIN uv-stack — managed block, do not edit by hand."
END_MARKER = "# END uv-stack"


def _pattern(config: ConfigRoot, path: Path) -> str:
    """Render a path accessor's result as a root-relative ignore pattern.

    Deriving the patterns from the accessors rather than writing them as
    literals means a future rename of a generated file updates the ignore block
    with it, instead of leaving a stale pattern that silently commits the file.

    :param config: The config root the path belongs to.
    :param path: A path under the root — either a root-level file or one built
        with the env name ``"*"``.
    :returns: The POSIX-style relative pattern.
    """
    return path.relative_to(config.root).as_posix()


def ignore_patterns(config: ConfigRoot) -> list[str]:
    """The entries the managed block holds, in written order.

    :param config: The config root.
    :returns: Ignore patterns, relative to the root.
    """
    return [
        # A directory pattern needs its trailing slash; the accessor has none.
        f"{_pattern(config, config.locks_dir)}/",
        _pattern(config, config.editor_path()),
        _pattern(config, config.variables_local_path()),
        _pattern(config, config.env_requirements_in("*")),
        _pattern(config, config.env_environment_yml("*")),
        _pattern(config, config.env_requirements_lock("*")),
        _pattern(config, config.env_local_path("*")),
        # Not a uv-stack path, so there is no accessor to derive it from.
        ".DS_Store",
    ]


def render_block(config: ConfigRoot, newline: str = "\n") -> str:
    """The managed block's text, without a trailing terminator.

    :param config: The config root.
    :param newline: The terminator to join lines with. A rewrite passes the
        terminator the target file already uses, so a CRLF ``.gitignore`` does
        not come back with a block of LF lines wedged into it.
    :returns: Marker, patterns, marker, joined by ``newline``.
    """
    return newline.join([BEGIN_MARKER, *ignore_patterns(config), END_MARKER])


@dataclass(frozen=True)
class PortableResult:
    """What ``stack config portable`` did, or would do under ``--dry-run``.

    :ivar path: The ignore file.
    :ivar outcome: ``"created"``, ``"updated"``, or ``"unchanged"``.
    :ivar block: The managed block's text, LF-joined. This is the form the CLI
        prints; what was written to disk uses the target file's own terminator.
    :ivar is_repository: Whether ``<root>/.git`` exists. Decided by a
        filesystem check rather than a git invocation, so the command works on
        a machine where git is not installed.
    """

    path: Path
    outcome: str
    block: str
    is_repository: bool


def _line_numbers(indexes: list[int]) -> str:
    """Render 0-based line indexes as a 1-based, human-readable list."""
    return ", ".join(str(index + 1) for index in indexes) if indexes else "none"


def _find_span(lines: list[str], path: Path) -> tuple[int, int] | None:
    """Locate the managed block, refusing any topology but the two legal ones.

    Legal: no markers at all, or exactly one BEGIN followed by exactly one END.
    Anything else — a lone marker, a reversed pair, a nested pair — is a file
    a rewrite could corrupt, so the command refuses rather than guessing.

    :param lines: The file split on newlines.
    :param path: The file, for the message.
    :returns: ``(begin index, end index)``, or ``None`` when there is no block.
    :raises ConfigError: On any other topology.
    """
    begins = [i for i, line in enumerate(lines) if line.rstrip() == BEGIN_MARKER]
    ends = [i for i, line in enumerate(lines) if line.rstrip() == END_MARKER]
    if not begins and not ends:
        return None
    if len(begins) == 1 and len(ends) == 1 and begins[0] < ends[0]:
        return (begins[0], ends[0])
    raise ConfigError(
        f"Malformed uv-stack managed block in {path}: BEGIN on line(s) "
        f"{_line_numbers(begins)}, END on line(s) {_line_numbers(ends)}.",
        hint=(
            "Leave exactly one BEGIN line followed by one END line, or delete "
            "both markers and re-run 'stack config portable'. Nothing was "
            "written."
        ),
    )


def _terminator(line: str) -> str:
    """The exact line terminator a line carries, or the empty string.

    :param line: One element of ``str.splitlines(keepends=True)``.
    :returns: The terminator, or ``""`` for an unterminated final line.
    """
    for candidate in ("\r\n", "\n", "\r"):
        if line.endswith(candidate):
            return candidate
    return ""


def _newline(text: str) -> str:
    """The line terminator a file already uses, defaulting to LF.

    Only the first terminator is consulted. A file with mixed endings has no
    single right answer, and matching its first line is the least surprising
    choice — the alternative, imposing LF, is exactly the whole-file rewrite
    this function exists to avoid.

    :param text: The file's contents, read without newline translation.
    :returns: CRLF, LF, or a bare CR.
    """
    index = text.find("\n")
    if index > 0 and text[index - 1] == "\r":
        return "\r\n"
    if index != -1:
        return "\n"
    return "\r" if "\r" in text else "\n"


def _append_block(original: str, block: str, newline: str) -> str:
    """Append the block to a file that has none, preserving its bytes.

    :param original: The file's current contents, byte-exact.
    :param block: The managed block, already joined with ``newline``.
    :param newline: The terminator the file uses.
    :returns: The new contents.
    """
    separator = "" if original == "" or original.endswith(("\n", "\r")) else newline
    return f"{original}{separator}{block}{newline}"


def write_portable_ignore(
    config: ConfigRoot, *, dry_run: bool = False
) -> PortableResult:
    """Create or refresh the managed block in ``<root>/.gitignore``.

    Idempotent: a second run over an up-to-date file reports ``unchanged`` and
    writes nothing. Everything outside the markers is preserved byte for byte,
    line endings included, so a root that already ignores other things keeps
    doing so and a CRLF file stays a CRLF file.

    :param config: The config root.
    :param dry_run: Compute the outcome but write nothing. Malformed-topology
        refusals still raise, because the point of the dry run is to find out.
    :returns: What happened, or would have.
    :raises ConfigError: When the existing file has an illegal marker topology,
        is not valid UTF-8, or is not a regular file.
    """
    path = config.root / ".gitignore"
    block = render_block(config)
    is_repository = (config.root / ".git").exists()

    original: str | None = None
    if path.exists():
        require_regular_file(path)
        original = read_text_utf8(path, exact_newlines=True)

    if original is None:
        outcome = "created"
        updated = block + "\n"
    else:
        # Everything below works on terminator-carrying lines, so the splice
        # can put back exactly the bytes it took out.
        newline = _newline(original)
        file_block = render_block(config, newline)
        lines = original.splitlines(keepends=True)
        span = _find_span(lines, path)
        if span is None:
            updated = _append_block(original, file_block, newline)
        else:
            begin, end = span
            # The END line's own terminator is reused rather than assumed: a
            # block that ends the file with no trailing newline must stay that
            # way, or every run reports 'updated'.
            spliced = file_block + _terminator(lines[end])
            updated = "".join([*lines[:begin], spliced, *lines[end + 1 :]])
        outcome = "unchanged" if updated == original else "updated"

    if not dry_run and updated != original:
        atomic_write(path, updated)
    return PortableResult(
        path=path, outcome=outcome, block=block, is_repository=is_repository
    )


def next_steps(config: ConfigRoot, result: PortableResult) -> list[str]:
    """The git commands to run after writing the block.

    uv-stack prints these rather than running them: the untracking step
    rewrites what the index holds, and a root may already have a remote,
    a branch policy, or uncommitted work that only its owner knows about.

    The branch keys on whether the root is a repository, not on what the write
    did — an ``unchanged`` outcome in an existing repository still needs the
    untracking step, because files committed before they were ignored stay
    tracked no matter how current the ignore file is.

    :param config: The config root.
    :param result: What the write did.
    :returns: One command (or note) per line, in the order to run them.
    """
    root = shlex.quote(str(config.root))
    if not result.is_repository:
        return [
            f"git -C {root} init",
            f"git -C {root} add .",
            f'git -C {root} commit -m "Initial config root"',
            f"git -C {root} remote add origin <url>",
            f"git -C {root} push -u origin HEAD",
        ]
    patterns = " ".join(shlex.quote(pattern) for pattern in ignore_patterns(config))
    return [
        "Some of these patterns may already be tracked from before they were "
        "ignored; untrack them first.",
        f"git -C {root} rm -r --cached --ignore-unmatch -- {patterns}",
        f"git -C {root} add .",
        f'git -C {root} commit -m "Stop tracking generated files"',
        f"git -C {root} push",
    ]
