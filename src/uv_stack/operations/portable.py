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

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.fsutil import atomic_write, read_text_utf8_nofollow

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


def enclosing_repository(path: Path) -> Path | None:
    """The top level of the git working tree ``path`` sits in, if any.

    A config root is as often a subdirectory of a repository as a repository
    itself — ``~/dotfiles`` with ``python-envs/`` inside it, or any monorepo
    layout. Probing ``<root>/.git`` alone reads that as "not a repository" for
    a root whose every file the enclosing repository already tracks, which
    drops the untracking advice from :func:`next_steps` and silences doctor's
    ignore-block finding on precisely the roots that need it. So the probe
    walks upward instead of asking once.

    No git is invoked, for the reason given on :class:`PortableResult`: the
    answer has to be the same on a machine where git is not installed. The
    price is that a ``.git`` in a directory this process may not search reads
    as absent, and that a symlinked root is walked through its own path
    rather than its target's.

    :param path: The directory to start from, absolute or relative. It is
        normalised before the walk and tested before its parents, so a root
        that *is* a top level answers with itself.
    :returns: The absolute, ``..``-free path of the nearest directory at or
        above ``path`` holding a ``.git`` entry, or ``None`` when there is
        none below the filesystem root. A ``.git`` *file* — the worktree and
        submodule spelling — counts as readily as the directory, since either
        one means the files are tracked.
    """
    # abspath, not the path as given: Path.parent is lexical and does not
    # collapse '..', so a root spelled '<repo>/sub/../../elsewhere' would have
    # the walk step through '<repo>/sub/..', find that repository's .git, and
    # name a top level the root is not under at all. abspath normalises '..'
    # away and anchors a relative root to the working directory, which is the
    # only reading of a relative --root that git could act on. It stays purely
    # lexical -- no symlink is resolved and no filesystem is consulted -- so
    # every property the walk below relies on is preserved.
    current = Path(os.path.abspath(path))
    while True:
        # os.path.exists, never Path.exists: before 3.14 the pathlib probe
        # re-raises any OSError whose errno is outside its small allowed set,
        # so one unsearchable ancestor would abort a write that has nothing to
        # do with it. The os.path spelling absorbs OSError and ValueError
        # alike and answers False, which is the honest reading of "cannot
        # tell" for a check whose only job is to pick which advice to print.
        if os.path.exists(current / ".git"):
            return current
        parent = current.parent
        # The walk is lexical, so no symlink cycle can extend it; this test is
        # what bounds it at the filesystem root, where a path is its own
        # parent. A relative path bounds out at '.' by the same rule.
        if parent == current:
            return None
        current = parent


@dataclass(frozen=True)
class PortableResult:
    """What ``stack config portable`` did, or would do under ``--dry-run``.

    :ivar path: The ignore file.
    :ivar outcome: ``"created"``, ``"updated"``, or ``"unchanged"``.
    :ivar block: The managed block's text, LF-joined. This is the form the CLI
        prints; what was written to disk uses the target file's own terminator.
    :ivar repository_root: The top level of the git working tree the config
        root sits in, or ``None`` when it sits in none. Decided by a filesystem
        walk rather than a git invocation, so the command works on a machine
        where git is not installed. The commands :func:`next_steps` prints
        name the config root and consult only whether a working tree was found
        — that is what :attr:`is_repository` derives — but the path is kept
        rather than collapsed to a flag, because which working tree already
        tracks this root is the fact to check when the printed advice is not
        the advice that was expected.
    """

    path: Path
    outcome: Literal["created", "updated", "unchanged"]
    block: str
    repository_root: Path | None

    @property
    def is_repository(self) -> bool:
        """Whether the config root is inside a git working tree.

        Derived rather than stored so the two cannot disagree: "inside a
        repository" is exactly "a top level was found".

        :returns: True when :attr:`repository_root` names a directory.
        """
        return self.repository_root is not None


def _line_numbers(indexes: list[int]) -> str:
    """Render 0-based line indexes as a 1-based, human-readable list."""
    return ", ".join(str(index + 1) for index in indexes) if indexes else "none"


def _find_span(lines: list[str], path: Path) -> tuple[int, int] | None:
    """Locate the managed block, refusing any topology but the two legal ones.

    Legal: no markers at all, or exactly one BEGIN followed by exactly one END.
    Anything else — a lone marker, a reversed pair, a nested pair — is a file
    a rewrite could corrupt, so the command refuses rather than guessing.

    A marker counts only when what follows it on the line is spaces, tabs, or
    a CR/LF terminator. A marker ended by any other ``splitlines`` separator
    is left unmatched. When the file's other marker still matches, that is a
    refusal; when neither does, the file reads as having no block at all and
    the block is appended below the stale markers.

    :param lines: The file split on newlines.
    :param path: The file, for the message.
    :returns: ``(begin index, end index)``, or ``None`` when there is no block.
    :raises ConfigError: On any other topology.
    """
    # The strip set is explicit because ``splitlines`` breaks on eleven
    # separators while :func:`_terminator` can put back only three. Under a
    # bare ``rstrip()`` an END marker ended by one of the other eight would
    # match, and the splice — which restores only the END line's terminator —
    # would eat that separator and glue the next line onto the marker. The
    # BEGIN half prevents no such corruption, since whatever follows that
    # marker on its line falls inside the replaced span, but it is held to
    # the same rule so that a marker means one thing in both comprehensions.
    # Either way, trailing spaces and tabs stay tolerated.
    begins = [i for i, line in enumerate(lines) if line.rstrip(" \t\r\n") == BEGIN_MARKER]
    ends = [i for i, line in enumerate(lines) if line.rstrip(" \t\r\n") == END_MARKER]
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
    :returns: CRLF, LF, or CR, the only three recognised. ``""`` both for
        an unterminated final line and for one ended by any other separator.
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
    this function exists to avoid. A CRLF counts as one terminator, not as
    a CR followed by an LF.

    :param text: The file's contents, read without newline translation.
    :returns: CRLF, LF, or a bare CR.
    """
    # The earliest of the two positions decides. Searching for the LF alone
    # would let a bare CR earlier in the file go unnoticed.
    carriage_return = text.find("\r")
    line_feed = text.find("\n")
    if carriage_return == -1 or (line_feed != -1 and line_feed < carriage_return):
        return "\n"
    return "\r\n" if line_feed == carriage_return + 1 else "\r"


def _append_block(original: str, block: str, newline: str) -> str:
    """Append the block to a file that has none, preserving its bytes.

    :param original: The file's current contents, byte-exact.
    :param block: The managed block, already joined with ``newline``.
    :param newline: The terminator the file uses.
    :returns: The new contents.
    """
    separator = "" if original == "" or original.endswith(("\n", "\r")) else newline
    return f"{original}{separator}{block}{newline}"


def _refuse_symlink(path: Path) -> None:
    """Refuse a symlinked ignore file, whatever its target.

    :param path: The ignore file.
    :raises ConfigError: When a symlink stands at ``path``.
    """
    if path.is_symlink():
        raise ConfigError(
            f"Symlinked ignore file: {path}",
            hint=(
                "Replace it with a regular file. Git commits the link itself, "
                "so the rules it points at would not reach another machine."
            ),
        )


def write_portable_ignore(
    config: ConfigRoot, *, dry_run: bool = False
) -> PortableResult:
    """Create or refresh the managed block in ``<root>/.gitignore``.

    Idempotent: a second run over an up-to-date file reports ``unchanged`` and
    writes nothing. Everything outside the markers is preserved byte for byte,
    line endings included, so a root that already ignores other things keeps
    doing so and a CRLF file stays a CRLF file.

    That guarantee covers the file's contents, not its identity: a write goes
    through :func:`atomic_write`, which publishes a new inode. Permissions
    revert to the process default and a hardlinked ignore file is de-linked.
    A *symlinked* ignore file is refused outright rather than written through
    or replaced, because git commits the link and not the rules. The read side
    of that refusal is race-free — it opens ``O_NOFOLLOW``, so a link planted
    after the check is refused rather than followed to a file the invoking user
    can read and the planter cannot. The write side is narrowed and not closed:
    ``os.replace`` never follows a link, so nothing is written through one, but
    a link planted in the instant before the rename is destroyed rather than
    refused, and POSIX has no rename that declines a symlinked target.

    :param config: The config root.
    :param dry_run: Compute the outcome but write nothing. Malformed-topology
        refusals still raise, because the point of the dry run is to find out.
    :returns: What happened, or would have.
    :raises ConfigError: When the existing file is a symlink, has an illegal
        marker topology, is not valid UTF-8, or is not a regular file.
    """
    path = config.root / ".gitignore"
    block = render_block(config)
    repository_root = enclosing_repository(config.root)

    # The symlink refusal sits above the read, not inside an exists() guard,
    # because a dangling link is not exists(): left to the code below, a
    # dangling or stale link would be quietly materialised into a regular
    # file while a link to already-correct content survived untouched. Git
    # commits a surviving link as a link — mode 120000, the target's path as
    # its content — so a clone gets a dangling or foreign link carrying none
    # of these rules and tracks the generated files the block exists to
    # exclude. Refusing is the only outcome that neither destroys a link the
    # user made on purpose nor publishes a root whose rules do not travel.
    #
    # This call is what makes the refusal legible: it names the ignore file and
    # says why git is the reason. It is not what makes it hold. Every refusal
    # below is a test on a name, and the read and the write that follow are
    # separate syscalls, so the guarantees are layered rather than resting here.
    _refuse_symlink(path)

    # The read is the half that can be closed outright, and is: it opens
    # O_NOFOLLOW, so a link planted after the test above is refused by the
    # kernel instead of resolved. Left to an ordinary open, that link's target
    # would be read with the invoking user's permissions — reaching a file the
    # planter cannot read themselves — spliced into the block, and written to a
    # .gitignore the printed sequence then tells the user to commit and push.
    original = read_text_utf8_nofollow(path)

    if original is None:
        outcome: Literal["created", "updated", "unchanged"] = "created"
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
        # The write is the half that cannot be closed. os.replace does not
        # follow a symlink — it replaces the link itself — so nothing is written
        # through to a target, but the link is destroyed rather than refused,
        # and POSIX offers no rename that declines one. Looking again here is
        # therefore a narrowing and not a guarantee: it moves the window from
        # "any time during the read and the splice" to the instant between this
        # lstat and the rename. Taking it costs one syscall on a path already
        # committed to a write.
        _refuse_symlink(path)
        atomic_write(path, updated)
    return PortableResult(
        path=path, outcome=outcome, block=block, repository_root=repository_root
    )


def next_steps(config: ConfigRoot, result: PortableResult) -> list[str]:
    """The git commands to run after writing the block.

    uv-stack prints these rather than running them: the untracking step
    rewrites what the index holds, and a root may already have a remote,
    a branch policy, or uncommitted work that only its owner knows about.

    Two sequences, not three. A root with no repository above it gets the
    bootstrap sequence. A root that *is* a working tree's top level and a root
    nested inside one get the same untracking sequence, because every command
    in it is addressed to the config root and git finds the repository from
    there. Only the first state may print ``git init``: offering it inside an
    existing working tree would advise nesting one repository in another.

    The branch keys on whether the root is in a repository, not on what the
    write did — an ``unchanged`` outcome in an existing repository still needs
    the untracking step, because files committed before they were ignored stay
    tracked no matter how current the ignore file is.

    The untracking sequence stages only ``.gitignore``, which costs the one
    case it does not serve: a root inside a repository whose own files have
    never been committed gets its ignore file staged and nothing else, so its
    owner adds the rest themselves after ``git status`` shows it to them. That
    is the trade taken deliberately, because the alternative — ``add .`` —
    stages every untracked file sitting in the root, and the sequence ends in
    a push.

    :param config: The config root. Every command names it, so the paths the
        user is asked to paste are the root they asked about.
    :param result: What the write did; only whether a working tree was found
        is consulted. It comes from a :func:`write_portable_ignore` call on
        ``config``, which is what ties the answer to this root.
    :returns: One command (or note) per line, in the order to run them.
    """
    root = shlex.quote(str(config.root))
    if not result.is_repository:
        return [
            f"git -C {root} init",
            f"git -C {root} add .",
            # No repository sits above a root being bootstrapped, so no parent
            # .gitignore can reach it — but core.excludesFile still does, and a
            # developer who wrote '.gitignore' into theirs has told git to skip
            # every generated ignore file on the machine. 'add .' honours that
            # silently, and the commit and push below then publish a root whose
            # block never travels. Forcing is safe for the same reason it is in
            # the untracking sequence: the path is this module's own constant
            # naming the file uv-stack just wrote. It is an addition to 'add .',
            # not a replacement — the whole root is staged here on purpose.
            f"git -C {root} add -f .gitignore",
            f'git -C {root} commit -m "Initial config root"',
            f"git -C {root} remote add origin <url>",
            f"git -C {root} push -u origin HEAD",
        ]
    # Addressed to the config root even when the working tree's top level is
    # somewhere above it, so that no part of the root's path ever lands in a
    # pathspec. git discovers the repository by walking up from its working
    # directory and resolves pathspecs against that same directory, so -C
    # <root> reaches the working tree the walk found while leaving the
    # patterns exactly as ignore_patterns produced them. Addressing the top
    # level instead would mean prefixing each pattern with the root's path
    # relative to it, and git reads a pathspec's leading characters as magic:
    # a root named ':(exclude)python-envs' turns every pattern into an
    # exclusion, at which point 'rm --cached' untracks the whole repository.
    # Quoting cannot prevent that — the argument reaches git intact and git is
    # what interprets it — and '--' separates options from pathspecs without
    # disabling magic. The literal '.gitignore' below is in a pathspec
    # position and is safe there for the same reason the patterns are: it is
    # this module's own constant, with no user-supplied text in it.
    #
    # It is also the only thing this step may stage. 'add .' is bounded by the
    # -C directory, but the root is exactly where a user keeps the private
    # values this block exists to keep out of a commit, and an untracked file
    # no pattern happens to cover — an '.env.secret' beside the generated
    # ones — would be staged by it, committed under a message about generated
    # files, and pushed by the last line of the same sequence. The untracking
    # step has already staged its deletions, so the ignore file is all that
    # the sequence's stated purpose still has left to stage.
    #
    # The commit stays unrestricted, and the note above it says so rather than
    # pretending otherwise: 'git commit -- <paths>' is not the narrowing it
    # looks like, because a pathspec switches the commit to --only semantics,
    # which records the working-tree state of those paths and would put back
    # the very entries 'rm --cached' just removed from the index.
    #
    # '-f' because the enclosing repository may ignore this very path — a
    # parent .gitignore holding 'python-envs/' or '*.gitignore' is enough —
    # and without it git refuses the add and says so, while the commit and
    # push two lines below still run. That publishes the untracking without
    # the rules that justify it, so the next clone tracks the generated files
    # again: the one outcome this command exists to prevent. Forcing is safe
    # here and nowhere else in the sequence, because the path is this module's
    # own constant naming the file uv-stack just wrote.
    patterns = " ".join(shlex.quote(pattern) for pattern in ignore_patterns(config))
    return [
        "Some of these patterns may already be tracked from before they were "
        "ignored; untrack them first.",
        "The commit records everything already staged in the repository, not "
        "just this root; check 'git status' first.",
        f"git -C {root} rm -r --cached --ignore-unmatch -- {patterns}",
        f"git -C {root} add -f .gitignore",
        f'git -C {root} commit -m "Stop tracking generated files"',
        f"git -C {root} push",
    ]
