"""Diagnostics for a uv-stack config tree.

``diagnose`` writes one probe lock at ``.locks/probe.lock`` to test whether
locking works at all; the path is resolved the same way ``name_lock`` resolves
it, so a ``.locks`` that is a symlink to a directory sends the write to that
directory. It returns a list of findings the CLI prints with suggested fixes. It
flags missing directories, legacy names (``*.in``, ``*.bundle``,
``profiles.txt``), env-like directories left at the root or left without the
``stack.txt`` that gets them listed, envs missing their source files, and a
config root whose filesystem cannot serve the advisory locks that serialize
concurrent creates.

It also reports portability problems: declared variables with no value here,
references that are undeclared, malformed, or in an entry that may not hold
one, editable checkouts that are absent, a ``project-python.txt`` value that
will not travel, and a missing or stale managed ``.gitignore`` block in a
config root that sits inside a git repository.
"""

from __future__ import annotations

import errno
import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import yaml

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.fsutil import (
    _LINK_FALLBACK_ERRNOS,
    Published,
    atomic_write_new,
    link_or_copy_no_replace,
    name_lock,
    nofollow_read_flags,
    probe_locking,
)
from uv_stack.operations.portable import enclosing_repository, write_portable_ignore
from uv_stack.operations.project import python_travel_problem
from uv_stack.parse import read_clean_lines
from uv_stack.variables import Variables, expand_all, placement_problem, referenced_names

_KNOWN_TOP_LEVEL = {"profiles", "bundles", "envs", "lib", ".locks"}

#: Files whose presence makes a directory an env directory rather than an
#: unrelated one. Matched against enumerated names, so a marker left as a
#: dangling symlink still marks its directory -- the entry is there, and
#: the repair moves the directory whole either way.
_ENV_MARKERS = {"requirements.in", "environment.yml"}

# Public-source fallback: narrower than the temp-file set, because on Linux
# with fs.protected_hardlinks=1, EPERM from os.link means "policy denies
# hardlinking a file you do not own," not "this filesystem has no hard links."
# Treating it as link-less turns a denial into copy-then-unlink. A caller
# publishing its own temp file wants the full set.
_PUBLIC_LINK_FALLBACK = _LINK_FALLBACK_ERRNOS - {errno.EPERM}

#: What to do about each placement refusal. The explanation returned by
#: :func:`~uv_stack.variables.placement_problem` names the condition that
#: failed; this names the form to use instead. Indexed rather than queried:
#: a refusal kind with no row here is a bug that must surface, not a finding
#: shipped with an empty fix.
_PLACEMENT_FIX = {
    "multiline-entry": "Split it into one requirement per line.",
    "continuation-entry": (
        "Remove the trailing backslash. A requirement entry is one line; "
        "there is nothing to continue onto."
    ),
    "malformed-reference": (
        "Write references as ${NAME}. There is no escape sequence, so a "
        "literal '${' cannot appear in a requirement entry."
    ),
    "misplaced-reference": (
        "Put the reference in an editable ('-e ${NAME}/pkg'), an option value, "
        "or a path operand ('${NAME}/pkg') instead."
    ),
}


@dataclass
class Finding:
    """A single diagnostic result.

    :param level: ``"error"`` or ``"warn"``.
    :param message: What was detected.
    :param fix: Optional suggested remediation.
    :param kind: Machine-readable finding type (used by :func:`repair`).
    :param path: The offending file or directory, when applicable.
    :param dest: The repair target path, when applicable.
    """

    level: str
    message: str
    fix: str | None = None
    kind: str = ""
    path: Path | None = None
    dest: Path | None = None


def _unparseable(path: Path, detail: str) -> Finding:
    """Report a source doctor could not read, rather than failing on it.

    :param path: The file that could not be read.
    :param detail: ``str(error)`` from whichever read failed. Both families the
        callers catch render the same way: ``UvStackError.__init__`` passes its
        ``message`` to ``Exception.__init__``, so ``str()`` is that message, and
        an ``OSError`` has nothing else to offer. A normalizing helper would add
        a branch no test could distinguish.
    :returns: A ``warn`` finding.
    """
    return Finding(
        "warn",
        f"Cannot read {path}: {detail}",
        fix="Fix or remove the file; the checks that read it were skipped.",
        kind="unparseable-source",
        path=path,
    )


def _children(directory: Path) -> tuple[list[Path], list[Finding]]:
    """List a directory's contents, or report why the walk failed.

    ``iterdir`` raises rather than yielding nothing when a directory cannot be
    read, and it is lazy — the error surfaces on the first iteration, not at
    the call. Materializing inside the guard is what makes the ``except``
    cover it.

    :param directory: The directory to walk.
    :returns: ``(children, findings)``. ``findings`` holds exactly one entry
        when the walk failed, and ``children`` is then empty.
    """
    try:
        return list(directory.iterdir()), []
    except OSError as error:
        return [], [_unparseable(directory, str(error))]


def _scaffold_directory_problem(name: str, path: Path) -> Finding | None:
    """Probe one of the root's own directories, distinguishing three answers.

    Absent and unreadable are not the same finding. ``os.path.isdir`` collapses
    both into ``False``, and stat'ing a directory needs search permission on its
    *parent* rather than on itself, so a whole config root one ``chmod`` from
    usable would report every directory under it absent. That answer is not
    merely imprecise: it is the one thing certainly untrue about a directory the
    kernel declined to describe, and ``doctor --fix`` acts on it by running a
    ``mkdir`` that fails for the same reason the stat did.

    So a residual errno gets an ``unparseable-source`` warn, and this probe is
    where it has to come from. Nothing downstream picks the path up: the scans
    below are guarded by ``os.path.isdir``, which answers ``False`` for exactly
    the errno that made the stat fail, so :func:`_children` never runs on it and
    no walk names it. A root whose ``profiles/`` is a self-referential symlink
    otherwise produces no findings whatsoever — a clean bill of health for a
    directory doctor never managed to look at.

    The warn describes a directory rather than a file, which is what
    :func:`_children` already does with the same helper.

    This warn overlaps the scan's on an unsearchable root, which fails this stat
    *and* the listings :func:`_scan_sources` does, rendering the same errno for
    the same path both times. :func:`_without_repeats` collapses the pair, which
    is the right place for it: the scan's warn is not redundant in general —
    a directory that stats but cannot be opened reaches only that one — so
    neither probe may be silenced, and conditioning either on what the other
    found would couple two checks that have no business knowing about each
    other.

    :param name: The directory's short name, for the missing-directory message.
    :param path: The directory to probe.
    :returns: A ``missing-dir`` error when nothing, or something that is not a
        directory, is at ``path``; an ``unparseable-source`` warn when the probe
        failed for any other reason; ``None`` when a directory is there.
    """
    try:
        if stat.S_ISDIR(os.stat(path).st_mode):
            return None
    except (FileNotFoundError, NotADirectoryError):
        # ENOENT and ENOTDIR are the two errnos that genuinely mean nothing is
        # there: the last component is missing, or a component above it is not
        # a directory, so no directory can be at the path either way.
        pass
    except OSError as error:
        return _unparseable(path, str(error))
    return Finding(
        "error",
        f"Missing {name} directory: {path}",
        fix=f"Create {path} (or run 'stack config init').",
        kind="missing-dir",
        path=path,
    )


def _is_root_directory(child: Path, config: ConfigRoot) -> bool:
    """Whether ``child`` is one of the config root's own directories.

    Asked of the filesystem, not answered by transforming the name. On a
    case-insensitive filesystem ``Envs`` IS ``envs`` — the directory every
    other read in the program resolves — and treating it as an unknown
    top-level entry ends in a ``mv`` that moves that directory into itself.

    Casefolding the name instead would be the mirror bug named in the comment
    below: on a case-sensitive filesystem ``Envs`` is a different directory,
    and a real misplaced env inside it has to stay reportable. Asking leaves
    each platform's answer to that platform.

    :param child: A top-level entry of the config root.
    :param config: The configuration root, for the known directory paths.
    :returns: ``True`` when ``child`` is one of the root's own directories.
    """
    if child.name in _KNOWN_TOP_LEVEL:
        return True
    return any(
        _same_directory(child, config.root / known) for known in _KNOWN_TOP_LEVEL
    )


def _same_directory(left: Path, right: Path) -> bool:
    """Whether two paths name one directory, answering False when unknowable.

    ``samefile`` raises when either side is absent, which for this question is
    simply "no" — a root that has no ``lib/`` cannot have a case variant of one.

    :param left: First path.
    :param right: Second path.
    :returns: ``True`` only when the filesystem says both name one directory.
    """
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _without_repeats(findings: list[Finding]) -> list[Finding]:
    """Drop any finding exactly equal to one already reported.

    Two findings agreeing on level, message, fix, kind, path and dest are
    indistinguishable by construction: the CLI renders them as the same line,
    so the second tells a reader nothing except that something went wrong
    twice — which it did not. :func:`repair` would also act on both, attempting
    the same move or write a second time on a tree the first one changed.

    A general guard, not a patch for one pair. The pair that motivated it is an
    unsearchable root, where the scaffold-directory stat and the listing
    :func:`_scan_sources` does both fail with the same errno on the same path
    and both render it through :func:`_unparseable`; independent probes reaching
    one conclusion is the shape, and nothing stops another pair from taking it.

    Comparison is by ``==`` over a list rather than through a ``set`` or
    ``dict.fromkeys``, because :class:`Finding` is a plain mutable dataclass: it
    has ``__eq__`` and no ``__hash__``. The list scan is quadratic in a sequence
    that is a handful of items long on any tree worth diagnosing.

    :param findings: The findings collected, in the order they were produced.
    :returns: The same findings with later exact repeats removed, first
        occurrence kept, order otherwise untouched.
    """
    unique: list[Finding] = []
    for finding in findings:
        if finding not in unique:
            unique.append(finding)
    return unique


def _bundle_for_stem(config: ConfigRoot, stem: str, children: list[Path]) -> Path:
    """The bundle ``stem`` reaches, spelled the way ``bundles/`` spells it.

    On a case-insensitive filesystem a profile stem of ``ds`` reaches a
    ``bundles/DS.yaml``, and reporting that collision as two ``ds.yaml`` paths
    would read as a bug in doctor rather than as the name clash it is. The
    exact spelling is preferred over a folded one so that a case-*sensitive*
    filesystem holding both files names the one the stem actually opens.

    :param config: The configuration root being diagnosed.
    :param stem: A profile stem that opens a file under ``bundles/``.
    :param children: ``bundles/`` as :func:`_children` saw it.
    :returns: The walked child the stem reaches, or the path it resolves to
        when the walk did not see it — which is the unreadable-directory case,
        already reported separately, and leaves the finding itself sound
        because ``bundle_exists`` reached the file.
    """
    candidates = [child for child in sorted(children) if child.suffix == ".yaml"]
    for child in candidates:
        if child.stem == stem:
            return child
    for child in candidates:
        if child.stem.casefold() == stem.casefold():
            return child
    return config.bundle_path(stem)


def diagnose(config: ConfigRoot) -> list[Finding]:
    """Inspect the config tree and return findings.

    :param config: The configuration root to inspect.
    :returns: A list of :class:`Finding` (empty if everything looks correct),
        in the order the probes ran, with exact repeats dropped by
        :func:`_without_repeats`.
    """
    findings: list[Finding] = []

    # Every probe below uses the os.path spelling, which answers an errno it
    # cannot act on with False, rather than the pathlib one, which re-raises
    # anything but a missing path. A directory this user cannot search is a
    # thing to report, not a reason to abort before reporting anything at all.
    #
    # The root's own probe keeps that spelling deliberately. Unlike the three
    # below, nothing else in the run would name an unstattable root: the early
    # return is there because no later check can say anything useful, so the
    # imprecise error is the only report there is, and it carries a fix.
    if not os.path.isdir(config.root):
        findings.append(
            Finding(
                "error",
                f"Config root does not exist: {config.root}",
                fix=f"Run 'stack config init' or create {config.root}.",
                kind="missing-root",
                path=config.root,
            )
        )
        return findings

    for name, directory in (
        ("profiles", config.profiles_dir),
        ("bundles", config.bundles_dir),
        ("envs", config.envs_dir),
    ):
        # _scaffold_directory_problem, not the isdir spelling: these three sit
        # under a root that may be listable without being searchable, and
        # calling them missing then contradicts the unparseable-source warns
        # the same run emits about the same paths -- with a mkdir that cannot
        # succeed. The probe says which of the three answers it got.
        problem = _scaffold_directory_problem(name, directory)
        if problem is not None:
            findings.append(problem)

    # Leftover pre-YAML config files (clean break: these are no longer read).
    # Walked rather than globbed: glob answers a directory it cannot read with
    # an empty match, which would report a clean bill of health for a tree
    # doctor never saw. The isdir guard stays because an absent directory is
    # already reported above as missing-dir.
    profile_children: list[Path] = []
    bundle_children: list[Path] = []
    if os.path.isdir(config.profiles_dir):
        profile_children, walk_findings = _children(config.profiles_dir)
        findings += walk_findings
        for legacy in profile_children:
            if legacy.suffix != ".in":
                continue
            findings.append(
                Finding(
                    "warn",
                    f"Legacy profile file: {legacy}",
                    fix=f"Convert it to {legacy.with_suffix('.yaml')} (YAML).",
                    kind="legacy-profile",
                    path=legacy,
                    dest=legacy.with_suffix(".yaml"),
                )
            )
    if os.path.isdir(config.bundles_dir):
        bundle_children, walk_findings = _children(config.bundles_dir)
        findings += walk_findings
        for legacy in bundle_children:
            if legacy.suffix != ".bundle":
                continue
            findings.append(
                Finding(
                    "warn",
                    f"Legacy bundle file: {legacy}",
                    fix=f"Convert it to {legacy.with_suffix('.yaml')} (YAML).",
                    kind="legacy-bundle",
                    path=legacy,
                    dest=legacy.with_suffix(".yaml"),
                )
            )

    # A name published under both kinds. The resolver already judges this worth
    # saying, but only once a stack names the token (see _warn_shadow); a root
    # carrying the collision is never told on its own account, which is the
    # gap.
    #
    # The profile side is read off the walk above rather than list_profiles,
    # which globs -- a directory doctor could not read would answer that it
    # holds nothing. The bundle side opens the stem instead of looking it up
    # among the walked names, because an intersection over those names answers
    # for the wrong filesystem: where case folds, 'ds' opens a 'bundles/DS.yaml'
    # that no string comparison matches, and doctor would stay silent about a
    # bundle _warn_shadow reports as shadowed. Whether this filesystem folds is
    # a question it is already answering.
    #
    # os.path.isfile rather than config.bundle_exists, which is the same stat
    # but through Path.is_file and so raises on a bundles/ that cannot be
    # searched -- doctor must not fail on a tree it was asked to diagnose. The
    # two agree wherever the stat succeeds; where it does not, this reports no
    # collision and the walk above has already reported the directory.
    #
    # '.yaml' on the profile side because that is what the resolver reads. A
    # legacy '.in' sharing a stem resolves nowhere, and the conversion that
    # would turn it into a real collision is already refused by the shadow
    # guard in _fix_convert_yaml.
    for profile in sorted(profile_children):
        if profile.suffix != ".yaml":
            continue
        if not os.path.isfile(config.bundle_path(profile.stem)):
            continue
        bundle = _bundle_for_stem(config, profile.stem, bundle_children)
        # warn, not error: both files stay reachable on the documented
        # precedence, so nothing here is broken, and the exit 1 an error earns
        # would fail doctor on a root using the '@name' escape the README
        # offers. The bundle is the finding's path because it is the shadowed
        # side, and so the one the reader has to act on.
        findings.append(
            Finding(
                "warn",
                f"Name collision: {profile} and {bundle} both answer to "
                f"'{profile.stem}'",
                fix=(
                    f"A bare '{profile.stem}' resolves to the profile; use "
                    f"'@{profile.stem}' for the bundle. Rename one of them to "
                    "remove the ambiguity."
                ),
                kind="name-collision",
                path=bundle,
            )
        )

    # Env-like directories left directly under root. Membership is decided
    # from the enumerated names rather than from exists() probes, for the
    # reason the two legacy scans above were rerouted: a directory this user
    # cannot search answers every probe inside it with False, so the marker
    # test reported nothing at all for a directory doctor never inspected --
    # while the same tree one chmod away yields a real misplaced-env.
    #
    # The price is that every unknown top-level directory is now enumerated,
    # so an unreadable one is reported even when it is not env-like at all.
    # That is the rule working: doctor could not read it, so it must not call
    # it clean.
    root_children, walk_findings = _children(config.root)
    findings += walk_findings
    for child in root_children:
        if not os.path.isdir(child) or _is_root_directory(child, config):
            continue
        child_entries, walk_findings = _children(child)
        findings += walk_findings
        # Two questions, because neither one answers for the other. The
        # enumerated names see an entry the filesystem will not resolve -- a
        # marker that is a dangling symlink -- and they see it in a directory
        # that can be read at all. The exists() probes see the filesystem's
        # own idea of the name: on a case-insensitive filesystem, APFS and the
        # macOS default, 'Requirements.in' IS the file config.py, render.py,
        # status.py and upgrade.py all open, so a name test alone reports
        # clean on a misplaced env the rest of the tool would use. Casefolding
        # the names instead of asking would be the mirror bug -- on a
        # case-sensitive filesystem that spelling is a different file uv-stack
        # never reads, and the mv fix would be wrong. Asking leaves each
        # platform's answer to that platform.
        names = {entry.name for entry in child_entries}
        if not names & _ENV_MARKERS and not any(
            os.path.exists(child / marker) for marker in _ENV_MARKERS
        ):
            continue
        findings.append(
            Finding(
                "warn",
                f"Env-like directory not under envs/: {child.name}",
                fix=f"Move it: mv {child} {config.envs_dir / child.name}",
                kind="misplaced-env",
                path=child,
                dest=config.envs_dir / child.name,
            )
        )

    # Per-env source-file checks.
    if os.path.isdir(config.envs_dir):
        env_children, walk_findings = _children(config.envs_dir)
        findings += walk_findings
        for env_dir in env_children:
            if not os.path.isdir(env_dir):
                continue
            has_profiles_txt = os.path.exists(env_dir / "profiles.txt")
            if has_profiles_txt:
                findings.append(
                    Finding(
                        "warn",
                        f"Legacy profiles.txt in env '{env_dir.name}'",
                        fix=f"Rename {env_dir / 'profiles.txt'} to stack.txt.",
                        kind="legacy-profiles-txt",
                        path=env_dir / "profiles.txt",
                        dest=env_dir / "stack.txt",
                    )
                )
            stack_txt = env_dir / "stack.txt"
            # lexists, not exists: a dangling symlink reads as absent to
            # exists(). list_envs keeps a child only when its stack.txt is a
            # regular file, so every shape caught here drops the env out of
            # every listing in the program; reporting it is the only way the
            # user learns why. Left as a report, not a repair -- doctor cannot
            # guess what the file was meant to contain.
            if os.path.lexists(stack_txt) and not os.path.isfile(stack_txt):
                findings.append(
                    Finding(
                        "error",
                        f"Env '{env_dir.name}' has a stack.txt that is not a regular file",
                        fix=f"Replace {stack_txt} with a regular file, or remove {env_dir}.",
                        kind="unusable-env",
                        path=stack_txt,
                    )
                )
            # An absent stack.txt is not a broken env on its own -- a
            # directory parked under envs/ holding notes is simply not one,
            # and saying otherwise reports a user's own file as a defect. A
            # marker is what distinguishes the two: it means a sync ran here,
            # so this is an env that list_envs drops while its compiled lock
            # sits beside it and 'stack env sync' answers that it does not
            # exist. Skipped when profiles.txt is there, because the legacy
            # finding above already names the remedy and this one would
            # contradict it -- create the file, against rename the one you
            # have. Report-only for unusable-env's reason: doctor cannot guess
            # what the file was meant to contain.
            elif not has_profiles_txt and any(
                os.path.lexists(env_dir / marker) for marker in _ENV_MARKERS
            ):
                findings.append(
                    Finding(
                        "error",
                        f"Env-like directory with no stack.txt: {env_dir.name}",
                        fix=f"Create {stack_txt}, or remove {env_dir}.",
                        kind="missing-stack-txt",
                        path=stack_txt,
                    )
                )
            python_txt = env_dir / "python.txt"
            # lexists, not isfile: a directory or a dangling symlink at this
            # path is emphatically present, and calling it missing offers a
            # repair -- create the file -- that cannot run, because
            # atomic_write_new opens O_EXCL and the entry is already there.
            # That contradiction is the one require_regular_file exists to
            # prevent, and load_env now raises it by name, which _scan_sources
            # turns into its own finding; so the shapes this test declines to
            # call missing are reported, not dropped.
            if os.path.isfile(stack_txt) and not os.path.lexists(python_txt):
                findings.append(
                    Finding(
                        "warn",
                        f"Env '{env_dir.name}' missing python.txt (will default to 3.12)",
                        fix=f"Create {python_txt}.",
                        kind="missing-python-txt",
                        path=python_txt,
                    )
                )

    if not probe_locking(config.probe_lock_path()):
        findings.append(
            Finding(
                "warn",
                f"Name locking is unavailable on {config.root}: concurrent "
                "'stack create' is not serialized.",
                fix=(
                    "Move the config root to a local filesystem, or avoid running "
                    "'stack create' concurrently against this root."
                ),
                kind="degraded-locks",
                path=config.locks_dir,
            )
        )

    findings.extend(_portability_findings(config))

    return _without_repeats(findings)


@dataclass
class RepairAction:
    """The outcome of attempting one finding's fix.

    :param finding: The finding that was addressed.
    :param description: Human description of what was (or would be) done.
    :param applied: Whether the fix was applied.
    :param reason: Why the fix was skipped, when it was.
    """

    finding: Finding
    description: str
    applied: bool
    reason: str | None = None


def _same_bytes(left: Path, right: Path) -> bool:
    """Report whether two paths hold identical bytes.

    Sizes first, then a chunked comparison that stops at the first difference.
    Deliberately not a timestamp comparison: the filesystems that reach the
    caller's copy branch are exactly the ones with coarse mtimes — FAT resolves
    to two seconds, exFAT to ten milliseconds — so an mtime test would miss the
    writes most likely to land in the millisecond-scale window it guards. Also
    deliberately not :func:`filecmp.cmp`, whose module-level cache is keyed on a
    stat signature containing mtime: the same weakness by another route.

    The comparison reads through guarded descriptors (``O_NOFOLLOW`` plus
    ``O_NONBLOCK``) so a symlink or FIFO planted after the caller's validation
    is refused rather than followed or hung on.

    :param left: First file.
    :param right: Second file.
    :returns: ``True`` when both hold the same bytes.
    :raises OSError: If the read guards are unavailable on this platform, if
        either path is not a regular file at open time, or if either file
        cannot be opened or read.
    """
    flags = nofollow_read_flags()
    if flags is None:
        raise OSError("cannot compare safely on this platform")
    lfd = os.open(left, flags)
    try:
        rfd = os.open(right, flags)
    except BaseException:
        os.close(lfd)
        raise
    try:
        lstat = os.fstat(lfd)
        rstat = os.fstat(rfd)
        if not (stat.S_ISREG(lstat.st_mode) and stat.S_ISREG(rstat.st_mode)):
            raise OSError("not a regular file")
        if lstat.st_size != rstat.st_size:
            return False
        with os.fdopen(lfd, "rb") as left_handle:
            lfd = -1  # ownership transferred to the file object
            with os.fdopen(rfd, "rb") as right_handle:
                rfd = -1  # ownership transferred to the file object
                while True:
                    left_chunk = left_handle.read(io.DEFAULT_BUFFER_SIZE)
                    right_chunk = right_handle.read(io.DEFAULT_BUFFER_SIZE)
                    if left_chunk != right_chunk:
                        return False
                    if not left_chunk:
                        return True
    finally:
        if lfd != -1:
            os.close(lfd)
        if rfd != -1:
            os.close(rfd)


def _finish_move(
    src: Path, dst: Path, moved_stat: os.stat_result, published: Published
) -> None:
    """Complete a move after publishing, checking both names before unlinking.

    Every unlink here names a path, not an inode — POSIX offers no
    unlink-by-inode — so each is preceded by an identity check that narrows,
    but cannot close, the window in which a concurrent writer could swap the
    file underneath us. Hence the success path's re-check of ``dst``: ``src``
    still naming the moved inode says nothing about what became of our link.

    Which rules apply is decided by ``published.kind`` and never by comparing
    stats. The two mechanisms differ in what they prove:

    - **link** — ``dst`` and ``src`` name one inode. Nothing published here is
      provably ours, because :func:`os.link` publishes whatever ``src`` named
      at link time, which may be a replacement. These rules are unchanged from
      before the copy path existed.
    - **copy** — ``dst`` is a *different* inode, made by an exclusive create,
      so it is provably ours and the rollback set narrows to it alone. But it
      is a snapshot, so identity is no longer sufficient to authorize removing
      ``src``: an in-place write leaves ``st_dev``/``st_ino`` untouched while
      the bytes diverge, and unlinking ``src`` would destroy them. The bytes
      are therefore compared before the unlink.

    The rollback withdraws ``dst`` on one condition and no other: ``dst``
    names an inode in the branch's rollback set — on the link path, the one we
    measured or the one ``src`` names now; on the copy path, the one the
    exclusive create made. Anything else is left alone — that link leaks, and
    only there is a deletion ruled out. Withdrawing at all is a choice, not a
    POSIX limit: ``dst`` is a name only we created, and leaving it would block
    every later move. On the link path the condition tests state, not
    provenance, so "rolled back" promises a clean destination and nothing more:

    - Neither the name at ``dst`` nor the inode it holds is provably ours. A
      pre-link replacement of ``src`` is what ``os.link`` publishes; a symlink
      planted after the caller's ``lstat`` publishes its target, which may be
      the moved inode; a stranger may re-link either inode there before we
      look; ``dst`` swapped after our ``lstat`` is withdrawn regardless.
      Illustrations, not an enumeration: any history ending in an ``ours``
      inode at ``dst`` takes the withdrawal.
    - Nor is another name for that inode guaranteed. The withdrawal may take
      its last, and the content is then gone: the file we moved, or a writer's
      file we could not tell from it.

    The paths that report success leave windows of their own:

    - ``dst`` swapped between its identity check and the unlink of ``src``
      that check guards, costing the moved inode what may be its last name.
    - ``src`` swapped between its check and its unlink, which removes the
      replacement rather than the file we moved.
    - On the copy path, a write landing between the byte comparison and the
      unlink is still lost. The comparison narrows that window to the interval
      between two adjacent statements; it does not close it, exactly as the
      identity checks above narrow rather than close theirs.
    - ``src`` already gone when we look — or gone by the time we unlink — with
      ``dst`` unlinked or replaced in the same interval. No name of ours is
      left to remove, so the function returns and the caller records the move
      as applied. That report can outrun the facts, and one ``dst.lstat()``
      before the return would narrow that window. It is deliberately not taken:
      a raise here reaches :func:`_fix_convert_yaml`, which answers a failed
      move by withdrawing the YAML it just published — and in this window that
      YAML is the last surviving copy, the source and its backup both having
      been removed by someone else. An inaccurate "applied" costs less than
      deleting the file we were asked to preserve.

    :param src: The source path that was published.
    :param dst: The destination path where it was published.
    :param moved_stat: The stat of the inode being moved, taken from ``src``
        BEFORE the publication. It must not come from ``dst`` afterwards: on
        the link path that records whatever ``dst`` names at that moment, which
        is our own link only if nobody intervened — the very thing this check
        must not assume.
    :param published: What :func:`~uv_stack.fsutil.link_or_copy_no_replace`
        reported. ``kind`` selects the branch; on the copy path ``stat`` is the
        identity ``dst`` was given, and is what the rollback set is built from.
    :raises OSError: If ``src`` or ``dst`` changed identity during the move, or
        — on the copy path — if the source's bytes changed under the copy.
        Every guarantee holds only as of the check that precedes the action it
        guards; the windows above say what each one costs past that point.
    """
    moved_ident = (moved_stat.st_dev, moved_stat.st_ino)
    if published.kind == "copy":
        _finish_copy_move(src, dst, moved_ident, published)
        return
    try:
        current = src.lstat()
    except FileNotFoundError:
        # src is already gone, so there is no name of ours left to remove and
        # nothing here can improve on whatever happened to dst.
        return
    if (current.st_dev, current.st_ino) == moved_ident:
        # src still names the moved inode, which says nothing about dst.
        # Confirm our link is the one published there before removing src's
        # name, so a dst that a third party unlinked or replaced aborts the
        # move with src intact rather than destroying the file.
        try:
            published_now = dst.lstat()
        except FileNotFoundError:
            raise OSError(f"{dst} vanished during move; nothing deleted") from None
        if (published_now.st_dev, published_now.st_ino) != moved_ident:
            raise OSError(f"{dst} changed during move; nothing deleted")
        # missing_ok: a third party can remove src between the check above and
        # this line, and dst already holds the moved inode by then. Raising
        # would report a completed move as failed, and _fix_convert_yaml
        # answers a failed move by withdrawing the YAML it just published.
        src.unlink(missing_ok=True)
        return
    # src no longer names the inode we measured, so something replaced it —
    # after we linked, or before, in which case os.link published whatever it
    # found there. Withdraw dst either way, but only while it names an inode
    # os.link could have published for us: the one we set out to move, or the
    # one src names now. Any other inode at dst is a concurrent writer's file,
    # and a bare "differs from moved_stat" test cannot tell that case apart
    # from ours — so it must not be the condition. See the docstring for what
    # this condition does not establish, and the windows it cannot close.
    ours = {moved_ident, (current.st_dev, current.st_ino)}
    withdrew = False
    try:
        dst_now = dst.lstat()
        if (dst_now.st_dev, dst_now.st_ino) in ours:
            dst.unlink(missing_ok=True)
            withdrew = True
    except FileNotFoundError:
        pass
    # Report the withdrawal that happened rather than a rule about when one
    # does: this branch leaves dst deleted, absent, or holding a third party's
    # file, and in the convert case the deleted one was the moved inode's last
    # name. A caller told "nothing deleted" there would be told wrong.
    outcome = f"{dst} withdrawn" if withdrew else "nothing deleted"
    raise OSError(f"{src} changed during move; {outcome}")


def _finish_copy_move(
    src: Path, dst: Path, moved_ident: tuple[int, int], published: Published
) -> None:
    """The copy branch of :func:`_finish_move`; see that docstring for the rules.

    Split out because its conditions differ from the link branch's in kind, not
    only in detail, and interleaving them in one body would make each harder to
    read than either is alone.

    :param src: The source path that was copied.
    :param dst: The destination holding the copy.
    :param moved_ident: ``(st_dev, st_ino)`` of the inode measured before the copy.
    :param published: The copy's own report; ``published.stat`` identifies the
        inode the exclusive create made.
    :raises OSError: If ``src`` changed identity, content, or mode under the
        copy, or if ``dst`` vanished or was replaced.
    """
    published_ident = (published.stat.st_dev, published.stat.st_ino)
    try:
        current = src.lstat()
    except FileNotFoundError:
        # Same reasoning as the link branch: no name of ours is left to remove,
        # and nothing here can improve on whatever became of dst.
        return
    if (current.st_dev, current.st_ino) == moved_ident:
        try:
            dst_now = dst.lstat()
        except FileNotFoundError:
            raise OSError(f"{dst} vanished during move; nothing deleted") from None
        if (dst_now.st_dev, dst_now.st_ino) != published_ident:
            raise OSError(f"{dst} changed during move; nothing deleted")
        try:
            identical = _same_bytes(src, dst)
        except OSError:
            # One of the two names became unreadable mid-comparison — a vanish,
            # a swap, a mode change. Not provably safe to unlink src, so take
            # the withdrawal path rather than guess.
            identical = False
        if identical:
            # Re-stat BOTH names after the comparison, not before it: a chmod
            # or a replacement landing while we read would otherwise be judged
            # against state captured before the read began. The window between
            # these lstats and the unlink cannot be closed without
            # unlinkat-with-identity, which Python does not expose portably.
            try:
                final = src.lstat()
            except FileNotFoundError:
                # src vanished: no name of ours is left to remove, and dst
                # holds the content.
                return
            try:
                dst_final = dst.lstat()
            except FileNotFoundError:
                # dst vanished: src still exists and the destination is gone, so
                # the move did not complete. Fall through to the withdrawal
                # block, which will find dst absent and raise.
                pass
            else:
                if (
                    (final.st_dev, final.st_ino) == moved_ident
                    and stat.S_IMODE(final.st_mode) == stat.S_IMODE(published.stat.st_mode)
                    and (dst_final.st_dev, dst_final.st_ino) == published_ident
                ):
                    src.unlink(missing_ok=True)
                    return
    # src was replaced, or its bytes or mode changed under the copy, or dst was
    # replaced after the comparison. Either way dst is a stale snapshot. Withdraw
    # it only while it still names the inode the exclusive create made: that
    # inode is provably ours, and unlike the link branch there is no second
    # candidate — we never published src's own inode, so whatever src names now
    # cannot be at dst by our doing.
    withdrew = False
    try:
        dst_now = dst.lstat()
        if (dst_now.st_dev, dst_now.st_ino) == published_ident:
            dst.unlink(missing_ok=True)
            withdrew = True
    except FileNotFoundError:
        pass
    outcome = f"{dst} withdrawn" if withdrew else "nothing deleted"
    # Name both: this arm is reached from a source-side or a destination-side
    # mismatch, and the reason string reaches the user as the whole diagnostic.
    raise OSError(f"{src} or {dst} changed during move; {outcome}")


def _move_no_replace(src: Path, dst: Path) -> None:
    """Move ``src`` to ``dst``, refusing to replace ``dst`` or a changed ``src``.

    Publishes via :func:`~uv_stack.fsutil.link_or_copy_no_replace` (fails if
    ``dst`` exists), then removes the source only while it still names the
    moved inode *and* ``dst`` still holds what we published — and, where the
    publication was a copy, only while the two still hold the same bytes. A
    source replaced mid-move is left untouched, a destination taken by someone
    else aborts the move with the source intact, and what was published at
    ``dst`` is withdrawn only under :func:`_finish_move`'s identity rules. On
    the link path that test cannot prove the link is ours; see
    :func:`_finish_move` for what it lets through.

    A symlink at ``src`` is refused, but only one that is there when we look:
    :func:`os.link` follows symlinks by default, so it would publish a link to
    the TARGET while ``src.lstat()`` describes the symlink. One planted after
    that check slips past and its target is published anyway, which silently
    relocates a third party's file — or destroys it, when the target is the
    moved inode and the rollback withdraws that inode's last name.

    :raises FileNotFoundError: If ``src`` does not exist when the move begins,
        or is removed in the window between that check and the publication.
    :raises FileExistsError: If ``dst`` already exists.
    :raises OSError: If ``src`` is not a regular file, if ``src`` or ``dst``
        changed identity during the move, or if ``src``'s bytes or mode changed
        under a copy publication. ``src`` is left in place on those paths, and
        anything published at ``dst`` is withdrawn only under
        :func:`_finish_move`'s rules — which, on the link path when ``src`` was
        replaced during the move, can withdraw the last name of the moved inode
        or of the replacement ``os.link`` published in its place. See that
        function's docstring for the residual windows those rules narrow but
        cannot close.
    """
    moved = src.lstat()
    if not stat.S_ISREG(moved.st_mode):
        raise OSError(f"{src} is not a regular file; nothing moved")
    published = link_or_copy_no_replace(src, dst, fallback_errnos=_PUBLIC_LINK_FALLBACK)
    _finish_move(src, dst, moved, published)


def _fix_mkdir(config: ConfigRoot, finding: Finding) -> RepairAction:
    assert finding.path is not None
    finding.path.mkdir(parents=True, exist_ok=True)
    return RepairAction(finding, f"created {finding.path}", applied=True)


def _fix_rename(config: ConfigRoot, finding: Finding) -> RepairAction:
    assert finding.path is not None and finding.dest is not None
    description = f"move {finding.path} to {finding.dest}"
    # Revalidate source exists.
    if not finding.path.exists():
        return RepairAction(
            finding, description, applied=False,
            reason=f"{finding.path.name} no longer exists",
        )
    if finding.dest.exists():
        return RepairAction(
            finding, description, applied=False,
            reason=f"{finding.dest.name} already exists",
        )
    finding.dest.parent.mkdir(parents=True, exist_ok=True)
    # For files: use no-replace move; for directories: reserve the destination
    # first to prevent replacing an empty directory created concurrently.
    if finding.path.is_file():
        try:
            _move_no_replace(finding.path, finding.dest)
        except FileExistsError:
            return RepairAction(
                finding, description, applied=False,
                reason=f"{finding.dest.name} already exists",
            )
        except OSError as error:
            return RepairAction(
                finding, description, applied=False, reason=str(error)
            )
    else:
        try:
            finding.dest.mkdir(exist_ok=False)
        except FileExistsError:
            return RepairAction(
                finding, description, applied=False,
                reason=f"{finding.dest.name} already exists",
            )
        try:
            os.rename(finding.path, finding.dest)
        except OSError as error:
            # Best-effort cleanup of the placeholder we created.
            try:
                finding.dest.rmdir()
            except OSError:
                pass
            return RepairAction(
                finding, description, applied=False, reason=str(error)
            )
    return RepairAction(
        finding, f"moved {finding.path} to {finding.dest}", applied=True
    )


def _fix_python_txt(config: ConfigRoot, finding: Finding) -> RepairAction:
    assert finding.path is not None
    description = f"write {finding.path} with default 3.12"
    try:
        atomic_write_new(finding.path, "3.12\n")
    except FileExistsError:
        return RepairAction(
            finding, description, applied=False,
            reason="python.txt already exists",
        )
    return RepairAction(finding, f"wrote {finding.path} with default 3.12", applied=True)


def _fix_convert_yaml(config: ConfigRoot, finding: Finding) -> RepairAction:
    assert finding.path is not None and finding.dest is not None
    description = f"convert {finding.path} to {finding.dest}"
    # A conversion publishes profiles/<stem>.yaml or bundles/<stem>.yaml, which
    # is a create in the same shared namespace scaffold's writers serialize on.
    # Their post-check cannot cover this one: it runs before doctor publishes,
    # so a bundle create that starts inside doctor's window commits, and doctor
    # then commits the profile on top of it — leaving both kinds of the stem on
    # disk with diagnose reporting nothing wrong. The collision spans two paths,
    # so no single no-clobber write can reserve it; only the shared lock can.
    stem = finding.path.stem
    try:
        with name_lock(config.stem_lock_path(stem), stem):
            return _convert_under_stem_lock(config, finding, description, stem)
    except ConfigError as error:
        # Either the lock — contended past its timeout, or standing on
        # something name_lock refuses — or the source read inside, which
        # refuses bytes that are not UTF-8. Both describe themselves, so this
        # skips the one finding and says why. Letting it out would abort the
        # whole pass — repair() catches only OSError — and cost every later
        # finding its fix over one unavailable stem. The hint
        # carries the actionable half of every one of these — which process to
        # look for, what not to delete — and the skip line is the only place
        # the user ever sees this error, so fold it in rather than drop it.
        reason = error.message if error.hint is None else f"{error.message}. {error.hint}"
        return RepairAction(finding, description, applied=False, reason=reason)


def _convert_under_stem_lock(
    config: ConfigRoot, finding: Finding, description: str, stem: str
) -> RepairAction:
    """Convert one legacy file to YAML, with the stem's create lock already held.

    :param config: The configuration root being repaired.
    :param finding: The ``legacy-profile`` or ``legacy-bundle`` finding.
    :param description: The unapplied-action wording for a skip.
    :param stem: The profile/bundle name both kinds share.
    :returns: The action taken, applied or skipped with a reason.
    """
    assert finding.path is not None and finding.dest is not None
    # Lstat the source for identity binding.
    try:
        src_identity_before = finding.path.lstat()
    except FileNotFoundError:
        return RepairAction(
            finding, description, applied=False,
            reason=f"{finding.path.name} no longer exists",
        )
    if finding.dest.exists():
        return RepairAction(
            finding, description, applied=False,
            reason=f"{finding.dest.name} already exists",
        )
    # Cross-kind shadow guard: skip if the opposite kind exists for the stem.
    if finding.kind == "legacy-profile":
        if config.bundle_exists(stem):
            return RepairAction(
                finding, description, applied=False,
                reason=f"profile '{stem}' would shadow the existing bundle",
            )
    elif finding.kind == "legacy-bundle":
        if config.profile_exists(stem):
            return RepairAction(
                finding, description, applied=False,
                reason=f"bundle '{stem}' would be shadowed by the existing profile",
            )

    # Path.rename would silently replace an existing backup on POSIX; a
    # repair pass must never destroy user content, so skip instead.
    backup = finding.path.with_name(finding.path.name + ".bak")
    if backup.exists():
        return RepairAction(
            finding, description, applied=False,
            reason=f"{finding.path.name}.bak already exists",
        )
    includes = read_clean_lines(finding.path)
    # A conversion is a durable writer of human-typed entries, so it owes the
    # same placement rule init, create, and edit enforce on their own writes.
    # Without it the pass reports a fix and leaves a generated YAML that every
    # later command refuses -- naming that file, while the legacy one the user
    # actually wrote sits renamed to '.bak'. Judged with placement_problem
    # rather than check_placement: a repair pass must not abort because the
    # thing it diagnoses is broken, so a refused entry skips this one finding.
    refused = [
        (entry, problem[1])
        for entry in includes
        if (problem := placement_problem(entry)) is not None
    ]
    if refused:
        entry, explanation = refused[0]
        noun = "entry" if len(refused) == 1 else "entries"
        return RepairAction(
            finding, description, applied=False,
            reason=(
                f"{finding.path.name} holds {len(refused)} {noun} uv-stack will "
                f"not write, starting with {entry!r}: {explanation}. Fix them "
                "there, then re-run"
            ),
        )
    # Publish the YAML with atomic_write_new; capture stat for identity check.
    try:
        dest_stat = atomic_write_new(
            finding.dest,
            yaml.safe_dump(
                {"includes": includes}, sort_keys=False, default_flow_style=False
            ),
        )
    except FileExistsError:
        return RepairAction(
            finding, description, applied=False,
            reason=f"{finding.dest.name} already exists",
        )
    # Lstat source again immediately before move to detect replacement.
    try:
        src_identity_after = finding.path.lstat()
    except FileNotFoundError:
        # Source vanished after read but before move.
        try:
            current_stat = finding.dest.lstat()
            if (current_stat.st_dev, current_stat.st_ino) == (
                dest_stat.st_dev,
                dest_stat.st_ino,
            ):
                finding.dest.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        return RepairAction(
            finding, description, applied=False,
            reason=f"{finding.path.name} vanished during conversion",
        )
    if (src_identity_before.st_dev, src_identity_before.st_ino) != (
        src_identity_after.st_dev,
        src_identity_after.st_ino
    ):
        # Source replaced after read: withdraw the YAML.
        try:
            current_stat = finding.dest.lstat()
            if (current_stat.st_dev, current_stat.st_ino) == (
                dest_stat.st_dev,
                dest_stat.st_ino,
            ):
                finding.dest.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
        return RepairAction(
            finding, description, applied=False,
            reason=f"{finding.path.name} changed during conversion",
        )
    # Move the source to backup with no-replace semantics.
    try:
        _move_no_replace(finding.path, backup)
    except OSError as error:
        # Backup appeared after our check OR the source vanished: remove the
        # just-published YAML only if it is still the file we published.
        skip_reason: str
        try:
            current_stat = finding.dest.lstat()
            if (current_stat.st_dev, current_stat.st_ino) == (
                dest_stat.st_dev,
                dest_stat.st_ino,
            ):
                finding.dest.unlink(missing_ok=True)
            skip_reason = (
                f"{finding.path.name}.bak already exists"
                if isinstance(error, FileExistsError)
                else str(error)
            )
        except FileNotFoundError:
            skip_reason = (
                f"{finding.path.name}.bak already exists"
                if isinstance(error, FileExistsError)
                else str(error)
            )
        return RepairAction(finding, description, applied=False, reason=skip_reason)
    return RepairAction(
        finding,
        f"converted {finding.path.name} to {finding.dest.name} "
        f"(original saved as {backup.name})",
        applied=True,
    )


_REPAIRS = {
    "missing-root": _fix_mkdir,
    "missing-dir": _fix_mkdir,
    "legacy-profile": _fix_convert_yaml,
    "legacy-bundle": _fix_convert_yaml,
    "misplaced-env": _fix_rename,
    "legacy-profiles-txt": _fix_rename,
    "missing-python-txt": _fix_python_txt,
}


def repair(config: ConfigRoot, findings: list[Finding]) -> list[RepairAction]:
    """Apply the safe fix for each finding that has one.

    Findings without a registered handler are ignored. Conversions preserve
    the original as ``*.bak``; renames skip when the destination exists.
    :func:`_move_no_replace` can withdraw the last name of the moved inode.

    :param config: The configuration root being repaired.
    :param findings: Findings from :func:`diagnose`.
    :returns: One :class:`RepairAction` per handled finding, in order.
    """
    actions: list[RepairAction] = []
    for finding in findings:
        handler = _REPAIRS.get(finding.kind)
        if handler is None:
            continue
        try:
            actions.append(handler(config, finding))
        except OSError as error:
            actions.append(
                RepairAction(
                    finding,
                    finding.fix or finding.message,
                    applied=False,
                    reason=str(error),
                )
            )
    return actions


def _scan_sources(config: ConfigRoot) -> tuple[list[tuple[Path, str]], list[Finding]]:
    """Collect every requirement entry the root declares, without resolving.

    A reference can appear in three kinds of file, not one: a profile's
    ``includes``, a bundle's ``includes`` entry that is a literal rather than a
    profile or bundle name, and a literal line in an env's ``stack.txt``. The
    resolver funnels the last two into the same inline-requirement path as the
    first, so scanning only profiles would leave two silent routes.

    Nothing is resolved. A root with a dangling ``profile:`` token must still
    get a useful variable diagnosis, and doctor must not fail because the thing
    it is diagnosing is broken. A source that cannot be loaded becomes its own
    finding and is then skipped — its absence from the reference results is not
    a clean bill of health.

    Enumeration is guarded as well as reading. ``list_envs`` walks the
    directory with ``iterdir``, which raises rather than yielding nothing when
    the directory cannot be read; the profile and bundle listings go through
    ``glob``, which swallows a permission error today but is not contracted to.
    One guard per listing costs nothing and removes the question.

    :param config: The configuration root to scan.
    :returns: ``(entries, findings)``, where each entry pairs its holding file
        with one unexpanded requirement string.
    """
    entries: list[tuple[Path, str]] = []
    findings: list[Finding] = []

    try:
        profiles = config.list_profiles()
    except OSError as error:
        profiles = []
        findings.append(_unparseable(config.profiles_dir, str(error)))
    for name in profiles:
        path = config.profile_path(name)
        try:
            includes = config.load_profile(name).includes
        except (UvStackError, OSError) as error:
            findings.append(_unparseable(path, str(error)))
            continue
        entries.extend((path, entry) for entry in includes)

    try:
        bundles = config.list_bundles()
    except OSError as error:
        bundles = []
        findings.append(_unparseable(config.bundles_dir, str(error)))
    for name in bundles:
        path = config.bundle_path(name)
        try:
            includes = config.load_bundle(name).includes
        except (UvStackError, OSError) as error:
            findings.append(_unparseable(path, str(error)))
            continue
        entries.extend((path, entry) for entry in includes)

    try:
        envs = config.list_envs()
    except OSError as error:
        envs = []
        findings.append(_unparseable(config.envs_dir, str(error)))
    for name in envs:
        path = config.env_stack_path(name)
        try:
            stack = config.load_env(name).stack
        except (UvStackError, OSError) as error:
            findings.append(_unparseable(path, str(error)))
            continue
        entries.extend((path, entry) for entry in stack)

    return entries, findings


def _placement_findings(entries: list[tuple[Path, str]]) -> list[Finding]:
    """Report entries the placement rule refuses, one finding per refusal kind.

    :param entries: Scanned ``(source, entry)`` pairs.
    :returns: ``error`` findings, empty when every entry is admitted.
    """
    findings: list[Finding] = []
    for source, entry in entries:
        problem = placement_problem(entry)
        if problem is None:
            continue
        kind, explanation = problem
        findings.append(
            Finding(
                "error",
                f"{source}: entry '{entry}' — {explanation}",
                fix=_PLACEMENT_FIX[kind],
                kind=kind,
                path=source,
            )
        )
    return findings


def _reference_findings(
    config: ConfigRoot, entries: list[tuple[Path, str]], variables: Variables
) -> list[Finding]:
    """Report undeclared references and declared names with no value here.

    Declaration is what makes a reference part of the root's portable
    contract, so a name the environment happens to define is still undeclared.

    :param config: The configuration root, for the two file paths named in fixes.
    :param entries: Scanned ``(source, entry)`` pairs.
    :param variables: The root's declared names and this machine's values.
    :returns: ``error`` findings, empty when the root is fully configured.
    """
    findings = [
        Finding(
            "error",
            f"Variable '{name}' is declared in {config.variables_path()} but "
            "has no value on this machine.",
            fix=(
                f"Add '{name}=<value>' to {config.variables_local_path()}, "
                f"or export {name}."
            ),
            kind="undefined-variable",
            path=config.variables_local_path(),
        )
        for name in variables.undefined()
    ]
    declared = set(variables.declared)
    for source, entry in entries:
        for name in referenced_names(entry):
            if name in declared:
                continue
            findings.append(
                Finding(
                    "error",
                    f"{source}: entry '{entry}' references '{name}', which "
                    f"{config.variables_path()} does not declare.",
                    fix=(
                        f"Add '{name}' to {config.variables_path()}, or remove "
                        "the reference."
                    ),
                    kind="undeclared-variable",
                    path=source,
                )
            )
    return findings


def _editable_target(entry: str) -> str | None:
    """The local path an editable entry installs from, if it has one.

    An entry counts as a local editable when its first whitespace-separated
    token is ``-e`` or ``--editable``, alone or with the operand attached by
    ``=``, and the value carries no URL scheme. Anything else is a remote
    install with no path to check.

    The operand is extracted verbatim from the original entry to preserve
    interior whitespace exactly as written — uv reads ``-e ./my  pkg`` as the
    single path ``my  pkg`` with two spaces, and ``split()`` would discard the
    run length. In both the attached (``-e=PATH``) and separated (``-e PATH``)
    forms, the operand runs to the end of the entry, except that it stops at
    the first whitespace run preceding a token that begins with ``-`` or ``#``.
    A requirements file treats ``#`` as a comment marker at the start of a line
    or after whitespace, so ``-e ./pkg # note`` installs from ``./pkg``, while
    a ``#`` inside a token is ordinary text and ``-e ./pkg#1`` installs from
    ``./pkg#1``.

    A trailing PEP 508 extras suffix is dropped from the operand.

    :param entry: One expanded requirement entry.
    :returns: The path operand, or ``None``.
    """
    stripped = entry.strip()
    if not stripped:
        return None
    parts = stripped.split()
    # uv is what consumes these entries, so its parser sets the boundary: it
    # accepts '-e=PATH' and '--editable=PATH' as readily as the separated
    # forms, but refuses '-ePATH' with "Expected '=' or whitespace". Reading a
    # path out of the glued form would report a missing checkout for an entry
    # that cannot install for an entirely different reason.
    flag, attached, operand_start = parts[0].partition("=")
    if flag not in ("-e", "--editable"):
        return None
    # An attached '=' with nothing after it is still a separator to uv, which
    # reads '-e= PATH' exactly as '-e PATH'. Treating the empty operand as the
    # value would drop a checkout doctor is supposed to be watching.
    if attached and operand_start:
        # The attached form: everything after the '=' in the original entry.
        # The operand_start from partition is only what sat in the first
        # token, so we slice the stripped entry to get the whole remainder.
        remainder = stripped[len(flag) + 1 :]
    elif len(parts) >= 2:
        # The separated form: everything after the flag and its trailing
        # whitespace. We slice from the original entry rather than rejoining
        # split() to preserve interior whitespace exactly as written.
        flag_text = parts[0]
        remainder = stripped[len(flag_text) :].lstrip()
        # When the remainder begins with a comment, the operand is empty. The
        # lstrip() ensures a leading '#' here genuinely followed whitespace, so
        # it is a comment marker per the requirements file line semantics rather
        # than ordinary text inside a token. In '-e=#note' the '#' is glued to
        # the '=' with no space, so the attached branch keeps it as a path. uv
        # sees a bare '-e' when the separated operand is only a comment, which
        # is a malformed entry, not a checkout.
        if remainder.startswith("#"):
            return None
    else:
        return None
    # uv reads everything after the flag as one path, but '-e ./my pkg --opt'
    # stops the path at the option, and '-e ./pkg # note' stops the path at the
    # comment. We scan the remainder for the first whitespace run followed by
    # a token starting with '-' or '#', and cut there. A requirements file
    # treats '#' as a comment marker at the start of a line or after whitespace,
    # not inside a token, so './pkg#1' keeps its '#'. An option or comment
    # cannot be part of a path uv would accept here, and a path that genuinely
    # begins with '-' is the first token, so this rule applies only to later
    # tokens.
    target = remainder
    for index, char in enumerate(remainder):
        if char.isspace():
            after = index
            while after < len(remainder) and remainder[after].isspace():
                after += 1
            # The cut lands before the whitespace run rather than before the
            # '-' or '#': the run separates the two tokens and belongs to
            # neither, so keeping it would leave the path with a trailing space.
            if after < len(remainder) and remainder[after] in ("-", "#"):
                target = remainder[:index]
                break
    if "://" in target or target.startswith("git+"):
        return None
    # pip reads '-e ./pkg[dev]' as the path './pkg' carrying extras, so probing
    # the whole token would report a checkout that is present as missing. The
    # suffix is stripped rather than parsed: the operand is a path here, and
    # nothing downstream has any use for the extras names.
    if target.endswith("]") and "[" in target:
        target = target[: target.rindex("[")]
    return target or None


def _expanded_entry_findings(
    config: ConfigRoot, entries: list[tuple[Path, str]], variables: Variables
) -> list[Finding]:
    """Report what expanding each scanned entry reveals.

    Two findings come out of the one substitution. An expansion the safety
    check refuses is an ``error``, and this is the only place doctor can say
    so: placement is judged before substitution and the reference checks are
    already satisfied, so nothing above sees it. An expanded editable whose
    checkout is absent is a ``warn``.

    The refusal quotes ``expand_all``'s own message with its whitespace
    collapsed. That function formats a multi-line block because it reports
    every refused entry at once, and a finding is one line; quoting sync's
    words rather than paraphrasing them keeps the two diagnoses from drifting.

    Missing checkouts are derived, not declared: there is no checkout registry
    and no cloning. This reports which checkouts are missing, not where to
    fetch them.

    The whole entry is expanded rather than the bare target, because a bare
    ``${DEV}`` is not an admitted entry on its own — only the entry it sits in
    is. A relative path resolves against the config root, matching how uv reads
    the generated ``requirements.in``.

    :param config: The configuration root, for resolving relative paths.
    :param entries: Scanned ``(source, entry)`` pairs.
    :param variables: The values to expand with.
    :returns: An ``error`` finding per refused expansion and a ``warn`` finding
        per missing checkout, in scan order.
    """
    findings: list[Finding] = []
    for source, entry in entries:
        try:
            expanded = expand_all([entry], variables)[0]
        except UvStackError as error:
            # Reachable despite the caller's guard, and by exactly one route: a
            # declared name with a value here whose substitution would rewrite
            # the entry. Dropping it would leave a root that 'stack sync'
            # hard-fails on looking clean to 'stack doctor' — the one command
            # whose job is to say otherwise before sync does.
            findings.append(
                Finding(
                    "error",
                    f"{source}: {' '.join(str(error).split())}",
                    fix=(
                        "Change the value so it fills a path or an option "
                        "value rather than introducing requirements-file "
                        "syntax of its own."
                    ),
                    kind="unsafe-expansion",
                    path=source,
                )
            )
            continue
        target = _editable_target(expanded)
        if target is None:
            continue
        # os.path.expanduser, not Path.expanduser: the pathlib spelling raises
        # RuntimeError on a '~user' it cannot resolve, and RuntimeError is
        # neither UvStackError nor OSError, so it would leave 'stack doctor' as
        # a traceback on a profile it is supposed to diagnose.
        #
        # The os.path spelling is not total either. It resolves the '~user'
        # form through pwd.getpwnam, which rejects an embedded NUL with
        # ValueError -- again neither family the CLI turns into a message. Both
        # failures want the same answer, so the except supplies by hand what
        # expanduser supplies for an unresolvable name: the text unchanged,
        # resolved against the root and reported as a checkout that does not
        # exist — which is the true answer, not a consolation prize.
        try:
            path = Path(os.path.expanduser(target))
        except ValueError:
            path = Path(target)
        if not path.is_absolute():
            path = config.root / path
        # os.path.exists for the same family of reason: the pathlib probe
        # re-raises every errno but a handful, and this path is built from a
        # variable value, so an ancestor this user cannot search is ordinary
        # input rather than a pathology.
        if os.path.exists(path):
            continue
        findings.append(
            Finding(
                "warn",
                f"{source}: editable checkout does not exist: {path}",
                fix=f"Clone or create {path}, or point the variable elsewhere.",
                kind="missing-checkout",
                path=path,
            )
        )
    return findings


def _project_python_findings(config: ConfigRoot) -> list[Finding]:
    """Report a ``project-python.txt`` value that works here but will not travel.

    Both findings are ``warn``, not ``error``: either selector is valid and
    works perfectly on the machine that wrote it.

    The guard wraps the whole body, not just the file read.
    :func:`~uv_stack.operations.project.python_travel_problem` decides
    ``undeclared-env`` by testing the spec against ``config.list_envs()``, and
    the fix line joins the same listing — two directory walks that read at the
    call site like pure computation. An unreadable ``envs/`` escapes from
    either one.

    :param config: The configuration root.
    :returns: At most one finding.
    """
    path = config.project_python_path()
    try:
        spec = config.default_project_python()
        if spec is None:
            return []
        problem = python_travel_problem(config, spec)
        if problem == "path":
            return [
                Finding(
                    "warn",
                    f"{path} holds an interpreter path: {spec}",
                    fix=(
                        "Use a Python version ('3.12'), a uv implementation form "
                        "('cpython@3.12'), or an environment name this root "
                        "declares."
                    ),
                    kind="project-python-path",
                    path=path,
                )
            ]
        if problem == "undeclared-env":
            declared = ", ".join(config.list_envs()) or "none"
            return [
                Finding(
                    "warn",
                    f"{path} names environment '{spec}', which this root does "
                    "not declare.",
                    fix=(
                        f"Declared environments: {declared}. Create it with "
                        f"'stack create env {spec} ...', or use a Python version."
                    ),
                    kind="project-python-undeclared-env",
                    path=path,
                )
            ]
    except (UvStackError, OSError) as error:
        return [_unparseable(path, str(error))]
    return []


def _ignore_block_findings(config: ConfigRoot) -> list[Finding]:
    """Report a missing or stale managed ignore block in a root git tracks.

    Staleness is decided by the writer itself, under ``dry_run``, so doctor's
    notion of stale can never drift from what ``stack config portable`` would
    actually write.

    :param config: The configuration root.
    :returns: At most one finding; empty when the root is inside no repository.
    """
    # The writer's own predicate, so the two can never disagree about which
    # roots the block is for. A root nested in a working tree is one of them:
    # its files are tracked by the enclosing repository, so a missing or stale
    # block there leaves another machine's generated files committed, and
    # reporting clean on that is the one thing this scan may not do. The walk
    # probes with os.path.exists, which is why it can sit above the guard
    # below rather than inside it.
    if enclosing_repository(config.root) is None:
        return []
    try:
        result = write_portable_ignore(config, dry_run=True)
    except (UvStackError, OSError) as error:
        # The writer raises ConfigError for a non-UTF-8 or non-regular
        # .gitignore, and OSError for one it cannot open at all. Both mean the
        # same thing here: the block's state is unknown, so say so.
        return [_unparseable(config.root / ".gitignore", str(error))]
    if result.outcome == "unchanged":
        return []
    state = "missing" if result.outcome == "created" else "out of date"
    return [
        Finding(
            "warn",
            f"The managed .gitignore block in {config.root} is {state}.",
            fix=(
                "Run 'stack config portable' to write it; it prints the "
                "untracking and commit steps to follow."
            ),
            kind="stale-ignore-block",
            path=result.path,
        )
    ]


def _variables_blame(config: ConfigRoot, error: UvStackError | OSError) -> Path:
    """Name the variables file a failed load was actually about.

    ``load_variables`` reads two files and consults the environment, so the
    call doctor made identifies none of them. Built from the declaration path
    unconditionally, the finding told a reader to fix ``variables.txt`` when
    the fault was in ``variables.local.txt`` -- and pointed a ``--json``
    consumer at the wrong one, with the message naming the real file only in
    its tail.

    :param config: The configuration root, for the fallback.
    :param error: Whatever escaped the load.
    :returns: The file the raiser blamed, the file the kernel named, or the
        declaration file. The last is the fallback rather than nothing because
        an environment override belongs to no file at all, and the declaration
        file is what admitted the name; the message says which variable it was.
    """
    if isinstance(error, ConfigError) and error.path is not None:
        return error.path
    if isinstance(error, OSError) and isinstance(error.filename, str):
        return Path(error.filename)
    return config.variables_path()


def _portability_findings(config: ConfigRoot) -> list[Finding]:
    """Every portability check, ordered so a broken root still reports usefully.

    :param config: The configuration root to inspect.
    :returns: The findings, in reporting order.
    """
    findings = _ignore_block_findings(config)
    findings.extend(_project_python_findings(config))

    entries, scan_findings = _scan_sources(config)
    findings.extend(scan_findings)
    placement = _placement_findings(entries)
    findings.extend(placement)

    try:
        variables = config.load_variables()
    except (UvStackError, OSError) as error:
        findings.append(_unparseable(_variables_blame(config, error), str(error)))
        return findings

    references = _reference_findings(config, entries, variables)
    findings.extend(references)
    if not placement and not references:
        # Expanding on top of a known-bad reference set produces noise, not
        # information: the findings above already name every cause.
        findings.extend(_expanded_entry_findings(config, entries, variables))
    return findings
