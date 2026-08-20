"""Detect-only diagnostics for a uv-stack config tree.

``diagnose`` never mutates the filesystem; it returns a list of findings the CLI
prints with suggested fixes. It flags missing directories, legacy names
(``*.in``, ``*.bundle``, ``profiles.txt``), env-like directories left at the root, and
envs missing their source files.
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
from uv_stack.errors import ConfigError
from uv_stack.fsutil import (
    _LINK_FALLBACK_ERRNOS,
    Published,
    atomic_write_new,
    link_or_copy_no_replace,
    name_lock,
    nofollow_read_flags,
)
from uv_stack.parse import read_clean_lines

_KNOWN_TOP_LEVEL = {"profiles", "bundles", "envs", "lib", ".locks"}

# Public-source fallback: narrower than the temp-file set, because on Linux
# with fs.protected_hardlinks=1, EPERM from os.link means "policy denies
# hardlinking a file you do not own," not "this filesystem has no hard links."
# Treating it as link-less turns a denial into copy-then-unlink. A caller
# publishing its own temp file wants the full set.
_PUBLIC_LINK_FALLBACK = _LINK_FALLBACK_ERRNOS - {errno.EPERM}


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


def diagnose(config: ConfigRoot) -> list[Finding]:
    """Inspect the config tree and return findings.

    :param config: The configuration root to inspect.
    :returns: A list of :class:`Finding` (empty if everything looks correct).
    """
    findings: list[Finding] = []

    if not config.root.is_dir():
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
        if not directory.is_dir():
            findings.append(
                Finding(
                    "error",
                    f"Missing {name} directory: {directory}",
                    fix=f"Create {directory} (or run 'stack config init').",
                    kind="missing-dir",
                    path=directory,
                )
            )

    # Leftover pre-YAML config files (clean break: these are no longer read).
    if config.profiles_dir.is_dir():
        for legacy in config.profiles_dir.glob("*.in"):
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
    if config.bundles_dir.is_dir():
        for legacy in config.bundles_dir.glob("*.bundle"):
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

    # Env-like directories left directly under root.
    for child in config.root.iterdir():
        if not child.is_dir() or child.name in _KNOWN_TOP_LEVEL:
            continue
        if (child / "requirements.in").exists() or (child / "environment.yml").exists():
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
    if config.envs_dir.is_dir():
        for env_dir in config.envs_dir.iterdir():
            if not env_dir.is_dir():
                continue
            if (env_dir / "profiles.txt").exists():
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
            if (env_dir / "stack.txt").is_file() and not (
                env_dir / "python.txt"
            ).is_file():
                findings.append(
                    Finding(
                        "warn",
                        f"Env '{env_dir.name}' missing python.txt (will default to 3.12)",
                        fix=f"Create {env_dir / 'python.txt'}.",
                        kind="missing-python-txt",
                        path=env_dir / "python.txt",
                    )
                )

    return findings


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
        # Nothing inside the block raises ConfigError, so this is the lock
        # itself: contended past its timeout, or standing on something
        # name_lock refuses. Skip this one finding and say why. Letting it out
        # would abort the whole pass — repair() catches only OSError — and cost
        # every later finding its fix over one unavailable stem. The hint
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
