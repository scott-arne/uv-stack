"""Detect-only diagnostics for a uv-stack config tree.

``diagnose`` never mutates the filesystem; it returns a list of findings the CLI
prints with suggested fixes. It flags missing directories, legacy names
(``*.in``, ``*.bundle``, ``profiles.txt``), env-like directories left at the root, and
envs missing their source files.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

import yaml

from uv_stack.config import ConfigRoot
from uv_stack.fsutil import atomic_write_new
from uv_stack.parse import read_clean_lines

_KNOWN_TOP_LEVEL = {"profiles", "bundles", "envs", "lib"}


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


def _finish_move(src: Path, dst: Path, moved_stat: os.stat_result) -> None:
    """Complete a move after linking, checking both names before unlinking.

    Every unlink here names a path, not an inode — POSIX offers no
    unlink-by-inode — so each is preceded by an identity check that narrows,
    but cannot close, the window in which a concurrent writer could swap the
    file underneath us. Both paths check before they act. The success path in
    particular re-checks ``dst``: ``src`` still naming the moved inode does
    not prove our link survived, and unlinking ``src`` on that assumption
    would drop the inode's last name while reporting the move as done.

    These residual windows stay open — narrowed where the POSIX file API
    allows, and in one case left open by choice:

    - A ``src`` replaced after the link, with ``dst`` still naming the moved
      inode. Withdrawing ``dst`` then removes what may be that inode's last
      name, and its content is lost. This one is a choice, not a POSIX limit:
      the withdrawal is deliberate — ``dst`` is a name only we created, and
      leaving it would block every later move to that destination — but
      "rolled back" here means the destination is left clean, not that the
      moved file survives.
    - Any replacement of ``src`` that leaves ``dst`` naming an inode we cannot
      attribute to our own link — a pre-link replacement followed by another
      after the link, or a pre-link replacement by a symlink, whose target
      ``os.link`` publishes. That link is left in place: a leak, never a
      deletion.
    - ``dst`` swapped between an identity check and the unlink that check
      guards. On the rollback path that withdraws a stranger's file; on the
      success path it costs the moved inode its last name.
    - ``src`` swapped between its check and its unlink on the success path,
      which removes the replacement rather than the file we moved.
    - ``src`` already gone when we look — or gone by the time we unlink — with
      ``dst`` unlinked or replaced in the same interval. No name of ours is
      left to remove, so the function returns and the caller records the move
      as applied. That report can outrun the facts, and one ``dst.lstat()``
      before the return would make it accurate. It is deliberately not taken:
      a raise here reaches :func:`_fix_convert_yaml`, which answers a failed
      move by withdrawing the YAML it just published — and in this window that
      YAML is the last surviving copy, the source and its backup both having
      been removed by someone else. An inaccurate "applied" costs less than
      deleting the file we were asked to preserve.

    :param src: The source path that was linked.
    :param dst: The destination path where the link was created.
    :param moved_stat: The stat of the inode being moved, taken from ``src``
        BEFORE the link. It must not come from ``dst`` after the link: that
        records whatever ``dst`` names at that moment, which is our own link
        only if nobody intervened — the very thing this check must not assume.
    :raises OSError: If ``src`` or ``dst`` changed identity during the move.
        Every guarantee here holds as of the identity check that precedes the
        action it guards, never after it: the withdrawal fires only while
        ``dst`` can be attributed to our own ``os.link``, and on the two
        ``dst`` paths ``src`` still holds the moved inode. Inside the
        check-to-unlink windows the third residual window governs instead, and
        a stranger's file can go. On the rollback path ``src`` does not hold
        the moved inode at all, and the withdrawal can take that inode's last
        name — see the first residual window above.
    """
    moved_ident = (moved_stat.st_dev, moved_stat.st_ino)
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
            published = dst.lstat()
        except FileNotFoundError:
            raise OSError(f"{dst} vanished during move; nothing deleted") from None
        if (published.st_dev, published.st_ino) != moved_ident:
            raise OSError(f"{dst} changed during move; nothing deleted")
        # missing_ok: a third party can remove src between the check above and
        # this line, and dst already holds the moved inode by then. Raising
        # would report a completed move as failed, and _fix_convert_yaml
        # answers a failed move by withdrawing the YAML it just published.
        src.unlink(missing_ok=True)
        return
    # src no longer names the inode we measured — either it was replaced after
    # we linked that inode, or it was replaced BEFORE the link and os.link
    # published the replacement. Withdraw dst in both cases, but only while it
    # names an inode os.link could have published for us: the one we set out to
    # move, or the one src names now. Any other inode at dst is a concurrent
    # writer's file, and a bare "differs from moved_stat" test cannot tell that
    # case apart from ours — so it must not be the condition. See the docstring
    # for the windows this narrows but cannot close.
    ours = {moved_ident, (current.st_dev, current.st_ino)}
    try:
        dst_now = dst.lstat()
        if (dst_now.st_dev, dst_now.st_ino) in ours:
            dst.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    raise OSError(f"{src} changed during move; nothing deleted")


def _move_no_replace(src: Path, dst: Path) -> None:
    """Move ``src`` to ``dst``, refusing to replace ``dst`` or a changed ``src``.

    Publishes via :func:`os.link` (fails if ``dst`` exists), then removes the
    source only while it still names the moved inode *and* ``dst`` still holds
    our link — a source replaced mid-move is left untouched, a destination
    taken by someone else aborts the move with the source intact, and the link
    published at ``dst`` is withdrawn when :func:`_finish_move` can still prove
    that link is ours.

    A symlinked source is refused: :func:`os.link` follows symlinks by default,
    so it would publish a link to the TARGET while ``src.lstat()`` describes the
    symlink, and the unlink would silently relocate a third party's file.

    :raises FileNotFoundError: If ``src`` does not exist when the move begins,
        or is removed in the window between that check and the link.
    :raises FileExistsError: If ``dst`` already exists.
    :raises OSError: If ``src`` is not a regular file, or ``src`` or ``dst``
        changed identity during the move. ``src`` is left in place on those
        paths, and any link published at ``dst`` is withdrawn only under
        :func:`_finish_move`'s identity rules — which, when ``src`` was
        replaced after the link, can withdraw the moved inode's last name. See
        that function's docstring for the residual windows those rules narrow
        but cannot close.
    """
    moved = src.lstat()
    if not stat.S_ISREG(moved.st_mode):
        raise OSError(f"{src} is not a regular file; nothing moved")
    os.link(src, dst)
    _finish_move(src, dst, moved)


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
    stem = finding.path.stem
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

    Findings without a registered handler are ignored. Nothing here deletes
    user content: conversions keep the original as ``*.bak`` and renames skip
    when the destination exists.

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
