"""Detect-only diagnostics for a uv-stack config tree.

``diagnose`` never mutates the filesystem; it returns a list of findings the CLI
prints with suggested fixes. It flags missing directories, legacy names
(``*.in``, ``*.bundle``, ``profiles.txt``), env-like directories left at the root, and
envs missing their source files.
"""

from __future__ import annotations

import os
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


def _finish_move(src: Path, dst: Path, linked_stat: os.stat_result) -> None:
    """Complete a move after linking, ensuring source identity before unlinking.

    :param src: The source path that was linked.
    :param dst: The destination path where the link was created.
    :param linked_stat: The stat result of ``dst`` immediately after linking.
    :raises OSError: If ``src`` changed identity during the move (the move is
        rolled back and nothing is deleted).
    """
    try:
        current = src.lstat()
    except FileNotFoundError:
        return  # src vanished after linking; dst preserves the inode
    if (current.st_dev, current.st_ino) == (linked_stat.st_dev, linked_stat.st_ino):
        src.unlink()
        return
    # src was replaced after we linked the old inode: withdraw our link
    # (identity-checked) and report without deleting anyone's file.
    try:
        dst_now = dst.lstat()
        if (dst_now.st_dev, dst_now.st_ino) == (linked_stat.st_dev, linked_stat.st_ino):
            dst.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    raise OSError(f"{src} changed during move; nothing deleted")


def _move_no_replace(src: Path, dst: Path) -> None:
    """Move ``src`` to ``dst``, refusing to replace ``dst`` or a changed ``src``.

    Publishes via :func:`os.link` (fails if ``dst`` exists), then removes the
    source only while it still names the linked inode — a source replaced
    mid-move is left untouched and the published link is withdrawn.

    :raises FileExistsError: If ``dst`` already exists.
    :raises OSError: If ``src`` changed identity during the move (the move is
        rolled back and nothing is deleted).
    """
    os.link(src, dst)
    linked = dst.lstat()
    _finish_move(src, dst, linked)


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
    # Revalidate the source first.
    if not finding.path.is_file():
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
