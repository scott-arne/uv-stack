from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import _deadline, _lock_held_by_another_process
from uv_stack.config import ConfigRoot
from uv_stack.fsutil import _LOCK_AVAILABLE
from uv_stack.operations.doctor import diagnose, repair

_IS_ROOT = getattr(os, "geteuid", lambda: -1)() == 0


def test_clean_tree_has_no_errors(config_tree: ConfigRoot):
    findings = diagnose(config_tree)
    assert [f for f in findings if f.level == "error"] == []


def test_missing_root_reports_error(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "does-not-exist")
    findings = diagnose(cfg)
    assert any(f.level == "error" for f in findings)


def test_legacy_profile_in_file_flagged(config_tree: ConfigRoot):
    (config_tree.profiles_dir / "old.in").write_text("numpy\n")
    messages = [f.message for f in diagnose(config_tree)]
    assert any("old.in" in m for m in messages)


def test_legacy_bundle_file_flagged(config_tree: ConfigRoot):
    (config_tree.bundles_dir / "old.bundle").write_text("ds\n")
    messages = [f.message for f in diagnose(config_tree)]
    assert any("old.bundle" in m for m in messages)


def test_legacy_profiles_txt_flagged(config_tree: ConfigRoot):
    (config_tree.env_dir("main") / "profiles.txt").write_text("ds\n")
    messages = [f.message for f in diagnose(config_tree)]
    assert any("profiles.txt" in m for m in messages)


def test_legacy_env_at_root_flagged(config_tree: ConfigRoot):
    # An env-like dir directly under root (not under envs/) with generated files.
    legacy = config_tree.root / "legacyenv"
    legacy.mkdir()
    (legacy / "requirements.in").write_text("# x\n")
    messages = [f.message for f in diagnose(config_tree)]
    assert any("legacyenv" in m for m in messages)


def test_env_missing_python_txt_flagged(config_tree: ConfigRoot):
    config_tree.env_python_path("main").unlink()
    messages = [f.message for f in diagnose(config_tree)]
    assert any("python.txt" in m for m in messages)


def test_findings_carry_kinds(tmp_path):
    config = ConfigRoot(tmp_path / "python-envs")
    findings = diagnose(config)
    assert findings[0].kind == "missing-root"
    assert findings[0].path == config.root


def test_repair_creates_missing_dirs(tmp_path):
    config = ConfigRoot(tmp_path / "python-envs")
    actions = repair(config, diagnose(config))
    assert all(a.applied for a in actions)
    assert config.root.is_dir()
    # Re-diagnosing after creating the root reveals the missing subdirs;
    # a second repair pass clears those too.
    repair(config, diagnose(config))
    assert config.profiles_dir.is_dir()
    assert config.bundles_dir.is_dir()
    assert config.envs_dir.is_dir()
    assert diagnose(config) == []


def test_repair_renames_profiles_txt(config_tree: ConfigRoot):
    env_dir = config_tree.env_dir("legacy")
    env_dir.mkdir(parents=True)
    (env_dir / "profiles.txt").write_text("@standard\n")
    actions = repair(config_tree, diagnose(config_tree))
    renamed = [a for a in actions if a.finding.kind == "legacy-profiles-txt"]
    assert renamed and renamed[0].applied
    assert (env_dir / "stack.txt").read_text() == "@standard\n"
    assert not (env_dir / "profiles.txt").exists()


def test_repair_skips_profiles_txt_when_stack_exists(config_tree: ConfigRoot):
    env_dir = config_tree.env_dir("main")
    (env_dir / "profiles.txt").write_text("@standard\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-profiles-txt"]
    assert skipped and not skipped[0].applied
    assert skipped[0].reason is not None
    assert (env_dir / "profiles.txt").exists()


def test_repair_converts_legacy_profile_to_yaml(config_tree: ConfigRoot):
    legacy = config_tree.profiles_dir / "old.in"
    legacy.write_text("# comment\nnumpy\npandas>=2\n")
    actions = repair(config_tree, diagnose(config_tree))
    converted = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert converted and converted[0].applied
    assert config_tree.load_profile("old").includes == ["numpy", "pandas>=2"]
    assert not legacy.exists()
    assert (config_tree.profiles_dir / "old.in.bak").is_file()


def test_repair_skips_conversion_when_yaml_exists(config_tree: ConfigRoot):
    legacy = config_tree.profiles_dir / "ds.in"
    legacy.write_text("numpy\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert legacy.exists()


def test_repair_skips_conversion_when_backup_exists(config_tree: ConfigRoot):
    legacy = config_tree.profiles_dir / "old.in"
    legacy.write_text("numpy\n")
    (config_tree.profiles_dir / "old.in.bak").write_text("precious\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "old.in.bak already exists" == skipped[0].reason
    assert (config_tree.profiles_dir / "old.in.bak").read_text() == "precious\n"
    assert legacy.exists()


def test_repair_converts_legacy_bundle(config_tree: ConfigRoot):
    legacy = config_tree.bundles_dir / "oldb.bundle"
    legacy.write_text("ds\nchem\n")
    repair(config_tree, diagnose(config_tree))
    assert config_tree.load_bundle("oldb").includes == ["ds", "chem"]
    assert (config_tree.bundles_dir / "oldb.bundle.bak").is_file()


def test_repair_skips_legacy_profile_when_bundle_exists_for_stem(config_tree: ConfigRoot):
    """Legacy profiles/old.in skipped when bundles/old.yaml exists (shadow guard)."""
    # Create existing bundle.
    (config_tree.bundles_dir / "old.yaml").write_text("includes:\n- pandas\n")
    # Create legacy profile with same stem.
    legacy = config_tree.profiles_dir / "old.in"
    legacy.write_text("numpy\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "would shadow the existing bundle" in skipped[0].reason
    # Legacy file intact, no profiles/old.yaml created.
    assert legacy.exists()
    assert not (config_tree.profiles_dir / "old.yaml").exists()


def test_repair_skips_legacy_bundle_when_profile_exists_for_stem(config_tree: ConfigRoot):
    """Legacy bundles/old.bundle skipped when profiles/old.yaml exists (shadow guard)."""
    # Create existing profile.
    (config_tree.profiles_dir / "old.yaml").write_text("includes:\n- numpy\n")
    # Create legacy bundle with same stem.
    legacy = config_tree.bundles_dir / "old.bundle"
    legacy.write_text("ds\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-bundle"]
    assert skipped and not skipped[0].applied
    assert "would be shadowed by the existing profile" in skipped[0].reason
    # Legacy file intact, no bundles/old.yaml created.
    assert legacy.exists()
    assert not (config_tree.bundles_dir / "old.yaml").exists()


def test_repair_moves_misplaced_env(config_tree: ConfigRoot):
    stray = config_tree.root / "straggler"
    stray.mkdir()
    (stray / "requirements.in").write_text("# x\n")
    repair(config_tree, diagnose(config_tree))
    assert not stray.exists()
    assert (config_tree.envs_dir / "straggler" / "requirements.in").is_file()


def test_repair_writes_default_python_txt(config_tree: ConfigRoot):
    config_tree.env_python_path("main").unlink()
    repair(config_tree, diagnose(config_tree))
    assert config_tree.env_python_path("main").read_text() == "3.12\n"


def test_repair_skips_stale_legacy_conversion(config_tree: ConfigRoot):
    """Stale conversion: legacy .in deleted after diagnose() → skip, no .yaml created."""
    legacy = config_tree.profiles_dir / "ephemeral.in"
    legacy.write_text("numpy\n")
    findings = diagnose(config_tree)
    # Delete the source after diagnosis.
    legacy.unlink()
    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "no longer exists" in skipped[0].reason
    assert not (config_tree.profiles_dir / "ephemeral.yaml").exists()


def test_repair_skips_python_txt_created_between_diagnose_and_repair(config_tree: ConfigRoot):
    """python.txt created between diagnose and repair → skipped, user content preserved."""
    config_tree.env_python_path("main").unlink()
    findings = diagnose(config_tree)
    # User creates python.txt with their own content after diagnosis.
    config_tree.env_python_path("main").write_text("3.11\n")
    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "missing-python-txt"]
    assert skipped and not skipped[0].applied
    assert "python.txt already exists" in skipped[0].reason
    assert config_tree.env_python_path("main").read_text() == "3.11\n"


def test_repair_skips_profiles_txt_rename_when_stack_txt_appears_after_diagnose(
    config_tree: ConfigRoot,
):
    """profiles.txt rename when stack.txt appears after diagnose → skipped, both intact."""
    env_dir = config_tree.env_dir("racy")
    env_dir.mkdir(parents=True)
    (env_dir / "profiles.txt").write_text("@standard\n")
    findings = diagnose(config_tree)
    # User creates stack.txt after diagnosis.
    (env_dir / "stack.txt").write_text("@stable\n")
    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profiles-txt"]
    assert skipped and not skipped[0].applied
    assert "stack.txt already exists" in skipped[0].reason
    assert (env_dir / "profiles.txt").exists()
    assert (env_dir / "stack.txt").read_text() == "@stable\n"


def test_repair_conversion_rollback_on_source_vanish(config_tree: ConfigRoot, monkeypatch):
    """Source vanishes after YAML publish → YAML removed (identity match), action skipped."""
    from uv_stack.operations import doctor

    legacy = config_tree.profiles_dir / "vanish.in"
    legacy.write_text("numpy\n")
    findings = diagnose(config_tree)

    # Monkeypatch _move_no_replace to unlink the source then raise FileNotFoundError.
    def fake_move(src, dst):
        src.unlink()
        raise FileNotFoundError(f"{src} disappeared")

    monkeypatch.setattr(doctor, "_move_no_replace", fake_move)

    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "disappeared" in skipped[0].reason
    # YAML was removed in rollback.
    assert not (config_tree.profiles_dir / "vanish.yaml").exists()
    # Source is gone (simulated vanish).
    assert not legacy.exists()


def test_repair_conversion_preserves_replaced_yaml(config_tree: ConfigRoot, monkeypatch):
    """Dest YAML replaced after publish → replacement survives, action skipped."""
    from uv_stack.operations import doctor

    legacy = config_tree.profiles_dir / "race.in"
    legacy.write_text("numpy\n")
    findings = diagnose(config_tree)

    def fake_move(src, dst):
        # Simulate concurrent replacement: unlink the YAML and rewrite it.
        yaml_path = config_tree.profiles_dir / "race.yaml"
        yaml_path.unlink()
        yaml_path.write_text("includes:\n- pandas\n")
        raise FileExistsError(f"{dst.with_suffix('.bak')} already exists")

    monkeypatch.setattr(doctor, "_move_no_replace", fake_move)

    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "race.in.bak already exists" in skipped[0].reason
    # Replacement YAML survives (identity check prevented deletion).
    yaml_path = config_tree.profiles_dir / "race.yaml"
    assert yaml_path.exists()
    assert "pandas" in yaml_path.read_text()
    # Source file untouched.
    assert legacy.exists()


def test_repair_misplaced_env_skips_when_dest_created_after_diagnose(config_tree: ConfigRoot):
    """Dest dir created after diagnose → skipped, both directories intact."""
    stray = config_tree.root / "concurrent"
    stray.mkdir()
    (stray / "requirements.in").write_text("# x\n")
    findings = diagnose(config_tree)
    # Destination appears after diagnosis.
    dest = config_tree.envs_dir / "concurrent"
    dest.mkdir()
    (dest / "environment.yml").write_text("name: x\n")
    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "misplaced-env"]
    assert skipped and not skipped[0].applied
    assert "concurrent already exists" in skipped[0].reason
    # Both directories intact.
    assert stray.exists()
    assert (stray / "requirements.in").exists()
    assert dest.exists()
    assert (dest / "environment.yml").read_text() == "name: x\n"


def test_finish_move_normal_case(tmp_path: Path):
    """Normal move: source identity matches linked inode → source removed."""
    import os

    from uv_stack.fsutil import Published
    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    _finish_move(src, dst, moved_stat, Published("link", moved_stat))
    assert not src.exists()
    assert dst.read_text() == "content\n"




def test_finish_move_src_replaced_after_link(tmp_path: Path):
    """Source replaced after link → link withdrawn, OSError raised, replacement survives."""
    import os

    from uv_stack.fsutil import Published
    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    # Replace source with new inode.
    src.unlink()
    src.write_text("replacement\n")
    try:
        _finish_move(src, dst, moved_stat, Published("link", moved_stat))
        raise AssertionError("Expected OSError")
    except OSError as e:
        assert "changed during move" in str(e)
    # Replacement survives.
    assert src.exists()
    assert src.read_text() == "replacement\n"
    # Link withdrawn.
    assert not dst.exists()


def test_finish_move_src_vanished_after_link(tmp_path: Path):
    """Source vanished after link → no error, dest preserves the inode."""
    import os

    from uv_stack.fsutil import Published
    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    src.unlink()
    # Should not raise.
    _finish_move(src, dst, moved_stat, Published("link", moved_stat))
    assert not src.exists()
    assert dst.read_text() == "content\n"


def test_finish_move_dst_replaced_after_link(tmp_path: Path):
    """Both src and dst replaced → the stranger's dst survives, OSError raised."""
    import os

    from uv_stack.fsutil import Published
    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    # Replace both src and dst.
    src.unlink()
    src.write_text("new-src\n")
    dst.unlink()
    dst.write_text("new-dst\n")
    try:
        _finish_move(src, dst, moved_stat, Published("link", moved_stat))
        raise AssertionError("Expected OSError")
    except OSError as e:
        assert "changed during move" in str(e)
    # Both replacements survive.
    assert src.exists()
    assert src.read_text() == "new-src\n"
    assert dst.exists()
    assert dst.read_text() == "new-dst\n"


def test_move_no_replace_identity_check_in_rename(config_tree: ConfigRoot, monkeypatch):
    """File rename with source replaced after link → skipped, replacement survives."""

    env_dir = config_tree.env_dir("race")
    env_dir.mkdir(parents=True)
    profiles_txt = env_dir / "profiles.txt"
    stack_txt = env_dir / "stack.txt"
    profiles_txt.write_text("@standard\n")
    findings = diagnose(config_tree)

    # Simulate race: monkeypatch _finish_move to replace source before unlinking.
    from uv_stack.fsutil import Published
    from uv_stack.operations import doctor

    original_finish = doctor._finish_move

    def race_finish(
        src: Path, dst: Path, moved_stat: os.stat_result, published: Published
    ) -> None:
        # Replace source with new inode before calling original.
        if src == profiles_txt:
            src.unlink()
            src.write_text("@replaced\n")
        original_finish(src, dst, moved_stat, published)

    monkeypatch.setattr(doctor, "_finish_move", race_finish)

    actions = repair(config_tree, findings)

    skipped = [a for a in actions if a.finding.kind == "legacy-profiles-txt"]
    assert skipped and not skipped[0].applied
    assert "changed during move" in skipped[0].reason
    # Replacement survives.
    assert profiles_txt.exists()
    assert profiles_txt.read_text() == "@replaced\n"
    # Link withdrawn.
    assert not stack_txt.exists()


def _link_less(monkeypatch) -> None:
    """Make os.link raise EOPNOTSUPP, as FAT/exFAT and some network mounts do."""
    import errno
    import os

    def _no_link(src, dst, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links not supported")

    monkeypatch.setattr(os, "link", _no_link)


def test_move_no_replace_completes_on_a_link_less_filesystem(tmp_path: Path, monkeypatch):
    """The whole point of C10: a repair that is impossible today now completes."""
    from uv_stack.operations.doctor import _move_no_replace

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    _move_no_replace(src, dst)
    assert not src.exists()
    assert dst.read_text() == "content\n"


@pytest.mark.parametrize(
    ("mutation", "label"),
    [(b"mutated-and-much-longer\n", "different-length"), (b"MUTATED!\n", "same-length")],
)
def test_move_no_replace_copy_path_refuses_an_in_place_rewrite(
    tmp_path: Path, monkeypatch, mutation: bytes, label: str
):
    """A write through the source's own inode after the copy must not be lost.

    On the copy path ``dst`` is a snapshot, so an in-place write leaves
    st_dev/st_ino untouched while the bytes diverge. Unlinking the source on
    identity alone would destroy the post-write content. The same-length case
    is here so the size short-circuit cannot be what carries the test.
    """
    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_mutate(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        with open(src, "r+b") as handle:
            handle.write(mutation)
            handle.truncate()
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_mutate)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # The source survives, still the same inode, holding the post-copy bytes.
    assert src.read_bytes() == mutation
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    # The stale snapshot is withdrawn.
    assert not dst.exists()


def test_move_no_replace_link_path_survives_an_in_place_rewrite(tmp_path: Path, monkeypatch):
    """The paired link case still succeeds: both names reach the mutated inode.

    The content check belongs to the copy branch only. Applying it to the link
    branch would turn a move that loses nothing into a spurious failure.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_mutate(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        with open(src, "r+b") as handle:
            handle.write(b"MUTATED!\n")
            handle.truncate()
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_mutate)
    doctor._move_no_replace(src, dst)
    assert not src.exists()
    assert dst.read_bytes() == b"MUTATED!\n"


def test_move_no_replace_copy_path_withdraws_when_the_source_is_replaced(
    tmp_path: Path, monkeypatch
):
    """A replaced source takes the withdrawal path, leaving the replacement intact."""
    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_replace(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        src.unlink()
        src.write_text("replacement\n")
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_replace)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    assert src.read_text() == "replacement\n"
    assert not dst.exists()


def test_move_no_replace_copy_path_leaves_a_foreign_destination_alone(
    tmp_path: Path, monkeypatch
):
    """The rollback set is exactly the inode the exclusive create made.

    A destination replaced by a third party after publication is not ours to
    delete, however the move ends.
    """
    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_hijack(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        src.unlink()
        src.write_text("replacement\n")
        dst.unlink()
        dst.write_text("stranger\n")
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_hijack)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "nothing deleted" in str(excinfo.value)
    assert dst.read_text() == "stranger\n"


def test_same_bytes_compares_content_not_timestamps(tmp_path: Path):
    """Sizes first, then bytes — never mtime, which FAT resolves to two seconds."""
    from uv_stack.operations.doctor import _same_bytes

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    d = tmp_path / "d.txt"
    a.write_bytes(b"same\n")
    b.write_bytes(b"same\n")
    c.write_bytes(b"diff\n")
    d.write_bytes(b"longer content\n")
    assert _same_bytes(a, b)
    assert not _same_bytes(a, c)
    assert not _same_bytes(a, d)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform lacks mkfifo")
def test_move_no_replace_copy_path_refuses_a_fifo_source_during_same_bytes(
    tmp_path: Path, monkeypatch
):
    """A FIFO swapped for src during the byte comparison is refused, not hung on.

    The FIFO must be empty (st_size == 0) to reach the open — a non-empty source
    short-circuits the comparison to False. The guarded open refuses it rather
    than blocking.
    """
    import os

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("")  # Empty so size check passes.
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_fifo(left, right):
        # Replace src with a FIFO before calling the real comparison.
        src.unlink()
        os.mkfifo(src)
        return real_same_bytes(left, right)

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_fifo)
    with _deadline(5.0):
        with pytest.raises(OSError) as excinfo:
            doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # FIFO survives, dst is withdrawn.
    assert src.exists()
    assert not dst.exists()


def test_move_no_replace_copy_path_refuses_a_symlink_source_during_same_bytes(
    tmp_path: Path, monkeypatch
):
    """A symlink swapped for src during the byte comparison is refused via O_NOFOLLOW."""

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    target.write_text("content\n")
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_symlink(left, right):
        # Replace src with a symlink before calling the real comparison.
        src.unlink()
        src.symlink_to(target)
        return real_same_bytes(left, right)

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_symlink)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # Symlink survives, dst is withdrawn.
    assert src.is_symlink()
    assert not dst.exists()


def test_move_no_replace_copy_path_refuses_a_chmod_after_same_bytes(
    tmp_path: Path, monkeypatch
):
    """A chmod after the byte comparison but before the mode check withdraws dst.

    The mode check now re-stats src immediately before the unlink, so a chmod
    landing after the byte comparison is caught.
    """
    import os
    import stat

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    os.chmod(src, 0o640)
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_chmod(left, right):
        result = real_same_bytes(left, right)
        # chmod after the byte comparison returns.
        os.chmod(src, 0o600)
        return result

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_chmod)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # The source survives with the post-chmod mode.
    assert src.read_text() == "content\n"
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    assert stat.S_IMODE(src.stat().st_mode) == 0o600
    # The stale snapshot is withdrawn.
    assert not dst.exists()


def test_move_no_replace_copy_path_refuses_a_dst_replacement_after_same_bytes(
    tmp_path: Path, monkeypatch
):
    """Dst replaced with a same-byte file after the comparison withdraws nothing.

    The replacement is not ours to delete, and src survives.
    """
    import os
    import stat

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    replacement = tmp_path / "replacement.txt"
    src.write_text("content\n")
    replacement.write_text("content\n")  # Same bytes.
    os.chmod(replacement, 0o600)  # Different mode from default.
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_replace_dst(left, right):
        result = real_same_bytes(left, right)
        # Replace dst after the comparison returns.
        dst.unlink()
        os.rename(replacement, dst)
        return result

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_replace_dst)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    assert "nothing deleted" in str(excinfo.value)
    # The source survives unchanged.
    assert src.read_text() == "content\n"
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    # The replacement at dst survives (not ours to delete).
    assert dst.read_text() == "content\n"
    assert stat.S_IMODE(dst.stat().st_mode) == 0o600


def test_move_no_replace_copy_path_refuses_when_dst_vanishes_after_same_bytes(
    tmp_path: Path, monkeypatch
):
    """Dst vanishing after the comparison raises an error rather than silently succeeding.

    Src still exists and the destination is gone, so the move did not complete.
    """

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_remove_dst(left, right):
        result = real_same_bytes(left, right)
        # Remove dst after the comparison returns.
        dst.unlink()
        return result

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_remove_dst)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # The source survives.
    assert src.read_text() == "content\n"
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    # Dst is gone.
    assert not dst.exists()


def test_move_no_replace_copy_path_refuses_a_chmod_under_the_copy(tmp_path: Path, monkeypatch):
    """A chmod on src after publication leaves src in place and withdraws dst.

    The copy is a snapshot of the pre-chmod bits. Unlinking src on identity and
    bytes alone would discard the mode change.
    """
    import os
    import stat

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    os.chmod(src, 0o640)
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_chmod(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        os.chmod(src, 0o600)
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_chmod)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # The source survives, still the same inode, holding the post-chmod mode.
    assert src.read_text() == "content\n"
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    assert stat.S_IMODE(src.stat().st_mode) == 0o600
    # The stale snapshot is withdrawn.
    assert not dst.exists()


def test_move_no_replace_propagates_eperm_from_a_public_source(tmp_path: Path, monkeypatch):
    """EPERM from os.link on a user-visible file is a denial, not a fallback trigger.

    On Linux with fs.protected_hardlinks=1, EPERM means "you do not own this
    file." Treating it as link-less turns a denial into copy-then-unlink.
    """
    import errno
    import os

    from uv_stack.operations.doctor import _move_no_replace

    def _eperm(a, b, **kwargs):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", _eperm)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    with pytest.raises(OSError) as excinfo:
        _move_no_replace(src, dst)
    assert excinfo.value.errno == errno.EPERM
    assert src.read_text() == "content\n"
    assert not dst.exists()


def test_repair_conversion_source_replaced_between_read_and_move(
    config_tree: ConfigRoot, monkeypatch
):
    """Source replaced after read but before move → skipped, no YAML left, replacement intact."""
    from uv_stack.operations import doctor
    from uv_stack.parse import read_clean_lines as original_read

    legacy = config_tree.profiles_dir / "race.in"
    legacy.write_text("numpy\n")
    findings = diagnose(config_tree)

    # Monkeypatch read_clean_lines to replace the source after reading.
    def race_read(path):
        result = original_read(path)
        if path == legacy:
            # Replace source with new inode.
            path.unlink()
            path.write_text("pandas\n")
        return result

    monkeypatch.setattr(doctor, "read_clean_lines", race_read)

    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "changed during conversion" in skipped[0].reason
    # No YAML left.
    assert not (config_tree.profiles_dir / "race.yaml").exists()
    # Replacement intact.
    assert legacy.exists()
    assert legacy.read_text() == "pandas\n"


def test_move_no_replace_never_adopts_a_foreign_destination(tmp_path: Path, monkeypatch):
    """A dst holding a stranger's file is not ours to delete.

    The os.link stub models the reachable end state — our link published and
    then replaced by a foreign writer, with src replaced too — rather than a
    literal call sequence, since a real os.link would have raised
    FileExistsError against an existing dst. The inode relationships driving
    the code path are the same either way.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")

    def racing_link(_a, _b, **_kwargs):
        dst.write_text("foreign\n")
        src.unlink()
        src.write_text("replacement\n")

    monkeypatch.setattr(doctor.os, "link", racing_link)

    with pytest.raises(OSError, match="source.txt changed during move"):
        doctor._move_no_replace(src, dst)

    assert dst.read_text() == "foreign\n"
    assert src.read_text() == "replacement\n"


def test_move_no_replace_withdraws_a_link_to_a_replaced_source(
    tmp_path: Path, monkeypatch
):
    """A source replaced BEFORE the link leaves no stray link at dst.

    The link then publishes the replacement inode, which is neither the inode
    we measured nor a stranger's file — it is ours, and leaving it behind
    would block every later move to this destination.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    real_link = doctor.os.link

    def racing_link(a, b, **kwargs):
        # A concurrent writer replaces src between our lstat and the link.
        src.unlink()
        src.write_text("replacement\n")
        real_link(a, b, **kwargs)

    monkeypatch.setattr(doctor.os, "link", racing_link)

    with pytest.raises(OSError, match="source.txt changed during move"):
        doctor._move_no_replace(src, dst)

    assert not dst.exists()
    assert src.read_text() == "replacement\n"


def test_move_no_replace_refuses_a_symlinked_source(tmp_path: Path):
    """os.link would publish the TARGET, and the unlink would relocate it."""
    from uv_stack.operations.doctor import _move_no_replace

    target = tmp_path / "target.txt"
    target.write_text("target\n")
    src = tmp_path / "link.txt"
    src.symlink_to(target)
    dst = tmp_path / "dest.txt"

    with pytest.raises(OSError, match="not a regular file"):
        _move_no_replace(src, dst)

    assert src.is_symlink()
    assert target.read_text() == "target\n"
    assert not dst.exists()


def test_move_no_replace_refuses_when_the_destination_vanishes(
    tmp_path: Path, monkeypatch
):
    """A dst removed after we publish it must not cost src its last name.

    src still names the moved inode, so checking src alone succeeds; only
    re-checking dst keeps the unlink from destroying the file outright while
    reporting the move as done.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("precious\n")
    real_link = doctor.os.link

    def link_then_dst_vanishes(a, b, **kwargs):
        real_link(a, b, **kwargs)
        dst.unlink()

    monkeypatch.setattr(doctor.os, "link", link_then_dst_vanishes)

    with pytest.raises(OSError, match="vanished during move"):
        doctor._move_no_replace(src, dst)

    assert src.read_text() == "precious\n"


def test_move_no_replace_refuses_when_the_destination_is_replaced(
    tmp_path: Path, monkeypatch
):
    """A dst swapped for a stranger's file after we link leaves both intact."""
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("precious\n")
    real_link = doctor.os.link

    def link_then_dst_replaced(a, b, **kwargs):
        real_link(a, b, **kwargs)
        dst.unlink()
        dst.write_text("foreign\n")

    monkeypatch.setattr(doctor.os, "link", link_then_dst_replaced)

    with pytest.raises(OSError, match="dest.txt changed during move"):
        doctor._move_no_replace(src, dst)

    assert src.read_text() == "precious\n"
    assert dst.read_text() == "foreign\n"


def test_move_no_replace_completes_when_the_source_vanishes_before_the_unlink(
    tmp_path: Path, monkeypatch
):
    """A src removed after its check has already been moved; do not report failure.

    dst holds the moved inode by then, so the move is complete. Raising here
    would make _fix_convert_yaml withdraw a YAML file it had already published,
    leaving the tree with neither the source nor its conversion.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("precious\n")
    real_lstat = Path.lstat
    src_checks = 0
    removed = False

    def remove_src_right_after_its_identity_check(self, *args, **kwargs):
        # Anchored on src's own check, not on dst's — which merely happens to
        # sit between it and the unlink today. Reordering the two stats must
        # not quietly turn this into a test of the early-return path.
        nonlocal src_checks, removed
        result = real_lstat(self, *args, **kwargs)
        if self == src:
            src_checks += 1
            # The first check is _move_no_replace's; the second is the one
            # _finish_move's unlink relies on.
            if src_checks == 2:
                src.unlink()
                removed = True
        return result

    monkeypatch.setattr(Path, "lstat", remove_src_right_after_its_identity_check)

    doctor._move_no_replace(src, dst)

    assert removed, "the race never fired; this no longer tests the window"
    assert not src.exists()
    assert dst.read_text() == "precious\n"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_repair_conversion_serializes_against_a_concurrent_create(
    config_tree: ConfigRoot, monkeypatch
):
    """A conversion is a create in the shared stem namespace, so it takes the lock.

    Publishing profiles/<stem>.yaml collides with a bundle create for the same
    stem, and the collision spans two paths, so no no-clobber write can reserve
    it. Without the lock the two commit on top of each other and diagnose then
    reports nothing wrong. Holding the stem lock from another process is what
    proves doctor asks for it: the conversion has no other reason to stop.

    The env's missing python.txt is repaired in the same pass on purpose. The
    lock raises ConfigError, which repair() does not catch, so an unavailable
    stem must cost its own finding and nothing else.
    """
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    legacy = config_tree.profiles_dir / "old.in"
    legacy.write_text("numpy\n")
    (config_tree.envs_dir / "main" / "python.txt").unlink()

    with _lock_held_by_another_process(config_tree.stem_lock_path("old")):
        actions = repair(config_tree, diagnose(config_tree))

    converted = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert converted and not converted[0].applied
    assert "another stack process" in converted[0].reason
    assert legacy.exists()
    assert not (config_tree.profiles_dir / "old.yaml").exists()

    unrelated = [a for a in actions if a.finding.kind == "missing-python-txt"]
    assert unrelated and unrelated[0].applied, (
        "one unavailable stem lock aborted the rest of the repair pass"
    )


def test_diagnose_reports_degraded_locks(config_tree: ConfigRoot, monkeypatch):
    """A root that cannot lock is reported once, as a warning, with no repair."""
    from uv_stack.operations import doctor

    monkeypatch.setattr(doctor, "probe_locking", lambda path: False)
    findings = diagnose(config_tree)
    degraded = [f for f in findings if f.kind == "degraded-locks"]
    assert len(degraded) == 1
    assert degraded[0].level == "warn"
    assert str(config_tree.root) in degraded[0].message
    assert "not serialized" in degraded[0].message
    assert degraded[0].path == config_tree.locks_dir
    assert degraded[0].kind not in doctor._REPAIRS


def test_diagnose_is_silent_when_locking_works(config_tree: ConfigRoot):
    findings = diagnose(config_tree)
    assert [f for f in findings if f.kind == "degraded-locks"] == []


def test_diagnose_never_raises_when_locks_is_a_plain_file(config_tree: ConfigRoot):
    """The probe's non-raising contract, exercised through the caller that needs it."""
    config_tree.locks_dir.write_text("not a directory\n")
    findings = diagnose(config_tree)
    assert any(f.kind == "degraded-locks" for f in findings)


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_diagnose_never_raises_on_an_unsearchable_locks_dir(config_tree: ConfigRoot):
    """A .locks this user cannot traverse into: the degrade path with no exception."""
    config_tree.locks_dir.mkdir()
    os.chmod(config_tree.locks_dir, 0o600)
    try:
        findings = diagnose(config_tree)
    finally:
        # Restore, or tmp_path teardown cannot remove the directory.
        os.chmod(config_tree.locks_dir, 0o700)
    assert any(f.kind == "degraded-locks" for f in findings)


def test_diagnose_never_raises_on_a_non_regular_probe_lock(config_tree: ConfigRoot):
    """A planted FIFO at probe.lock is the shape name_lock refuses outright."""
    config_tree.locks_dir.mkdir()
    os.mkfifo(config_tree.probe_lock_path())
    findings = diagnose(config_tree)
    assert any(f.kind == "degraded-locks" for f in findings)


def test_repair_leaves_degraded_locks_alone(config_tree: ConfigRoot, monkeypatch):
    """Nothing here is machine-fixable, so --fix must report no action for it."""
    from uv_stack.operations import doctor

    monkeypatch.setattr(doctor, "probe_locking", lambda path: False)
    findings = diagnose(config_tree)
    actions = repair(config_tree, findings)
    assert [a for a in actions if a.finding.kind == "degraded-locks"] == []
