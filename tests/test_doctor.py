from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.operations.doctor import diagnose, repair


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

    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    _finish_move(src, dst, moved_stat)
    assert not src.exists()
    assert dst.read_text() == "content\n"




def test_finish_move_src_replaced_after_link(tmp_path: Path):
    """Source replaced after link → link withdrawn, OSError raised, replacement survives."""
    import os

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
        _finish_move(src, dst, moved_stat)
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

    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    src.unlink()
    # Should not raise.
    _finish_move(src, dst, moved_stat)
    assert not src.exists()
    assert dst.read_text() == "content\n"


def test_finish_move_dst_replaced_after_link(tmp_path: Path):
    """Both src and dst replaced → the stranger's dst survives, OSError raised."""
    import os

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
        _finish_move(src, dst, moved_stat)
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
    import os

    env_dir = config_tree.env_dir("race")
    env_dir.mkdir(parents=True)
    profiles_txt = env_dir / "profiles.txt"
    stack_txt = env_dir / "stack.txt"
    profiles_txt.write_text("@standard\n")
    findings = diagnose(config_tree)

    # Simulate race: monkeypatch _finish_move to replace source before unlinking.
    from uv_stack.operations import doctor

    original_finish = doctor._finish_move

    def race_finish(src: Path, dst: Path, moved_stat: os.stat_result) -> None:
        # Replace source with new inode before calling original.
        if src == profiles_txt:
            src.unlink()
            src.write_text("@replaced\n")
        original_finish(src, dst, moved_stat)

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

    with pytest.raises(OSError, match="changed during move"):
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

    with pytest.raises(OSError, match="changed during move"):
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
