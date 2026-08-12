from __future__ import annotations

from pathlib import Path

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
