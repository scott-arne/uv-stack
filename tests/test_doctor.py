from __future__ import annotations

from pathlib import Path

from uvstack.config import ConfigRoot
from uvstack.operations.doctor import Finding, diagnose


def test_clean_tree_has_no_errors(config_tree: ConfigRoot):
    findings = diagnose(config_tree)
    assert [f for f in findings if f.level == "error"] == []


def test_missing_root_reports_error(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "does-not-exist")
    findings = diagnose(cfg)
    assert any(f.level == "error" for f in findings)


def test_legacy_bundle_extension_flagged(config_tree: ConfigRoot):
    (config_tree.bundles_dir / "old.profiles").write_text("ds\n")
    messages = [f.message for f in diagnose(config_tree)]
    assert any("old.profiles" in m for m in messages)


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
