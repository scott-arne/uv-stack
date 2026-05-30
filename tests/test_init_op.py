from __future__ import annotations

from pathlib import Path

from uv_stack.config import ConfigRoot
from uv_stack.operations.init import seed_defaults


def test_seed_creates_profiles_and_bundles(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "envs-root")
    written = seed_defaults(cfg)
    assert cfg.profile_path("ds").is_file()
    assert cfg.bundle_path("standard").is_file()
    assert cfg.profile_path("ds") in written
    # envs/ directory is created too
    assert cfg.envs_dir.is_dir()


def test_seed_does_not_clobber_existing(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "envs-root")
    cfg.profiles_dir.mkdir(parents=True)
    cfg.profile_path("ds").write_text("custom-content\n")
    written = seed_defaults(cfg)
    assert cfg.profile_path("ds").read_text() == "custom-content\n"
    assert cfg.profile_path("ds") not in written


def test_seed_is_idempotent(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "envs-root")
    seed_defaults(cfg)
    written_second = seed_defaults(cfg)
    assert written_second == []
