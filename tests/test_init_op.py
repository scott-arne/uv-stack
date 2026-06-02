from __future__ import annotations

from pathlib import Path

from uv_stack.config import ConfigRoot
from uv_stack.operations.init import init_config_root


def test_init_creates_directories(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "envs-root")
    created = init_config_root(cfg)
    assert cfg.profiles_dir.is_dir()
    assert cfg.bundles_dir.is_dir()
    assert cfg.envs_dir.is_dir()
    assert set(created) == {cfg.profiles_dir, cfg.bundles_dir, cfg.envs_dir}


def test_init_seeds_no_profiles_or_bundles(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "envs-root")
    init_config_root(cfg)
    assert list(cfg.profiles_dir.iterdir()) == []
    assert list(cfg.bundles_dir.iterdir()) == []


def test_init_is_idempotent(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "envs-root")
    init_config_root(cfg)
    assert init_config_root(cfg) == []


def test_init_only_creates_missing(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "envs-root")
    cfg.profiles_dir.mkdir(parents=True)
    created = init_config_root(cfg)
    assert cfg.profiles_dir not in created
    assert set(created) == {cfg.bundles_dir, cfg.envs_dir}
