from __future__ import annotations

import re
from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.operations.init import init_config_root


def test_init_creates_directories(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "envs-root")
    created = init_config_root(cfg)
    assert cfg.profiles_dir.is_dir()
    assert cfg.bundles_dir.is_dir()
    assert cfg.envs_dir.is_dir()
    assert set(created) == {cfg.profiles_dir, cfg.bundles_dir, cfg.envs_dir, cfg.locks_dir}
    assert cfg.locks_dir.is_dir()


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
    assert set(created) == {cfg.bundles_dir, cfg.envs_dir, cfg.locks_dir}


@pytest.mark.parametrize("below", ["", "python-envs"])
def test_init_creates_the_missing_target_of_a_dangling_root_link(tmp_path: Path, below: str):
    target = tmp_path / "checkout"
    link = tmp_path / "root"
    link.symlink_to(target)
    cfg = ConfigRoot(link / below)
    created = init_config_root(cfg)
    assert created[0] == target
    assert set(created[1:]) == {cfg.profiles_dir, cfg.bundles_dir, cfg.envs_dir, cfg.locks_dir}
    assert link.is_symlink()
    assert (target / below / "profiles").is_dir()


@pytest.mark.parametrize("below", ["", "python-envs"])
def test_init_refuses_a_root_at_or_under_a_file(tmp_path: Path, below: str):
    blocker = tmp_path / "root"
    blocker.write_text("")
    with pytest.raises(ConfigError, match=f"^Not a directory: {re.escape(str(blocker))}$"):
        init_config_root(ConfigRoot(blocker / below))
