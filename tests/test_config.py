from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError


def test_discover_precedence_explicit_over_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "from-env"))
    cfg = ConfigRoot.discover(root=tmp_path / "explicit")
    assert cfg.root == (tmp_path / "explicit")


def test_discover_uses_env_when_no_explicit(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "from-env"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "from-env")


def test_discover_default(monkeypatch):
    monkeypatch.delenv("UV_ENV_ROOT", raising=False)
    cfg = ConfigRoot.discover()
    assert cfg.root == (Path.home() / ".config" / "python-envs")


def test_path_helpers(config_tree: ConfigRoot):
    root = config_tree.root
    assert config_tree.profile_path("ds") == root / "profiles" / "ds.yaml"
    assert config_tree.bundle_path("qsar") == root / "bundles" / "qsar.yaml"
    assert config_tree.env_stack_path("main") == root / "envs" / "main" / "stack.txt"
    assert config_tree.env_lock("main") == root / "envs" / "main" / "requirements.lock.txt"


def test_existence_checks(config_tree: ConfigRoot):
    assert config_tree.profile_exists("ds")
    assert not config_tree.profile_exists("nope")
    assert config_tree.bundle_exists("standard")
    assert not config_tree.bundle_exists("nope")
    assert config_tree.env_exists("main")
    assert not config_tree.env_exists("ghost")


def test_listing(config_tree: ConfigRoot):
    assert config_tree.list_profiles() == ["chem", "ds", "utils"]
    assert config_tree.list_bundles() == ["qsar", "standard"]
    assert config_tree.list_envs() == ["main"]


def test_load_profile(config_tree: ConfigRoot):
    p = config_tree.load_profile("ds")
    assert p.includes == ["numpy", "pandas"]
    assert p.description == "Core data-science stack"
    assert p.tags == ["data", "core"]


def test_load_profile_missing_raises(config_tree: ConfigRoot):
    with pytest.raises(ConfigError):
        config_tree.load_profile("nope")


def test_load_bundle(config_tree: ConfigRoot):
    b = config_tree.load_bundle("standard")
    assert b.includes == ["ds", "chem", "utils"]


def test_load_profile_malformed_yaml_raises(config_tree: ConfigRoot):
    config_tree.profile_path("ds").write_text("includes: [unterminated\n")
    with pytest.raises(ConfigError):
        config_tree.load_profile("ds")


def test_load_profile_empty_file_raises(config_tree: ConfigRoot):
    config_tree.profile_path("ds").write_text("")
    with pytest.raises(ConfigError):
        config_tree.load_profile("ds")


def test_load_profile_unknown_key_raises(config_tree: ConfigRoot):
    config_tree.profile_path("ds").write_text("includes: [numpy]\nbogus: true\n")
    with pytest.raises(ConfigError):
        config_tree.load_profile("ds")


def test_load_env(config_tree: ConfigRoot):
    env = config_tree.load_env("main")
    assert env.python == "3.12"
    assert env.stack == ["@standard"]
    assert env.micromamba == ["graphviz"]
    assert env.channels == ["bioconda"]


def test_load_env_without_channels(config_tree: ConfigRoot):
    config_tree.env_channels_path("main").unlink()
    env = config_tree.load_env("main")
    assert env.channels == []


def test_load_env_missing_stack_raises(config_tree: ConfigRoot):
    (config_tree.root / "envs" / "broken").mkdir()
    with pytest.raises(ConfigError):
        config_tree.load_env("broken")
