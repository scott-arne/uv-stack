from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError


def test_discover_precedence_explicit_over_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("UV_STACK_ROOT", raising=False)
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "from-env"))
    cfg = ConfigRoot.discover(root=tmp_path / "explicit")
    assert cfg.root == (tmp_path / "explicit")


def test_discover_uses_env_when_no_explicit(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("UV_STACK_ROOT", raising=False)
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "from-env"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "from-env")


def test_discover_default(monkeypatch):
    monkeypatch.delenv("UV_STACK_ROOT", raising=False)
    monkeypatch.delenv("UV_ENV_ROOT", raising=False)
    cfg = ConfigRoot.discover()
    assert cfg.root == (Path.home() / ".config" / "python-envs")


def test_discover_prefers_uv_stack_root(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("UV_ENV_ROOT", raising=False)
    monkeypatch.setenv("UV_STACK_ROOT", str(tmp_path / "stack-root"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "stack-root")


def test_discover_uv_stack_root_wins_over_legacy(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("UV_STACK_ROOT", str(tmp_path / "stack-root"))
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "legacy-root"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "stack-root")


def test_discover_falls_back_to_legacy_uv_env_root(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("UV_STACK_ROOT", raising=False)
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "legacy-root"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "legacy-root")


def test_discover_empty_uv_stack_root_falls_through_to_legacy(monkeypatch, tmp_path: Path):
    """An empty UV_STACK_ROOT is treated as unset, matching the existing
    ``if env:`` guard's handling of an empty UV_ENV_ROOT."""
    monkeypatch.setenv("UV_STACK_ROOT", "")
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "legacy-root"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "legacy-root")


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


def test_load_bundle_malformed_yaml_raises(config_tree: ConfigRoot):
    config_tree.bundle_path("standard").write_text("includes: [unterminated\n")
    with pytest.raises(ConfigError):
        config_tree.load_bundle("standard")


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


def test_load_env_missing_hint_mentions_create_env_tokens(tmp_path):
    from uv_stack.config import ConfigRoot
    from uv_stack.errors import ConfigError

    cfg = ConfigRoot(tmp_path)
    try:
        cfg.load_env("ghost")
    except ConfigError as error:
        assert error.hint is not None
        assert "pass TOKENS: stack create env ghost TOKENS..." in error.hint
    else:
        raise AssertionError("expected ConfigError")
