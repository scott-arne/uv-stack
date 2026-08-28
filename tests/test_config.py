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


def test_discover_explicit_root_wins_over_both_env_vars(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("UV_STACK_ROOT", str(tmp_path / "stack-root"))
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "env-root"))
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
    assert (
        config_tree.env_requirements_lock("main")
        == root / "envs" / "main" / "requirements.lock.txt"
    )
    assert config_tree.locks_dir == root / ".locks"
    assert config_tree.stem_lock_path("x") == root / ".locks" / "stem-x.lock"
    assert config_tree.env_lock_path("myenv") == root / ".locks" / "env-myenv.lock"


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


def test_load_env_hint_shell_quotes_the_name(tmp_path: Path):
    """The hint is a runnable command, so the name must survive a paste intact."""
    root = ConfigRoot(tmp_path / "python-envs")
    hostile = "a b; rm -rf /"
    with pytest.raises(ConfigError) as excinfo:
        root.load_env(hostile)
    assert "stack create env 'a b; rm -rf /' TOKENS..." in excinfo.value.hint


def test_load_env_hint_leaves_a_plain_name_unquoted(tmp_path: Path):
    root = ConfigRoot(tmp_path / "python-envs")
    with pytest.raises(ConfigError) as excinfo:
        root.load_env("ghost")
    assert "stack create env ghost TOKENS..." in excinfo.value.hint


def test_load_env_hint_prefixes_leading_dash_name(tmp_path: Path):
    """A name starting with - must be prefixed with -- to remain positional."""
    root = ConfigRoot(tmp_path / "python-envs")
    with pytest.raises(ConfigError) as excinfo:
        root.load_env("--recreate")
    assert "stack create env -- --recreate TOKENS..." in excinfo.value.hint


def test_probe_lock_path_cannot_collide_with_a_user_name(config_tree):
    """The stem and env locks are prefixed, so a fixed bare name is safe."""
    assert config_tree.probe_lock_path() == config_tree.locks_dir / "probe.lock"
    assert config_tree.probe_lock_path() != config_tree.stem_lock_path("probe")
    assert config_tree.probe_lock_path() != config_tree.env_lock_path("probe")


def test_editor_path_is_root_scoped(tmp_path: Path):
    # editor.txt is a property of the config root, not of any one env.
    assert ConfigRoot(tmp_path).editor_path() == tmp_path / "editor.txt"


def test_default_editor_reads_first_clean_line(tmp_path: Path):
    (tmp_path / "editor.txt").write_text("# my editor\nvim -f\n", encoding="utf-8")
    assert ConfigRoot(tmp_path).default_editor() == "vim -f"


def test_default_editor_is_none_when_absent(tmp_path: Path):
    assert ConfigRoot(tmp_path).default_editor() is None


def test_default_editor_is_none_when_blank(tmp_path: Path):
    (tmp_path / "editor.txt").write_text("   \n# nothing here\n", encoding="utf-8")
    assert ConfigRoot(tmp_path).default_editor() is None


def test_default_editor_reports_undecodable_file_as_config_error(tmp_path: Path):
    # UnicodeDecodeError is a ValueError, so unwrapped it escapes the CLI edge,
    # which renders only UvStackError and OSError.
    (tmp_path / "editor.txt").write_bytes(b"\xff\xfe vim")
    with pytest.raises(ConfigError) as excinfo:
        ConfigRoot(tmp_path).default_editor()
    assert "not valid UTF-8" in excinfo.value.message
    assert str(tmp_path / "editor.txt") in excinfo.value.message


def test_require_env_accepts_an_env_with_a_stack_file(config_tree: ConfigRoot):
    config_tree.require_env("main")


def test_require_env_reports_a_missing_stack_file(tmp_path: Path):
    with pytest.raises(ConfigError) as excinfo:
        ConfigRoot(tmp_path).require_env("ghost")
    assert "Missing stack file for env 'ghost'" in excinfo.value.message
    assert "stack create env ghost TOKENS..." in (excinfo.value.hint or "")
