from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from uvstack.cli import cli


def _seeded_root(tmp_path: Path) -> Path:
    from uvstack.config import ConfigRoot
    from uvstack.operations.init import seed_defaults

    root = tmp_path / "python-envs"
    seed_defaults(ConfigRoot(root))
    return root


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_profile_list(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "profile", "list"])
    assert result.exit_code == 0
    assert "ds" in result.output
    assert "chem" in result.output


def test_resolve_command(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "resolve", "standard"])
    assert result.exit_code == 0
    assert "ds" in result.output


def test_config_error_renders_panel_and_exit_1(tmp_path: Path):
    root = _seeded_root(tmp_path)
    # No env named 'ghost' exists -> ConfigError from load_env via show.
    result = CliRunner().invoke(cli, ["--root", str(root), "env", "show", "ghost"])
    assert result.exit_code == 1
    assert "ghost" in result.output


def test_bundle_show(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "bundle", "show", "standard"])
    assert result.exit_code == 0
    assert "ds" in result.output
