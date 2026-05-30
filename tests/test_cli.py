from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from uv_stack.cli import cli


def _seeded_root(tmp_path: Path) -> Path:
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import seed_defaults

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


def _env_root(tmp_path: Path):
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import seed_defaults

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    seed_defaults(cfg)
    env_dir = cfg.env_dir("main")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    (env_dir / "stack.txt").write_text("@standard\n")
    (env_dir / "micromamba.txt").write_text("graphviz\n")
    return root


def test_env_list(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "env", "list"])
    assert result.exit_code == 0
    assert "main" in result.output


def test_env_update_dry_run(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "env", "update", "--dry-run", "main"]
    )
    assert result.exit_code == 0
    assert "compile" in result.output
    # Dry run wrote requirements.in but not the lock.
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    assert cfg.env_requirements_in("main").is_file()
    assert not cfg.env_lock("main").is_file()


def test_project_init_dry_invocation():
    # Verify the command wires up and --help lists its options.
    result = CliRunner().invoke(cli, ["project", "init", "--help"])
    assert result.exit_code == 0
    assert "--python" in result.output
    assert "--no-sync" in result.output


def _two_failing_envs_root(tmp_path: Path) -> Path:
    """A config root with two envs whose stacks fail at resolution.

    Each stack references an explicit, missing profile, so ``update`` raises a
    ResolutionError during the pure resolve step before any subprocess call —
    keeping the batch-behavior tests hermetic.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import seed_defaults

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    seed_defaults(cfg)
    for name in ("alpha", "beta"):
        env_dir = cfg.env_dir(name)
        env_dir.mkdir(parents=True)
        (env_dir / "python.txt").write_text("3.12\n")
        (env_dir / "stack.txt").write_text("profile:ghost\n")
    return root


def test_env_update_batch_continues_on_failure_and_summarizes(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "env", "update", "alpha", "beta"]
    )
    # Non-empty failures exit 1.
    assert result.exit_code == 1
    # By default the batch continues: the second env was still attempted.
    assert "Updating alpha" in result.output
    assert "Updating beta" in result.output
    # A summary marks both as failed.
    assert "Summary" in result.output
    assert "2 of 2 environment(s) failed." in result.output


def test_env_update_stop_on_error_aborts_after_first(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "env", "update", "--stop-on-error", "alpha", "beta"]
    )
    assert result.exit_code == 1
    # The batch aborted at the first failure: beta was never attempted.
    assert "Updating alpha" in result.output
    assert "Updating beta" not in result.output


def test_update_dry_run(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "update", "--dry-run", "main"]
    )
    assert result.exit_code == 0
    assert "compile" in result.output
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    assert cfg.env_requirements_in("main").is_file()
    assert not cfg.env_lock("main").is_file()


def test_create_env_passes_create_option(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    captured: dict = {}

    def fake_run_update(config, names, options, *, stop_on_error=False):
        captured["names"] = names
        captured["options"] = options

    monkeypatch.setattr("uv_stack.cli.create._run_update", fake_run_update)
    result = CliRunner().invoke(cli, ["--root", str(root), "create", "env", "main"])
    assert result.exit_code == 0
    assert captured["names"] == ["main"]
    assert captured["options"].create is True
    assert captured["options"].recreate is False


def test_create_env_recreate_passes_recreate_option(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    captured: dict = {}

    def fake_run_update(config, names, options, *, stop_on_error=False):
        captured["options"] = options

    monkeypatch.setattr("uv_stack.cli.create._run_update", fake_run_update)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "create", "env", "main", "--recreate"]
    )
    assert result.exit_code == 0
    assert captured["options"].recreate is True
    assert captured["options"].create is False


def test_create_no_subcommand_shows_help():
    result = CliRunner().invoke(cli, ["create"])
    # Click exits 2 and lists the subcommands when a group is invoked bare.
    assert result.exit_code == 2
    assert "env" in result.output
    assert "project" in result.output


def test_create_project_help():
    result = CliRunner().invoke(cli, ["create", "project", "--help"])
    assert result.exit_code == 0
    assert "--python" in result.output
    assert "--no-sync" in result.output


def test_list_env(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "env"])
    assert result.exit_code == 0
    assert "main" in result.output


def test_list_profile(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "profile"])
    assert result.exit_code == 0
    assert "ds" in result.output


def test_list_bundle(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "bundle"])
    assert result.exit_code == 0
    assert "standard" in result.output


def test_list_bad_kind(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "widget"])
    assert result.exit_code == 2
    assert "widget" in result.output


def test_show_env(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "env", "main"])
    assert result.exit_code == 0
    assert "Environment: main" in result.output


def test_show_env_defaults_to_main(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "env"])
    assert result.exit_code == 0
    assert "Environment: main" in result.output


def test_show_profile(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "profile", "ds"])
    assert result.exit_code == 0
    assert "Profile: ds" in result.output


def test_show_bundle(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "bundle", "standard"])
    assert result.exit_code == 0
    assert "Bundle: standard" in result.output


def test_show_profile_requires_name(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "profile"])
    assert result.exit_code == 2
    assert "NAME" in result.output


def test_show_bad_kind(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "widget", "x"])
    assert result.exit_code == 2
    assert "widget" in result.output


def test_show_missing_env_errors(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "env", "ghost"])
    assert result.exit_code == 1
    assert "ghost" in result.output
