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


def _env_root(tmp_path: Path) -> Path:
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


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


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


def test_update_batch_continues_on_failure_and_summarizes(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "update", "alpha", "beta"])
    assert result.exit_code == 1
    assert "Updating alpha" in result.output
    assert "Updating beta" in result.output
    assert "Summary" in result.output
    assert "2 of 2 environment(s) failed." in result.output


def test_update_stop_on_error_aborts_after_first(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "update", "--stop-on-error", "alpha", "beta"]
    )
    assert result.exit_code == 1
    assert "Updating alpha" in result.output
    assert "Updating beta" not in result.output


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


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
    assert result.exit_code == 2
    assert "env" in result.output
    assert "project" in result.output


def test_create_project_help():
    result = CliRunner().invoke(cli, ["create", "project", "--help"])
    assert result.exit_code == 0
    assert "--python" in result.output
    assert "--no-sync" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# resolve / doctor
# ---------------------------------------------------------------------------


def test_resolve_command(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "resolve", "standard"])
    assert result.exit_code == 0
    assert "ds" in result.output


def test_doctor_clean(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "doctor"])
    assert result.exit_code == 0
    assert "No problems detected" in result.output


def test_doctor_reports_missing_dirs(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    result = CliRunner().invoke(cli, ["--root", str(root), "doctor"])
    assert result.exit_code == 0
    assert "Missing" in result.output


# ---------------------------------------------------------------------------
# clean-break guards: the old noun groups must no longer exist
# ---------------------------------------------------------------------------


def test_old_env_group_is_gone():
    result = CliRunner().invoke(cli, ["env", "update", "main"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_old_profile_group_is_gone():
    result = CliRunner().invoke(cli, ["profile", "list"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_old_project_group_is_gone():
    result = CliRunner().invoke(cli, ["project", "init", "ds"])
    assert result.exit_code == 2
    assert "No such command" in result.output
