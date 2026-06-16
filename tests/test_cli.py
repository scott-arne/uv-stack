from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from uv_stack.cli import cli


def _row_cells(output: str, name: str) -> list[str]:
    """Return the trimmed cell values of the rich table row containing ``name``.

    Splits the matching line on the box-drawing column separator so a count
    assertion targets the intended column rather than matching a digit that
    appears incidentally elsewhere in the row (e.g. the ``3.12`` Python cell).
    """
    line = next(line for line in output.splitlines() if name in line)
    return [cell.strip() for cell in line.strip().strip("│").split("│")]


def _seeded_root(tmp_path: Path) -> Path:
    """A config root with the profiles and bundles the CLI tests reference.

    Mirrors the ``config_tree`` fixture: profiles ds/chem/utils and a
    ``standard`` bundle (ds chem utils). ``config init`` no longer seeds any
    profiles or bundles, so the tests author the ones they need.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    cfg.profile_path("ds").write_text("numpy\npandas\n")
    cfg.profile_path("chem").write_text("rdkit\n")
    cfg.profile_path("utils").write_text("rich\n")
    cfg.bundle_path("standard").write_text("ds\nchem\nutils\n")
    return root


def _env_root(tmp_path: Path) -> Path:
    from uv_stack.config import ConfigRoot

    root = _seeded_root(tmp_path)
    cfg = ConfigRoot(root)
    env_dir = cfg.env_dir("main")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    (env_dir / "stack.txt").write_text("@standard\n")
    (env_dir / "micromamba.txt").write_text("graphviz\n")
    (env_dir / "channels.txt").write_text("bioconda\n")
    return root


def _two_failing_envs_root(tmp_path: Path) -> Path:
    """A config root with two envs whose stacks fail at resolution.

    Each stack references an explicit, missing profile, so ``upgrade`` raises a
    ResolutionError during the pure resolve step before any subprocess call —
    keeping the batch-behavior tests hermetic.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
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
    assert "stack, version 0.1.3" in result.output


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------


def test_upgrade_dry_run(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "upgrade", "--dry-run", "main"]
    )
    assert result.exit_code == 0
    assert "compile" in result.output
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    assert cfg.env_requirements_in("main").is_file()
    assert not cfg.env_lock("main").is_file()


def test_upgrade_all_dry_run_targets_every_env(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "--all", "--dry-run"])
    assert result.exit_code == 0
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" in result.output


def test_upgrade_all_does_not_prompt(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    # No stdin supplied: an unconditional confirm() prompt would abort the batch.
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "--all"])
    assert result.exit_code == 1
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" in result.output


def test_upgrade_all_rejects_explicit_names(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "--all", "alpha"])
    assert result.exit_code == 2
    assert "--all cannot be combined" in result.output


def test_upgrade_all_no_envs(tmp_path: Path):
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    init_config_root(ConfigRoot(root))
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "--all"])
    assert result.exit_code == 0
    assert "No environments discovered." in result.output


def test_upgrade_batch_continues_on_failure_and_summarizes(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "alpha", "beta"])
    assert result.exit_code == 1
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" in result.output
    assert "Summary" in result.output
    assert "2 of 2 environment(s) failed." in result.output


def test_upgrade_stop_on_error_aborts_after_first(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "upgrade", "--stop-on-error", "alpha", "beta"]
    )
    assert result.exit_code == 1
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" not in result.output


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_env_passes_create_option(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    captured: dict = {}

    def fake_run_upgrade(config, names, options, *, stop_on_error=False):
        captured["names"] = names
        captured["options"] = options

    monkeypatch.setattr("uv_stack.cli.create._run_upgrade", fake_run_upgrade)
    result = CliRunner().invoke(cli, ["--root", str(root), "create", "env", "main"])
    assert result.exit_code == 0
    assert captured["names"] == ["main"]
    assert captured["options"].create is True
    assert captured["options"].recreate is False


def test_create_env_recreate_passes_recreate_option(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    captured: dict = {}

    def fake_run_upgrade(config, names, options, *, stop_on_error=False):
        captured["options"] = options

    monkeypatch.setattr("uv_stack.cli.create._run_upgrade", fake_run_upgrade)
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
    # The --python help advertises micromamba env-name support.
    assert "micromamba" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_env(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "env"])
    assert result.exit_code == 0
    assert "main" in result.output
    # New columns: Python version and a raw stack-token count.
    assert "Python" in result.output
    assert "Stack" in result.output
    assert "3.12" in result.output
    cells = _row_cells(result.output, "main")
    assert cells[-1] == "1"  # @standard -> one stack token


def test_list_profile(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "profile"])
    assert result.exit_code == 0
    assert "ds" in result.output
    # The profiles directory is indicated in the output.
    assert "profiles" in result.output
    # New column: raw package count.
    assert "Packages" in result.output
    cells = _row_cells(result.output, "ds")
    assert cells[-1] == "2"  # ds -> numpy, pandas


def test_list_bundle(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "bundle"])
    assert result.exit_code == 0
    assert "standard" in result.output
    # New column: raw token count.
    assert "Tokens" in result.output
    cells = _row_cells(result.output, "standard")
    assert cells[-1] == "3"  # standard -> ds, chem, utils


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
    assert "Channels:" in result.output
    assert "conda-forge" in result.output
    assert "bioconda" in result.output


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


def test_resolve_classifies_without_expanding(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "resolve", "standard", "ds", "numpy"]
    )
    assert result.exit_code == 0
    lines = result.output.split()
    assert lines == ["bundle:standard", "profile:ds", "package:numpy"]


def test_resolve_full_expands_to_flat_packages(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "resolve", "--full", "standard"]
    )
    assert result.exit_code == 0
    # standard -> ds (numpy, pandas), chem (rdkit), utils (rich); no prefixes.
    lines = result.output.split()
    assert lines == ["numpy", "pandas", "rdkit", "rich"]


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
    result = CliRunner().invoke(cli, ["env", "upgrade", "main"])
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
