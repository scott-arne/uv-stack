from __future__ import annotations

from pathlib import Path

import rich_click
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


def _combined_output(result) -> str:
    """stdout plus stderr, tolerant of click versions that separate them."""
    try:
        return result.output + result.stderr
    except (ValueError, AttributeError):
        return result.output


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
    cfg.profile_path("ds").write_text(
        "description: Core data-science stack with numpy scipy pandas polars and friends\n"
        "tags: [data, core]\n"
        "includes:\n  - numpy\n  - pandas\n"
    )
    cfg.profile_path("chem").write_text(
        "description: Cheminformatics\ntags: [chem, bio]\nincludes:\n  - rdkit\n"
    )
    cfg.profile_path("utils").write_text("includes:\n  - rich\n")
    cfg.bundle_path("standard").write_text(
        "description: Everything for daily work\n"
        "tags: [core]\n"
        "includes:\n  - ds\n  - chem\n  - utils\n"
    )
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
    assert "stack, version 0.2.0" in result.output


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
    # Both envs fail at resolution, so dry-run exits nonzero.
    assert result.exit_code == 1
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
    # Each ✗ row is annotated with a one-line reason drawn from the error.
    summary = result.output.split("Summary", 1)[1]
    assert "Missing profile" in summary


def test_upgrade_stop_on_error_aborts_after_first(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "upgrade", "--stop-on-error", "alpha", "beta"]
    )
    assert result.exit_code == 1
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" not in result.output


def test_upgrade_dry_run_strict_exits_nonzero_on_failure(tmp_path: Path):
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    env_dir = cfg.env_dir("test")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    # "numpyy" is a near-miss typo that will fail strict mode.
    (env_dir / "stack.txt").write_text("numpyy\n")
    result = CliRunner().invoke(
        cli, ["--root", str(root), "upgrade", "--dry-run", "--strict", "test"]
    )
    assert result.exit_code == 1


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
    assert "Python" in result.output
    assert "Stack" in result.output
    assert "3.12" in result.output
    cells = _row_cells(result.output, "main")
    assert cells[-1] == "1"  # @standard -> one stack token


def test_list_profile_columns(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "list", "profile"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0
    assert "Packages" in result.output
    assert "Tags" in result.output
    assert "Description" in result.output
    cells = _row_cells(result.output, "chem")
    assert cells[0] == "chem"
    assert cells[1] == "1"
    assert "chem" in cells[2] and "bio" in cells[2]
    assert cells[3] == "Cheminformatics"


def test_list_profile_truncates_long_description(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "list", "profile"], env={"COLUMNS": "200"}
    )
    cells = _row_cells(result.output, "ds")
    assert cells[-1].endswith("…")
    assert len(cells[-1]) <= 40


def test_list_bundle_renames_count_to_entries(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "list", "bundle"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0
    assert "Entries" in result.output
    assert "Tokens" not in result.output
    cells = _row_cells(result.output, "standard")
    assert cells[0] == "standard"
    assert cells[1] == "3"


def test_list_profile_tag_filter(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "list", "profile", "--tag", "data"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "ds" in result.output
    assert "rdkit" not in result.output
    assert "chem" not in result.output
    assert "utils" not in result.output


def test_list_profile_tag_filter_or(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli,
        ["--root", str(root), "list", "profile", "--tag", "data", "--tag", "bio"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "ds" in result.output
    assert "chem" in result.output
    assert "utils" not in result.output


def test_list_env_tag_is_usage_error(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "env", "--tag", "x"])
    assert result.exit_code == 2
    assert "tag" in result.output.lower()


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


def test_resolve_prints_near_miss_warning(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "resolve", "standrd"])
    assert result.exit_code == 0
    assert "package:standrd" in result.output
    assert "did you mean 'standard'" in _combined_output(result)


def test_resolve_strict_fails_on_bare_literal(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "resolve", "--strict", "numpyy"]
    )
    assert result.exit_code == 1


def test_resolve_full_strict_and_warnings(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "resolve", "--full", "standrd"]
    )
    assert result.exit_code == 0
    assert "did you mean 'standard'" in _combined_output(result)


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


def test_create_profile_writes_yaml(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "create", "profile", "viz",
         "matplotlib", "seaborn", "--description", "Plotting", "--tag", "viz"],
    )
    assert result.exit_code == 0
    assert "Wrote" in result.output
    from uv_stack.config import ConfigRoot

    prof = ConfigRoot(root).load_profile("viz")
    assert prof.includes == ["matplotlib", "seaborn"]
    assert prof.description == "Plotting"
    assert prof.tags == ["viz"]


def test_create_profile_requires_packages(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "create", "profile", "viz"])
    assert result.exit_code == 2


def test_create_profile_refuses_overwrite(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "ds", "numpy"]
    )
    assert result.exit_code == 1


def test_create_profile_refuses_collision_with_bundle(tmp_path: Path):
    """Creating a profile named 'standard' (existing bundle) exits with code 1."""
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "standard", "numpy"]
    )
    assert result.exit_code == 1
    assert "would shadow the existing bundle" in _combined_output(result)
    # Verify the file was not created.
    from uv_stack.config import ConfigRoot
    assert not ConfigRoot(root).profile_path("standard").exists()


def test_create_bundle_writes_yaml_and_warns(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "create", "bundle", "daily", "ds", "standrd"],
    )
    assert result.exit_code == 0
    assert "did you mean 'standard'" in _combined_output(result)
    from uv_stack.config import ConfigRoot

    assert ConfigRoot(root).load_bundle("daily").includes == ["ds", "standrd"]


def test_create_bundle_strict_rejects_typo(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "create", "bundle", "daily", "--strict", "numpyy"],
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot

    assert not ConfigRoot(root).bundle_exists("daily")


def test_create_bundle_rejects_missing_explicit_profile(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "daily", "profile:ghost"]
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot

    assert not ConfigRoot(root).bundle_exists("daily")


def test_create_bundle_rejects_missing_explicit_bundle(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "daily", "@ghost"]
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot

    assert not ConfigRoot(root).bundle_exists("daily")


def test_create_bundle_strict_accepts_existing_bundle_with_bare_package(tmp_path: Path):
    """Strict mode applies to direct tokens only; existing bundle with bare package is OK."""
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    # Create bundle 'web' containing bare 'httpx'.
    cfg.bundle_path("web").write_text("includes:\n  - httpx\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "app", "--strict", "web"]
    )
    assert result.exit_code == 0
    assert cfg.bundle_exists("app")
    assert cfg.load_bundle("app").includes == ["web"]


def test_create_bundle_strict_rejects_direct_bare_near_miss(tmp_path: Path):
    """Strict mode rejects direct bare near-miss even if existing bundles are fine."""
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.bundle_path("web").write_text("includes:\n  - httpx\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "app", "--strict", "numpyy"]
    )
    assert result.exit_code == 1
    assert not cfg.bundle_exists("app")


def test_create_env_with_tokens_scaffolds_and_builds(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    calls: list[tuple[list[str], object]] = []
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade",
        lambda config, names, options, **kw: calls.append((names, options)),
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "create", "env", "fresh",
         "@standard", "httpx", "--python", "3.13"],
    )
    assert result.exit_code == 0
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    assert cfg.env_stack_path("fresh").read_text() == "@standard\nhttpx\n"
    assert cfg.env_python_path("fresh").read_text() == "3.13\n"
    assert calls and calls[0][0] == ["fresh"]
    assert calls[0][1].create is True
    assert "micromamba activate fresh" in result.output


def test_create_env_python_without_tokens_is_usage_error(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "--python", "3.13"]
    )
    assert result.exit_code == 2


def test_create_env_tokens_refuse_existing_stack(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade", lambda *a, **kw: None
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "main", "ds"]
    )
    assert result.exit_code == 1


def test_create_env_without_tokens_still_works(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade",
        lambda config, names, options, **kw: calls.append(names),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "create", "env", "main"])
    assert result.exit_code == 0
    assert calls == [["main"]]
    assert "micromamba activate main" in result.output


def test_create_env_strict_typo_exits_before_scaffolding(tmp_path: Path, monkeypatch):
    """Strict failure pre-validates so stack.txt is never written."""
    root = _seeded_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade", lambda *a, **kw: None
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "--strict", "numpyy"]
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot
    # stack.txt must not exist; the pre-validation failed before any write.
    assert not ConfigRoot(root).env_stack_path("fresh").exists()


def test_create_env_missing_explicit_profile_exits_before_scaffolding(tmp_path: Path, monkeypatch):
    """Missing explicit reference pre-validates so stack.txt is never written."""
    root = _seeded_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade", lambda *a, **kw: None
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "profile:ghost"]
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot
    assert not ConfigRoot(root).env_stack_path("fresh").exists()


def test_create_env_python_empty_string_is_usage_error(tmp_path: Path):
    """--python '' with tokens exits 2 (UsageError) before writing anything."""
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "ds", "--python", ""]
    )
    assert result.exit_code == 2
    from uv_stack.config import ConfigRoot
    assert not ConfigRoot(root).env_stack_path("fresh").exists()


def test_create_env_python_empty_string_without_tokens_is_usage_error(tmp_path: Path):
    """--python '' without tokens also exits 2."""
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "--python", ""]
    )
    assert result.exit_code == 2


def test_create_env_rejects_malformed_profile_before_scaffolding(
    tmp_path: Path, monkeypatch
):
    """Malformed profile YAML pre-validates so stack.txt is never written."""
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.profile_path("broken").write_text("includes: {not: [valid\n")
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "broken"]
    )
    assert result.exit_code == 1
    assert not cfg.env_stack_path("fresh").exists()


# ---------------------------------------------------------------------------
# activation hints
# ---------------------------------------------------------------------------


def test_activation_hint_shell_quotes_names(capsys):
    from uv_stack.cli._render import print_activation_hint

    print_activation_hint("safe; touch pwn")
    out = capsys.readouterr().out
    assert "micromamba activate 'safe; touch pwn'" in out
    assert "-n 'safe; touch pwn'" in out


def test_activation_hint_leaves_safe_names_unquoted(capsys):
    from uv_stack.cli._render import print_activation_hint

    print_activation_hint("main")
    out = capsys.readouterr().out
    assert "micromamba activate main" in out
    assert "-n main" in out


# ---------------------------------------------------------------------------
# doctor --fix and JSON
# ---------------------------------------------------------------------------


def test_doctor_fix_repairs_and_reports(tmp_path: Path):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    (cfg.profiles_dir / "old.in").write_text("numpy\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix"])
    assert result.exit_code == 0
    assert "fixed:" in result.output
    assert "No problems detected." in result.output
    assert cfg.profile_exists("old")


def test_doctor_fix_iterates_to_fixed_point(tmp_path: Path):
    # A completely absent root: fixing the root reveals the missing subdirectories,
    # which are then fixed in subsequent rounds.
    root = tmp_path / "python-envs"
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix"])
    assert result.exit_code == 0
    # Expect multiple "fixed:" lines (one for root, one for each subdir).
    assert result.output.count("fixed:") >= 4  # root + profiles + bundles + envs
    assert "No problems detected." in result.output
    # Verify all directories were created.
    from uv_stack.config import ConfigRoot
    cfg = ConfigRoot(root)
    assert cfg.root.is_dir()
    assert cfg.profiles_dir.is_dir()
    assert cfg.bundles_dir.is_dir()
    assert cfg.envs_dir.is_dir()


def test_doctor_fix_json_iterates_to_fixed_point(tmp_path: Path):
    import json
    # Same as above, but verify JSON output structure.
    root = tmp_path / "python-envs"
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    # Expect multiple actions: missing-root, then missing-dir for each subdir.
    action_kinds = [a["kind"] for a in payload["actions"]]
    assert "missing-root" in action_kinds
    assert action_kinds.count("missing-dir") >= 3  # profiles, bundles, envs
    assert all(a["applied"] for a in payload["actions"])
    assert payload["remaining"] == []


def test_doctor_fix_exit_1_when_errors_remain(tmp_path: Path, monkeypatch):
    import json
    # A root whose parent is read-only cannot be created; simulate by pointing
    # repair at a path under a file (mkdir raises, so the error finding stays).
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(blocker / "python-envs"), "doctor", "--fix", "--json"]
    )
    assert result.exit_code == 1
    # Exit 1 must be from graceful failure reporting, not an unhandled OSError.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    payload = json.loads(result.output)
    # Verify the repair handler converted the mkdir OSError into a skipped action.
    assert payload["actions"][0]["applied"] is False
    assert payload["actions"][0]["reason"]  # non-empty reason
    assert payload["remaining"][0]["kind"] == "missing-root"


def test_doctor_json_lists_findings(tmp_path: Path):
    import json

    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(tmp_path / "nope"), "doctor", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["kind"] == "missing-root"
    assert payload[0]["level"] == "error"


def test_doctor_fix_json_shape(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    (cfg.bundles_dir / "old.bundle").write_text("ds\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload.keys()) == {"actions", "remaining"}
    action = payload["actions"][0]
    assert action["kind"] == "legacy-bundle"
    assert action["applied"] is True
    assert action["reason"] is None
    assert payload["remaining"] == []


# ---------------------------------------------------------------------------
# show env config and interpreter
# ---------------------------------------------------------------------------


class _FakeProbeRunner:
    """Stands in for SubprocessRunner in probe-only CLI paths."""

    def __init__(self, stdout: str = "/envs/main/bin/python\n", returncode: int = 0):
        self._stdout = stdout
        self._returncode = returncode

    def run(self, command, *, capture=False, check=True):
        from uv_stack.runner import CommandResult

        return CommandResult(returncode=self._returncode, stdout=self._stdout)


def test_show_env_prints_config_and_interpreter(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.show.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "show", "env", "main"])
    assert result.exit_code == 0
    assert f"Config: {root}/envs/main" in result.output
    assert "Interpreter: /envs/main/bin/python" in result.output


def test_show_env_interpreter_not_created(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.show.SubprocessRunner",
        lambda: _FakeProbeRunner(stdout="", returncode=1),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "show", "env", "main"])
    assert result.exit_code == 0
    assert "Interpreter: not created (run 'stack create env main')" in result.output


def test_status_table(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.status_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "status"])
    assert result.exit_code == 0
    cells = _row_cells(result.output, "main")
    assert cells[0] == "main"
    assert cells[1] == "3.12"
    assert cells[2] == "yes"          # Created
    assert cells[3] == "no"           # Lock (never upgraded here)
    assert cells[4] == "never built"  # State


def test_status_config_error_row_prints_message(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    from uv_stack.config import ConfigRoot

    ConfigRoot(root).env_stack_path("main").unlink()
    monkeypatch.setattr(
        "uv_stack.cli.status_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "status", "main"])
    assert result.exit_code == 0
    assert "config error" in result.output
    assert "main: Missing stack file" in result.output


def test_status_json(tmp_path: Path, monkeypatch):
    import json

    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.status_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "status", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == [
        {
            "name": "main",
            "python": "3.12",
            "created": True,
            "lock": False,
            "state": "never built",
            "message": None,
        }
    ]


def test_list_env_json(tmp_path: Path):
    import json

    root = _env_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "env", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {"name": "main", "python": "3.12", "stack": ["@standard"]}
    ]


def test_list_profile_json_full_lists(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "list", "profile", "--json", "--tag", "chem"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {
            "name": "chem",
            "packages": ["rdkit"],
            "tags": ["chem", "bio"],
            "description": "Cheminformatics",
        }
    ]


def test_show_env_json_has_no_interpreter_probe(tmp_path: Path, monkeypatch):
    import json

    root = _env_root(tmp_path)

    def _boom():
        raise AssertionError("JSON mode must not construct a runner")

    monkeypatch.setattr("uv_stack.cli.show.SubprocessRunner", _boom)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "show", "env", "main", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "main"
    assert payload["config_dir"] == str(root / "envs" / "main")
    assert payload["channels"][0] == "conda-forge"
    assert payload["profiles"] == ["ds", "chem", "utils"]


def test_show_env_surfaces_resolver_warnings_json(tmp_path: Path, monkeypatch):
    import json

    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    env_dir = cfg.env_dir("main")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    # Use a near-miss typo (standrd -> standard) to trigger a warning.
    (env_dir / "stack.txt").write_text("standrd\n")

    def _boom():
        raise AssertionError("JSON mode must not construct a runner")

    monkeypatch.setattr("uv_stack.cli.show.SubprocessRunner", _boom)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "show", "env", "main", "--json"]
    )
    assert result.exit_code == 0
    # Extract JSON from output (warnings may appear before/after JSON in combined output).
    # Try parsing lines until we find a valid JSON object.
    combined = _combined_output(result)
    lines = combined.split('\n')
    json_lines = []
    in_json = False
    for line in lines:
        if line.startswith('{'):
            in_json = True
        if in_json:
            json_lines.append(line)
        if in_json and line.strip() == '}':
            break
    json_text = '\n'.join(json_lines)
    payload = json.loads(json_text)
    assert payload["name"] == "main"
    # Verify the warning is in the combined output.
    assert "did you mean 'standard'" in combined


def test_show_profile_json(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "show", "profile", "utils", "--json"]
    )
    assert json.loads(result.output) == {
        "name": "utils",
        "description": None,
        "tags": [],
        "includes": ["rich"],
    }


def test_resolve_json(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "resolve", "--json", "standard", "ds", "numpy"]
    )
    assert json.loads(result.output) == [
        {"input": "standard", "kind": "bundle", "name": "standard"},
        {"input": "ds", "kind": "profile", "name": "ds"},
        {"input": "numpy", "kind": "package", "name": "numpy"},
    ]


def test_resolve_full_json(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "resolve", "--full", "--json", "standard"]
    )
    assert json.loads(result.output) == {
        "packages": ["numpy", "pandas", "rdkit", "rich"]
    }


def test_list_bundle_json_full_lists(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "bundle", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {
            "name": "standard",
            "entries": ["ds", "chem", "utils"],
            "tags": ["core"],
            "description": "Everything for daily work",
        }
    ]


def test_show_bundle_json(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "show", "bundle", "standard", "--json"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "name": "standard",
        "description": "Everything for daily work",
        "tags": ["core"],
        "includes": ["ds", "chem", "utils"],
    }


def test_list_env_empty_hint(tmp_path: Path):
    root = _seeded_root(tmp_path)  # profiles/bundles but no envs
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "env"])
    assert result.exit_code == 0
    assert "No environments yet" in result.output
    assert "stack create env" in result.output


def test_list_profile_empty_hint(tmp_path: Path):
    root = tmp_path / "python-envs"
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    init_config_root(ConfigRoot(root))
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "profile"])
    assert result.exit_code == 0
    assert "No profiles yet" in result.output


def test_list_tag_filter_no_match_hint(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "list", "profile", "--tag", "nonexistent"]
    )
    assert result.exit_code == 0
    assert "No profiles match tags: nonexistent." in result.output


def test_list_env_empty_json_is_empty_list(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "env", "--json"])
    assert json.loads(result.output) == []


def test_init_yes_on_empty_root_seeds_and_builds(tmp_path: Path, monkeypatch):
    root = tmp_path / "python-envs"
    calls: list[tuple[list[str], object]] = []
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda config, names, options, **kw: calls.append((names, options)),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "init", "--yes"])
    assert result.exit_code == 0
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    assert cfg.profile_exists("starter")
    starter_text = cfg.profile_path("starter").read_text()
    assert starter_text.startswith("# A profile is a reusable")
    assert cfg.env_stack_path("main").read_text() == "starter\n"
    assert cfg.env_python_path("main").read_text() == "3.12\n"
    assert calls and calls[0][0] == ["main"] and calls[0][1].create is True
    assert "micromamba activate main" in result.output


def test_init_yes_is_idempotent(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)  # profiles and env 'main' already exist
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "init", "--yes"])
    assert result.exit_code == 0
    from uv_stack.config import ConfigRoot

    # Nothing new was seeded: profiles existed, so no starter profile.
    assert not ConfigRoot(root).profile_exists("starter")
    assert "Config root:" in result.output


def test_init_interactive_decline_build_prints_next_step(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "python-envs"
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    # Prompts: seed starter? y | create env? y | name [main] | tokens [starter]
    # | python [3.12] | build now? n
    result = runner.invoke(
        cli,
        ["--root", str(root), "init"],
        input="y\ny\n\n\n\nn\n",
    )
    assert result.exit_code == 0
    assert "Build it with: stack create env main" in result.output
    assert "micromamba activate" not in result.output


def test_init_rejects_bad_tokens_before_writing(tmp_path: Path, monkeypatch):
    root = tmp_path / "python-envs"
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    # Prompts: seed starter? y | create env? y | name [main] | tokens: profile:ghost
    # | python [3.12] — command errors at pre-validation before writing.
    result = runner.invoke(
        cli,
        ["--root", str(root), "init"],
        input="y\ny\n\nprofile:ghost\n\n",
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot

    assert not ConfigRoot(root).env_stack_path("main").exists()


def test_init_decline_build_still_surfaces_warnings(tmp_path: Path, monkeypatch):
    root = tmp_path / "python-envs"
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    # Prompts: seed starter? y | create env? y | name [main] | tokens: startr
    # (near-miss of the seeded 'starter' profile) | python [3.12] | build now? n
    result = runner.invoke(
        cli,
        ["--root", str(root), "init"],
        input="y\ny\n\nstartr\n\nn\n",
    )
    assert result.exit_code == 0
    assert result.output.count("did you mean 'starter'") == 1
    assert "Build it with: stack create env main" in result.output


# ---------------------------------------------------------------------------
# completion
# ---------------------------------------------------------------------------


def test_completion_zsh_prints_script(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "zsh"])
    assert result.exit_code == 0
    assert result.output.startswith('# Add to ~/.zshrc: eval "$(stack completion zsh)"')
    assert "_STACK_COMPLETE" in result.output


def test_completion_bash_prints_script():
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "bash"])
    assert result.exit_code == 0
    assert result.output.startswith("# Requires bash >= 4.4.")
    assert "_STACK_COMPLETE" in result.output


def test_completion_fish_prints_script():
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "fish"])
    assert result.exit_code == 0
    assert result.output.startswith("# Add to ~/.config/fish/config.fish")
    assert "_STACK_COMPLETE" in result.output


def test_completion_requires_shell_argument():
    runner = CliRunner()
    result = runner.invoke(cli, ["completion"])
    assert result.exit_code == 2


def test_complete_env_names(tmp_path: Path):
    import click as _click

    from uv_stack.cli._complete import complete_env_names

    root = _env_root(tmp_path)
    ctx = _click.Context(cli)
    ctx.params = {"root": str(root)}
    assert complete_env_names(ctx, None, "") == ["main"]
    assert complete_env_names(ctx, None, "ma") == ["main"]
    assert complete_env_names(ctx, None, "zz") == []


def test_complete_env_names_survives_bad_root():
    import click as _click

    from uv_stack.cli._complete import complete_env_names

    ctx = _click.Context(cli)
    ctx.params = {}
    # No root param at all: fall back to discovery, never raise.
    assert isinstance(complete_env_names(ctx, None, "zzz-no-such"), list)


def test_complete_show_names_dispatches_on_kind(tmp_path: Path):
    import click as _click

    from uv_stack.cli._complete import complete_show_names

    root = _seeded_root(tmp_path)
    ctx = _click.Context(cli)
    ctx.params = {"root": str(root), "kind": "profile"}
    assert complete_show_names(ctx, None, "d") == ["ds"]


def test_help_contains_no_rest_double_backticks():
    runner = CliRunner()
    for args in (
        ["--help"],
        ["upgrade", "--help"],
        ["create", "--help"],
        ["create", "env", "--help"],
        ["create", "project", "--help"],
        ["create", "profile", "--help"],
        ["create", "bundle", "--help"],
        ["list", "--help"],
        ["show", "--help"],
        ["resolve", "--help"],
        ["status", "--help"],
        ["doctor", "--help"],
        ["config", "--help"],
        ["init", "--help"],
        ["completion", "--help"],
    ):
        result = runner.invoke(cli, args)
        assert result.exit_code == 0
        assert "``" not in result.output, f"double backticks in: {args}"


def test_create_project_no_track_flag(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    captured: dict[str, object] = {}

    def _fake_init(config, runner, tokens, options, *, cwd):
        captured["track"] = options.track
        return []

    monkeypatch.setattr("uv_stack.cli.create.init_project", _fake_init)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "project", "ds", "--no-track"]
    )
    assert result.exit_code == 0
    assert captured["track"] is False


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


def _tracked_project_dir(tmp_path: Path) -> Path:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas", "rdkit", "rich"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["standard"]\n'
        'applied = ["numpy", "pandas", "rdkit", "rich"]\n'
    )
    return project_dir


def test_refresh_outside_project_fails_with_hint(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "refresh"])
    assert result.exit_code == 1
    assert "No tracked project here." in _combined_output(result)


def test_refresh_happy_path_prints_summary(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    ConfigRoot(root).profile_path("chem").write_text("includes:\n  - chemprop\n")
    project_dir = _tracked_project_dir(tmp_path)
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        "uv_stack.cli.refresh_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "refresh", "--python", "3.12"])
    assert result.exit_code == 0
    assert "Removed (1): rdkit" in result.output
    assert "chemprop" in result.output  # listed under Added
    assert "Project refreshed." in result.output


def test_refresh_dry_run_prints_plan(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    project_dir = _tracked_project_dir(tmp_path)
    monkeypatch.chdir(project_dir)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "refresh", "--dry-run"])
    assert result.exit_code == 0
    assert "Planned commands:" in result.output
    assert "uv add" in result.output


def test_refresh_flags_pass_through_to_refresh_project(tmp_path: Path, monkeypatch):
    from uv_stack.operations.project import RefreshOptions, RefreshResult
    root = _seeded_root(tmp_path)
    project_dir = _tracked_project_dir(tmp_path)
    monkeypatch.chdir(project_dir)
    captured: dict = {}

    def fake_refresh(config, runner, options, *, cwd):
        captured["options"] = options
        captured["cwd"] = cwd
        return RefreshResult()

    monkeypatch.setattr("uv_stack.cli.refresh_cmd.refresh_project", fake_refresh)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "refresh", "--python", "3.13", "--strict", "--no-sync"]
    )
    assert result.exit_code == 0
    opts: RefreshOptions = captured["options"]
    assert opts.python == "3.13"
    assert opts.strict is True
    assert opts.no_sync is True
    assert opts.dry_run is False
    assert captured["cwd"] == project_dir

    result2 = runner.invoke(cli, ["--root", str(root), "refresh", "--dry-run"])
    assert result2.exit_code == 0
    opts2: RefreshOptions = captured["options"]
    assert opts2.dry_run is True


def test_refresh_in_help_panels():
    """Refresh command must appear in the Environments panel.

    Asserts structurally against COMMAND_GROUPS so the test fails if refresh is
    registered but placed in a different panel.
    """
    # Structural assertion: the Environments group must contain exactly the
    # three listed commands in order.
    command_groups = rich_click.rich_click.COMMAND_GROUPS.get("stack", [])
    env_group = next((g for g in command_groups if g["name"] == "Environments"), None)
    assert env_group is not None, "Environments panel not found in COMMAND_GROUPS"
    assert env_group["commands"] == ["upgrade", "create", "refresh"], (
        f"Expected ['upgrade', 'create', 'refresh'], got {env_group['commands']}"
    )

    # Lightweight smoke assertion: help renders correctly and refresh appears.
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"], prog_name="stack")
    assert result.exit_code == 0
    assert "refresh" in result.output


def test_create_profile_warns_on_bare_token_retarget(tmp_path: Path):
    root = _env_root(tmp_path)  # env 'main' stack.txt contains '@standard'
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.env_stack_path("main").write_text("@standard\nhttpx\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "httpx", "httpx>=0.27"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)
    # Styled rendering may wrap; assert distinctive unwrappable fragments.
    assert "'httpx' is used as a bare token" in combined
    assert "envs/main/stack.txt" in combined
    assert "use pkg:httpx there" in combined


def test_create_bundle_warns_on_bare_token_in_other_bundle(tmp_path: Path):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.bundle_path("web").write_text("includes:\n  - httpx\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "httpx", "pkg:httpx"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)
    # Styled rendering may wrap; assert distinctive unwrappable fragments.
    assert "'httpx' is used as a bare token" in combined
    assert "bundles/web.yaml" in combined
    assert "this bundle" in combined


def test_create_bundle_does_not_warn_about_itself(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    # Create bundle 'daily' that includes its own name as a bare token.
    # Self-exclusion logic must prevent a warning about bundles/daily.yaml.
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "daily", "pkg:numpy", "daily"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)
    assert "used as a bare token in bundles/daily.yaml" not in combined


def test_create_profile_no_warning_without_bare_usage(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "viz2", "matplotlib"]
    )
    assert result.exit_code == 0
    assert "used as a bare token" not in _combined_output(result)
