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
    assert "stack, version 0.1.6" in result.output


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


def test_doctor_fix_exit_1_when_errors_remain(tmp_path: Path, monkeypatch):
    # A root whose parent is read-only cannot be created; simulate by pointing
    # repair at a path under a file (mkdir raises, so the error finding stays).
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(blocker / "python-envs"), "doctor", "--fix"]
    )
    assert result.exit_code == 1


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
