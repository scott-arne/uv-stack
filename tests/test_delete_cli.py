"""Tests for ``stack delete``."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from tests.test_cli import _combined_output, _flat_panel
from uv_stack.cli import cli
from uv_stack.commands import micromamba_remove
from uv_stack.config import ConfigRoot
from uv_stack.operations.pyproject import read_tracking
from uv_stack.runner import Command, CommandResult, RecordingRunner


def _invoke(root: ConfigRoot, *args: str, input: str | None = None) -> Result:
    return CliRunner().invoke(cli, ["--root", str(root.root), *args], input=input)


def _env_present(cmd: Command) -> CommandResult:
    if "run" in cmd.args:
        return CommandResult(returncode=0, stdout="/envs/main/bin/python\n")
    return CommandResult(returncode=0, stdout="")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> RecordingRunner:
    """Stand in for the SubprocessRunner the delete commands construct."""
    rec = RecordingRunner(responder=_env_present)
    monkeypatch.setattr("uv_stack.cli.delete.SubprocessRunner", lambda: rec)
    return rec


# ---------------------------------------------------------------------------
# group
# ---------------------------------------------------------------------------


def test_delete_no_subcommand_shows_help():
    result = CliRunner().invoke(cli, ["delete"])
    assert result.exit_code == 2
    for kind in ("env", "project", "profile", "bundle"):
        assert kind in result.output


def test_delete_is_listed_beside_create(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COLUMNS", "200")
    # The command panels are keyed on the program name, so the help must be
    # rendered as 'stack' for them to appear at all.
    result = CliRunner().invoke(cli, ["--help"], prog_name="stack")
    assert result.exit_code == 0
    assert "Create and delete" in result.output
    assert "delete" in result.output


def test_delete_project_help_keeps_the_bracketed_table_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["delete", "project", "--help"])
    assert result.exit_code == 0
    assert "--no-sync" in result.output
    # rich-click parses help as markup; the table name must survive rather
    # than being eaten as an unknown style tag.
    assert "[tool.uv-stack]" in result.output


# ---------------------------------------------------------------------------
# profile and bundle
# ---------------------------------------------------------------------------


def test_delete_profile_yes_deletes_without_a_prompt(config_tree: ConfigRoot):
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")

    result = _invoke(config_tree, "delete", "profile", "solo", "-y")

    assert result.exit_code == 0, result.output
    assert f"Deleted {config_tree.profile_path('solo')}" in result.output
    assert "Delete profile" not in result.output
    assert not config_tree.profile_path("solo").exists()


def test_delete_profile_prompts_and_accepts(config_tree: ConfigRoot):
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")

    result = _invoke(config_tree, "delete", "profile", "solo", input="y\n")

    assert result.exit_code == 0, result.output
    assert "Delete profile 'solo'" in result.output
    assert not config_tree.profile_path("solo").exists()


def test_delete_profile_declined_prompt_aborts(config_tree: ConfigRoot):
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")

    result = _invoke(config_tree, "delete", "profile", "solo", input="n\n")

    assert result.exit_code == 1
    assert "Aborted" in _combined_output(result)
    assert config_tree.profile_path("solo").exists()


def test_delete_profile_without_input_aborts(config_tree: ConfigRoot):
    """A script that forgot -y gets a refusal, not a hang or a deletion."""
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")

    result = _invoke(config_tree, "delete", "profile", "solo")

    assert result.exit_code == 1
    assert config_tree.profile_path("solo").exists()


def test_delete_profile_referenced_is_refused_before_any_prompt(config_tree: ConfigRoot):
    result = _invoke(config_tree, "delete", "profile", "ds", input="y\n")

    assert result.exit_code == 1
    flat = _flat_panel(result)
    assert "Profile 'ds' is still referenced" in flat
    assert "bundles/standard.yaml: ds" in flat
    assert "--force" in flat
    assert "Delete profile" not in result.output
    assert config_tree.profile_path("ds").exists()


def test_delete_profile_force_shows_the_warnings_before_the_prompt(config_tree: ConfigRoot):
    result = _invoke(config_tree, "delete", "profile", "ds", "--force", input="y\n")

    assert result.exit_code == 0, result.output
    combined = _combined_output(result)
    assert "warning:" in combined
    assert "bundles/standard.yaml" in combined
    assert combined.index("warning:") < combined.index("Delete profile 'ds'")
    assert not config_tree.profile_path("ds").exists()


def test_delete_profile_missing_renders_a_panel(config_tree: ConfigRoot):
    result = _invoke(config_tree, "delete", "profile", "nope", "-y")

    assert result.exit_code == 1
    assert "Missing profile" in _flat_panel(result)


def test_delete_bundle_yes_deletes(config_tree: ConfigRoot):
    result = _invoke(config_tree, "delete", "bundle", "qsar", "-y")

    assert result.exit_code == 0, result.output
    assert f"Deleted {config_tree.bundle_path('qsar')}" in result.output
    assert not config_tree.bundle_path("qsar").exists()


def test_delete_bundle_referenced_is_refused(config_tree: ConfigRoot):
    result = _invoke(config_tree, "delete", "bundle", "standard", "-y")

    assert result.exit_code == 1
    assert "envs/main/stack.txt: @standard" in _flat_panel(result)
    assert config_tree.bundle_path("standard").exists()


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------


def test_delete_env_yes_removes_micromamba_env_and_sources(
    config_tree: ConfigRoot, recorder: RecordingRunner
):
    result = _invoke(config_tree, "delete", "env", "main", "-y")

    assert result.exit_code == 0, result.output
    assert "Removed micromamba environment main" in result.output
    assert f"Deleted {config_tree.env_dir('main')}" in result.output
    assert recorder.commands[-1] == micromamba_remove("main")
    assert not config_tree.env_dir("main").exists()


def test_delete_env_prompt_names_the_micromamba_env_when_built(
    config_tree: ConfigRoot, recorder: RecordingRunner
):
    result = _invoke(config_tree, "delete", "env", "main", input="n\n")

    assert result.exit_code == 1
    assert "Delete environment 'main'" in result.output
    assert "micromamba environment" in result.output
    assert not any("remove" in cmd.args for cmd in recorder.commands)
    assert config_tree.env_stack_path("main").is_file()


def test_delete_env_prompt_omits_micromamba_when_not_built(
    config_tree: ConfigRoot, recorder: RecordingRunner
):
    recorder.responder = lambda _cmd: CommandResult(returncode=1, stdout="")

    result = _invoke(config_tree, "delete", "env", "main", input="y\n")

    assert result.exit_code == 0, result.output
    assert "micromamba environment" not in result.output
    assert "Removed micromamba" not in result.output
    assert not config_tree.env_dir("main").exists()


def test_delete_env_missing_renders_a_panel(config_tree: ConfigRoot, recorder: RecordingRunner):
    result = _invoke(config_tree, "delete", "env", "nope", "-y")

    assert result.exit_code == 1
    assert "Missing stack file" in _flat_panel(result)
    assert recorder.commands == []


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------

_TRACKING = (
    "\n[tool.uv-stack]\nversion = 1\n"
    'stack = ["standard"]\n'
    'applied = ["numpy", "pandas", "rdkit", "rich"]\n'
)


def _tracked_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking: str) -> Path:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "proj"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas", "rdkit", "rich", "user-extra"]\n' + tracking
    )
    monkeypatch.chdir(project_dir)
    return project_dir


def test_delete_project_yes_removes_applied_and_drops_the_table(
    config_tree: ConfigRoot, recorder: RecordingRunner, tmp_path: Path, monkeypatch
):
    project_dir = _tracked_project(tmp_path, monkeypatch, _TRACKING)

    result = _invoke(config_tree, "delete", "project", "-y")

    assert result.exit_code == 0, result.output
    assert "Removing (4): numpy, pandas, rdkit, rich" in result.output
    assert "[tool.uv-stack]" in result.output
    assert [c.args for c in recorder.commands] == [
        ["uv", "remove", "--no-sync", "numpy", "pandas", "rdkit", "rich"],
        ["uv", "sync"],
    ]
    assert read_tracking(project_dir / "pyproject.toml") is None


def test_delete_project_prompt_lists_the_packages_then_declines(
    config_tree: ConfigRoot, recorder: RecordingRunner, tmp_path: Path, monkeypatch
):
    project_dir = _tracked_project(tmp_path, monkeypatch, _TRACKING)

    result = _invoke(config_tree, "delete", "project", input="n\n")

    assert result.exit_code == 1
    assert "Removing (4): numpy, pandas, rdkit, rich" in result.output
    assert recorder.commands == []
    assert read_tracking(project_dir / "pyproject.toml") is not None


def test_delete_project_no_sync(
    config_tree: ConfigRoot, recorder: RecordingRunner, tmp_path: Path, monkeypatch
):
    _tracked_project(tmp_path, monkeypatch, _TRACKING)

    result = _invoke(config_tree, "delete", "project", "-y", "--no-sync")

    assert result.exit_code == 0, result.output
    assert [c.args[:2] for c in recorder.commands] == [["uv", "remove"]]


def test_delete_project_with_nothing_applied_says_so(
    config_tree: ConfigRoot, recorder: RecordingRunner, tmp_path: Path, monkeypatch
):
    empty = '\n[tool.uv-stack]\nversion = 1\nstack = ["standard"]\napplied = []\n'
    _tracked_project(tmp_path, monkeypatch, empty)

    result = _invoke(config_tree, "delete", "project", "-y")

    assert result.exit_code == 0, result.output
    assert "Nothing to remove" in result.output
    assert [c.args for c in recorder.commands] == [["uv", "sync"]]


def test_delete_project_untracked_renders_a_panel(
    config_tree: ConfigRoot, recorder: RecordingRunner, tmp_path: Path, monkeypatch
):
    project_dir = tmp_path / "untracked"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text('[project]\nname = "x"\n')
    monkeypatch.chdir(project_dir)

    result = _invoke(config_tree, "delete", "project", "-y")

    assert result.exit_code == 1
    assert "No tracked project" in _flat_panel(result)
    assert recorder.commands == []
