"""Tests for the ``stack delete`` operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import _lock_held_by_another_process
from uv_stack.commands import micromamba_remove
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, ToolError
from uv_stack.fsutil import _LOCK_AVAILABLE
from uv_stack.operations.delete import (
    DeleteResult,
    EnvDeleteResult,
    WithdrawResult,
    delete_bundle,
    delete_env,
    delete_profile,
    withdraw_project,
)
from uv_stack.operations.pyproject import read_tracking
from uv_stack.runner import Command, CommandResult, RecordingRunner


def test_delete_profile_removes_an_unreferenced_profile(config_tree: ConfigRoot):
    path = config_tree.profile_path("solo")
    path.write_text("includes:\n  - rich\n")

    result = delete_profile(config_tree, "solo")

    assert result.path == path
    assert result is not None
    assert result.warnings == []
    assert not path.exists()


def test_delete_profile_refuses_one_a_bundle_includes(config_tree: ConfigRoot):
    # config_tree's 'standard' bundle lists 'ds' as a bare token.
    with pytest.raises(ConfigError) as excinfo:
        delete_profile(config_tree, "ds")

    message = str(excinfo.value)
    assert "bundles/standard.yaml: ds" in message
    assert "--force" in (excinfo.value.hint or "")
    assert config_tree.profile_path("ds").exists()


def test_delete_profile_refuses_a_qualified_env_reference(config_tree: ConfigRoot):
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")
    config_tree.env_stack_path("main").write_text("profile:solo\n")

    with pytest.raises(ConfigError) as excinfo:
        delete_profile(config_tree, "solo")

    assert "envs/main/stack.txt: profile:solo" in str(excinfo.value)
    assert config_tree.profile_path("solo").exists()


def test_delete_profile_ignores_a_package_token_of_the_same_name(config_tree: ConfigRoot):
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")
    config_tree.bundle_path("extra").write_text("includes:\n  - pkg:solo\n")

    result = delete_profile(config_tree, "solo")

    assert result is not None
    assert result.warnings == []
    assert not config_tree.profile_path("solo").exists()


def test_delete_profile_force_deletes_and_warns_per_reference(config_tree: ConfigRoot):
    config_tree.env_stack_path("main").write_text("profile:ds\n")

    result = delete_profile(config_tree, "ds", force=True)

    assert result is not None
    assert not config_tree.profile_path("ds").exists()
    assert len(result.warnings) == 2
    bundle_warning = next(w for w in result.warnings if "bundles/standard.yaml" in w)
    env_warning = next(w for w in result.warnings if "envs/main/stack.txt" in w)
    # A bare token silently changes meaning; a qualified one fails loudly.
    assert "pip package 'ds'" in bundle_warning
    assert "no longer resolves" in env_warning


def test_delete_profile_refuses_a_missing_profile(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        delete_profile(config_tree, "nope")

    assert "Missing profile" in str(excinfo.value)


def test_delete_profile_points_at_a_bundle_of_that_name(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        delete_profile(config_tree, "standard")

    assert "stack delete bundle standard" in (excinfo.value.hint or "")


def test_delete_profile_rejects_a_traversing_name(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        delete_profile(config_tree, "../solo")

    assert "Invalid profile name" in str(excinfo.value)


def test_delete_profile_refuses_when_a_source_cannot_be_read(config_tree: ConfigRoot):
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")
    config_tree.bundle_path("broken").write_text("includes: [\n")

    with pytest.raises(ConfigError) as excinfo:
        delete_profile(config_tree, "solo")

    assert "could not be checked" in str(excinfo.value)
    assert "bundles/broken.yaml" in str(excinfo.value)
    assert config_tree.profile_path("solo").exists()


def test_delete_profile_force_warns_about_an_unreadable_source(config_tree: ConfigRoot):
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")
    config_tree.bundle_path("broken").write_text("includes: [\n")

    result = delete_profile(config_tree, "solo", force=True)

    assert result is not None
    assert any("bundles/broken.yaml" in w for w in result.warnings)
    assert not config_tree.profile_path("solo").exists()


def test_delete_profile_counts_a_case_variant_that_reaches_the_file(config_tree: ConfigRoot):
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")
    if not config_tree.profile_path("SOLO").is_file():
        pytest.skip("requires a case-folding filesystem")
    config_tree.env_stack_path("main").write_text("SOLO\n")

    with pytest.raises(ConfigError) as excinfo:
        delete_profile(config_tree, "solo")

    assert "envs/main/stack.txt: SOLO" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_delete_profile_waits_on_the_stem_lock(config_tree: ConfigRoot, monkeypatch):
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    config_tree.profile_path("solo").write_text("includes:\n  - rich\n")

    with _lock_held_by_another_process(config_tree.stem_lock_path("solo")):
        with pytest.raises(ConfigError) as excinfo:
            delete_profile(config_tree, "solo")

    assert "another stack process" in str(excinfo.value)
    assert config_tree.profile_path("solo").exists()


# ----------------------------------------------------------------------------
# bundles
# ----------------------------------------------------------------------------


def test_delete_bundle_removes_an_unreferenced_bundle(config_tree: ConfigRoot):
    path = config_tree.bundle_path("qsar")

    result = delete_bundle(config_tree, "qsar")

    assert result.path == path
    assert result is not None
    assert result.warnings == []
    assert not path.exists()


def test_delete_bundle_refuses_env_and_bundle_references(config_tree: ConfigRoot):
    # main's stack.txt says '@standard'; qsar.yaml says 'standard'.
    with pytest.raises(ConfigError) as excinfo:
        delete_bundle(config_tree, "standard")

    message = str(excinfo.value)
    assert "envs/main/stack.txt: @standard" in message
    assert "bundles/qsar.yaml: standard" in message
    assert config_tree.bundle_path("standard").exists()


def test_delete_bundle_refuses_the_bundle_prefix_form(config_tree: ConfigRoot):
    config_tree.env_stack_path("main").write_text("bundle:qsar\n")

    with pytest.raises(ConfigError) as excinfo:
        delete_bundle(config_tree, "qsar")

    assert "envs/main/stack.txt: bundle:qsar" in str(excinfo.value)


def test_delete_bundle_does_not_count_its_own_self_reference(config_tree: ConfigRoot):
    config_tree.bundle_path("loop").write_text("includes:\n  - loop\n")

    result = delete_bundle(config_tree, "loop")

    assert result is not None
    assert result.warnings == []


def test_delete_bundle_force_distinguishes_bare_from_qualified(config_tree: ConfigRoot):
    result = delete_bundle(config_tree, "standard", force=True)

    assert not config_tree.bundle_path("standard").exists()
    env_warning = next(w for w in result.warnings if "envs/main/stack.txt" in w)
    bundle_warning = next(w for w in result.warnings if "bundles/qsar.yaml" in w)
    assert "no longer resolves" in env_warning
    assert "pip package 'standard'" in bundle_warning


def test_delete_bundle_points_at_a_profile_of_that_name(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        delete_bundle(config_tree, "ds")

    assert "Missing bundle" in str(excinfo.value)
    assert "stack delete profile ds" in (excinfo.value.hint or "")


# ----------------------------------------------------------------------------
# environments
# ----------------------------------------------------------------------------


def _env_present(cmd: Command) -> CommandResult:
    if "run" in cmd.args:
        return CommandResult(returncode=0, stdout="/envs/main/bin/python\n")
    return CommandResult(returncode=0, stdout="")


def _env_absent(cmd: Command) -> CommandResult:
    if "run" in cmd.args:
        return CommandResult(returncode=1, stdout="")
    return CommandResult(returncode=0, stdout="")


def test_delete_env_removes_the_micromamba_env_then_the_sources(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_env_present)

    result = delete_env(config_tree, rec, "main")

    assert result.directory == config_tree.env_dir("main")
    assert result.removed_micromamba is True
    assert not config_tree.env_dir("main").exists()
    assert rec.commands[-1] == micromamba_remove("main")


def test_delete_env_skips_micromamba_when_the_env_is_not_built(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_env_absent)

    result = delete_env(config_tree, rec, "main")

    assert result.removed_micromamba is False
    assert not config_tree.env_dir("main").exists()
    assert not any("remove" in cmd.args for cmd in rec.commands)


def test_delete_env_keeps_the_sources_when_micromamba_remove_fails(config_tree: ConfigRoot):
    def _remove_fails(cmd: Command) -> CommandResult:
        if "remove" in cmd.args:
            raise ToolError("micromamba remove failed", command=cmd.args, returncode=1)
        return _env_present(cmd)

    with pytest.raises(ToolError):
        delete_env(config_tree, RecordingRunner(responder=_remove_fails), "main")

    assert config_tree.env_stack_path("main").is_file()


def test_delete_env_keeps_the_sources_when_micromamba_cannot_be_probed(config_tree: ConfigRoot):
    def _spawn_failure(cmd: Command) -> CommandResult:
        raise ToolError("Could not run micromamba.", command=cmd.args, returncode=127)

    with pytest.raises(ToolError):
        delete_env(config_tree, RecordingRunner(responder=_spawn_failure), "main")

    assert config_tree.env_stack_path("main").is_file()


def test_delete_env_refuses_an_env_without_a_stack_file(config_tree: ConfigRoot):
    (config_tree.envs_dir / "stray").mkdir()
    rec = RecordingRunner(responder=_env_present)

    with pytest.raises(ConfigError) as excinfo:
        delete_env(config_tree, rec, "stray")

    # Only what uv-stack manages is deleted: no micromamba command ran.
    assert "Missing stack file" in str(excinfo.value)
    assert rec.commands == []
    assert (config_tree.envs_dir / "stray").is_dir()


def test_delete_env_refuses_a_linked_env_directory(config_tree: ConfigRoot, tmp_path):
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "stack.txt").write_text("rich\n")
    config_tree.env_dir("linked").symlink_to(real)
    rec = RecordingRunner(responder=_env_present)

    with pytest.raises(ConfigError) as excinfo:
        delete_env(config_tree, rec, "linked")

    assert "symbolic link" in str(excinfo.value)
    assert rec.commands == []
    assert config_tree.env_dir("linked").is_symlink()
    assert (real / "stack.txt").is_file()


def test_delete_env_rejects_a_traversing_name(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        delete_env(config_tree, RecordingRunner(), "../main")

    assert "Invalid environment name" in str(excinfo.value)


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_delete_env_waits_on_the_env_lock(config_tree: ConfigRoot, monkeypatch):
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    rec = RecordingRunner(responder=_env_present)

    with _lock_held_by_another_process(config_tree.env_lock_path("main")):
        with pytest.raises(ConfigError) as excinfo:
            delete_env(config_tree, rec, "main")

    assert "another stack process" in str(excinfo.value)
    assert rec.commands == []
    assert config_tree.env_stack_path("main").is_file()


# ----------------------------------------------------------------------------
# projects
# ----------------------------------------------------------------------------

_DIRECT_REF = "httpx @ https://example.com/httpx-1.0-py3-none-any.whl"


def _tracked_project(tmp_path: Path, tracking_text: str, *, deps: tuple[str, ...]) -> Path:
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    listed = ", ".join(f'"{dep}"' for dep in deps)
    (project_dir / "pyproject.toml").write_text(
        f'[project]\nname = "proj"\nversion = "0.1.0"\ndependencies = [{listed}]\n'
        + tracking_text
    )
    return project_dir


_DEPS = ("numpy", "pandas", "rdkit", "rich", "user-extra")
_TRACKING = (
    "\n[tool.uv-stack]\nversion = 1\n"
    'stack = ["standard"]\n'
    'applied = ["numpy", "pandas", "rdkit", "rich"]\n'
)


def test_withdraw_project_removes_applied_drops_the_table_and_syncs(
    config_tree: ConfigRoot, tmp_path: Path
):
    project_dir = _tracked_project(tmp_path, _TRACKING, deps=_DEPS)
    rec = RecordingRunner()

    result = withdraw_project(config_tree, rec, cwd=project_dir)

    assert [c.args for c in rec.commands] == [
        ["uv", "remove", "--no-sync", "numpy", "pandas", "rdkit", "rich"],
        ["uv", "sync"],
    ]
    assert all(c.cwd == project_dir for c in rec.commands)
    assert result.removed == ["numpy", "pandas", "rdkit", "rich"]
    assert result.skipped_removals == []
    assert read_tracking(project_dir / "pyproject.toml") is None
    # The user's own dependency is never named to uv.
    assert "user-extra" not in rec.commands[0].args


def test_withdraw_project_no_sync_skips_the_sync(config_tree: ConfigRoot, tmp_path: Path):
    project_dir = _tracked_project(tmp_path, _TRACKING, deps=_DEPS)
    rec = RecordingRunner()

    withdraw_project(config_tree, rec, cwd=project_dir, no_sync=True)

    assert [c.args[:2] for c in rec.commands] == [["uv", "remove"]]


def test_withdraw_project_requires_a_tracked_project(config_tree: ConfigRoot, tmp_path: Path):
    project_dir = tmp_path / "untracked"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text('[project]\nname = "x"\n')

    with pytest.raises(ConfigError) as excinfo:
        withdraw_project(config_tree, RecordingRunner(), cwd=project_dir)

    assert f"No tracked project in {project_dir}." in str(excinfo.value)


def test_withdraw_project_refuses_a_newer_schema(config_tree: ConfigRoot, tmp_path: Path):
    project_dir = _tracked_project(
        tmp_path, "\n[tool.uv-stack]\nversion = 2\nstack = []\napplied = []\n", deps=_DEPS
    )

    with pytest.raises(ConfigError) as excinfo:
        withdraw_project(config_tree, RecordingRunner(), cwd=project_dir)

    assert "newer uv-stack (schema 2)" in str(excinfo.value)


def test_withdraw_project_names_only_packages_still_present(
    config_tree: ConfigRoot, tmp_path: Path
):
    # 'rdkit' is in the ledger but the user already removed it by hand.
    project_dir = _tracked_project(
        tmp_path, _TRACKING, deps=("numpy", "pandas", "rich", "user-extra")
    )
    rec = RecordingRunner()

    result = withdraw_project(config_tree, rec, cwd=project_dir)

    assert rec.commands[0].args == ["uv", "remove", "--no-sync", "numpy", "pandas", "rich"]
    assert result.removed == ["numpy", "pandas", "rich"]


def test_withdraw_project_removes_pending_entries_too(config_tree: ConfigRoot, tmp_path: Path):
    tracking = _TRACKING + 'pending = ["numpy", "pandas", "rdkit", "rich", "extra-pending"]\n'
    project_dir = _tracked_project(tmp_path, tracking, deps=(*_DEPS, "extra-pending"))
    rec = RecordingRunner()

    result = withdraw_project(config_tree, rec, cwd=project_dir)

    assert "extra-pending" in rec.commands[0].args
    assert result.removed == ["numpy", "pandas", "rdkit", "rich", "extra-pending"]


def test_withdraw_project_skips_a_direct_reference_with_the_notice(
    config_tree: ConfigRoot, tmp_path: Path
):
    tracking = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["standard"]\n'
        f'applied = ["numpy", "{_DIRECT_REF}"]\n'
    )
    project_dir = _tracked_project(tmp_path, tracking, deps=("numpy", _DIRECT_REF))
    rec = RecordingRunner()

    result = withdraw_project(config_tree, rec, cwd=project_dir)

    assert rec.commands[0].args == ["uv", "remove", "--no-sync", "numpy"]
    assert result.skipped_removals == [_DIRECT_REF]
    assert read_tracking(project_dir / "pyproject.toml") is None


def test_withdraw_project_with_nothing_applied_only_drops_the_table(
    config_tree: ConfigRoot, tmp_path: Path
):
    tracking = '\n[tool.uv-stack]\nversion = 1\nstack = ["standard"]\napplied = []\n'
    project_dir = _tracked_project(tmp_path, tracking, deps=("user-extra",))
    rec = RecordingRunner()

    result = withdraw_project(config_tree, rec, cwd=project_dir)

    assert [c.args for c in rec.commands] == [["uv", "sync"]]
    assert result.removed == []
    assert read_tracking(project_dir / "pyproject.toml") is None


def test_withdraw_project_keeps_the_table_when_uv_remove_fails(
    config_tree: ConfigRoot, tmp_path: Path
):
    project_dir = _tracked_project(tmp_path, _TRACKING, deps=_DEPS)

    def _remove_fails(cmd: Command) -> CommandResult:
        if cmd.args[:2] == ["uv", "remove"]:
            raise ToolError("uv remove failed", command=cmd.args, returncode=1)
        return CommandResult(returncode=0, stdout="")

    with pytest.raises(ToolError):
        withdraw_project(config_tree, RecordingRunner(responder=_remove_fails), cwd=project_dir)

    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert tracking.applied == ["numpy", "pandas", "rdkit", "rich"]


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_withdraw_project_waits_on_the_project_lock(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    project_dir = _tracked_project(tmp_path, _TRACKING, deps=_DEPS)
    rec = RecordingRunner()

    with _lock_held_by_another_process(config_tree.project_lock_path(project_dir)):
        with pytest.raises(ConfigError) as excinfo:
            withdraw_project(config_tree, rec, cwd=project_dir)

    assert "another stack process" in str(excinfo.value)
    assert rec.commands == []
    assert read_tracking(project_dir / "pyproject.toml") is not None


# ----------------------------------------------------------------------------
# the confirm callback
# ----------------------------------------------------------------------------


def test_delete_profile_declined_confirm_deletes_nothing(config_tree: ConfigRoot):
    config_tree.env_stack_path("main").write_text("profile:ds\n")
    seen: list[DeleteResult] = []

    def _decline(plan: DeleteResult) -> bool:
        seen.append(plan)
        return False

    result = delete_profile(config_tree, "ds", force=True, confirm=_decline)

    assert result is None
    assert config_tree.profile_path("ds").exists()
    # The callback saw what the delete would have reported, warnings included.
    assert [plan.path for plan in seen] == [config_tree.profile_path("ds")]
    assert any("envs/main/stack.txt" in w for w in seen[0].warnings)


def test_delete_profile_confirm_is_not_asked_for_a_refused_delete(config_tree: ConfigRoot):
    asked: list[DeleteResult] = []

    with pytest.raises(ConfigError):
        delete_profile(config_tree, "ds", confirm=lambda plan: asked.append(plan) or True)

    assert asked == []


def test_delete_env_declined_confirm_removes_nothing(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_env_present)
    seen: list[EnvDeleteResult] = []

    def _decline(plan: EnvDeleteResult) -> bool:
        seen.append(plan)
        return False

    result = delete_env(config_tree, rec, "main", confirm=_decline)

    assert result is None
    assert config_tree.env_stack_path("main").is_file()
    assert not any("remove" in cmd.args for cmd in rec.commands)
    assert [plan.removed_micromamba for plan in seen] == [True]


def test_withdraw_project_declined_confirm_runs_nothing(config_tree: ConfigRoot, tmp_path: Path):
    project_dir = _tracked_project(tmp_path, _TRACKING, deps=_DEPS)
    rec = RecordingRunner()
    seen: list[WithdrawResult] = []

    def _decline(plan: WithdrawResult) -> bool:
        seen.append(plan)
        return False

    result = withdraw_project(config_tree, rec, cwd=project_dir, confirm=_decline)

    assert result is None
    assert rec.commands == []
    assert read_tracking(project_dir / "pyproject.toml") is not None
    assert [plan.removed for plan in seen] == [["numpy", "pandas", "rdkit", "rich"]]
