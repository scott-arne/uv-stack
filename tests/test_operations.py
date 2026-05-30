from __future__ import annotations

import pytest

from uv_stack.commands import micromamba_create, micromamba_remove
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, EnvError
from uv_stack.operations.create import ensure_env, env_micromamba_exists
from uv_stack.operations.project import ProjectOptions, init_project
from uv_stack.operations.update import UpdateOptions, update_env
from uv_stack.runner import Command, CommandResult, RecordingRunner


def _missing_env_responder(cmd: Command) -> CommandResult:
    # Simulate "env does not exist": the python-path probe fails.
    if "run" in cmd.args:
        return CommandResult(returncode=1, stdout="")
    return CommandResult(returncode=0, stdout="")


def _existing_env_responder(cmd: Command) -> CommandResult:
    if "run" in cmd.args:
        return CommandResult(returncode=0, stdout="/envs/main/bin/python\n")
    return CommandResult(returncode=0, stdout="")


def test_env_micromamba_exists_true(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    assert env_micromamba_exists(config_tree, rec, "main") is True


def test_env_micromamba_exists_false(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_missing_env_responder)
    assert env_micromamba_exists(config_tree, rec, "main") is False


def test_ensure_env_missing_without_create_raises(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError):
        ensure_env(config_tree, rec, "main", create=False, recreate=False)


def test_ensure_env_missing_with_create_creates(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_missing_env_responder)
    # environment.yml must exist for create; write it first
    config_tree.env_environment_yml("main").write_text("name: main\n")
    ensure_env(config_tree, rec, "main", create=True, recreate=False)
    assert micromamba_create(config_tree.env_environment_yml("main")) in rec.commands


def test_ensure_env_recreate_removes_then_creates(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    config_tree.env_environment_yml("main").write_text("name: main\n")
    ensure_env(config_tree, rec, "main", create=False, recreate=True)
    assert micromamba_remove("main") in rec.commands
    assert micromamba_create(config_tree.env_environment_yml("main")) in rec.commands
    # remove must precede create
    assert rec.commands.index(micromamba_remove("main")) < rec.commands.index(
        micromamba_create(config_tree.env_environment_yml("main"))
    )


def test_ensure_env_existing_no_flags_is_noop(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    ensure_env(config_tree, rec, "main", create=False, recreate=False)
    # only the existence probe ran; no create/remove
    assert micromamba_create(config_tree.env_environment_yml("main")) not in rec.commands
    assert micromamba_remove("main") not in rec.commands


# ============================================================================
# update operation tests
# ============================================================================


def test_update_writes_generated_files_and_runs_sequence(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    result = update_env(config_tree, rec, "main", UpdateOptions())

    # Generated files are written.
    assert config_tree.env_requirements_in("main").is_file()
    assert config_tree.env_environment_yml("main").is_file()
    assert config_tree.env_lock("main").is_file()

    # Command sequence: probe -> compile -> install -> check.
    argv = [" ".join(c.args) for c in rec.commands]
    compile_idx = next(i for i, a in enumerate(argv) if "pip compile" in a)
    install_idx = next(i for i, a in enumerate(argv) if "pip install" in a)
    check_idx = next(i for i, a in enumerate(argv) if "pip check" in a)
    assert compile_idx < install_idx < check_idx
    # Default behavior forces --upgrade on compile.
    assert "--upgrade" in rec.commands[compile_idx].args
    assert result.env_name == "main"


def test_update_no_upgrade_omits_flag(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    update_env(config_tree, rec, "main", UpdateOptions(no_upgrade=True))
    compile_cmd = next(c for c in rec.commands if "compile" in c.args)
    assert "--upgrade" not in compile_cmd.args


def test_update_upgrade_packages(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    update_env(config_tree, rec, "main", UpdateOptions(upgrade_packages=["pandas"]))
    compile_cmd = next(c for c in rec.commands if "compile" in c.args)
    assert "--upgrade" not in compile_cmd.args
    assert "--upgrade-package" in compile_cmd.args
    assert "pandas" in compile_cmd.args


def test_update_dry_run_writes_files_but_runs_nothing(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    result = update_env(config_tree, rec, "main", UpdateOptions(dry_run=True))
    assert config_tree.env_requirements_in("main").is_file()
    assert config_tree.env_environment_yml("main").is_file()
    # No commands executed.
    assert rec.commands == []
    # But a plan is returned.
    assert result.planned
    assert any("compile" in c.args for c in result.planned)
    # Dry run never wrote a lock file.
    assert not config_tree.env_lock("main").is_file()


# ============================================================================
# project init operation tests
# ============================================================================


def test_init_project_runs_init_add_sync(config_tree: ConfigRoot, tmp_path):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    rec = RecordingRunner()
    init_project(
        config_tree, rec, ["standard"], ProjectOptions(python="3.12"), cwd=project_dir
    )
    argv = [" ".join(c.args) for c in rec.commands]
    assert any("uv init --bare" in a for a in argv)
    assert any("uv add --no-sync" in a for a in argv)
    assert any(a == "uv sync" for a in argv)


def test_init_project_no_sync_skips_sync(config_tree: ConfigRoot, tmp_path):
    project_dir = tmp_path / "proj2"
    project_dir.mkdir()
    rec = RecordingRunner()
    init_project(
        config_tree, rec, ["ds"], ProjectOptions(no_sync=True), cwd=project_dir
    )
    argv = [" ".join(c.args) for c in rec.commands]
    assert all(a != "uv sync" for a in argv)


def test_init_project_existing_pyproject_without_force_raises(
    config_tree: ConfigRoot, tmp_path
):
    project_dir = tmp_path / "proj3"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='x'\n")
    rec = RecordingRunner()
    with pytest.raises(ConfigError):
        init_project(config_tree, rec, ["ds"], ProjectOptions(), cwd=project_dir)


def test_init_project_existing_pyproject_with_force_skips_init(
    config_tree: ConfigRoot, tmp_path
):
    project_dir = tmp_path / "proj4"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='x'\n")
    rec = RecordingRunner()
    init_project(config_tree, rec, ["ds"], ProjectOptions(force=True), cwd=project_dir)
    argv = [" ".join(c.args) for c in rec.commands]
    assert not any("uv init" in a for a in argv)
    assert any("uv add --no-sync" in a for a in argv)
