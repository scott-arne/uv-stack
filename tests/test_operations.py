from __future__ import annotations

import pytest

from uvstack.commands import micromamba_create, micromamba_python_path, micromamba_remove
from uvstack.config import ConfigRoot
from uvstack.errors import EnvError
from uvstack.operations.create import ensure_env, env_micromamba_exists
from uvstack.runner import Command, CommandResult, RecordingRunner


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

from uvstack.operations.update import UpdateOptions, update_env


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
