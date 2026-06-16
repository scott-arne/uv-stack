from __future__ import annotations

import pytest

from uv_stack.commands import micromamba_create, micromamba_remove
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, EnvError
from uv_stack.operations.create import ensure_env, env_micromamba_exists
from uv_stack.operations.project import (
    PROJECT_PYTHON_ENV,
    ProjectOptions,
    _is_python_passthrough,
    init_project,
    resolve_project_python,
    select_project_python,
)
from uv_stack.operations.upgrade import UpgradeOptions, upgrade_env
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


def test_upgrade_writes_generated_files_and_runs_sequence(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    result = upgrade_env(config_tree, rec, "main", UpgradeOptions())

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


def test_upgrade_no_upgrade_omits_flag(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    upgrade_env(config_tree, rec, "main", UpgradeOptions(no_upgrade=True))
    compile_cmd = next(c for c in rec.commands if "compile" in c.args)
    assert "--upgrade" not in compile_cmd.args


def test_upgrade_upgrade_packages(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    upgrade_env(config_tree, rec, "main", UpgradeOptions(upgrade_packages=["pandas"]))
    compile_cmd = next(c for c in rec.commands if "compile" in c.args)
    assert "--upgrade" not in compile_cmd.args
    assert "--upgrade-package" in compile_cmd.args
    assert "pandas" in compile_cmd.args


def test_upgrade_dry_run_writes_files_but_runs_nothing(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    result = upgrade_env(config_tree, rec, "main", UpgradeOptions(dry_run=True))
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


def test_init_project_runs_init_add_sync(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    rec = RecordingRunner()
    init_project(
        config_tree, rec, ["standard"], ProjectOptions(python="3.12"), cwd=project_dir
    )
    argv = [" ".join(c.args) for c in rec.commands]
    assert any("uv init --bare" in a for a in argv)
    assert any("uv add --no-sync" in a for a in argv)
    # A version passes through unchanged and reaches both init and sync.
    assert any("uv init --bare --python 3.12" in a for a in argv)
    assert any(a == "uv sync --python 3.12" for a in argv)


def test_init_project_no_sync_skips_sync(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj2"
    project_dir.mkdir()
    rec = RecordingRunner()
    init_project(
        config_tree, rec, ["ds"], ProjectOptions(no_sync=True), cwd=project_dir
    )
    argv = [" ".join(c.args) for c in rec.commands]
    # Resolution still runs (uv init gets the fallback) but no sync is recorded.
    assert any("uv init --bare --python 3.12" in a for a in argv)
    assert not any(a.startswith("uv sync") for a in argv)


def test_init_project_force_skip_init_still_pins_sync(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_force_sync"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='x'\n")
    rec = RecordingRunner()
    init_project(config_tree, rec, ["ds"], ProjectOptions(force=True), cwd=project_dir)
    argv = [" ".join(c.args) for c in rec.commands]
    # No init runs, but the resolved interpreter still pins sync.
    assert not any("uv init" in a for a in argv)
    assert any(a == "uv sync --python 3.12" for a in argv)


def test_init_project_resolves_micromamba_env_for_both_commands(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_env"
    project_dir.mkdir()
    # The responder maps any "micromamba run" probe to a path with a newline.
    rec = RecordingRunner(responder=_existing_env_responder)
    init_project(
        config_tree, rec, ["ds"], ProjectOptions(python="main"), cwd=project_dir
    )
    argv = [" ".join(c.args) for c in rec.commands]
    # The env name resolved to the (stripped) interpreter path for init and sync.
    assert any("uv init --bare --python /envs/main/bin/python" in a for a in argv)
    assert any(a == "uv sync --python /envs/main/bin/python" for a in argv)


def test_init_project_unresolvable_env_fails_fast(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_bad_env"
    project_dir.mkdir()
    # The probe fails, so resolution must raise before any scaffolding.
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError):
        init_project(
            config_tree, rec, ["ds"], ProjectOptions(python="nope"), cwd=project_dir
        )
    argv = [" ".join(c.args) for c in rec.commands]
    assert not any(a.startswith("uv ") for a in argv)
    assert not (project_dir / "pyproject.toml").exists()


@pytest.mark.parametrize(
    "spec, passthrough",
    [
        ("3", True),
        ("3.12", True),
        ("3.12.4", True),
        ("/abs/path/python", True),
        ("rel\\path\\python", True),
        ("cpython@3.12", True),
        ("pypy@3.10", True),
        ("cpython-3.12.4-macos-aarch64-none", True),
        ("main", False),
        ("py311", False),
    ],
)
def test_is_python_passthrough(spec, passthrough):
    assert _is_python_passthrough(spec) is passthrough


def test_select_project_python_flag_wins(config_tree: ConfigRoot, monkeypatch):
    monkeypatch.setenv(PROJECT_PYTHON_ENV, "envvar")
    config_tree.project_python_path().write_text("fromfile\n")
    assert select_project_python(config_tree, "3.11") == "3.11"


def test_select_project_python_env_over_file(config_tree: ConfigRoot, monkeypatch):
    monkeypatch.setenv(PROJECT_PYTHON_ENV, "envvar")
    config_tree.project_python_path().write_text("fromfile\n")
    assert select_project_python(config_tree, None) == "envvar"


def test_select_project_python_file_over_default(config_tree: ConfigRoot, monkeypatch):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.project_python_path().write_text("main\n")
    assert select_project_python(config_tree, None) == "main"


def test_select_project_python_empty_file_falls_back(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.project_python_path().write_text("# just a comment\n\n")
    assert select_project_python(config_tree, None) == "3.12"


def test_select_project_python_default(config_tree: ConfigRoot, monkeypatch):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    assert select_project_python(config_tree, None) == "3.12"


def test_resolve_project_python_passes_version_through(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    rec = RecordingRunner(responder=_existing_env_responder)
    assert resolve_project_python(config_tree, rec, "3.12") == "3.12"
    # No env probe ran for a version.
    assert not any("run" in c.args for c in rec.commands)


def test_resolve_project_python_resolves_env_name(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    rec = RecordingRunner(responder=_existing_env_responder)
    # Trailing newline from the probe must be stripped.
    assert resolve_project_python(config_tree, rec, "main") == "/envs/main/bin/python"


def test_resolve_project_python_bad_env_raises(config_tree: ConfigRoot, monkeypatch):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError):
        resolve_project_python(config_tree, rec, "nope")


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
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj4"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='x'\n")
    rec = RecordingRunner()
    init_project(config_tree, rec, ["ds"], ProjectOptions(force=True), cwd=project_dir)
    argv = [" ".join(c.args) for c in rec.commands]
    assert not any("uv init" in a for a in argv)
    assert any("uv add --no-sync" in a for a in argv)
