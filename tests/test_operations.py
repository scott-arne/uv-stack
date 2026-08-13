from __future__ import annotations

from pathlib import Path

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


def test_env_interpreter_path_and_none(config_tree: ConfigRoot):
    from uv_stack.operations.create import env_interpreter

    ok = RecordingRunner(responder=_existing_env_responder)
    assert env_interpreter(config_tree, ok, "main") == "/envs/main/bin/python"
    missing = RecordingRunner(responder=_missing_env_responder)
    assert env_interpreter(config_tree, missing, "main") is None


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

    # Command sequence: probe -> compile -> sync -> check.
    argv = [" ".join(c.args) for c in rec.commands]
    compile_idx = next(i for i, a in enumerate(argv) if "pip compile" in a)
    sync_idx = next(i for i, a in enumerate(argv) if "pip sync" in a)
    check_idx = next(i for i, a in enumerate(argv) if "pip check" in a)
    assert compile_idx < sync_idx < check_idx
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


def test_upgrade_result_carries_warnings(config_tree: ConfigRoot):
    config_tree.env_stack_path("main").write_text("standrd\n")
    rec = RecordingRunner(responder=_existing_env_responder)
    result = upgrade_env(config_tree, rec, "main", UpgradeOptions())
    assert result.warnings and "did you mean 'standard'" in result.warnings[0]


def test_upgrade_strict_fails_on_bare_literal(config_tree: ConfigRoot):
    from uv_stack.errors import ResolutionError

    config_tree.env_stack_path("main").write_text("numpyy\n")
    rec = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ResolutionError):
        upgrade_env(config_tree, rec, "main", UpgradeOptions(strict=True))
    # Strict failure happens at resolution, before any command runs.
    assert rec.commands == []


def test_init_project_returns_warnings(config_tree: ConfigRoot, tmp_path, monkeypatch):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_warn"
    project_dir.mkdir()
    rec = RecordingRunner()
    warnings = init_project(
        config_tree, rec, ["standrd"], ProjectOptions(python="3.12"),
        cwd=project_dir,
    )
    assert warnings and "did you mean 'standard'" in warnings[0]


def test_init_project_strict_fails_fast(config_tree: ConfigRoot, tmp_path, monkeypatch):
    from uv_stack.errors import ResolutionError

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_strict"
    project_dir.mkdir()
    rec = RecordingRunner()
    with pytest.raises(ResolutionError):
        init_project(
            config_tree, rec, ["numpyy"],
            ProjectOptions(python="3.12", strict=True), cwd=project_dir,
        )
    assert not (project_dir / "pyproject.toml").exists()


def test_init_project_strict_fails_before_env_probe(config_tree: ConfigRoot, tmp_path, monkeypatch):
    from uv_stack.errors import ResolutionError

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_strict_probe"
    project_dir.mkdir()
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(ResolutionError):
        init_project(
            config_tree, rec, ["numpyy"],
            ProjectOptions(python="main", strict=True), cwd=project_dir,
        )
    # No commands should have run (not even the env probe).
    assert rec.commands == []


def test_upgrade_env_attaches_warnings_to_tool_error(config_tree: ConfigRoot):
    from uv_stack.errors import ToolError

    # Stack with a near-miss token that will generate a warning.
    config_tree.env_stack_path("main").write_text("standrd\n")

    def _compile_failure_responder(cmd: Command) -> CommandResult:
        # The env-existence probe succeeds.
        if "run" in cmd.args:
            return CommandResult(returncode=0, stdout="/envs/main/bin/python\n")
        # The compile command fails.
        if "compile" in cmd.args:
            raise ToolError(
                "uv pip compile failed.",
                command=cmd.args,
                returncode=1,
                detail="ERROR: Could not find a version that satisfies the requirement",
            )
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=_compile_failure_responder)
    with pytest.raises(ToolError) as excinfo:
        upgrade_env(config_tree, rec, "main", UpgradeOptions())

    # The error should carry the resolution warning.
    assert excinfo.value.resolution_warnings
    assert "did you mean 'standard'" in excinfo.value.resolution_warnings[0]


# ============================================================================
# project tracking tests
# ============================================================================


def test_init_project_records_tracking(config_tree: ConfigRoot, tmp_path, monkeypatch):
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_track"
    project_dir.mkdir()
    rec = RecordingRunner()
    init_project(
        config_tree, rec, ["standard", "pkg:httpx"],
        ProjectOptions(python="3.12"), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert tracking.stack == ["standard", "pkg:httpx"]
    assert tracking.python == "3.12"
    # applied is the flattened list: ds+chem+utils profiles then inline.
    assert tracking.applied == ["numpy", "pandas", "rdkit", "rich", "httpx"]


def test_init_project_no_python_flag_records_none(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_nopython"
    project_dir.mkdir()
    init_project(config_tree, RecordingRunner(), ["ds"], ProjectOptions(), cwd=project_dir)
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None and tracking.python is None


def test_init_project_no_track_writes_nothing(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_notrack"
    project_dir.mkdir()
    init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(track=False), cwd=project_dir,
    )
    assert read_tracking(project_dir / "pyproject.toml") is None


def test_init_project_force_no_track_removes_stale_table(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_forcenotrack"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = []\n'
        "\n[tool.uv-stack]\nversion = 1\nstack = [\"ds\"]\napplied = []\n"
    )
    init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(force=True, track=False), cwd=project_dir,
    )
    assert read_tracking(project_dir / "pyproject.toml") is None


def test_init_project_tracking_not_written_when_add_fails(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.errors import ToolError
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_addfail"
    project_dir.mkdir()

    def _fail_add(cmd: Command) -> CommandResult:
        if "add" in cmd.args:
            raise ToolError("add failed", command=cmd.args, returncode=1)
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=_fail_add)
    with pytest.raises(ToolError):
        init_project(
            config_tree, rec, ["ds"], ProjectOptions(python="3.12"), cwd=project_dir
        )
    assert read_tracking(project_dir / "pyproject.toml") is None


def test_init_project_no_track_removes_table_even_when_add_fails(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.errors import ToolError
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_notrack_addfail"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = []\n'
        "\n[tool.uv-stack]\nversion = 1\nstack = [\"ds\"]\napplied = []\n"
    )

    def _fail_add(cmd: Command) -> CommandResult:
        if "add" in cmd.args:
            raise ToolError("add failed", command=cmd.args, returncode=1)
        return CommandResult(returncode=0, stdout="")

    with pytest.raises(ToolError):
        init_project(
            config_tree,
            RecordingRunner(responder=_fail_add),
            ["ds"],
            ProjectOptions(python="3.12", force=True, track=False),
            cwd=project_dir,
        )
    assert read_tracking(project_dir / "pyproject.toml") is None


def test_init_project_force_does_not_adopt_user_dependencies(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_force_adopt"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = ["numpy"]\n'
    )
    init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(python="3.12", force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "pandas" in tracking.applied  # ds brings numpy+pandas
    assert "numpy" not in tracking.applied  # user-owned, never adopted


def test_init_project_force_keeps_previously_owned_packages(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_force_owned"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = ["numpy", "pandas"]\n'
        "\n[tool.uv-stack]\nversion = 1\nstack = [\"ds\"]\n"
        'applied = ["numpy", "pandas"]\n'
    )
    init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(python="3.12", force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    # Previously uv-stack-owned names remain in the ledger.
    assert "numpy" in tracking.applied and "pandas" in tracking.applied


def test_init_project_force_ownership_is_name_normalized(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.profile_path("norm").write_text("includes:\n  - my.pkg\n")
    project_dir = tmp_path / "proj_norm"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = ["My_Pkg"]\n'
    )
    init_project(
        config_tree, RecordingRunner(), ["norm"],
        ProjectOptions(python="3.12", force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert tracking.applied == []  # my.pkg == My_Pkg canonically → user-owned


# ============================================================================
# project refresh tests
# ============================================================================


def _tracked_project(tmp_path, tracking_text: str) -> Path:
    project_dir = tmp_path / "proj_refresh"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas", "rdkit", "rich", "user-extra"]\n'
        + tracking_text
    )
    return project_dir


_TRACKING = (
    "\n[tool.uv-stack]\nversion = 1\n"
    'stack = ["standard"]\n'
    'applied = ["numpy", "pandas", "rdkit", "rich"]\n'
)


def test_refresh_requires_tracked_project(config_tree: ConfigRoot, tmp_path):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    project_dir = tmp_path / "untracked"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text('[project]\nname = "x"\n')
    with pytest.raises(ConfigError) as excinfo:
        refresh_project(
            config_tree, RecordingRunner(), RefreshOptions(), cwd=project_dir
        )
    assert "No tracked project here." in str(excinfo.value)


def test_refresh_schema_version_guard(config_tree: ConfigRoot, tmp_path):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    project_dir = _tracked_project(
        tmp_path, "\n[tool.uv-stack]\nversion = 2\nstack = []\napplied = []\n"
    )
    with pytest.raises(ConfigError) as excinfo:
        refresh_project(
            config_tree, RecordingRunner(), RefreshOptions(), cwd=project_dir
        )
    assert "newer uv-stack (schema 2)" in str(excinfo.value)


def test_refresh_removes_dropped_and_updates_ledger(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Drop rdkit from the chem profile: refresh must remove it.
    config_tree.profile_path("chem").write_text("includes:\n  - chemprop\n")
    project_dir = _tracked_project(tmp_path, _TRACKING)
    rec = RecordingRunner()
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    argv = [" ".join(c.args) for c in rec.commands]
    assert any(a == "uv remove --no-sync rdkit" for a in argv)
    assert any("uv add --no-sync -r" in a for a in argv)
    assert any(a == "uv sync --python 3.12" for a in argv)
    assert result.removed == ["rdkit"]
    assert "chemprop" in result.added
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "chemprop" in tracking.applied and "rdkit" not in tracking.applied
    # user-extra was never in applied → untouched by construction.
    assert "user-extra" not in result.removed


def test_refresh_no_sync_runs_no_sync_anywhere(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.profile_path("chem").write_text("includes:\n  - chemprop\n")
    project_dir = _tracked_project(tmp_path, _TRACKING)
    rec = RecordingRunner()
    refresh_project(
        config_tree, rec, RefreshOptions(python="3.12", no_sync=True), cwd=project_dir
    )
    argv = [" ".join(c.args) for c in rec.commands]
    assert any(a.startswith("uv remove --no-sync") for a in argv)
    assert not any(a.startswith("uv sync") for a in argv)


def test_refresh_skips_non_name_removals(config_tree: ConfigRoot, tmp_path, monkeypatch):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["numpy", "pandas", "-e ./tool", "pkg @ https://h/x.whl"]\n'
    )
    project_dir = _tracked_project(tmp_path, tracking_text)
    rec = RecordingRunner()
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    assert result.skipped_removals == ["-e ./tool", "pkg @ https://h/x.whl"]
    argv = [" ".join(c.args) for c in rec.commands]
    assert not any("./tool" in a or "x.whl" in a for a in argv if a.startswith("uv remove"))


def test_refresh_retry_after_partial_failure_is_idempotent(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.errors import ToolError
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.profile_path("chem").write_text("includes:\n  - chemprop\n")
    project_dir = _tracked_project(tmp_path, _TRACKING)

    def _fail_add(cmd: Command) -> CommandResult:
        if "add" in cmd.args:
            raise ToolError("add failed", command=cmd.args, returncode=1)
        return CommandResult(returncode=0, stdout="")

    with pytest.raises(ToolError):
        refresh_project(
            config_tree,
            RecordingRunner(responder=_fail_add),
            RefreshOptions(python="3.12"),
            cwd=project_dir,
        )
    # Ledger unchanged after the failure.
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None and "rdkit" in tracking.applied
    # Simulate uv having removed rdkit from [project.dependencies] before the
    # failure, WITHOUT touching the [tool.uv-stack] ledger.
    pyproject = project_dir / "pyproject.toml"
    text = pyproject.read_text()
    deps_line = 'dependencies = ["numpy", "pandas", "rdkit", "rich", "user-extra"]'
    assert deps_line in text
    pyproject.write_text(
        text.replace(
            deps_line, 'dependencies = ["numpy", "pandas", "rich", "user-extra"]', 1
        )
    )
    tracking = read_tracking(pyproject)
    assert tracking is not None and "rdkit" in tracking.applied  # ledger intact
    # Retry with a healthy runner: presence filter skips the absent name.
    rec = RecordingRunner()
    refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    argv = [" ".join(c.args) for c in rec.commands]
    assert not any(a.startswith("uv remove") for a in argv)


def test_refresh_dry_run_plans_without_mutation(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.profile_path("chem").write_text("includes:\n  - chemprop\n")
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["standard"]\npython = "main"\n'
        'applied = ["numpy", "pandas", "rdkit", "rich"]\n'
    )
    project_dir = _tracked_project(tmp_path, tracking_text)
    rec = RecordingRunner()
    result = refresh_project(
        config_tree, rec, RefreshOptions(dry_run=True), cwd=project_dir
    )
    assert rec.commands == []  # no probe, no uv execution
    assert result.planned
    plan_text = [" ".join(c.args) for c in result.planned]
    # Recorded python "main" is an env name → placeholder, never probed.
    assert any("<project-python>" in a for a in plan_text)
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None and "rdkit" in tracking.applied  # unchanged


def test_refresh_python_flag_overrides_and_records(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\npython = "3.12"\n'
        'applied = ["numpy", "pandas"]\n'
    )
    project_dir = _tracked_project(tmp_path, tracking_text)
    rec = RecordingRunner()
    refresh_project(
        config_tree, rec, RefreshOptions(python="3.13"), cwd=project_dir
    )
    argv = [" ".join(c.args) for c in rec.commands]
    assert any(a == "uv sync --python 3.13" for a in argv)
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None and tracking.python == "3.13"
