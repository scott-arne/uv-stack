from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.commands import micromamba_create, micromamba_remove
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, EnvError
from uv_stack.models import ProjectTracking
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


def test_ensure_env_shell_quotes_name_in_hint(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError) as exc_info:
        ensure_env(config_tree, rec, "bad;touch", create=False, recreate=False)
    assert "stack create env 'bad;touch'" in str(exc_info.value.hint)


def test_ensure_env_prefixes_leading_dash_name_in_hint(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError) as exc_info:
        ensure_env(config_tree, rec, "--recreate", create=False, recreate=False)
    assert "stack create env -- --recreate" in str(exc_info.value.hint)


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


def test_resolve_project_python_shell_quotes_env_name_in_hint(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError) as exc_info:
        resolve_project_python(config_tree, rec, "bad;touch")
    assert "stack create env 'bad;touch'" in str(exc_info.value.hint)


def test_resolve_project_python_prefixes_leading_dash_name_in_hint(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError) as exc_info:
        resolve_project_python(config_tree, rec, "--recreate")
    assert "stack create env -- --recreate" in str(exc_info.value.hint)


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


def test_init_project_interrupted_pending_hint(config_tree: ConfigRoot, tmp_path):
    """Test that a pyproject with pending state shows a resume hint."""
    project_dir = tmp_path / "proj_interrupted"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n'
        "\n[tool.uv-stack]\nversion = 1\nstack = [\"ds\"]\napplied = []\n"
        'pending = ["numpy"]\n'
    )
    rec = RecordingRunner()
    with pytest.raises(ConfigError) as excinfo:
        init_project(config_tree, rec, ["ds"], ProjectOptions(), cwd=project_dir)
    assert "pyproject.toml already exists." in str(excinfo.value)
    assert "--force to resume" in str(excinfo.value.hint)


def test_init_project_existing_untracked_original_hint(
    config_tree: ConfigRoot, tmp_path
):
    """Test that an untracked pyproject shows the original hint."""
    project_dir = tmp_path / "proj_untracked"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("[project]\nname='x'\n")
    rec = RecordingRunner()
    with pytest.raises(ConfigError) as excinfo:
        init_project(config_tree, rec, ["ds"], ProjectOptions(), cwd=project_dir)
    assert "pyproject.toml already exists." in str(excinfo.value)
    assert "Use --force to add to the existing project." in str(excinfo.value.hint)


def test_init_project_v2_tracked_without_force_raises_newer_schema(
    config_tree: ConfigRoot, tmp_path
):
    """Test that init WITHOUT force over a v2 project raises the newer-schema error."""
    project_dir = tmp_path / "proj_v2"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n'
        "\n[tool.uv-stack]\nversion = 2\nstack = [\"ds\"]\napplied = []\n"
    )
    rec = RecordingRunner()
    with pytest.raises(ConfigError) as excinfo:
        init_project(config_tree, rec, ["ds"], ProjectOptions(), cwd=project_dir)
    assert "newer uv-stack (schema 2)" in str(excinfo.value)


def test_init_project_corrupt_tracking_without_force_shows_original_hint(
    config_tree: ConfigRoot, tmp_path
):
    """Test that corrupt tracking (non-newer-schema) WITHOUT force shows generic hint."""
    project_dir = tmp_path / "proj_corrupt"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n'
        '\n[tool.uv-stack]\nversion = "x"\nstack = []\napplied = []\n'
    )
    rec = RecordingRunner()
    with pytest.raises(ConfigError) as excinfo:
        init_project(config_tree, rec, ["ds"], ProjectOptions(), cwd=project_dir)
    assert "pyproject.toml already exists." in str(excinfo.value)
    assert "Use --force to add to the existing project." in str(excinfo.value.hint)


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


def test_init_project_force_no_track_over_pending_skips_adoption(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """--no-track deletes the table, so nothing may be adopted into it.

    The orphan warnings tell the user the next 'stack refresh' will report the
    entry — advice that cannot hold once the project is untracked.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_notrack_pending"
    project_dir.mkdir()
    # A crashed tracked run: chemprop reached the dependencies and the pending
    # record, and the stack (ds) no longer provides it — the adoption trigger.
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["chemprop"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = []\n'
        'pending = ["chemprop"]\n'
    )
    warnings = init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(force=True, track=False), cwd=project_dir,
    )
    assert read_tracking(project_dir / "pyproject.toml") is None
    assert not any("interrupted run" in w for w in warnings), warnings
    assert not any("refresh" in w for w in warnings), warnings


def test_init_project_force_tracked_over_pending_still_adopts(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """The tracked path is unchanged: the same state still adopts and warns."""
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_track_pending"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["chemprop"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = []\n'
        'pending = ["chemprop"]\n'
    )
    warnings = init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None and "chemprop" in tracking.applied
    assert any("interrupted run" in w for w in warnings), warnings


def test_init_add_failure_leaves_pending_intent(
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
    stale = read_tracking(project_dir / "pyproject.toml")
    assert stale is not None
    assert stale.applied == []
    assert stale.pending == ["numpy", "pandas"]


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


def test_init_project_no_track_probe_failure_preserves_ledger(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A failed interpreter probe leaves the ledger alone, even with --no-track.

    Adjudicated, not incidental. --no-track removes the table as part of
    creating the project; the probe runs before anything is created, so a bad
    --python value aborts the command with the directory untouched. The
    opposite pin is test_init_project_no_track_removes_table_even_when_add_fails:
    once the run has begun mutating the project, no tool failure may leave the
    table behind. Moving the removal above the probe would break this test --
    that is the intended signal, not a stale assertion.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_notrack_probe_fail"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = []\n'
        "\n[tool.uv-stack]\nversion = 1\nstack = [\"ds\"]\napplied = []\n"
    )

    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError):
        init_project(
            config_tree,
            rec,
            ["ds"],
            ProjectOptions(python="nope", force=True, track=False),
            cwd=project_dir,
        )
    # The probe failed, so the ledger must still be present.
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert tracking.stack == ["ds"]
    assert tracking.applied == []


def test_init_project_no_track_rejects_malformed_toml_before_scaffolding(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """Malformed TOML must abort before any scaffolding, even with --no-track.

    Adjudicated boundary. With --track, validate_tracking_write rejects the
    malformed input before anything runs. With --no-track that pre-flight is
    skipped, so the refusal lands at the remove_tracking call instead — but it
    must still happen before any uv command runs. This is the one path where
    the malformed-input error surfaces at remove_tracking rather than at the
    validate_tracking_write pre-flight. Wrapping that call in try/except to
    restore the old "returned False" behavior would leave every other test
    green. Dropping _read_exact's parse guard also fails this test, which is
    the only place that refusal is pinned from the operations layer.
    """
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_notrack_malformed"
    project_dir.mkdir()
    # A lone \r mid-line: read_tracking uses universal newlines and accepts it,
    # but _read_exact uses newline="" and rejects it. That asymmetry is why
    # the failure lands at remove_tracking instead of an earlier pre-flight.
    (project_dir / "pyproject.toml").write_bytes(
        b'[project]\nname = "demo"\n\n[tool.uv-stack]\n'
        b'version = 1\nstack = ["a"]\rapplied = []\n'
    )
    original_bytes = (project_dir / "pyproject.toml").read_bytes()

    rec = RecordingRunner()
    with pytest.raises(ConfigError):
        init_project(
            config_tree,
            rec,
            ["a"],
            ProjectOptions(python="3.12", force=True, track=False),
            cwd=project_dir,
        )
    # No uv command ran.
    assert rec.commands == []
    # The malformed file is unchanged.
    assert (project_dir / "pyproject.toml").read_bytes() == original_bytes


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
    # Declining to adopt must not mean removing: the user's own declaration
    # stays in [project].dependencies.
    assert '"numpy"' in (project_dir / "pyproject.toml").read_text()


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


def test_init_project_force_filters_user_owned_from_temp_file(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_user_filter"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = ["numpy<2"]\n'
    )
    captured_req: str | None = None

    def capture_responder(cmd: Command) -> CommandResult:
        nonlocal captured_req
        if "add" in cmd.args and "-r" in cmd.args:
            req_idx = cmd.args.index("-r")
            req_path = Path(cmd.args[req_idx + 1])
            if req_path.exists():
                captured_req = req_path.read_text()
        return CommandResult(returncode=0, stdout="")

    warnings = init_project(
        config_tree, RecordingRunner(responder=capture_responder), ["ds"],
        ProjectOptions(python="3.12", force=True), cwd=project_dir,
    )
    # ds brings both numpy and pandas
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "pandas" in tracking.applied
    assert "numpy" not in tracking.applied  # user-owned, excluded from ledger
    # temp requirements file must not contain numpy
    assert captured_req is not None
    assert "pandas" in captured_req
    assert "numpy" not in captured_req
    # warning emitted for the excluded entry
    assert any("numpy" in w and "user-owned" in w for w in warnings)


def test_init_project_force_schema_version_guard_track_true(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """init_project --force with track=True must reject newer tracking schemas."""
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_schema_guard_track"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n'
        "\n[tool.uv-stack]\nversion = 2\nstack = [\"ds\"]\napplied = []\n"
    )
    rec = RecordingRunner()
    with pytest.raises(ConfigError) as excinfo:
        init_project(
            config_tree, rec, ["ds"],
            ProjectOptions(python="3.12", force=True, track=True),
            cwd=project_dir,
        )
    assert "newer uv-stack (schema 2)" in str(excinfo.value)
    # Zero uv commands should have run (guard fires before any mutation).
    assert len(rec.commands) == 0


def test_init_project_force_schema_version_guard_track_false(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """init_project --force with track=False must also reject newer tracking schemas."""
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_schema_guard_notrack"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n'
        "\n[tool.uv-stack]\nversion = 2\nstack = [\"ds\"]\napplied = []\n"
    )
    rec = RecordingRunner()
    with pytest.raises(ConfigError) as excinfo:
        init_project(
            config_tree, rec, ["ds"],
            ProjectOptions(python="3.12", force=True, track=False),
            cwd=project_dir,
        )
    assert "newer uv-stack (schema 2)" in str(excinfo.value)
    # Zero uv commands should have run (guard fires before any mutation).
    assert len(rec.commands) == 0


# ============================================================================
# project refresh tests
# ============================================================================


def _tracked_project(
    tmp_path, tracking_text: str, *, extra_dependencies: tuple[str, ...] = ()
) -> Path:
    """Write a tracked project whose deps cover the stack plus a user-owned one.

    :param tracking_text: The ``[tool.uv-stack]`` table to append verbatim.
    :param extra_dependencies: Further ``[project.dependencies]`` entries, for
        scenarios whose tracking table claims something the default list does
        not carry (a direct reference, say).
    """
    project_dir = tmp_path / "proj_refresh"
    project_dir.mkdir()
    deps = ", ".join(
        f'"{dep}"'
        for dep in ("numpy", "pandas", "rdkit", "rich", "user-extra", *extra_dependencies)
    )
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        f"dependencies = [{deps}]\n" + tracking_text
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
    assert f"No tracked project in {project_dir}." in str(excinfo.value)


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
    pyproject = project_dir / "pyproject.toml"
    stale = read_tracking(pyproject)
    assert stale is not None and stale.pending is not None  # intent record present
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


def _mutating_responder(project_dir: Path):
    """Create a responder that simulates uv add/remove mutating [project.dependencies]."""
    import json
    import re

    pyproject = project_dir / "pyproject.toml"

    def responder(cmd: Command) -> CommandResult:
        if "remove" in cmd.args and "--no-sync" in cmd.args:
            # Extract names after --no-sync.
            idx = cmd.args.index("--no-sync")
            names_to_remove = set(cmd.args[idx + 1 :])
            text = pyproject.read_text()
            # Parse dependencies array from the line.
            match = re.search(r'dependencies\s*=\s*(\[.*?\])', text, re.DOTALL)
            if match:
                deps = json.loads(match.group(1))
                deps = [d for d in deps if d not in names_to_remove]
                new_line = f"dependencies = {json.dumps(deps)}"
                text = text[: match.start()] + new_line + text[match.end() :]
                pyproject.write_text(text)
            # Assert cwd equals project dir.
            assert cmd.cwd == project_dir
            return CommandResult(returncode=0, stdout="")
        elif "add" in cmd.args and "--no-sync" in cmd.args:
            # Find the temp requirements file.
            req_file = None
            for arg in cmd.args:
                if arg.startswith("/") and arg.endswith(".txt"):
                    req_file = Path(arg)
                    break
            if req_file and req_file.exists():
                text = pyproject.read_text()
                match = re.search(r'dependencies\s*=\s*(\[.*?\])', text, re.DOTALL)
                if match:
                    deps = json.loads(match.group(1))
                    # Add non-comment lines from the requirements file.
                    for line in req_file.read_text().splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            if line not in deps:
                                deps.append(line)
                    new_line = f"dependencies = {json.dumps(deps)}"
                    text = text[: match.start()] + new_line + text[match.end() :]
                    pyproject.write_text(text)
            assert cmd.cwd == project_dir
            return CommandResult(returncode=0, stdout="")
        elif cmd.args[0] == "uv" and cmd.args[1] == "sync":
            assert cmd.cwd == project_dir
            return CommandResult(returncode=0, stdout="")
        return CommandResult(returncode=0, stdout="")

    return responder


def test_refresh_convergence_simulates_dependency_file_mutation(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    import json
    import re

    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Drop rdkit from the chem profile: refresh should remove it and add chemprop.
    config_tree.profile_path("chem").write_text("includes:\n  - chemprop\n")
    project_dir = _tracked_project(tmp_path, _TRACKING)
    pyproject = project_dir / "pyproject.toml"

    rec = RecordingRunner(responder=_mutating_responder(project_dir))
    refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    # After refresh: [project.dependencies] should contain chemprop and not rdkit.
    text = pyproject.read_text()
    match = re.search(r'dependencies\s*=\s*(\[.*?\])', text, re.DOTALL)
    assert match
    deps = json.loads(match.group(1))
    assert "chemprop" in deps
    assert "rdkit" not in deps
    # Tracking.applied should match (excludes user-extra which was never in applied).
    tracking = read_tracking(pyproject)
    assert tracking is not None
    assert "chemprop" in tracking.applied and "rdkit" not in tracking.applied
    # All uv commands ran with cwd=project_dir (verified by the responder assertions).


def test_refresh_canonical_presence_filter_uses_original_name(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.profile_path("norm").write_text("includes:\n  - my.pkg\n")
    # Tracking has old applied entry `my.pkg` (stack drops it).
    # [project.dependencies] spells it `My_Pkg` (canonical match, different spelling).
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["norm"]\n'
        'applied = ["my.pkg"]\n'
    )
    project_dir = tmp_path / "proj_canonical"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["My_Pkg"]\n' + tracking_text
    )
    rec = RecordingRunner()
    # Drop my.pkg from the profile so refresh removes it.
    config_tree.profile_path("norm").write_text("includes:\n  - other-package\n")
    refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    argv = [" ".join(c.args) for c in rec.commands]
    # uv remove must receive the ORIGINAL extracted name `my.pkg`, not the canonical form.
    assert any("my.pkg" in a for a in argv if a.startswith("uv remove"))


def test_refresh_direct_reference_skipped_before_requirement_name(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Applied contains a direct reference with '@' but no slash: `pkg @ file:wheel.whl`.
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["numpy", "pandas", "pkg @ file:wheel.whl"]\n'
    )
    project_dir = tmp_path / "proj_directref"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas", "pkg @ file:wheel.whl"]\n' + tracking_text
    )
    rec = RecordingRunner()
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    # The direct reference must be in skipped_removals, not removed.
    assert "pkg @ file:wheel.whl" in result.skipped_removals
    argv = [" ".join(c.args) for c in rec.commands]
    # uv remove argv must NOT mention "pkg".
    assert not any("remove" in a and "pkg" in a for a in argv)


def test_refresh_preserves_user_owned_dependencies(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Tracked project: old applied lacks numpy.
    # [project.dependencies] contains user-owned numpy.
    # Stack resolves to include numpy → after refresh, new applied must still exclude numpy.
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["pandas"]\n'
    )
    project_dir = tmp_path / "proj_ownership"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas"]\n' + tracking_text
    )
    rec = RecordingRunner()
    refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    # numpy is user-owned → must NOT be in applied.
    assert "numpy" not in tracking.applied
    # pandas was in old applied and is in new_flat → should be in applied.
    assert "pandas" in tracking.applied


def test_refresh_user_owned_never_in_temp_requirements_file(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # User owns numpy<2 in [project.dependencies] (absent from old applied).
    # Stack resolves to include numpy and pandas.
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = []\n'
    )
    project_dir = tmp_path / "proj_ownership_file"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy<2"]\n' + tracking_text
    )

    captured_file_content = None

    def capture_responder(cmd: Command) -> CommandResult:
        nonlocal captured_file_content
        if "add" in cmd.args and "--no-sync" in cmd.args:
            # Capture the temp requirements file content.
            for arg in cmd.args:
                if arg.startswith("/") and arg.endswith(".txt"):
                    req_file = Path(arg)
                    if req_file.exists():
                        captured_file_content = req_file.read_text()
                    break
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=capture_responder)
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    # Assert temp file content lacks any numpy line, contains pandas.
    assert captured_file_content is not None
    lines = [ln.strip() for ln in captured_file_content.splitlines() if ln.strip()]
    assert "pandas" in lines
    assert not any("numpy" in ln for ln in lines)
    # Assert result.warnings has one message naming numpy.
    assert any("numpy" in w and "user-owned" in w for w in result.warnings)
    # Final tracking.applied excludes numpy.
    from uv_stack.operations.pyproject import read_tracking

    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "numpy" not in tracking.applied
    assert "pandas" in tracking.applied


def test_refresh_user_owned_direct_reference_not_adopted(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # User owns `numpy @ https://h/x.whl` in [project.dependencies] (absent from old applied).
    # Stack resolves to include numpy and pandas (ds stack).
    # After refresh: numpy must not land in applied (user-owned direct ref).
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = []\n'
    )
    project_dir = tmp_path / "proj_directref_ownership"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy @ https://h/x.whl"]\n' + tracking_text
    )

    captured_file_content = None

    def capture_responder(cmd: Command) -> CommandResult:
        nonlocal captured_file_content
        if "add" in cmd.args and "--no-sync" in cmd.args:
            # Capture the temp requirements file content.
            for arg in cmd.args:
                if arg.startswith("/") and arg.endswith(".txt"):
                    req_file = Path(arg)
                    if req_file.exists():
                        captured_file_content = req_file.read_text()
                    break
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=capture_responder)
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    # Assert temp file content has no numpy line, contains pandas.
    assert captured_file_content is not None
    lines = [ln.strip() for ln in captured_file_content.splitlines() if ln.strip()]
    assert "pandas" in lines
    assert not any("numpy" in ln for ln in lines)
    # Assert result.warnings has one message naming numpy.
    assert any("numpy" in w and "user-owned" in w for w in result.warnings)
    # Final tracking.applied excludes numpy.
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "numpy" not in tracking.applied
    assert "pandas" in tracking.applied


def test_refresh_direct_reference_in_skipped_removals_not_removed(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Applied containing `pkg @ file:wheel.whl` that the stack drops.
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["numpy", "pandas", "pkg @ file:wheel.whl"]\n'
    )
    project_dir = tmp_path / "proj_directref_removed"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas", "pkg @ file:wheel.whl"]\n' + tracking_text
    )
    rec = RecordingRunner()
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    # Assert it appears in skipped_removals and NOT in removed.
    assert "pkg @ file:wheel.whl" in result.skipped_removals
    assert "pkg @ file:wheel.whl" not in result.removed


def test_refresh_direct_reference_owned_not_classified_user_owned(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """Old ledger and current deps both have 'pkg @ url', stack resolves
    pkg==1.0 → NOT user-owned."""
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.profile_path("directref").write_text("includes:\n  - pkg==1.0\n")
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["directref"]\n'
        'applied = ["pkg @ https://h/x.whl"]\n'
    )
    project_dir = tmp_path / "proj_directref_owned"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["pkg @ https://h/x.whl"]\n' + tracking_text
    )

    captured_file_content = None

    def capture_responder(cmd: Command) -> CommandResult:
        nonlocal captured_file_content
        if "add" in cmd.args and "--no-sync" in cmd.args:
            for arg in cmd.args:
                if arg.startswith("/") and arg.endswith(".txt"):
                    req_file = Path(arg)
                    if req_file.exists():
                        captured_file_content = req_file.read_text()
                    break
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=capture_responder)
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    # pkg should NOT be classified user-owned (it was in old ledger).
    assert not any("pkg" in w and "user-owned" in w for w in result.warnings)
    # Temp file should contain pkg==1.0.
    assert captured_file_content is not None
    assert "pkg==1.0" in captured_file_content
    # New ledger should contain pkg==1.0.
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "pkg==1.0" in tracking.applied


def test_refresh_sync_failure_after_add_ledger_written(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """Ledger written after uv add succeeds, before uv sync → sync failure keeps new ledger."""
    from uv_stack.errors import ToolError
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.profile_path("chem").write_text("includes:\n  - chemprop\n")
    project_dir = _tracked_project(tmp_path, _TRACKING)

    def _fail_sync(cmd: Command) -> CommandResult:
        responder = _mutating_responder(project_dir)
        result = responder(cmd)
        if cmd.args[0] == "uv" and cmd.args[1] == "sync":
            raise ToolError("sync failed", command=cmd.args, returncode=1)
        return result

    rec = RecordingRunner(responder=_fail_sync)
    with pytest.raises(ToolError):
        refresh_project(
            config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
        )
    # Ledger was written after successful add → contains chemprop.
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "chemprop" in tracking.applied
    # Second refresh converges: chemprop is NOT classified user-owned.
    rec2 = RecordingRunner(responder=_mutating_responder(project_dir))
    result = refresh_project(
        config_tree, rec2, RefreshOptions(python="3.12"), cwd=project_dir
    )
    assert not any("chemprop" in w and "user-owned" in w for w in result.warnings)
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "chemprop" in tracking.applied


def test_refresh_preflight_validation_blocks_mutation(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """Pre-flight validation failure raises before any uv commands run."""
    from uv_stack.errors import ConfigError
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    rec = RecordingRunner()
    # Monkeypatch validate_tracking_write to raise ConfigError.
    monkeypatch.setattr(
        "uv_stack.operations.project.validate_tracking_write",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ConfigError("Refusing to write pyproject.toml: result would not parse.")
        ),
    )
    with pytest.raises(ConfigError) as excinfo:
        refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    assert "would not parse" in str(excinfo.value)
    # Zero uv commands should have been recorded (pre-flight blocks mutation).
    assert len(rec.commands) == 0


def test_refresh_writes_pending_before_first_uv_command(config_tree: ConfigRoot, tmp_path):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    project_dir = _tracked_project(tmp_path, _TRACKING)
    pyproject = project_dir / "pyproject.toml"
    snapshots: list[ProjectTracking | None] = []

    def responder(cmd: Command) -> CommandResult:
        snapshots.append(read_tracking(pyproject))
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=responder)
    refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    first = snapshots[0]
    assert first is not None
    assert first.pending == ["numpy", "pandas", "rdkit", "rich"]
    assert "rdkit" in first.applied


def test_refresh_pending_cleared_on_success(config_tree: ConfigRoot, tmp_path):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    project_dir = _tracked_project(tmp_path, _TRACKING)
    rec = RecordingRunner()
    refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert tracking.pending is None
    assert "pending" not in (project_dir / "pyproject.toml").read_text()


def test_refresh_final_write_failure_retry_converges(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Drop rdkit from the chem profile: refresh should remove it and add chemprop.
    config_tree.profile_path("chem").write_text("includes:\n  - chemprop\n")
    # The motivating window: add succeeds, the CLEARING write fails once.
    # The pending record written before mutations must shield the retry.
    project_dir = _tracked_project(tmp_path, _TRACKING)
    pyproject = project_dir / "pyproject.toml"
    import uv_stack.operations.project as project_mod

    real_write = project_mod.write_tracking

    def flaky_write(path, tracking):
        if tracking.pending is None:  # the clearing write, post-add
            raise OSError("disk full")
        real_write(path, tracking)

    monkeypatch.setattr(project_mod, "write_tracking", flaky_write)
    rec = RecordingRunner(responder=_mutating_responder(project_dir))
    with pytest.raises(OSError):
        refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    assert any(c.args[:2] == ["uv", "add"] for c in rec.commands)
    crashed = read_tracking(pyproject)
    assert crashed is not None and crashed.pending is not None
    assert "rdkit" in crashed.applied  # old ledger still on disk

    monkeypatch.setattr(project_mod, "write_tracking", real_write)
    rec2 = RecordingRunner(responder=_mutating_responder(project_dir))
    result = refresh_project(config_tree, rec2, RefreshOptions(python="3.12"), cwd=project_dir)
    final = read_tracking(pyproject)
    assert final is not None and final.pending is None
    assert "chemprop" in final.applied  # ownership retained across the crash
    assert not any("user-owned" in w for w in result.warnings)


def test_refresh_orphan_adoption_three_run_convergence(
    config_tree: ConfigRoot, tmp_path: Path
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    # Crashed state: chemprop applied to deps + pending, then the profile
    # drops chemprop before the retry.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "chemprop"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = ["numpy"]\n'
        'pending = ["numpy", "chemprop"]\n'
    )
    pyproject = project_dir / "pyproject.toml"
    snapshots: list[ProjectTracking | None] = []

    def responder(cmd: Command) -> CommandResult:
        snapshots.append(read_tracking(pyproject))
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=responder)
    result = refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    # Adoption is durable from the FIRST write.
    first = snapshots[0]
    assert first is not None and "chemprop" in first.applied
    assert any("interrupted run" in w and "chemprop" in w for w in result.warnings)
    after = read_tracking(pyproject)
    assert after is not None
    assert "chemprop" in after.applied and after.pending is None
    # Third refresh removes the orphan through the normal dropped-diff.
    rec2 = RecordingRunner()
    result2 = refresh_project(config_tree, rec2, RefreshOptions(python="3.12"), cwd=project_dir)
    assert "chemprop" in result2.removed
    remove_cmds = [c for c in rec2.commands if c.args[:2] == ["uv", "remove"]]
    assert remove_cmds and "chemprop" in remove_cmds[0].args


def test_refresh_never_applied_pending_cleared_without_adoption(
    config_tree: ConfigRoot, tmp_path: Path
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    # Pending name absent from dependencies: crash happened before its add
    # took effect — cleared silently, no adoption, no warning.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = ["numpy"]\n'
        'pending = ["numpy", "ghost-package"]\n'
    )
    rec = RecordingRunner()
    result = refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    after = read_tracking(project_dir / "pyproject.toml")
    assert after is not None and after.pending is None
    assert not any("ghost-package" in e for e in after.applied)
    assert not any("interrupted run" in w for w in result.warnings)


def test_refresh_direct_reference_orphan_gets_manual_warning(
    config_tree: ConfigRoot, tmp_path: Path
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pkg @ https://h/x.whl"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = ["numpy"]\n'
        'pending = ["numpy", "pkg @ https://h/x.whl"]\n'
    )
    rec = RecordingRunner()
    result = refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    assert any("will not be auto-removed" in w for w in result.warnings)
    after = read_tracking(project_dir / "pyproject.toml")
    assert after is not None and "pkg @ https://h/x.whl" in after.applied
    # The following refresh reports it as skipped, never uv-removed.
    rec2 = RecordingRunner()
    result2 = refresh_project(config_tree, rec2, RefreshOptions(python="3.12"), cwd=project_dir)
    assert "pkg @ https://h/x.whl" in result2.skipped_removals
    assert not any("pkg" in c.args for c in rec2.commands if c.args[:2] == ["uv", "remove"])


def test_refresh_nameless_pending_entry_warns_without_adoption(
    config_tree: ConfigRoot, tmp_path
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    # Editables/paths have no ownership name — nothing to adopt, but the
    # stale pending entry must be surfaced, not silently dropped.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = ["numpy"]\n'
        'pending = ["numpy", "-e ./local-pkg"]\n'
    )
    rec = RecordingRunner()
    result = refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    assert any("cannot be verified by name" in w for w in result.warnings)
    after = read_tracking(project_dir / "pyproject.toml")
    assert after is not None and after.pending is None
    assert "-e ./local-pkg" not in after.applied


def test_refresh_crash_mid_adoption_resumes_without_duplicate_warning(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    # Adopting retry's FINAL write fails once; orphan already in applied,
    # so the next run treats it as ordinary ledger content.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "chemprop"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = ["numpy"]\n'
        'pending = ["numpy", "chemprop"]\n'
    )
    pyproject = project_dir / "pyproject.toml"
    import uv_stack.operations.project as project_mod

    real_write = project_mod.write_tracking

    def flaky_write(path, tracking):
        if tracking.pending is None:  # the clearing write, post-add
            raise OSError("disk full")
        real_write(path, tracking)

    monkeypatch.setattr(project_mod, "write_tracking", flaky_write)
    rec = RecordingRunner()
    with pytest.raises(OSError):
        refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    assert any(c.args[:2] == ["uv", "add"] for c in rec.commands)
    crashed = read_tracking(pyproject)
    assert crashed is not None and "chemprop" in crashed.applied  # durable adoption

    monkeypatch.setattr(project_mod, "write_tracking", real_write)
    rec2 = RecordingRunner()
    result = refresh_project(config_tree, rec2, RefreshOptions(python="3.12"), cwd=project_dir)
    assert not any("interrupted run" in w for w in result.warnings)
    assert any("chemprop" in entry for entry in result.removed)
    assert any(
        c.args[:2] == ["uv", "remove"] and "chemprop" in c.args for c in rec2.commands
    )


def test_refresh_direct_ref_update_orphan_surfaces_via_skipped_removals(
    config_tree: ConfigRoot, tmp_path
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    # Interrupted direct-ref UPDATE: pending holds the new ref, applied the
    # old one, deps the new one; the stack no longer provides pkg. Adoption
    # skips (name already ledger-owned) — visibility comes from the loud
    # skipped-removals report on this same run.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pkg @ https://h/new.whl"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = ["numpy", "pkg @ https://h/old.whl"]\n'
        'pending = ["numpy", "pkg @ https://h/new.whl"]\n'
    )
    rec = RecordingRunner()
    result = refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    assert "pkg @ https://h/old.whl" in result.skipped_removals
    assert not any(
        "pkg" in c.args for c in rec.commands if c.args[:2] == ["uv", "remove"]
    )
    after = read_tracking(project_dir / "pyproject.toml")
    assert after is not None and after.pending is None
    assert not any("pkg" in e for e in after.applied)


def test_refresh_plain_to_direct_interrupted_update_converges_by_removal(
    config_tree: ConfigRoot, tmp_path
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    # The pending record proves OUR crashed run applied the direct ref, so
    # name-based removal of the stack-dropped name is safe and convergent —
    # and loud (reported in removed). Skipping would silently orphan pkg.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pkg @ https://h/new.whl"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = ["numpy", "pkg==1.0"]\n'
        'pending = ["numpy", "pkg @ https://h/new.whl"]\n'
    )
    rec = RecordingRunner()
    result = refresh_project(config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir)
    assert "pkg==1.0" in result.removed
    remove_cmds = [c for c in rec.commands if c.args[:2] == ["uv", "remove"]]
    assert remove_cmds and "pkg" in remove_cmds[0].args
    after = read_tracking(project_dir / "pyproject.toml")
    assert after is not None and after.pending is None
    assert not any("pkg" in e for e in after.applied)


def test_init_fresh_runs_uv_init_before_any_table(config_tree: ConfigRoot, tmp_path):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    states: list[tuple[list[str], bool]] = []

    def responder(cmd: Command) -> CommandResult:
        states.append((list(cmd.args), pyproject.is_file()))
        if cmd.args[:2] == ["uv", "init"]:
            pyproject.write_text('[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n')
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=responder)
    init_project(config_tree, rec, ["numpy"], ProjectOptions(python="3.12"), cwd=project_dir)
    init_calls = [s for s in states if s[0][:2] == ["uv", "init"]]
    add_calls = [s for s in states if s[0][:2] == ["uv", "add"]]
    assert init_calls and init_calls[0][1] is False  # no pyproject before uv init
    assert add_calls  # the pending-at-add assertion lives in the next test


def test_init_pending_between_init_and_add(config_tree: ConfigRoot, tmp_path: Path):
    from uv_stack.operations.pyproject import read_tracking

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pending_at_add: list[ProjectTracking | None] = []

    def responder(cmd: Command) -> CommandResult:
        if cmd.args[:2] == ["uv", "init"]:
            pyproject.write_text('[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n')
        if cmd.args[:2] == ["uv", "add"]:
            pending_at_add.append(read_tracking(pyproject))
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=responder)
    init_project(config_tree, rec, ["numpy"], ProjectOptions(python="3.12"), cwd=project_dir)
    assert pending_at_add and pending_at_add[0] is not None
    assert pending_at_add[0].applied == []
    assert pending_at_add[0].pending == ["numpy"]
    final = read_tracking(pyproject)
    assert final is not None and final.pending is None


def test_init_force_orphan_adoption(config_tree: ConfigRoot, tmp_path: Path):
    from uv_stack.operations.pyproject import read_tracking

    # Crashed tracked init left applied=[] + pending=["chemprop"]; a --force
    # retry whose stack no longer includes chemprop adopts it at the FIRST
    # write.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["chemprop"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["chemprop"]\n'
        'applied = []\n'
        'pending = ["chemprop"]\n'
    )
    snapshots: list[ProjectTracking | None] = []

    def responder(cmd: Command) -> CommandResult:
        snapshots.append(read_tracking(pyproject))
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=responder)
    warnings = init_project(
        config_tree,
        rec,
        ["numpy"],
        ProjectOptions(python="3.12", force=True),
        cwd=project_dir,
    )
    first = snapshots[0]
    assert first is not None and "chemprop" in first.applied  # durable from first write
    assert any("interrupted run" in w and "chemprop" in w for w in warnings)
    final = read_tracking(pyproject)
    assert final is not None
    assert "chemprop" in final.applied and final.pending is None


def test_init_force_user_pin_survives(config_tree: ConfigRoot, tmp_path: Path):
    # Sharpen the existing ownership test: the user's pin text survives
    # verbatim in [project.dependencies] (write_tracking never touches it).
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy<2"]\n'
    )
    rec = RecordingRunner()
    init_project(
        config_tree,
        rec,
        ["numpy", "pandas"],
        ProjectOptions(python="3.12", force=True),
        cwd=project_dir,
    )
    assert '"numpy<2"' in pyproject.read_text()


def test_init_force_triple_crash_carries_owned_orphan(
    config_tree: ConfigRoot, tmp_path: Path
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    # Triple-crash scenario: crashed init left applied=["chemprop"],
    # pending=["numpy"] (fresh target), deps contain both, stack resolves ["numpy"].
    # --force retry completes → final applied contains BOTH chemprop AND numpy,
    # pending None; a follow-up refresh removes chemprop (reported in result.removed).
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["chemprop", "numpy"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = ["chemprop"]\n'
        'pending = ["numpy"]\n'
    )
    rec = RecordingRunner()
    init_project(
        config_tree,
        rec,
        ["numpy"],
        ProjectOptions(python="3.12", force=True),
        cwd=project_dir,
    )
    after = read_tracking(pyproject)
    assert after is not None
    assert "chemprop" in after.applied
    assert "numpy" in after.applied
    assert after.pending is None
    # Follow-up refresh removes chemprop.
    rec2 = RecordingRunner()
    result = refresh_project(config_tree, rec2, RefreshOptions(python="3.12"), cwd=project_dir)
    assert "chemprop" in result.removed
    final = read_tracking(pyproject)
    assert final is not None
    assert "chemprop" not in final.applied
    assert "numpy" in final.applied


def test_init_force_same_stack_retry_keeps_pending_package_owned(
    config_tree: ConfigRoot, tmp_path
):
    from uv_stack.operations.pyproject import read_tracking
    from uv_stack.parse import ownership_name

    # Crashed init, stack UNCHANGED: the pending package must ride the
    # ownership union into stack_adds — not be misclassified user-owned and
    # falsely adopted as an orphan.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["numpy"]\n'
        'applied = []\n'
        'pending = ["numpy"]\n'
    )
    rec = RecordingRunner()
    warnings = init_project(
        config_tree,
        rec,
        ["numpy"],
        ProjectOptions(python="3.12", force=True),
        cwd=project_dir,
    )
    assert not any("user-owned" in w for w in warnings)
    assert not any("interrupted run" in w for w in warnings)
    final = read_tracking(pyproject)
    assert final is not None and final.pending is None
    assert any(ownership_name(e) == "numpy" for e in final.applied)


def test_init_preflight_blocks_before_any_command(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    from uv_stack.errors import ConfigError

    # Pre-flight fires before uv init/uv add AND before any env probe.
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    import uv_stack.operations.project as project_mod

    def exploding_validate(path, tracking):
        raise ConfigError("pre-flight rejected")

    monkeypatch.setattr(project_mod, "validate_tracking_write", exploding_validate)
    rec = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ConfigError):
        init_project(config_tree, rec, ["numpy"], ProjectOptions(python="main"), cwd=project_dir)
    assert len(rec.commands) == 0


def test_refresh_profile_direct_reference_respects_user_ownership(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """Profile containing 'pkg @ https://h/x.whl' respects user ownership filter."""
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Create profile with a direct reference and another package.
    config_tree.profile_path("directref").write_text(
        "includes:\n  - pkg @ https://h/x.whl\n  - numpy\n"
    )
    # User owns pkg (in deps, absent from applied).
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["directref"]\n'
        'applied = []\n'
    )
    project_dir = tmp_path / "proj_directref_user"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["pkg==1.0"]\n' + tracking_text
    )

    captured_file_content = None

    def capture_responder(cmd: Command) -> CommandResult:
        nonlocal captured_file_content
        if "add" in cmd.args and "--no-sync" in cmd.args:
            for arg in cmd.args:
                if arg.startswith("/") and arg.endswith(".txt"):
                    req_file = Path(arg)
                    if req_file.exists():
                        captured_file_content = req_file.read_text()
                    break
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=capture_responder)
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    # Temp file must contain numpy, not pkg.
    assert captured_file_content is not None
    lines = [ln.strip() for ln in captured_file_content.splitlines() if ln.strip()]
    assert "numpy" in lines
    assert not any("pkg" in ln for ln in lines)
    # Warning emitted for the excluded entry.
    assert any("pkg" in w and "user-owned" in w for w in result.warnings)
    # Final tracking.applied excludes pkg.
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "numpy" in tracking.applied
    # Not `"pkg" not in [...]`: that only rules out the bare token, while the
    # entry actually at risk is the full 'pkg @ https://h/x.whl' direct reference.
    assert not any("pkg" in e for e in tracking.applied)


def test_init_force_profile_direct_reference_respects_user_ownership(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """init --force with profile containing 'pkg @ url' respects user ownership."""
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Create profile with a direct reference.
    config_tree.profile_path("directref").write_text(
        "includes:\n  - pkg @ https://h/x.whl\n"
    )
    project_dir = tmp_path / "proj_init_directref"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    # Existing project with user-owned pkg==1.0.
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["pkg==1.0"]\n'
    )

    captured_file_content = None

    def capture_responder(cmd: Command) -> CommandResult:
        nonlocal captured_file_content
        if "add" in cmd.args and "--no-sync" in cmd.args:
            for arg in cmd.args:
                if arg.startswith("/") and arg.endswith(".txt"):
                    req_file = Path(arg)
                    if req_file.exists():
                        captured_file_content = req_file.read_text()
                    break
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=capture_responder)
    warnings = init_project(
        config_tree,
        rec,
        ["directref"],
        ProjectOptions(python="3.12", force=True),
        cwd=project_dir,
    )
    # Temp file must not contain pkg.
    assert captured_file_content is not None
    lines = [ln.strip() for ln in captured_file_content.splitlines() if ln.strip()]
    assert not any("pkg" in ln for ln in lines)
    # Warning emitted for the excluded entry.
    assert any("pkg" in w and "user-owned" in w for w in warnings)
    # Final tracking.applied excludes pkg.
    tracking = read_tracking(pyproject)
    assert tracking is not None
    assert not any("pkg" in e for e in tracking.applied)
    # User's pkg==1.0 survives verbatim.
    assert '"pkg==1.0"' in pyproject.read_text()


def test_refresh_nameless_entries_survive_ownership_filter(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """Editables, VCS URLs, and paths carry no distribution name, so they always pass.

    ``ownership_name`` returns None for these, and the filter compares
    ``canonical_name(name or "")`` — an empty string no dependency can
    produce. Tightening the filter to require a name would silently drop
    every editable from the stack, so pin that they reach both the temp
    requirements file and the ledger while a genuinely user-owned neighbour
    is still excluded.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config_tree.profile_path("nameless").write_text(
        "includes:\n"
        "  - -e ./libs/foo\n"
        "  - git+https://h/y.git\n"
        "  - numpy\n"
        "  - pkg\n"
    )
    project_dir = tmp_path / "proj_nameless"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    # User owns pkg (in deps, absent from applied); everything else is ours.
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["pkg==1.0"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["nameless"]\n'
        "applied = []\n"
    )

    captured_file_content = None

    def capture_responder(cmd: Command) -> CommandResult:
        nonlocal captured_file_content
        if "add" in cmd.args and "--no-sync" in cmd.args:
            for arg in cmd.args:
                if arg.startswith("/") and arg.endswith(".txt"):
                    req_file = Path(arg)
                    if req_file.exists():
                        captured_file_content = req_file.read_text()
                    break
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=capture_responder)
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )
    assert captured_file_content is not None
    lines = [ln.strip() for ln in captured_file_content.splitlines() if ln.strip()]
    assert "-e ./libs/foo" in lines
    assert "git+https://h/y.git" in lines
    assert "numpy" in lines
    assert "pkg" not in lines  # user-owned, filtered
    assert any("pkg" in w and "user-owned" in w for w in result.warnings)
    # A nameless entry must never be reported as user-owned.
    assert not any("libs/foo" in w for w in result.warnings)
    tracking = read_tracking(pyproject)
    assert tracking is not None
    assert "-e ./libs/foo" in tracking.applied
    assert "git+https://h/y.git" in tracking.applied


_DIRECT_REF = "torch @ https://example.invalid/torch-2.0-py3-none-any.whl"

_TRACKING_WITH_DIRECT_REF = (
    "\n[tool.uv-stack]\nversion = 1\n"
    'stack = ["standard"]\n'
    f'applied = ["numpy", "pandas", "rdkit", "rich", "{_DIRECT_REF}"]\n'
)


def test_refresh_attaches_advisories_to_a_failed_sync(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A sync failure must still deliver the advisories the run computed.

    The skipped-removals notice and the ownership warnings would otherwise only
    reach the caller on the RefreshResult, which a raised error never produces,
    so refresh_project attaches both to the error. Nothing else reports the
    notice once the run has got this far: the final ledger lands before uv sync
    already stripped of the skipped entry, so the retry re-resolves against a
    ledger that no longer mentions it. The ownership warning is not in that
    position — the retry does recompute it — so the two are pinned differently
    below.
    """
    from uv_stack.errors import ToolError
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Put a user-owned dependency in the stack so the failing run drops an
    # ownership warning too, not just the skipped-removals notice.
    config_tree.profile_path("utils").write_text("includes:\n  - rich\n  - user-extra\n")
    # A real project carrying this ledger has the direct reference installed.
    project_dir = _tracked_project(
        tmp_path, _TRACKING_WITH_DIRECT_REF, extra_dependencies=(_DIRECT_REF,)
    )

    # The entry IS classified as skipped — the dry run proves the notice exists.
    planned = refresh_project(
        config_tree,
        RecordingRunner(),
        RefreshOptions(python="3.12", dry_run=True),
        cwd=project_dir,
    )
    assert _DIRECT_REF in planned.skipped_removals

    def _fail_sync(cmd: Command) -> CommandResult:
        result = _mutating_responder(project_dir)(cmd)
        if cmd.args[0] == "uv" and cmd.args[1] == "sync":
            raise ToolError("sync failed", command=cmd.args, returncode=1)
        return result

    with pytest.raises(ToolError) as excinfo:
        refresh_project(
            config_tree,
            RecordingRunner(responder=_fail_sync),
            RefreshOptions(python="3.12"),
            cwd=project_dir,
        )

    # Both advisory kinds survive on the error, the notice worded as the
    # success path would have printed it.
    attached = excinfo.value.resolution_warnings
    assert f"Not auto-removed (edit pyproject.toml manually): {_DIRECT_REF}" in attached
    assert any("user-extra" in w and "user-owned" in w for w in attached)

    # The final ledger landed before sync, without the skipped entry.
    after = read_tracking(project_dir / "pyproject.toml")
    assert after is not None and _DIRECT_REF not in after.applied

    # The retry cannot recompute the notice — hence attaching it at the
    # failure. The ownership warning it can recompute, so only the notice is
    # unrecoverable here.
    retried = refresh_project(
        config_tree,
        RecordingRunner(responder=_mutating_responder(project_dir)),
        RefreshOptions(python="3.12"),
        cwd=project_dir,
    )
    assert _DIRECT_REF not in retried.skipped_removals
    assert any("user-extra" in w and "user-owned" in w for w in retried.warnings)
