from __future__ import annotations

import errno
import os
import shutil
import stat
from pathlib import Path

import pytest

from tests.conftest import _deadline
from uv_stack.commands import (
    micromamba_create,
    micromamba_python_info,
    micromamba_remove,
    uv_pip_check,
    uv_pip_compile,
    uv_pip_compile_for_version,
    uv_pip_sync,
)
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, EnvError, ToolError
from uv_stack.models import ProjectTracking
from uv_stack.operations.create import ensure_env, env_micromamba_exists
from uv_stack.operations.project import (
    PROJECT_PYTHON_ENV,
    PYTHON_TRAVEL_PROSPECTIVE,
    PYTHON_TRAVEL_RECORDED,
    ProjectOptions,
    _is_python_passthrough,
    init_project,
    python_travel_problem,
    resolve_project_python,
    select_project_python,
)
from uv_stack.operations.scaffold import validate_name
from uv_stack.operations.upgrade import UpgradeOptions, _new_candidate_lock, upgrade_env
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


def test_env_interpreter_spawn_failure_returns_none(config_tree: ConfigRoot):
    from uv_stack.errors import ToolError
    from uv_stack.operations.create import env_interpreter

    def _spawn_failure_responder(cmd: Command) -> CommandResult:
        raise ToolError(
            "Could not run micromamba: No such file or directory.",
            command=["micromamba"],
            returncode=127,
        )

    rec = RecordingRunner(responder=_spawn_failure_responder)
    assert env_interpreter(config_tree, rec, "main") is None


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


def _compile_output(cmd: Command) -> Path:
    """The path a recorded ``uv pip compile`` was told to write."""
    return Path(cmd.args[cmd.args.index("-o") + 1])


def _assert_compiles_to_candidate(cmd: Command, lock: Path) -> None:
    """Pin that a compile targets a sibling candidate, not the published lock.

    Without this, a compile that wrote straight into the lock would satisfy
    every other assertion about the recreate path.
    """
    target = _compile_output(cmd)
    assert target != lock
    assert target.parent == lock.parent
    assert target.name.startswith(lock.name + ".")
    assert target.suffix == ".tmp"


def test_upgrade_writes_generated_files_and_runs_sequence(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    result = upgrade_env(config_tree, rec, "main", UpgradeOptions())

    # Generated files are written.
    assert config_tree.env_requirements_in("main").is_file()
    assert config_tree.env_environment_yml("main").is_file()
    assert config_tree.env_requirements_lock("main").is_file()

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
    # Only the drift guard's read-only probe ran; nothing was executed.
    assert rec.commands == [micromamba_python_info("main")]
    # But a plan is returned.
    assert result.planned
    assert any("compile" in c.args for c in result.planned)
    # Dry run never wrote a lock file.
    assert not config_tree.env_requirements_lock("main").is_file()


def _drifted_responder(cmd: Command) -> CommandResult:
    """An env whose interpreter reports 3.13.1 against a configured 3.12."""
    if "run" in cmd.args:
        return CommandResult(returncode=0, stdout="/envs/main/bin/python\n3.13.1\n")
    return CommandResult(returncode=0, stdout="")


def test_upgrade_refuses_python_version_drift(config_tree: ConfigRoot):
    """upgrade_env raises when the running version differs from python.txt."""
    def _mismatched_responder(cmd: Command) -> CommandResult:
        if "run" in cmd.args:
            return CommandResult(returncode=0, stdout="/envs/main/bin/python\n3.13.1\n")
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=_mismatched_responder)
    with pytest.raises(EnvError) as exc_info:
        upgrade_env(config_tree, rec, "main", UpgradeOptions())

    # Error message mentions both versions.
    assert "3.13.1" in str(exc_info.value)
    assert "3.12" in str(exc_info.value)
    # Hint mentions recreate.
    assert "--recreate" in str(exc_info.value.hint)

    # Generated files were NOT written (this is the regression that matters).
    assert not config_tree.env_requirements_in("main").exists()
    assert not config_tree.env_environment_yml("main").exists()

    # No uv commands were recorded.
    assert not any("uv" in " ".join(c.args) for c in rec.commands)


def test_upgrade_drift_exempt_when_recreate(config_tree: ConfigRoot):
    """upgrade_env with recreate=True does not raise on version drift."""
    def _mismatched_responder(cmd: Command) -> CommandResult:
        if "run" in cmd.args:
            return CommandResult(returncode=0, stdout="/envs/main/bin/python\n3.13.1\n")
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=_mismatched_responder)
    # Should not raise.
    upgrade_env(config_tree, rec, "main", UpgradeOptions(recreate=True))


def test_upgrade_dry_run_refuses_python_version_drift(config_tree: ConfigRoot):
    """A dry run is refused exactly where the real command would be.

    The plan a dry run prints must not describe an upgrade the tool declines to
    perform, and the dry run rewrites the generated files on its way past.
    """
    rec = RecordingRunner(responder=_drifted_responder)
    with pytest.raises(EnvError) as exc_info:
        upgrade_env(config_tree, rec, "main", UpgradeOptions(dry_run=True))

    assert "3.13.1" in str(exc_info.value)
    # The refusal lands above the writes, so no generated file was rewritten.
    assert not config_tree.env_requirements_in("main").exists()
    assert not config_tree.env_environment_yml("main").exists()


def test_upgrade_refuses_python_version_drift_when_creating(config_tree: ConfigRoot):
    """The guard applies to create=True too, not only to a plain upgrade.

    Exempting create would mean 'stack create env NAME' silently re-syncs the
    pip layer onto an interpreter python.txt no longer describes.
    """
    rec = RecordingRunner(responder=_drifted_responder)
    with pytest.raises(EnvError) as exc_info:
        upgrade_env(config_tree, rec, "main", UpgradeOptions(create=True))

    assert "3.13.1" in str(exc_info.value)
    assert not any("uv" in " ".join(c.args) for c in rec.commands)


def test_upgrade_drift_refusal_offers_both_remedies(config_tree: ConfigRoot):
    """The hint must name the non-destructive remedy, not only the wipe.

    python.txt defaults to 3.12 when absent, so the destructive remedy alone
    tells the owner of a working 3.13 env to rebuild it at a version nobody
    chose. Naming the path lets them act whether or not the file exists.
    """
    config_tree.env_python_path("main").unlink()

    rec = RecordingRunner(responder=_drifted_responder)
    with pytest.raises(EnvError) as exc_info:
        upgrade_env(config_tree, rec, "main", UpgradeOptions())

    hint = str(exc_info.value.hint)
    # Remedy one: keep the interpreter by writing the version it reports.
    assert str(config_tree.env_python_path("main")) in hint
    assert "3.13.1" in hint
    # Remedy two: rebuild the interpreter, said to be destructive.
    assert "stack create env --recreate main" in hint
    assert "wipes" in hint


def test_upgrade_drift_refusal_hint_survives_a_flag_shaped_env_name(
    config_tree: ConfigRoot,
):
    """The rendered command must not turn its own flag into a positional.

    render_positional_arg emits '-- --recreate' for this name, so a hint that
    appended the flag after it would read 'stack create env -- --recreate
    --recreate' and paste as a request for an env named '--recreate'.
    """
    env_dir = config_tree.env_dir("--recreate")
    env_dir.mkdir(parents=True)
    (env_dir / "stack.txt").write_text("@standard\n")
    (env_dir / "python.txt").write_text("3.12\n")

    rec = RecordingRunner(responder=_drifted_responder)
    with pytest.raises(EnvError) as exc_info:
        upgrade_env(config_tree, rec, "--recreate", UpgradeOptions())

    assert "stack create env --recreate -- --recreate" in str(exc_info.value.hint)


def test_upgrade_drift_refusal_carries_resolution_warnings(config_tree: ConfigRoot):
    """The refusal sits outside the execution handler, so it attaches its own.

    Nothing else attaches them on this path: the advisories the resolve
    computed would be dropped silently if the guard forgot to.
    """
    config_tree.env_stack_path("main").write_text("standrd\n")

    rec = RecordingRunner(responder=_drifted_responder)
    with pytest.raises(EnvError) as exc_info:
        upgrade_env(config_tree, rec, "main", UpgradeOptions())

    assert any("did you mean 'standard'" in w for w in exc_info.value.resolution_warnings)


def test_upgrade_recreate_refusal_carries_resolution_warnings(config_tree: ConfigRoot):
    """The recreate preflight is outside the handler too, and attaches its own."""
    config_tree.env_python_path("main").write_text("3.12.*\n")
    config_tree.env_stack_path("main").write_text("standrd\n")

    rec = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ConfigError) as exc_info:
        upgrade_env(config_tree, rec, "main", UpgradeOptions(recreate=True))

    assert any("did you mean 'standard'" in w for w in exc_info.value.resolution_warnings)


def test_upgrade_drift_guard_fails_open_when_the_probe_raises(config_tree: ConfigRoot):
    """A probe that cannot run must not block an otherwise valid upgrade.

    micromamba missing from PATH makes the version unknowable, and an unknown
    version is not evidence of drift.
    """
    from uv_stack.errors import ToolError

    def _probe_failure_responder(cmd: Command) -> CommandResult:
        if cmd == micromamba_python_info("main"):
            raise ToolError(
                "Could not run micromamba: No such file or directory.",
                command=cmd.args,
                returncode=127,
            )
        return _existing_env_responder(cmd)

    rec = RecordingRunner(responder=_probe_failure_responder)
    upgrade_env(config_tree, rec, "main", UpgradeOptions())

    assert micromamba_python_info("main") in rec.commands
    assert any("compile" in c.args for c in rec.commands)
    assert uv_pip_sync("/envs/main/bin/python", config_tree.env_requirements_lock("main")) in (
        rec.commands
    )


def test_upgrade_recreate_failed_compile_never_destroys_the_env(config_tree: ConfigRoot):
    """An unsatisfiable recreate resolve must leave the environment standing."""
    from uv_stack.errors import ToolError

    lock = config_tree.env_requirements_lock("main")
    lock.write_bytes(b"old-pinned-lock\n")

    def _compile_failure_responder(cmd: Command) -> CommandResult:
        if "run" in cmd.args:
            return CommandResult(returncode=0, stdout="/envs/main/bin/python\n")
        if "compile" in cmd.args:
            # Write to whatever target the command names before failing, the
            # way a half-finished uv resolve would. A responder that never
            # writes cannot tell "compiled to a candidate" from "compiled
            # straight into the published lock", which is the property the
            # byte comparison below exists to prove.
            _compile_output(cmd).write_bytes(b"half-written-candidate\n")
            raise ToolError(
                "uv pip compile failed.",
                command=cmd.args,
                returncode=1,
                detail="ERROR: Could not find a version that satisfies the requirement",
            )
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=_compile_failure_responder)
    with pytest.raises(ToolError):
        upgrade_env(config_tree, rec, "main", UpgradeOptions(recreate=True))

    # The compile really was attempted; the raise below it proves something.
    compile_cmd = next(c for c in rec.commands if "compile" in c.args)
    _assert_compiles_to_candidate(compile_cmd, lock)
    # The guarantee that matters: the destructive pair never ran.
    assert micromamba_remove("main") not in rec.commands
    assert micromamba_create(config_tree.env_environment_yml("main")) not in rec.commands
    # The previous resolution survives, and no candidate lock is left behind.
    assert lock.read_bytes() == b"old-pinned-lock\n"
    assert list(lock.parent.glob("*.tmp")) == []


def test_upgrade_recreate_failed_create_leaves_no_candidate_lock(config_tree: ConfigRoot):
    """The temp-lock guard spans ensure_env, not just the compile."""
    from uv_stack.errors import ToolError

    lock = config_tree.env_requirements_lock("main")
    lock.write_bytes(b"old-pinned-lock\n")

    def _create_failure_responder(cmd: Command) -> CommandResult:
        if "run" in cmd.args:
            return CommandResult(returncode=0, stdout="/envs/main/bin/python\n")
        if "create" in cmd.args:
            raise ToolError(
                "micromamba create failed.", command=cmd.args, returncode=1
            )
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=_create_failure_responder)
    with pytest.raises(ToolError):
        upgrade_env(config_tree, rec, "main", UpgradeOptions(recreate=True))

    # This failure lands after the remove, so the env really is gone — the
    # window the reorder narrows but cannot close. What must still hold: the
    # candidate lock is cleaned up, and the published lock was never touched.
    assert list(lock.parent.glob("*.tmp")) == []
    assert lock.read_bytes() == b"old-pinned-lock\n"


def test_upgrade_recreate_syncs_against_the_rebuilt_interpreter(config_tree: ConfigRoot):
    """Sync and check must use a probe taken after the rebuild, never before."""
    rebuilt = False

    def _rebuilding_responder(cmd: Command) -> CommandResult:
        nonlocal rebuilt
        if "remove" in cmd.args:
            rebuilt = True
            return CommandResult(returncode=0, stdout="")
        if "run" in cmd.args:
            path = "/envs/main/bin/python3.13" if rebuilt else "/envs/main/bin/python3.12"
            return CommandResult(returncode=0, stdout=path + "\n")
        return CommandResult(returncode=0, stdout="")

    rec = RecordingRunner(responder=_rebuilding_responder)
    lock = config_tree.env_requirements_lock("main")
    upgrade_env(config_tree, rec, "main", UpgradeOptions(recreate=True))

    assert uv_pip_sync("/envs/main/bin/python3.13", lock) in rec.commands
    assert uv_pip_check("/envs/main/bin/python3.13") in rec.commands
    # The interpreter that existed before the rebuild reaches nothing.
    assert not any("python3.12" in " ".join(c.args) for c in rec.commands)


def test_upgrade_recreate_rejects_non_plain_python_version(config_tree: ConfigRoot):
    """A conda match spec cannot be resolved against, so recreate refuses first."""
    config_tree.env_python_path("main").write_text("3.12.*\n")

    rec = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ConfigError) as exc_info:
        upgrade_env(config_tree, rec, "main", UpgradeOptions(recreate=True))

    assert "3.12.*" in str(exc_info.value)
    assert "python.txt" in str(exc_info.value)
    assert str(config_tree.env_python_path("main")) in str(exc_info.value.hint)
    # Refused before resolving and, above all, before destroying.
    assert not any("uv" in c.args for c in rec.commands)
    assert micromamba_remove("main") not in rec.commands
    assert list(config_tree.env_requirements_lock("main").parent.glob("*.tmp")) == []
    # Refused above the writes, so the "sources changed" signal is preserved.
    assert not config_tree.env_requirements_in("main").exists()
    assert not config_tree.env_environment_yml("main").exists()


def test_upgrade_dry_run_recreate_rejects_non_plain_python_version(
    config_tree: ConfigRoot,
):
    """A plan must never describe commands the real run would refuse to issue."""
    config_tree.env_python_path("main").write_text("3.12.*\n")

    rec = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ConfigError) as exc_info:
        upgrade_env(
            config_tree, rec, "main", UpgradeOptions(dry_run=True, recreate=True)
        )

    assert "3.12.*" in str(exc_info.value)
    assert rec.commands == []
    assert not config_tree.env_requirements_in("main").exists()
    assert not config_tree.env_environment_yml("main").exists()


def test_upgrade_recreate_compiles_for_version_then_rebuilds(config_tree: ConfigRoot):
    """The recreate sequence: compile, remove, create, sync, check."""
    rec = RecordingRunner(responder=_existing_env_responder)
    lock = config_tree.env_requirements_lock("main")
    upgrade_env(config_tree, rec, "main", UpgradeOptions(recreate=True))

    # Drop the interpreter probes; what remains is the ordering under test.
    steps = [c for c in rec.commands if "run" not in c.args]
    # The compile writes to a temp path, so only its head is comparable.
    assert steps[0].args[:5] == ["uv", "pip", "compile", "--python-version", "3.12"]
    # The target interpreter does not exist yet, so it cannot be named.
    assert "--python" not in steps[0].args
    _assert_compiles_to_candidate(steps[0], lock)
    assert steps[1:] == [
        micromamba_remove("main"),
        micromamba_create(config_tree.env_environment_yml("main")),
        uv_pip_sync("/envs/main/bin/python", lock),
        uv_pip_check("/envs/main/bin/python"),
    ]
    # The candidate was published, and nothing was left behind.
    assert lock.is_file()
    assert list(lock.parent.glob("*.tmp")) == []


def test_upgrade_dry_run_recreate_plan_leads_with_the_compile(config_tree: ConfigRoot):
    rec = RecordingRunner(responder=_existing_env_responder)
    result = upgrade_env(
        config_tree, rec, "main", UpgradeOptions(dry_run=True, recreate=True)
    )
    assert rec.commands == []
    lock = config_tree.env_requirements_lock("main")
    assert result.planned == [
        uv_pip_compile_for_version(
            "3.12", config_tree.env_requirements_in("main"), lock, upgrade=True
        ),
        micromamba_remove("main"),
        micromamba_create(config_tree.env_environment_yml("main")),
        uv_pip_sync("<env-python>", lock),
        uv_pip_check("<env-python>"),
    ]


def test_upgrade_dry_run_omits_the_create_for_an_env_that_is_already_there(
    config_tree: ConfigRoot,
):
    """A plan must never describe commands the real run would refuse to issue.

    ensure_env probes first and creates only what is missing, so --create
    against an existing env issues no create at all. The plan claimed one
    anyway, off the flag alone, and 'micromamba create' over a live env is not
    the no-op that omission would make it look like.
    """
    rec = RecordingRunner(responder=_existing_env_responder)
    result = upgrade_env(
        config_tree, rec, "main", UpgradeOptions(dry_run=True, create=True)
    )
    lock = config_tree.env_requirements_lock("main")
    assert result.planned == [
        uv_pip_compile(
            "<env-python>", config_tree.env_requirements_in("main"), lock, upgrade=True
        ),
        uv_pip_sync("<env-python>", lock),
        uv_pip_check("<env-python>"),
    ]


def test_upgrade_dry_run_keeps_the_create_when_the_env_cannot_be_probed(
    config_tree: ConfigRoot,
):
    # The suppression is conditioned on a positive answer, not on the absence
    # of a negative one. Without micromamba there is no answer, and the honest
    # plan is the one the run would follow if the env turns out to be missing.
    def _spawn_failure(cmd: Command) -> CommandResult:
        raise ToolError("Could not run micromamba.", command=cmd.args, returncode=127)

    rec = RecordingRunner(responder=_spawn_failure)
    result = upgrade_env(
        config_tree, rec, "main", UpgradeOptions(dry_run=True, create=True)
    )
    assert result.planned[0] == micromamba_create(
        config_tree.env_environment_yml("main")
    )


def test_upgrade_dry_run_create_plan_is_unchanged(config_tree: ConfigRoot):
    """Only the recreate plan was reordered; --create still creates first."""
    rec = RecordingRunner(responder=_missing_env_responder)
    result = upgrade_env(
        config_tree, rec, "main", UpgradeOptions(dry_run=True, create=True)
    )
    lock = config_tree.env_requirements_lock("main")
    assert result.planned == [
        micromamba_create(config_tree.env_environment_yml("main")),
        uv_pip_compile(
            "<env-python>", config_tree.env_requirements_in("main"), lock, upgrade=True
        ),
        uv_pip_sync("<env-python>", lock),
        uv_pip_check("<env-python>"),
    ]


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


def test_resolve_project_python_near_miss_version_hint(
    config_tree: ConfigRoot, monkeypatch
):
    """A botched version must not be answered with "create an env by that name".

    '3.12.x' reaches the probe only because it failed the version test, so the
    default hint tells the user to run 'stack create env 3.12.x' — an
    instruction nobody wants carried out, and one that hides the actual typo.
    """
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError) as exc_info:
        resolve_project_python(config_tree, rec, "3.12.x")
    hint = str(exc_info.value.hint)
    assert "looks like a Python version" in hint
    assert "stack create env" not in hint


def test_resolve_project_python_plausible_env_name_keeps_default_hint(
    config_tree: ConfigRoot, monkeypatch
):
    """The near-miss arm must not swallow the case it was carved out of.

    'nope' is indistinguishable from an env the user has yet to create, so the
    create-it hint is still the right answer there.
    """
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    rec = RecordingRunner(responder=_missing_env_responder)
    with pytest.raises(EnvError) as exc_info:
        resolve_project_python(config_tree, rec, "nope")
    hint = str(exc_info.value.hint)
    assert "stack create env nope" in hint
    assert "looks like a Python version" not in hint


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


def test_init_resume_carries_a_nameless_applied_entry(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A reference-bearing applied entry must survive a resumed init.

    ownership_name returns None for an editable, and the carry's name guard
    read that as "do not carry", so the entry left the ledger while 'oldpkg'
    stayed in [project.dependencies]. A dependency that has left the ledger is
    user-owned from then on, so no later refresh reports or removes it: the
    record is wrong, stays wrong, and nothing is printed. The assertion is on
    the exact unexpanded spelling because that form is the entry's only
    identity, and storing it unexpanded is what the ledger is for.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_carry_nameless"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "oldpkg"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["-e ${DEV}/oldpkg"]\n'
        'pending = ["numpy"]\n'
    )
    init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "-e ${DEV}/oldpkg" in tracking.applied


def test_init_resume_does_not_duplicate_a_nameless_entry_the_stack_provides(
    tmp_path: Path, monkeypatch
):
    """The carry must not re-add a nameless entry that arrives via the stack.

    With no name to compare, the carry falls back to the exact spelling. When
    the stack still supplies that spelling the entry is already in the new
    applied list, and carrying it as well would write a duplicate row into a
    durable record. Counting occurrences rather than asserting membership is
    the whole point: 'in' passes either way.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    config = _portable_root(tmp_path, "root-carry", "/checkouts/a")
    project_dir = tmp_path / "proj_carry_dup"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["rich", "widget"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["dev"]\n'
        'applied = ["-e ${DEV}/widget"]\n'
        'pending = ["rich"]\n'
    )
    init_project(
        config, RecordingRunner(), ["dev"],
        ProjectOptions(python="3.12", force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert list(tracking.applied).count("-e ${DEV}/widget") == 1


def test_init_resume_drops_a_named_entry_that_left_the_dependencies(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A named applied entry no longer in [project.dependencies] must not carry.

    The carry exists to keep a claim on something still installed. Once the
    dependency is gone from the project, keeping its ledger row resurrects a
    claim on a package that is not there and hands the next refresh a removal
    to attempt. test_init_force_triple_crash_carries_owned_orphan pins the
    positive case; this pins the condition that separates the two, which
    nothing else did -- deleting it left the suite green.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_carry_gone"
    project_dir.mkdir()
    # 'chemprop' is in the ledger but deliberately absent from dependencies.
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["chemprop"]\n'
        'pending = ["numpy"]\n'
    )
    init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert "chemprop" not in tracking.applied


def test_init_resume_does_not_duplicate_a_named_entry_the_stack_provides(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """The carry must not re-add a named entry that arrives via the stack.

    The nameless branch has its own duplicate guard, pinned above by counting
    occurrences of the spelling. This is the named branch's equivalent, and it
    was pinned by nothing: with the stack_names check removed the ledger takes
    a second 'numpy' row and every other test still passes. Counting rather
    than asserting membership is again the point -- a durable record with a
    duplicate row reads as correct to 'in'.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_carry_named_dup"
    project_dir.mkdir()
    # numpy is in applied AND in the stack the resume re-resolves to.
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["numpy"]\n'
        'pending = ["pandas"]\n'
    )
    init_project(
        config_tree, RecordingRunner(), ["ds"],
        ProjectOptions(force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert list(tracking.applied).count("numpy") == 1


def test_init_force_reset_without_pending_does_not_carry_the_old_stack(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A --force reset with nothing pending starts the ledger from the new stack.

    The carry is resume-only: it exists to rescue entries a crashed retry would
    strand, and a project with no pending run stranded nothing. Without the
    previous_pending half of the guard a --force onto a different stack keeps
    the old stack's packages as stack-owned, which is the opposite of what a
    reset means. Nothing pinned that half, so the guard could have been halved
    silently -- including by the rewrite this test now sits beside.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_force_reset"
    project_dir.mkdir()
    # Tracked by the 'ds' stack, no pending run, re-initialized onto 'utils'.
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["numpy", "pandas"]\n'
    )
    init_project(
        config_tree, RecordingRunner(), ["utils"],
        ProjectOptions(force=True), cwd=project_dir,
    )
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert list(tracking.applied) == ["rich"]


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


def test_refresh_options_stack_overrides_the_recorded_stack(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)

    refresh_project(
        config_tree,
        RecordingRunner(),
        RefreshOptions(python="3.12", stack=["standard", "@qsar"]),
        cwd=project_dir,
    )

    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert tracking.stack == ["standard", "@qsar"]
    assert "umap-learn" in tracking.applied
    assert tracking.pending is None


def test_refresh_leaves_the_recorded_stack_alone_by_default(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)

    refresh_project(
        config_tree, RecordingRunner(), RefreshOptions(python="3.12"), cwd=project_dir
    )

    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert tracking.stack == ["standard"]


def test_refresh_dry_run_returns_the_target_stack(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    before = (project_dir / "pyproject.toml").read_bytes()
    rec = RecordingRunner()

    result = refresh_project(
        config_tree,
        rec,
        RefreshOptions(python="3.12", stack=["standard", "@qsar"], dry_run=True),
        cwd=project_dir,
    )

    assert result.stack == ["standard", "@qsar"]
    assert result.added == ["umap-learn"]
    assert rec.commands == []
    assert (project_dir / "pyproject.toml").read_bytes() == before


def test_refresh_dry_run_reports_a_stack_change_with_an_empty_delta(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    # 'chem' is already inside 'standard', so the stack grows while the
    # dependency delta stays empty -- the case RefreshResult.stack exists for.
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)

    result = refresh_project(
        config_tree,
        RecordingRunner(),
        RefreshOptions(python="3.12", stack=["standard", "chem"], dry_run=True),
        cwd=project_dir,
    )

    assert result.stack == ["standard", "chem"]
    assert result.added == []
    assert result.removed == []


def test_refresh_dry_run_leaves_stack_none_without_an_override(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)

    result = refresh_project(
        config_tree,
        RecordingRunner(),
        RefreshOptions(python="3.12", dry_run=True),
        cwd=project_dir,
    )

    assert result.stack is None


def test_union_project_stack_appends_only_new_tokens(tmp_path):
    from uv_stack.operations.project import union_project_stack

    project_dir = _tracked_project(tmp_path, _TRACKING)
    pyproject = project_dir / "pyproject.toml"

    assert union_project_stack(pyproject, project_dir, ["@qsar"]) == ["standard", "@qsar"]
    assert union_project_stack(pyproject, project_dir, ["standard"]) == ["standard"]
    assert union_project_stack(pyproject, project_dir, ["@qsar", "@qsar"]) == [
        "standard",
        "@qsar",
    ]


def test_union_project_stack_requires_a_tracked_project(tmp_path):
    from uv_stack.operations.project import union_project_stack

    project_dir = tmp_path / "untracked"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text('[project]\nname = "x"\n')

    with pytest.raises(ConfigError) as excinfo:
        union_project_stack(project_dir / "pyproject.toml", project_dir, ["@qsar"])
    assert excinfo.value.message == f"No tracked project in {project_dir}."
    assert "stack create project" in (excinfo.value.hint or "")


def test_the_union_is_in_the_pending_record(config_tree: ConfigRoot, tmp_path, monkeypatch):
    """A crash after the first durable write must retry against the union."""
    from uv_stack.operations import project as project_module
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    real_write = project_module.write_tracking
    calls = {"n": 0}

    def crash_after_first_write(pyproject, tracking):
        calls["n"] += 1
        real_write(pyproject, tracking)
        if calls["n"] == 1:
            raise RuntimeError("crash after the pending write")

    monkeypatch.setattr(project_module, "write_tracking", crash_after_first_write)

    with pytest.raises(RuntimeError, match="crash after the pending write"):
        refresh_project(
            config_tree,
            RecordingRunner(),
            RefreshOptions(python="3.12", stack=["standard", "@qsar"]),
            cwd=project_dir,
        )

    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None
    assert tracking.stack == ["standard", "@qsar"]
    assert tracking.pending is not None and "umap-learn" in tracking.pending


def test_a_retry_after_a_failed_uv_add_converges(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """The existing crash-safety property, with a stack that differs."""
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    options = RefreshOptions(python="3.12", stack=["standard", "@qsar"])

    def _fail_add(cmd: Command) -> CommandResult:
        if "add" in cmd.args:
            raise ToolError("add failed", command=cmd.args, returncode=1)
        return CommandResult(returncode=0, stdout="")

    crashed_root = tmp_path / "crashed"
    crashed_root.mkdir()
    crashed_dir = _tracked_project(crashed_root, _TRACKING)
    with pytest.raises(ToolError):
        refresh_project(
            config_tree, RecordingRunner(responder=_fail_add), options, cwd=crashed_dir
        )
    refresh_project(config_tree, RecordingRunner(), options, cwd=crashed_dir)

    # _tracked_project always writes '<parent>/proj_refresh', so the two
    # projects need separate parents.
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean_dir = _tracked_project(clean_root, _TRACKING)
    refresh_project(config_tree, RecordingRunner(), options, cwd=clean_dir)

    crashed = read_tracking(crashed_dir / "pyproject.toml")
    clean = read_tracking(clean_dir / "pyproject.toml")
    assert crashed is not None and clean is not None
    assert crashed == clean
    assert crashed.stack == ["standard", "@qsar"]
    assert crashed.pending is None


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


_TRACKING_DIRECT_REF_AND_PLAIN = (
    "\n[tool.uv-stack]\nversion = 1\n"
    'stack = ["standard"]\n'
    f'applied = ["numpy", "pandas", "rdkit", "rich", "scipy", "{_DIRECT_REF}"]\n'
)


def test_refresh_never_passes_a_direct_reference_to_uv_remove(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A dropped `name @ url` is reported, never removed by its bare name.

    `scipy` is dropped alongside it so a real `uv remove` runs: the point is not
    that no removal happens, but that the removal which does happen carries the
    plain name and not the direct reference's owned head.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    # Both dropped entries must be real dependencies: removal candidates are
    # filtered against [project.dependencies], so a ledger-only entry would
    # produce no uv remove at all and leave the assertions below vacuous.
    project_dir = _tracked_project(
        tmp_path,
        _TRACKING_DIRECT_REF_AND_PLAIN,
        extra_dependencies=("scipy", _DIRECT_REF),
    )

    rec = RecordingRunner(responder=_mutating_responder(project_dir))
    result = refresh_project(
        config_tree, rec, RefreshOptions(python="3.12"), cwd=project_dir
    )

    assert _DIRECT_REF in result.skipped_removals
    assert _DIRECT_REF not in result.removed
    assert "scipy" in result.removed
    remove_args = [cmd.args for cmd in rec.commands if "remove" in cmd.args]
    # Pin the whole payload rather than membership. A fragment check passes even
    # when the direct reference is handed over verbatim, because "torch" is never
    # an argv element in its own right.
    assert remove_args == [["uv", "remove", "--no-sync", "scipy"]]


def test_upgrade_writes_an_expanded_requirements_in(config_tree: ConfigRoot):
    # The upgrade path is where this machine's values actually reach uv, and
    # it is the only call site that writes the rendered text to disk. dry_run
    # writes the generated files and runs no commands, which isolates the
    # render from the rest of the pipeline.
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=/home/me/code\n")
    config_tree.profile_path("ds").write_text("includes:\n  - -e ${DEV}/mypkg\n")
    rec = RecordingRunner(responder=_existing_env_responder)
    upgrade_env(config_tree, rec, "main", UpgradeOptions(dry_run=True))
    text = config_tree.env_requirements_in("main").read_text()
    assert "-e /home/me/code/mypkg" in text
    assert "${DEV}" not in text


def test_candidate_lock_is_seeded_from_the_published_lock(config_tree: ConfigRoot):
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")
    candidate, copied = _new_candidate_lock(lock, seed=True)
    try:
        assert copied is True
        assert candidate.read_text() == "numpy==1.26.0\n"
        assert candidate != lock
    finally:
        candidate.unlink(missing_ok=True)


def test_candidate_lock_carries_the_conventional_file_mode(config_tree: ConfigRoot):
    """The candidate comes back at 0o666 & ~umask, not at mkstemp's 0600.

    That is _new_candidate_lock's own contract and the whole of what this
    asserts. It is not a check on the published lock's mode: uv pip compile
    replaces its output file rather than writing into this inode, so the mode
    that reaches the user is uv's to set. The relax guards the case where that
    stops holding -- a uv that wrote in place would publish 0600 onto a root
    whose every other generated file is readable.
    """
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    candidate, _ = _new_candidate_lock(lock, seed=False)
    try:
        umask = os.umask(0)
        os.umask(umask)
        assert stat.S_IMODE(candidate.stat().st_mode) == 0o666 & ~umask
    finally:
        candidate.unlink(missing_ok=True)


def test_candidate_lock_is_empty_when_no_lock_is_published(config_tree: ConfigRoot):
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    assert not lock.exists()
    candidate, copied = _new_candidate_lock(lock, seed=True)
    try:
        # Seeding was requested; the hint wording keys off this, not the mode.
        assert copied is False
        assert candidate.read_text() == ""
    finally:
        candidate.unlink(missing_ok=True)


def test_compile_sees_the_existing_pins(config_tree: ConfigRoot):
    """uv reads prior pins from its output file, so --no-upgrade needs them there."""
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")
    seen: list[str] = []

    def responder(cmd: Command) -> CommandResult:
        if "compile" in cmd.args:
            _assert_compiles_to_candidate(cmd, lock)
            seen.append(_compile_output(cmd).read_text())
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    upgrade_env(config_tree, runner, "main", UpgradeOptions(no_upgrade=True))
    assert seen == ["numpy==1.26.0\n"]


def test_the_recreate_branch_also_sees_the_existing_pins(config_tree: ConfigRoot):
    # The spec requires both branches covered. upgrade_env calls
    # _new_candidate_lock from two places -- the recreate branch compiles with
    # uv_pip_compile_for_version before the environment is torn down -- and a
    # regression could reach one without the other.
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")
    seen: list[str] = []

    def responder(cmd: Command) -> CommandResult:
        if "compile" in cmd.args:
            _assert_compiles_to_candidate(cmd, lock)
            seen.append(_compile_output(cmd).read_text())
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    upgrade_env(
        config_tree, runner, "main", UpgradeOptions(recreate=True, no_upgrade=True)
    )
    assert seen == ["numpy==1.26.0\n"]
    assert lock.read_text() == "numpy==1.26.0\n"
    assert list(lock.parent.glob(lock.name + ".*.tmp")) == []


def test_a_failed_compile_leaves_the_published_lock_untouched(config_tree: ConfigRoot):
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")

    def responder(cmd: Command) -> CommandResult:
        if "compile" in cmd.args:
            raise ToolError("uv pip compile failed.", command=cmd.args, returncode=1)
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    with pytest.raises(ToolError):
        upgrade_env(config_tree, runner, "main", UpgradeOptions())
    assert lock.read_text() == "numpy==1.26.0\n"
    leftovers = list(lock.parent.glob(lock.name + ".*.tmp"))
    assert leftovers == []


def test_candidate_lock_is_empty_when_lock_path_is_a_directory(config_tree: ConfigRoot):
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.mkdir()  # A directory stands in for any non-regular file.
    candidate, copied = _new_candidate_lock(lock, seed=True)
    try:
        assert copied is False
        assert candidate.read_text() == ""
    finally:
        candidate.unlink(missing_ok=True)


def test_candidate_lock_survives_concurrent_lock_removal(
    config_tree: ConfigRoot, monkeypatch
):
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")

    original_open = os.open

    def open_losing_the_race(path, flags, *args, **kwargs):
        # Removed between the caller deciding to seed and the open landing.
        if path == lock:
            raise FileNotFoundError(f"{path}")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_losing_the_race)

    candidate, copied = _new_candidate_lock(lock, seed=True)
    try:
        assert copied is False
        assert candidate.read_text() == ""
    finally:
        candidate.unlink(missing_ok=True)


def _assert_seeded_hint(hint: str) -> None:
    """Pin the wording a seeded candidate earns.

    :param hint: The hint attached to the failed compile.
    """
    assert "copy of" in hint
    assert "re-run as a full upgrade" in hint
    # Both spellings, because one defect this replaced was advice that fit
    # only one of the two commands reaching it. Either half going missing is
    # that defect again, pointed the other way.
    assert "drop --no-upgrade/--upgrade-package from 'stack upgrade'" in hint
    assert "add --upgrade to 'stack sync env'" in hint
    # And no runnable bare command: 'stack sync --upgrade' on its own
    # names no environment, so it turns a scoped repair into an unprompted
    # root-wide force-upgrade.
    assert "keeping the same environment names" in hint
    assert "new empty file" not in hint


def test_a_failed_compile_hint_names_the_published_lock(config_tree: ConfigRoot):
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")

    def responder(cmd: Command) -> CommandResult:
        if "compile" in cmd.args:
            raise ToolError("uv pip compile failed.", command=cmd.args, returncode=1)
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    with pytest.raises(ToolError) as caught:
        upgrade_env(config_tree, runner, "main", UpgradeOptions(no_upgrade=True))
    hint = caught.value.hint
    assert hint is not None
    assert str(lock) in hint
    _assert_seeded_hint(hint)
    assert list(lock.parent.glob(lock.name + ".*.tmp")) == []


def test_a_failed_compile_hint_names_the_published_lock_recreate_branch(
    config_tree: ConfigRoot,
):
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")

    def responder(cmd: Command) -> CommandResult:
        if "compile" in cmd.args:
            raise ToolError("uv pip compile failed.", command=cmd.args, returncode=1)
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    with pytest.raises(ToolError) as caught:
        upgrade_env(
            config_tree, runner, "main", UpgradeOptions(recreate=True, no_upgrade=True)
        )
    hint = caught.value.hint
    assert hint is not None
    assert str(lock) in hint
    _assert_seeded_hint(hint)
    assert list(lock.parent.glob(lock.name + ".*.tmp")) == []


@pytest.mark.parametrize("recreate", [False, True])
def test_a_failed_compile_hint_describes_an_unseeded_candidate(
    config_tree: ConfigRoot, recreate: bool
):
    # A full upgrade compiles into an empty candidate, so neither half of the
    # seeded wording is true of it: nothing was copied, and telling the user to
    # re-run without --no-upgrade/--upgrade-package names the mode they are
    # already in. Both branches are covered because upgrade_env attaches the
    # hint from two separate unwinds, either of which could be left behind.
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")

    def responder(cmd: Command) -> CommandResult:
        if "compile" in cmd.args:
            raise ToolError("uv pip compile failed.", command=cmd.args, returncode=1)
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    with pytest.raises(ToolError) as caught:
        upgrade_env(config_tree, runner, "main", UpgradeOptions(recreate=recreate))
    hint = caught.value.hint
    assert hint is not None
    assert str(lock) in hint
    assert "new empty file" in hint
    # Multi-word phrases: the lock path is interpolated into the hint, and a
    # bare "copy" could match a tmp_path component rather than the wording.
    assert "copy of" not in hint
    assert "re-run without" not in hint
    assert list(lock.parent.glob(lock.name + ".*.tmp")) == []


@pytest.mark.parametrize("recreate", [False, True])
def test_a_failed_compile_hint_does_not_claim_a_copy_that_never_happened(
    config_tree: ConfigRoot, recreate: bool
):
    """Seeding was requested, but there was no published lock to seed from.

    The wording follows what the candidate actually holds, not the mode that
    asked for it. An absent lock leaves the candidate empty whatever
    --no-upgrade requested, and the seeded advice would send the user to fix a
    parse error in a file that is not there.
    """
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    assert not lock.exists()

    def responder(cmd: Command) -> CommandResult:
        if "compile" in cmd.args:
            raise ToolError("uv pip compile failed.", command=cmd.args, returncode=1)
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    with pytest.raises(ToolError) as caught:
        upgrade_env(
            config_tree,
            runner,
            "main",
            UpgradeOptions(recreate=recreate, no_upgrade=True),
        )
    hint = caught.value.hint
    assert hint is not None
    assert str(lock) in hint
    assert "new empty file" in hint
    assert "copy of" not in hint
    assert "re-run as a full upgrade" not in hint
    # Nor the full-upgrade explanation: the pins were not ignored, there were
    # none. Saying so is the only way the user learns why a --no-upgrade run
    # came back with everything re-resolved.
    assert "a full upgrade ignores the existing pins" not in hint
    assert "nothing to seed" in hint
    assert list(lock.parent.glob(lock.name + ".*.tmp")) == []


def test_a_failed_rebuild_gets_no_compile_hint(config_tree: ConfigRoot):
    # The recreate branch's try also covers ensure_env and the interpreter
    # probe. A micromamba failure there must not be explained as a compile
    # writing into a temp copy.
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")

    def responder(cmd: Command) -> CommandResult:
        if "create" in cmd.args and "micromamba" in cmd.args[0]:
            raise ToolError(
                "Command failed (1): micromamba create",
                command=cmd.args,
                returncode=1,
            )
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    with pytest.raises(ToolError) as caught:
        upgrade_env(config_tree, runner, "main", UpgradeOptions(recreate=True))
    assert caught.value.hint is None
    assert list(lock.parent.glob(lock.name + ".*.tmp")) == []


def test_an_env_named_compile_does_not_claim_the_compile_hint(config_tree: ConfigRoot):
    # The env name reaches argv as a bare element in 'micromamba remove -n
    # <env>', so a membership test for "compile" would hand this hint to every
    # recreate-path failure for one pathological name.
    shutil.copytree(
        config_tree.root / "envs" / "main", config_tree.root / "envs" / "compile"
    )
    lock = config_tree.env_requirements_lock("compile")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")

    def responder(cmd: Command) -> CommandResult:
        if "remove" in cmd.args:
            raise ToolError(
                "Command failed (1): micromamba remove",
                command=cmd.args,
                returncode=1,
            )
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    with pytest.raises(ToolError) as caught:
        upgrade_env(config_tree, runner, "compile", UpgradeOptions(recreate=True))
    assert caught.value.command[1] == "remove"
    assert caught.value.hint is None


@pytest.mark.parametrize("recreate", [False, True])
def test_full_upgrade_recovers_unreadable_lock(config_tree: ConfigRoot, recreate: bool):
    # A plain upgrade passes --upgrade, so uv ignores the output file. An
    # unreadable lock must not fail the operation. Both branches are covered
    # for the same reason as test_the_recreate_branch_also_sees_the_existing_pins:
    # upgrade_env decides whether to seed at two separate call sites, and a
    # regression could reach one without the other.
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")
    lock.chmod(0o000)

    try:
        runner = RecordingRunner(responder=_existing_env_responder)
        upgrade_env(config_tree, runner, "main", UpgradeOptions(recreate=recreate))
        # The unseeded candidate replaced the lock: the old pins are gone,
        # which exists() alone would not have shown.
        assert lock.read_text() == ""
    finally:
        lock.chmod(0o644)


def test_full_upgrade_does_not_read_the_lock(config_tree: ConfigRoot):
    # A full upgrade seeds with an empty candidate, so the compile must not
    # see the published lock's contents.
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")
    seen: list[str] = []

    def responder(cmd: Command) -> CommandResult:
        if "compile" in cmd.args:
            seen.append(_compile_output(cmd).read_text())
        return _existing_env_responder(cmd)

    runner = RecordingRunner(responder=responder)
    upgrade_env(config_tree, runner, "main", UpgradeOptions())
    assert seen == [""]


def test_candidate_lock_is_empty_when_lock_path_is_a_fifo(config_tree: ConfigRoot):
    # What this pins is O_NONBLOCK, not the S_ISREG check: a FIFO with no
    # writer reads EOF, so removing S_ISREG still yields an empty candidate --
    # the directory test is what pins that. Without O_NONBLOCK the open never
    # returns, which is a hang rather than a failure, so _deadline turns it
    # into an ordinary assertion error.
    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)

    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo not available on this platform")

    os.mkfifo(lock)
    try:
        with _deadline(5.0):
            candidate, copied = _new_candidate_lock(lock, seed=True)
        try:
            assert copied is False
            assert candidate.read_text() == ""
        finally:
            candidate.unlink(missing_ok=True)
    finally:
        os.unlink(lock)


def test_seed_failure_surfaces_its_own_error(config_tree: ConfigRoot, monkeypatch):
    # The read hands the descriptor to a file object, so the unwind must not
    # close it a second time: a stray EBADF would replace the real reason the
    # seed failed, and in a threaded process could close an unrelated file.
    import errno

    lock = config_tree.env_requirements_lock("main")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("numpy==1.26.0\n")

    original_write_bytes = Path.write_bytes

    def full_disk(self: Path, data: bytes) -> int:
        if self.name.endswith(".tmp"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return original_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", full_disk)

    with pytest.raises(OSError) as caught:
        _new_candidate_lock(lock, seed=True)
    assert caught.value.errno == errno.ENOSPC
    # The unwind removed the candidate it could not fill.
    assert list(lock.parent.glob("*.tmp")) == []


def _portable_root(tmp_path: Path, name: str, dev: str) -> ConfigRoot:
    """A config root whose 'dev' profile installs an editable under ``dev``."""
    root = tmp_path / name
    (root / "profiles").mkdir(parents=True)
    (root / "bundles").mkdir(parents=True)
    (root / "envs").mkdir(parents=True)
    (root / "profiles" / "dev.yaml").write_text(
        "includes:\n  - rich\n  - -e ${DEV}/widget\n"
    )
    (root / "variables.txt").write_text("DEV\n")
    (root / "variables.local.txt").write_text(f"DEV={dev}\n")
    return ConfigRoot(root)


def test_the_ledger_keeps_the_unexpanded_entry(tmp_path: Path):
    from uv_stack.operations.pyproject import read_tracking

    config = _portable_root(tmp_path, "root-a", "/checkouts/a")
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    pyproject = project_dir / "pyproject.toml"
    captured: list[str] = []
    # 'pending' is as durable as 'applied': it is written before 'uv add' and
    # a crash in that window leaves it committed. The only place to observe it
    # is inside the command the window brackets.
    mid_run: list[list[str] | None] = []

    def responder(cmd: Command) -> CommandResult:
        if "add" in cmd.args and "-r" in cmd.args:
            captured.append(Path(cmd.args[cmd.args.index("-r") + 1]).read_text())
            during = read_tracking(pyproject)
            mid_run.append(during.pending if during is not None else None)
        return CommandResult(returncode=0, stdout="")

    init_project(
        config, RecordingRunner(responder=responder), ["dev"],
        ProjectOptions(python="3.12"), cwd=project_dir,
    )
    tracking = read_tracking(pyproject)
    assert tracking is not None
    assert "-e ${DEV}/widget" in tracking.applied
    assert "-e /checkouts/a/widget" not in tracking.applied
    assert "-e /checkouts/a/widget\n" in captured[0]
    pending = mid_run[0]
    assert pending is not None
    assert "-e ${DEV}/widget" in pending
    assert "-e /checkouts/a/widget" not in pending


def test_two_roots_with_different_values_write_identical_ledgers(tmp_path: Path):
    """The surface uv-stack controls is portable; assert it directly."""
    from uv_stack.operations.pyproject import read_tracking

    ledgers = []
    for name, dev in (("root-a", "/checkouts/a"), ("root-b", "/elsewhere/b")):
        config = _portable_root(tmp_path, name, dev)
        project_dir = tmp_path / f"proj-{name}"
        project_dir.mkdir()
        init_project(
            config, RecordingRunner(responder=_existing_env_responder), ["dev"],
            ProjectOptions(python="3.12"), cwd=project_dir,
        )
        tracking = read_tracking(project_dir / "pyproject.toml")
        assert tracking is not None
        ledgers.append((list(tracking.applied), tracking.pending))
    assert ledgers[0] == ledgers[1]


def test_an_undefined_variable_aborts_before_anything_is_created(tmp_path: Path):
    config = _portable_root(tmp_path, "root-c", "/checkouts/c")
    config.variables_local_path().unlink()
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    runner = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ConfigError) as excinfo:
        init_project(
            config, runner, ["dev"], ProjectOptions(python="3.12"), cwd=project_dir
        )
    assert "DEV" in str(excinfo.value)
    assert runner.commands == []
    assert not (project_dir / "pyproject.toml").exists()


def _portable_tracked_project(tmp_path: Path, name: str) -> Path:
    """A tracked project whose stack is the portable root's 'dev' profile.

    The ledger already holds the unexpanded editable, which is the steady
    state a second refresh must reproduce: nothing is added, nothing is
    dropped, and both tables come back byte-identical.

    :param tmp_path: Parent directory.
    :param name: Project directory name, unique within ``tmp_path``.
    :returns: The project directory.
    """
    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["rich"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["dev"]\n'
        'applied = ["rich", "-e ${DEV}/widget"]\n'
    )
    return project_dir


def _refresh_ledger_tables(
    config: ConfigRoot, tmp_path: Path, name: str
) -> tuple[list[str] | None, list[str]]:
    """Refresh a tracked project and return both ledger tables it wrote.

    ``pending`` only exists between the first write and the clearing write, so
    it has to be read from inside a command. A helper rather than a loop body
    because the responder closes over ``pyproject`` (B023).

    :param config: The root to refresh against.
    :param tmp_path: Parent for the project directory.
    :param name: Project directory name, unique within ``tmp_path``.
    :returns: ``(pending mid-run, applied after the run)``.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    project_dir = _portable_tracked_project(tmp_path, name)
    pyproject = project_dir / "pyproject.toml"
    mid_run: list[ProjectTracking | None] = []

    def responder(cmd: Command) -> CommandResult:
        mid_run.append(read_tracking(pyproject))
        return CommandResult(returncode=0, stdout="")

    refresh_project(
        config, RecordingRunner(responder=responder),
        RefreshOptions(python="3.12"), cwd=project_dir,
    )
    first = mid_run[0]
    final = read_tracking(pyproject)
    assert first is not None and final is not None
    return first.pending, list(final.applied)


def test_refresh_keeps_the_ledger_unexpanded_and_expands_the_temp_file(tmp_path: Path):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    config = _portable_root(tmp_path, "root-r", "/checkouts/r")
    project_dir = _portable_tracked_project(tmp_path, "proj-r")
    pyproject = project_dir / "pyproject.toml"
    captured: list[str] = []
    mid_run: list[ProjectTracking | None] = []

    def responder(cmd: Command) -> CommandResult:
        mid_run.append(read_tracking(pyproject))
        if "add" in cmd.args and "-r" in cmd.args:
            captured.append(Path(cmd.args[cmd.args.index("-r") + 1]).read_text())
        return CommandResult(returncode=0, stdout="")

    refresh_project(
        config, RecordingRunner(responder=responder),
        RefreshOptions(python="3.12"), cwd=project_dir,
    )
    assert "-e /checkouts/r/widget\n" in captured[0]
    # 'uv add' is the first command this run issues, so snapshot 0 is taken
    # with the pending table already on disk.
    first = mid_run[0]
    assert first is not None
    assert first.pending == ["rich", "-e ${DEV}/widget"]
    final = read_tracking(pyproject)
    assert final is not None
    assert list(final.applied) == ["rich", "-e ${DEV}/widget"]


def test_refresh_with_an_undefined_variable_writes_no_pending_table(tmp_path: Path):
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    config = _portable_root(tmp_path, "root-u", "/checkouts/u")
    config.variables_local_path().unlink()
    project_dir = _portable_tracked_project(tmp_path, "proj-u")
    pyproject = project_dir / "pyproject.toml"
    before = pyproject.read_text()
    runner = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ConfigError) as excinfo:
        refresh_project(config, runner, RefreshOptions(python="3.12"), cwd=project_dir)
    assert "DEV" in str(excinfo.value)
    assert runner.commands == []
    tracking = read_tracking(pyproject)
    assert tracking is not None and tracking.pending is None
    assert pyproject.read_text() == before


def test_two_roots_refresh_to_identical_ledgers(tmp_path: Path):
    """Both tables refresh writes must match across roots with different values."""
    observed = []
    for name, dev in (("root-x", "/checkouts/x"), ("root-y", "/elsewhere/y")):
        config = _portable_root(tmp_path, name, dev)
        observed.append(_refresh_ledger_tables(config, tmp_path, f"proj-{name}"))
    assert observed[0] == observed[1]
    assert observed[0] == (["rich", "-e ${DEV}/widget"], ["rich", "-e ${DEV}/widget"])


def test_refreshing_an_unchanged_reference_bearing_project_is_a_no_op(tmp_path: Path):
    """Both diffs must compare the ledger against the same spelling it holds.

    'dropped' and 'added' run against the unexpanded 'stack_adds'. Diffing
    the unexpanded ledger against the expanded list instead compares two
    spellings of one requirement and concludes the project changed: every
    refresh of a stable project would report a spurious addition, and -- since
    an editable is never auto-removed -- a spurious 'Not auto-removed'
    instruction to hand-edit a file that is already correct.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project

    config = _portable_root(tmp_path, "root-noop", "/checkouts/noop")
    project_dir = tmp_path / "proj-noop"
    project_dir.mkdir()
    init_project(
        config, RecordingRunner(responder=_existing_env_responder), ["dev"],
        ProjectOptions(python="3.12"), cwd=project_dir,
    )
    result = refresh_project(
        config, RecordingRunner(responder=_existing_env_responder),
        RefreshOptions(python="3.12"), cwd=project_dir,
    )
    assert result.added == []
    assert result.removed == []
    assert result.skipped_removals == []


def test_python_travel_problem_classifies_every_selector_shape(config_tree: ConfigRoot):
    # 'cpython@3.12' and 'pypy-3.10' are the regression guard: both are uv
    # implementation forms, and neither may be reclassified as an env name.
    assert python_travel_problem(config_tree, "/opt/envs/x/bin/python") == "path"
    assert python_travel_problem(config_tree, "scratch") == "undeclared-env"
    assert python_travel_problem(config_tree, "main") is None
    assert python_travel_problem(config_tree, "3.12") is None
    assert python_travel_problem(config_tree, "cpython@3.12") is None
    assert python_travel_problem(config_tree, "pypy-3.10") is None


#: (spec, the phrase its advisory must carry, or None for "stay silent").
#: 'main' is the environment 'config_tree' declares; 'scratch' is not.
_TRAVEL_MATRIX = [
    ("/opt/envs/x/bin/python", "does not travel"),
    ("scratch", "declares no environment"),
    ("main", None),
    ("3.12", None),
    ("cpython@3.12", None),
    ("pypy-3.10", None),
]


#: The openings a travel advisory can take, derived from the constants rather
#: than restated, so a reworded advisory cannot quietly empty the filter below.
_TRAVEL_OPENINGS = tuple(
    opening.split("{spec}")[0]
    for opening in (PYTHON_TRAVEL_RECORDED, PYTHON_TRAVEL_PROSPECTIVE)
)


def _travel_warnings(warnings: list[str]) -> list[str]:
    """The travel advisories in a warning list, in either tense.

    The two advisories no longer share one opening: a completed run says
    'Recording interpreter ' and every prospective delivery says 'Would record
    interpreter '. Nothing else either operation emits opens either way.
    Naming pyproject.toml would be the obvious discriminator and is the wrong
    one: SKIPPED_REMOVAL_NOTICE mentions the file too, and refresh puts both
    kinds in one list on its error path.

    :param warnings: Warnings returned by init or refresh.
    :returns: Only the travel advisories, in order.
    """
    return [w for w in warnings if w.startswith(_TRAVEL_OPENINGS)]


@pytest.mark.parametrize(("spec", "expected"), _TRAVEL_MATRIX)
def test_init_warns_only_for_an_interpreter_that_will_not_travel(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch, spec: str, expected: str | None
):
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    warnings = init_project(
        config_tree, RecordingRunner(responder=_existing_env_responder), ["ds"],
        ProjectOptions(python=spec), cwd=project_dir,
    )
    travel = _travel_warnings(warnings)
    if expected is None:
        assert travel == []
    else:
        assert len(travel) == 1 and expected in travel[0]


@pytest.mark.parametrize(("spec", "expected"), _TRAVEL_MATRIX)
def test_refresh_warns_only_for_an_interpreter_that_will_not_travel(
    config_tree: ConfigRoot, tmp_path, monkeypatch, spec: str, expected: str | None
):
    # refresh judges 'spec_flag', not options.python: the value actually
    # written to the table is the value that has to travel.
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    result = refresh_project(
        config_tree, RecordingRunner(responder=_existing_env_responder),
        RefreshOptions(python=spec), cwd=project_dir,
    )
    travel = _travel_warnings(result.warnings)
    if expected is None:
        assert travel == []
    else:
        assert len(travel) == 1 and expected in travel[0]


def test_refresh_warns_for_the_recorded_spec_when_no_flag_is_given(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    # No --python, so spec_flag falls back to tracking.python, which is the
    # stale path a clone inherited. That is exactly the case worth naming.
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    tracking_text = (
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["standard"]\n'
        'python = "/opt/envs/x/bin/python"\n'
        'applied = ["numpy", "pandas", "rdkit", "rich"]\n'
    )
    project_dir = _tracked_project(tmp_path, tracking_text)
    result = refresh_project(
        config_tree, RecordingRunner(responder=_existing_env_responder),
        RefreshOptions(), cwd=project_dir,
    )
    assert len(_travel_warnings(result.warnings)) == 1


def test_init_no_track_stays_silent_about_an_interpreter_it_never_records(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    """--no-track stores no spec, so there is no trip for one to survive.

    Neither wording is true on this path: nothing is being recorded, and
    nothing would be either. Asserting the table is absent as well as the
    advisory keeps the test from passing on a run that merely warned about
    something else.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    warnings = init_project(
        config_tree, RecordingRunner(responder=_existing_env_responder), ["ds"],
        ProjectOptions(python="/opt/envs/x/bin/python", track=False),
        cwd=project_dir,
    )
    assert _travel_warnings(warnings) == []
    assert read_tracking(project_dir / "pyproject.toml") is None


def _fail_on(word: str):
    """A responder that raises ToolError for the uv subcommand ``word``.

    :param word: An argument that identifies the command to fail, e.g. 'add'.
    """

    def responder(cmd: Command) -> CommandResult:
        if word in cmd.args:
            raise ToolError(f"{word} failed", command=cmd.args, returncode=1)
        return _existing_env_responder(cmd)

    return responder


def test_init_failure_past_the_write_still_reports_the_spec_it_recorded(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    """A tracked init that dies after the pending write must still say so.

    The advisory used to ride only on the success return, so a run that wrote
    the untravelable spec and then failed told nobody. Asserting the recorded
    value alongside the wording is what makes the claim checkable: the present
    tense is honest only because the table really is on disk.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_travel_addfail"
    project_dir.mkdir()
    with pytest.raises(ToolError) as excinfo:
        init_project(
            config_tree, RecordingRunner(responder=_fail_on("add")), ["ds"],
            ProjectOptions(python="/opt/envs/x/bin/python"), cwd=project_dir,
        )
    travel = _travel_warnings(excinfo.value.resolution_warnings)
    assert len(travel) == 1
    assert travel[0].startswith("Recording interpreter '/opt/envs/x/bin/python'")
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None and tracking.python == "/opt/envs/x/bin/python"


def test_init_failure_above_the_write_only_predicts_the_recording(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    """A fresh init that dies in uv init recorded nothing, and must say so.

    uv init runs one line above the pending write, which makes this the only
    reachable path where the advisory is delivered before the spec is on disk.
    Without it, hardcoding the completed tense would pass every other travel
    test here. The absent table is half the assertion: the future tense is
    correct only because nothing was written.
    """
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_travel_initfail"
    project_dir.mkdir()
    with pytest.raises(ToolError) as excinfo:
        init_project(
            config_tree, RecordingRunner(responder=_fail_on("init")), ["ds"],
            ProjectOptions(python="/opt/envs/x/bin/python"), cwd=project_dir,
        )
    travel = _travel_warnings(excinfo.value.resolution_warnings)
    assert len(travel) == 1
    assert travel[0].startswith("Would record interpreter '/opt/envs/x/bin/python'")
    assert read_tracking(project_dir / "pyproject.toml") is None


def test_init_success_says_it_recorded_the_spec(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    """A completed init wrote the spec, so the advisory stays in the present.

    Suppressing the advisory outright, or moving everything to the future
    tense, would satisfy the two failure tests above. This is the pin that
    keeps the completed wording alive on the path that earns it.
    """
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_travel_ok"
    project_dir.mkdir()
    warnings = init_project(
        config_tree, RecordingRunner(responder=_existing_env_responder), ["ds"],
        ProjectOptions(python="/opt/envs/x/bin/python"), cwd=project_dir,
    )
    travel = _travel_warnings(warnings)
    assert len(travel) == 1
    assert travel[0].startswith("Recording interpreter '/opt/envs/x/bin/python'")


def test_refresh_dry_run_only_predicts_the_recording(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A dry run changes nothing, so it may not claim to have recorded anything.

    The advisory used to be appended above the dry-run guard, so the one run
    that is defined to touch nothing still said 'Recording'. Comparing the
    file bytes either side is what separates a wording change from a fix: the
    present tense is wrong precisely because the write did not happen.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    pyproject = project_dir / "pyproject.toml"
    before = pyproject.read_bytes()
    result = refresh_project(
        config_tree, RecordingRunner(responder=_existing_env_responder),
        RefreshOptions(python="/opt/envs/x/bin/python", dry_run=True), cwd=project_dir,
    )
    travel = _travel_warnings(result.warnings)
    assert len(travel) == 1
    assert travel[0].startswith("Would record interpreter '/opt/envs/x/bin/python'")
    assert pyproject.read_bytes() == before


def test_refresh_failure_above_the_write_only_predicts_the_recording(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """An unresolvable env aborts refresh before the write, so nothing is recorded.

    resolve_project_python sits one line above write_tracking, so the ledger
    is untouched when it raises. The advisory reaches the user only on the
    error, and the byte comparison is what proves the future tense is the true
    reading rather than a guess about where the run died.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    pyproject = project_dir / "pyproject.toml"
    before = pyproject.read_bytes()
    with pytest.raises(EnvError) as excinfo:
        refresh_project(
            config_tree, RecordingRunner(responder=_missing_env_responder),
            RefreshOptions(python="scratch"), cwd=project_dir,
        )
    travel = _travel_warnings(excinfo.value.resolution_warnings)
    assert len(travel) == 1
    assert travel[0].startswith("Would record interpreter 'scratch'")
    assert pyproject.read_bytes() == before


def test_refresh_success_says_it_recorded_the_spec(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A completed refresh wrote the spec, so the advisory stays in the present.

    Counterpart to the init pin above, and for the same reason: without it,
    suppressing the advisory on every path would pass the dry-run and
    pre-write tests. Reading the table back ties the wording to the write.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    result = refresh_project(
        config_tree, RecordingRunner(responder=_existing_env_responder),
        RefreshOptions(python="/opt/envs/x/bin/python"), cwd=project_dir,
    )
    travel = _travel_warnings(result.warnings)
    assert len(travel) == 1
    assert travel[0].startswith("Recording interpreter '/opt/envs/x/bin/python'")
    tracking = read_tracking(project_dir / "pyproject.toml")
    assert tracking is not None and tracking.python == "/opt/envs/x/bin/python"


def _break_env_listing(monkeypatch) -> None:
    """Make every ``list_envs`` call raise, as a failing mount does.

    ``list_envs`` walks ``envs_dir`` with ``iterdir``, so an unreadable
    directory or a dead network mount surfaces as OSError rather than an empty
    list. Patching the class rather than one instance keeps the injection from
    being sidestepped by an operation that builds its own ConfigRoot.

    :param monkeypatch: The pytest monkeypatch fixture.
    """

    def unreadable(self: ConfigRoot) -> list[str]:
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(ConfigRoot, "list_envs", unreadable)


def _watch_env_listing(monkeypatch, runner: RecordingRunner) -> list[list[str]]:
    """Log what the run had already executed at each ``list_envs`` call.

    One entry per lookup, holding the program name of every command issued so
    far. The length therefore counts the lookups and each entry dates one:
    an entry containing 'uv' is a lookup that happened after the run started
    mutating the project, which is the shape the delivery-point classifier had.

    :param monkeypatch: The pytest monkeypatch fixture.
    :param runner: The runner whose recorded commands date each lookup.
    :returns: The log, appended to as the run proceeds.
    """
    real_list_envs = ConfigRoot.list_envs
    log: list[list[str]] = []

    def spy(self: ConfigRoot) -> list[str]:
        log.append([command.args[0] for command in runner.commands])
        return real_list_envs(self)

    monkeypatch.setattr(ConfigRoot, "list_envs", spy)
    return log


def test_init_env_listing_failure_aborts_before_anything_is_mutated(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    """An unreadable envs directory must abort the run rather than outlive it.

    Classifying the travel advisory at the delivery point put this OSError on
    the success return, where uv add had run, the final ledger was written and
    uv sync had completed: the CLI reported a failed run that had in fact done
    everything it meant to. Asserting the absent pyproject and the unrun uv
    commands is what separates an honest abort from a late complaint.
    """
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    _break_env_listing(monkeypatch)
    project_dir = tmp_path / "proj_travel_unreadable"
    project_dir.mkdir()
    runner = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(OSError) as excinfo:
        init_project(
            config_tree, runner, ["ds"],
            ProjectOptions(python="scratch"), cwd=project_dir,
        )
    # The injected failure, not an incidental one.
    assert excinfo.value.errno == errno.EACCES
    assert not (project_dir / "pyproject.toml").exists()
    assert [c.args[0] for c in runner.commands if c.args[0] == "uv"] == []


def test_refresh_env_listing_failure_aborts_before_anything_is_mutated(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """The refresh counterpart: the same OSError, the same required timing.

    refresh classifies above its own pre-flight, so the dry run and the real
    run share one lookup and a failing mount costs the project nothing. The
    byte comparison is the pin: before the hoist, this run rewrote the ledger
    twice and ran uv before the OSError escaped.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    _break_env_listing(monkeypatch)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    pyproject = project_dir / "pyproject.toml"
    before = pyproject.read_bytes()
    runner = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(OSError) as excinfo:
        refresh_project(
            config_tree, runner, RefreshOptions(python="scratch"), cwd=project_dir,
        )
    assert excinfo.value.errno == errno.EACCES
    assert pyproject.read_bytes() == before
    assert runner.commands == []


def test_refresh_reports_the_unwritable_ledger_ahead_of_the_unreadable_envs(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """When both pre-flights would fail, the ledger's refusal is the useful one.

    The travel classifier reads the config root and the pre-flight reads the
    project, so a run can be blocked by both at once. The user's actual blocker
    is the unwritable ledger -- an unreadable envs directory only costs them an
    advisory -- so the pre-flight has to come first. This pins the ordering that
    keeps refresh agreeing with init, which has always reported the ConfigError
    here; nothing else does, and the classification has already drifted above
    the pre-flight once.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    _break_env_listing(monkeypatch)
    project_dir = tmp_path / "proj_both_preflights_fail"
    project_dir.mkdir()
    # A lone \r mid-table: read_tracking uses universal newlines and accepts
    # it, while the pre-flight's _read_exact uses newline="" and refuses.
    pyproject = project_dir / "pyproject.toml"
    pyproject.write_bytes(
        b'[project]\nname = "demo"\ndependencies = []\n\n[tool.uv-stack]\n'
        b'version = 1\nstack = ["ds"]\rapplied = []\n'
    )
    before = pyproject.read_bytes()
    runner = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ConfigError):
        refresh_project(
            config_tree, runner, RefreshOptions(python="scratch"), cwd=project_dir,
        )
    assert pyproject.read_bytes() == before
    assert runner.commands == []


def test_init_consults_the_declared_envs_once_and_before_it_starts(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    """One lookup per run, above the scaffolding, on the path that succeeds.

    The advisory still has to be delivered twice in principle (both tenses),
    so the guard has to be the lookup rather than the wording: classify once,
    format twice.
    """
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_travel_once"
    project_dir.mkdir()
    runner = RecordingRunner(responder=_existing_env_responder)
    log = _watch_env_listing(monkeypatch, runner)
    warnings = init_project(
        config_tree, runner, ["ds"],
        ProjectOptions(python="scratch"), cwd=project_dir,
    )
    assert len(_travel_warnings(warnings)) == 1
    assert len(log) == 1
    assert "uv" not in log[0]


def test_init_failure_reuses_the_lookup_instead_of_repeating_it(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch
):
    """The error handler formats a tense; it may not go back to the filesystem.

    A lookup inside the handler can raise, and an OSError raised there replaces
    the UvStackError being handled — the user loses the real failure and every
    advisory attached to it. Pinning the surviving error and its advisory
    alongside the single early lookup is what makes that unrepeatable.
    """
    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = tmp_path / "proj_travel_once_failed"
    project_dir.mkdir()
    runner = RecordingRunner(responder=_fail_on("add"))
    log = _watch_env_listing(monkeypatch, runner)
    with pytest.raises(ToolError) as excinfo:
        init_project(
            config_tree, runner, ["ds"],
            ProjectOptions(python="scratch"), cwd=project_dir,
        )
    assert "add failed" in str(excinfo.value)
    assert len(log) == 1
    assert "uv" not in log[0]
    travel = _travel_warnings(excinfo.value.resolution_warnings)
    assert len(travel) == 1
    assert travel[0].startswith("Recording interpreter 'scratch'")


def test_refresh_temp_file_failure_leaves_the_ledger_untouched(
    config_tree: ConfigRoot, tmp_path, monkeypatch
):
    """A refusal to build the temp requirements file must land above the write.

    An OSError is not a UvStackError, so it carries no resolution_warnings and
    the travel advisory has no way out on it. Building and filling the temp
    file ahead of the durable write is what keeps that from mattering: the
    only filesystem failure the run can still produce happens before anything
    is recorded. Asserting the ledger rather than the exception type is the
    stronger pin, and it survives any further file work this region grows.

    The patch is narrowed to refresh's own prefix because atomic_write calls
    mkstemp too; a global patch would abort the run above the write for an
    unrelated reason and hide the defect rather than expose it.
    """
    import tempfile as tempfile_module

    from uv_stack.operations.project import RefreshOptions, refresh_project
    from uv_stack.operations.pyproject import read_tracking

    monkeypatch.delenv(PROJECT_PYTHON_ENV, raising=False)
    project_dir = _tracked_project(tmp_path, _TRACKING)
    pyproject = project_dir / "pyproject.toml"
    before = pyproject.read_bytes()
    real_mkstemp = tempfile_module.mkstemp

    def _no_space(*args, **kwargs):
        if str(kwargs.get("prefix", "")).startswith("uv-stack-refresh"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(tempfile_module, "mkstemp", _no_space)
    with pytest.raises(OSError) as excinfo:
        refresh_project(
            config_tree, RecordingRunner(responder=_existing_env_responder),
            RefreshOptions(python="/opt/envs/x/bin/python"), cwd=project_dir,
        )
    # The injected failure, not an incidental one, and not a silent success.
    assert excinfo.value.errno == errno.ENOSPC
    assert pyproject.read_bytes() == before
    tracking = read_tracking(pyproject)
    assert tracking is not None
    assert tracking.python is None
    assert tracking.pending is None


def test_refresh_dry_run_still_runs_the_placement_check(tmp_path: Path):
    """A dry run exists to find out, so the refusal comes before the plan.

    The check sits outside refresh's ``dry_run`` guard, which is the same
    placement that keeps it ahead of the pending write on the real path.
    """
    from uv_stack.operations.project import RefreshOptions, refresh_project

    root = tmp_path / "root-dry"
    (root / "profiles").mkdir(parents=True)
    (root / "bundles").mkdir(parents=True)
    (root / "envs").mkdir(parents=True)
    # A trailing backslash continues onto the next line, so this entry would
    # swallow whichever requirement the render writes after it.
    (root / "profiles" / "dev.yaml").write_text(
        "includes:\n  - rich\n  - -e /checkouts/widget\\\n"
    )
    config = ConfigRoot(root)
    project_dir = _portable_tracked_project(tmp_path, "proj-dry")
    pyproject = project_dir / "pyproject.toml"
    before = pyproject.read_text()
    runner = RecordingRunner(responder=_existing_env_responder)
    with pytest.raises(ConfigError) as excinfo:
        refresh_project(config, runner, RefreshOptions(dry_run=True), cwd=project_dir)
    assert "backslash" in str(excinfo.value)
    assert runner.commands == []
    assert pyproject.read_text() == before


def test_validate_name_hint_explains_why_a_listed_resource_is_refused():
    """A hand-made env or profile stays listed but cannot be named; say so."""
    with pytest.raises(ConfigError) as excinfo:
        validate_name("environment", "my env")
    hint = excinfo.value.hint or ""
    assert "stays listed" in hint
    assert "renaming it on disk" in hint
    # The same hint serves the create verbs, where nothing is listed yet, and
    # the profile and bundle kinds, whose resource is a .yaml file. Naming a
    # directory would be wrong for both.
    assert "directory" not in hint.lower()
