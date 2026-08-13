from __future__ import annotations

import os

from uv_stack.config import ConfigRoot
from uv_stack.operations.status import compute_status, env_status
from uv_stack.operations.upgrade import UpgradeOptions, upgrade_env
from uv_stack.runner import Command, CommandResult, RecordingRunner


def _existing_env_responder(cmd: Command) -> CommandResult:
    if "run" in cmd.args:
        return CommandResult(returncode=0, stdout="/envs/main/bin/python\n")
    return CommandResult(returncode=0, stdout="")


def _missing_env_responder(cmd: Command) -> CommandResult:
    if "run" in cmd.args:
        return CommandResult(returncode=1, stdout="")
    return CommandResult(returncode=0, stdout="")


class _ExplodingRunner:
    def run(self, command, *, capture=False, check=True):
        raise OSError("micromamba not installed")


def _built(config_tree: ConfigRoot) -> None:
    """Bring env 'main' to a fully built state (files + lock on disk)."""
    rec = RecordingRunner(responder=_existing_env_responder)
    upgrade_env(config_tree, rec, "main", UpgradeOptions())


def test_status_ok(config_tree: ConfigRoot):
    _built(config_tree)
    status = env_status(
        config_tree, RecordingRunner(responder=_existing_env_responder), "main"
    )
    assert status.state == "ok"
    assert status.python == "3.12"
    assert status.created is True
    assert status.lock_present is True
    assert status.message is None


def test_status_sources_changed(config_tree: ConfigRoot):
    _built(config_tree)
    with config_tree.env_stack_path("main").open("a") as handle:
        handle.write("httpx\n")
    status = env_status(
        config_tree, RecordingRunner(responder=_existing_env_responder), "main"
    )
    assert status.state == "sources changed"


def test_status_lock_stale(config_tree: ConfigRoot):
    _built(config_tree)
    lock = config_tree.env_lock("main")
    req = config_tree.env_requirements_in("main")
    old = req.stat().st_mtime - 100
    os.utime(lock, (old, old))
    status = env_status(
        config_tree, RecordingRunner(responder=_existing_env_responder), "main"
    )
    assert status.state == "lock stale"


def test_status_never_built(config_tree: ConfigRoot):
    _built(config_tree)
    config_tree.env_lock("main").unlink()
    status = env_status(
        config_tree, RecordingRunner(responder=_existing_env_responder), "main"
    )
    assert status.state == "never built"


def test_status_not_created_wins_over_never_built(config_tree: ConfigRoot):
    _built(config_tree)
    config_tree.env_lock("main").unlink()
    status = env_status(
        config_tree, RecordingRunner(responder=_missing_env_responder), "main"
    )
    assert status.state == "not created"
    assert status.created is False


def test_status_probe_unavailable_is_none(config_tree: ConfigRoot):
    _built(config_tree)
    status = env_status(config_tree, _ExplodingRunner(), "main")
    assert status.created is None
    assert status.state == "ok"


def test_status_config_error(config_tree: ConfigRoot):
    config_tree.env_stack_path("main").unlink()
    status = env_status(
        config_tree, RecordingRunner(responder=_existing_env_responder), "main"
    )
    assert status.state == "config error"
    assert status.python is None
    assert status.message is not None
    assert "Missing stack file" in status.message


def test_compute_status_defaults_to_all_envs(config_tree: ConfigRoot):
    _built(config_tree)
    statuses = compute_status(
        config_tree, RecordingRunner(responder=_existing_env_responder)
    )
    assert [s.name for s in statuses] == ["main"]


def test_compute_status_named_missing_env_is_config_error(config_tree: ConfigRoot):
    statuses = compute_status(
        config_tree,
        RecordingRunner(responder=_existing_env_responder),
        ["ghost"],
    )
    assert statuses[0].state == "config error"


def test_status_local_requirements_change_makes_lock_stale(config_tree: ConfigRoot):
    """Regression: edits to requirements.local.in must trigger lock stale."""
    _built(config_tree)
    local_req = config_tree.env_local_path("main")
    local_req.parent.mkdir(parents=True, exist_ok=True)
    local_req.write_text("httpx\n")
    # Bump mtime explicitly to ensure it exceeds the lock's mtime.
    future = config_tree.env_lock("main").stat().st_mtime + 100
    os.utime(local_req, (future, future))
    status = env_status(
        config_tree, RecordingRunner(responder=_existing_env_responder), "main"
    )
    assert status.state == "lock stale"


def test_compute_status_empty_list_returns_empty(config_tree: ConfigRoot):
    """Regression: empty list must return empty, not all envs."""
    _built(config_tree)
    rec = RecordingRunner(responder=_existing_env_responder)
    statuses = compute_status(config_tree, rec, [])
    assert statuses == []
    assert len(rec.commands) == 0


def test_status_not_created_early_return_when_requirements_missing(
    config_tree: ConfigRoot,
):
    """Regression: 'not created' must win even when generated files missing."""
    _built(config_tree)
    config_tree.env_requirements_in("main").unlink()
    status = env_status(
        config_tree, RecordingRunner(responder=_missing_env_responder), "main"
    )
    assert status.state == "not created"
    assert status.created is False


def test_status_requirements_in_deleted_after_build_is_sources_changed(
    config_tree: ConfigRoot,
):
    """Generated file deleted after build → sources changed (not crash)."""
    _built(config_tree)
    config_tree.env_requirements_in("main").unlink()
    status = env_status(
        config_tree, RecordingRunner(responder=_existing_env_responder), "main"
    )
    assert status.state == "sources changed"


def test_status_lock_deleted_after_build_is_never_built(config_tree: ConfigRoot):
    """Lock deleted but sources unchanged → never built (not ok)."""
    _built(config_tree)
    lock = config_tree.env_lock("main")
    # Sanity check: lock exists.
    assert lock.is_file()
    lock.unlink()
    status = env_status(
        config_tree, RecordingRunner(responder=_existing_env_responder), "main"
    )
    assert status.state == "never built"
    assert status.lock_present is False


def test_status_lock_race_condition(config_tree: ConfigRoot):
    """Race condition: lock vanishes between is_file() and stat().

    When the lock passes the initial is_file() check but disappears before
    _mtime_or_none() stats it, lock_present is set to False and state should
    be "never built", not "ok".
    """
    from pathlib import Path
    from unittest.mock import patch

    _built(config_tree)
    lock = config_tree.env_lock("main")
    assert lock.is_file()

    # Monkeypatch Path.stat to raise FileNotFoundError on the call from _mtime_or_none(lock)
    original_stat = Path.stat
    call_count = {"count": 0}

    def counting_stat(self, *, follow_symlinks=True):
        call_count["count"] += 1
        # Raise on the call to stat the lock file via _mtime_or_none (after is_file check)
        if self == lock and call_count["count"] > 2:
            raise FileNotFoundError(f"Lock {self} vanished")
        return original_stat(self, follow_symlinks=follow_symlinks)

    with patch.object(Path, "stat", counting_stat):
        status = env_status(
            config_tree, RecordingRunner(responder=_existing_env_responder), "main"
        )

    assert status.state == "never built"
    assert status.lock_present is False
    assert status.created is True
    assert status.python == "3.12"
