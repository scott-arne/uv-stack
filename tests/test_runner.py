from __future__ import annotations

import os
import sys

import pytest

from uv_stack.errors import ToolError
from uv_stack.runner import (
    _PTY_AVAILABLE,
    Command,
    CommandResult,
    RecordingRunner,
    SubprocessRunner,
    _spawn_error,
)


def test_command_equality():
    assert Command(["uv", "pip", "check"]) == Command(["uv", "pip", "check"])


def test_recording_runner_records_and_returns_default():
    rec = RecordingRunner()
    result = rec.run(Command(["uv", "pip", "check"]))
    assert result.returncode == 0
    assert rec.commands == [Command(["uv", "pip", "check"])]


def test_recording_runner_uses_responder():
    def responder(cmd: Command) -> CommandResult:
        return CommandResult(returncode=0, stdout="/fake/python")

    rec = RecordingRunner(responder=responder)
    out = rec.run(Command(["micromamba", "run"]), capture=True)
    assert out.stdout == "/fake/python"


def test_subprocess_runner_captures_stdout():
    runner = SubprocessRunner()
    result = runner.run(
        Command([sys.executable, "-c", "print('hello')"]), capture=True
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


def test_subprocess_runner_raises_tool_error_on_failure():
    runner = SubprocessRunner()
    with pytest.raises(ToolError) as exc:
        runner.run(Command([sys.executable, "-c", "import sys; sys.exit(3)"]))
    assert exc.value.returncode == 3


def test_subprocess_runner_captures_stderr_tail_in_detail():
    # The streaming path (capture=False) still tees stderr to a bounded buffer so
    # the raised ToolError can explain *why* the command failed.
    runner = SubprocessRunner()
    script = "import sys; sys.stderr.write('boom: it broke\\n'); sys.exit(1)"
    with pytest.raises(ToolError) as exc:
        runner.run(Command([sys.executable, "-c", script]))
    assert exc.value.detail is not None
    assert "boom: it broke" in exc.value.detail


def test_subprocess_runner_detail_from_captured_stderr():
    runner = SubprocessRunner()
    script = "import sys; sys.stderr.write('nope\\n'); sys.exit(1)"
    with pytest.raises(ToolError) as exc:
        runner.run(Command([sys.executable, "-c", script]), capture=True)
    assert exc.value.detail is not None
    assert "nope" in exc.value.detail


@pytest.mark.skipif(not _PTY_AVAILABLE, reason="pty unavailable on this platform")
def test_run_with_pty_captures_stripped_detail():
    # The pty path gives uv a terminal so it keeps rendering colour/progress
    # bars; the captured failure detail must still be plain text (no ANSI).
    runner = SubprocessRunner()
    script = (
        "import sys; "
        "sys.stderr.write('\\x1b[31mred error\\x1b[0m\\n'); "
        "sys.exit(2)"
    )
    returncode, detail = runner._run_with_pty(Command([sys.executable, "-c", script]))
    assert returncode == 2
    assert "red error" in detail
    assert "\x1b" not in detail


def test_subprocess_runner_check_false_does_not_raise():
    runner = SubprocessRunner()
    result = runner.run(
        Command([sys.executable, "-c", "import sys; sys.exit(3)"]),
        capture=True,
        check=False,
    )
    assert result.returncode == 3


def test_subprocess_runner_missing_binary_raises_tool_error():
    runner = SubprocessRunner()
    with pytest.raises(ToolError) as exc:
        runner.run(
            Command(["uv-stack-no-such-binary-xyz", "arg"]), capture=True
        )
    assert exc.value.returncode == 127
    assert "uv-stack-no-such-binary-xyz" in str(exc.value)


def test_subprocess_runner_missing_binary_check_false_still_raises():
    # check=False suppresses exit-code failures, but a spawn failure has no
    # exit code — the binary never ran — so it still raises.
    runner = SubprocessRunner()
    with pytest.raises(ToolError) as exc:
        runner.run(
            Command(["uv-stack-no-such-binary-xyz"]), capture=True, check=False
        )
    assert exc.value.returncode == 127


def test_subprocess_runner_streaming_missing_binary_raises():
    # The streaming path (capture=False) is branch-agnostic on purpose: under
    # pytest stderr is not a tty so it takes the Popen branch, but with -s on a
    # terminal it takes the pty branch. Both must raise.
    runner = SubprocessRunner()
    with pytest.raises(ToolError) as exc:
        runner.run(Command(["uv-stack-no-such-binary-xyz"]), capture=False)
    assert exc.value.returncode == 127


@pytest.mark.skipif(not _PTY_AVAILABLE, reason="pty unavailable on this platform")
def test_run_with_pty_missing_binary_raises():
    runner = SubprocessRunner()
    with pytest.raises(ToolError) as exc:
        runner._run_with_pty(Command(["uv-stack-no-such-binary-xyz"]))
    assert exc.value.returncode == 127


@pytest.mark.skipif(not _PTY_AVAILABLE, reason="pty unavailable on this platform")
@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="no /dev/fd to count against")
def test_run_with_pty_cleans_up_fds_on_spawn_failure():
    # A spawn that never starts still leaves both ends of the pty open, and
    # upgrade catches per environment and keeps going — so a missing uv would
    # leak two descriptors per environment for the length of the run. Counting
    # rather than asserting on the closes themselves keeps this a test of the
    # leak: either close going missing leaves FAILURES descriptors behind.
    def open_fds() -> int:
        return len(os.listdir("/dev/fd"))

    runner = SubprocessRunner()
    failures = 10
    before = open_fds()
    for _ in range(failures):
        with pytest.raises(ToolError):
            runner._run_with_pty(Command(["uv-stack-no-such-binary-xyz"]))
    leaked = open_fds() - before

    # One descriptor of slack for anything the interpreter opens incidentally;
    # a single missing close costs ten.
    assert leaked <= 1, f"{leaked} descriptors leaked over {failures} failed spawns"


def test_spawn_error_hints_chmod_for_a_non_executable_binary(tmp_path):
    """The message says Permission denied; the hint must not say "install it"."""
    import errno

    shim = tmp_path / "uv"
    shim.write_text("#!/bin/sh\n")
    command = Command([str(shim), "pip", "compile"])
    error = OSError(errno.EACCES, "Permission denied", str(shim))
    tool_error = _spawn_error(command, error)
    assert tool_error.returncode == 127
    assert "not executable" in tool_error.hint
    assert f"chmod +x {shim}" in tool_error.hint


def test_spawn_error_hints_path_for_a_missing_binary():
    """The pre-existing hint is unchanged for the case it was written for."""
    import errno

    command = Command(["nosuchtool", "--version"])
    error = OSError(errno.ENOENT, "No such file or directory", "nosuchtool")
    tool_error = _spawn_error(command, error)
    assert tool_error.hint == "Is nosuchtool installed and on PATH?"
