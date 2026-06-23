from __future__ import annotations

import sys

import pytest

from uv_stack.errors import ToolError
from uv_stack.runner import (
    _PTY_AVAILABLE,
    Command,
    CommandResult,
    RecordingRunner,
    SubprocessRunner,
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
