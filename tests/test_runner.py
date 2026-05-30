from __future__ import annotations

import sys

import pytest

from uv_stack.errors import ToolError
from uv_stack.runner import Command, CommandResult, RecordingRunner, SubprocessRunner


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


def test_subprocess_runner_check_false_does_not_raise():
    runner = SubprocessRunner()
    result = runner.run(
        Command([sys.executable, "-c", "import sys; sys.exit(3)"]),
        capture=True,
        check=False,
    )
    assert result.returncode == 3
