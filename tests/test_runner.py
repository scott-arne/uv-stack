from __future__ import annotations

import inspect
import os
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from uv_stack.errors import ToolError, UvStackError
from uv_stack.runner import (
    _PTY_AVAILABLE,
    _STDERR_TAIL_LINES,
    Command,
    CommandResult,
    InteractiveRunner,
    RecordingRunner,
    SubprocessRunner,
    _spawn_error,
)


def test_command_equality():
    assert Command(["uv", "pip", "check"]) == Command(["uv", "pip", "check"])


@pytest.mark.parametrize("implementation", [SubprocessRunner, RecordingRunner])
def test_interactive_runner_protocol_is_satisfied(implementation):
    """Both implementations match the InteractiveRunner protocol's signature.

    mypy is configured over ``src`` only, so a ``list[InteractiveRunner]``
    annotation here would assert nothing, and ``callable()`` passes for any
    method of any shape. Comparing the signatures is what actually catches the
    drift that matters — a renamed parameter, an added required one, a changed
    return type — in the one direction static checking does not cover, since
    neither class inherits the protocol and nothing forces them to keep up
    with it.

    The import is the second guard: deleting or renaming the protocol fails
    here rather than at the first CLI call site.
    """
    reference = inspect.signature(InteractiveRunner.run_interactive)
    assert inspect.signature(implementation.run_interactive) == reference


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


def test_spawn_error_hints_directory_permission_when_cwd_unenterable(tmp_path):
    """An EACCES naming the directory gets the cwd hint, not the chmod hint."""
    import errno

    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    some_exe = tmp_path / "uv"
    some_exe.write_text("#!/bin/sh\n")
    command = Command([str(some_exe)], cwd=blocked_dir)
    error = OSError(errno.EACCES, "Permission denied", blocked_dir)
    tool_error = _spawn_error(command, error)
    assert tool_error.returncode == 127
    assert str(blocked_dir) in tool_error.hint
    assert "Cannot enter the working directory" in tool_error.hint
    assert "chmod +x" not in tool_error.hint


def test_spawn_error_hints_path_for_a_missing_cwd(tmp_path):
    """A missing cwd gets the PATH hint, not the permissions hint."""
    import errno

    missing_dir = tmp_path / "does-not-exist"
    some_exe = "uv"
    command = Command([some_exe, "pip", "compile"], cwd=missing_dir)
    error = OSError(errno.ENOENT, "No such file or directory", missing_dir)
    tool_error = _spawn_error(command, error)
    assert tool_error.hint == f"Is {some_exe} installed and on PATH?"
    assert "permissions" not in tool_error.hint


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root ignores directory permissions",
)
def test_subprocess_runner_unenterable_cwd_hints_the_directory(tmp_path):
    """An unenterable working directory produces the cwd hint, not the chmod hint."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o000)
    runner = SubprocessRunner()
    try:
        with pytest.raises(ToolError) as exc:
            runner.run(Command([sys.executable, "-c", "pass"], cwd=blocked), capture=True)
    finally:
        blocked.chmod(0o755)
    assert exc.value.returncode == 127
    assert "Cannot enter the working directory" in exc.value.hint
    assert "chmod +x" not in exc.value.hint


def test_recording_runner_records_interactive_runs():
    rec = RecordingRunner()
    assert rec.run_interactive(Command(["vim", "/tmp/x.txt"])) == 0
    assert rec.commands == [Command(["vim", "/tmp/x.txt"])]


def test_recording_runner_interactive_uses_responder():
    rec = RecordingRunner(responder=lambda command: CommandResult(returncode=3))
    assert rec.run_interactive(Command(["vim", "/tmp/x.txt"])) == 3
    assert rec.commands == [Command(["vim", "/tmp/x.txt"])]


def test_subprocess_runner_interactive_returns_child_status():
    runner = SubprocessRunner()
    assert runner.run_interactive(Command([sys.executable, "-c", "raise SystemExit(0)"])) == 0
    assert runner.run_interactive(Command([sys.executable, "-c", "raise SystemExit(7)"])) == 7


def test_subprocess_runner_interactive_reports_a_missing_binary():
    runner = SubprocessRunner()
    with pytest.raises(ToolError) as excinfo:
        runner.run_interactive(Command(["uv-stack-no-such-editor-xyz"]))
    assert excinfo.value.returncode == 127


def test_subprocess_runner_interactive_honours_cwd(tmp_path):
    runner = SubprocessRunner()
    marker = tmp_path / "here.txt"
    status = runner.run_interactive(
        Command(
            [sys.executable, "-c", "import pathlib; pathlib.Path('here.txt').write_text('x')"],
            cwd=tmp_path,
        )
    )
    assert status == 0
    assert marker.is_file()


def test_subprocess_runner_interactive_reports_a_non_executable_binary(tmp_path):
    """A non-executable file fails with the chmod hint, not the PATH hint."""
    shim = tmp_path / "editor"
    shim.write_text("#!/bin/sh\n")
    runner = SubprocessRunner()
    with pytest.raises(ToolError) as excinfo:
        runner.run_interactive(Command([str(shim)]))
    assert excinfo.value.returncode == 127
    assert "not executable" in excinfo.value.hint
    assert f"chmod +x {shim}" in excinfo.value.hint


def test_subprocess_runner_interactive_shell_quotes_chmod_hint(tmp_path):
    """A non-executable editor with spaces and metacharacters gets quoted."""
    from uv_stack.hints import render_positional_arg

    shim = tmp_path / "bad editor; rm -rf /"
    shim.write_text("#!/bin/sh\n")
    runner = SubprocessRunner()
    with pytest.raises(ToolError) as excinfo:
        runner.run_interactive(Command([str(shim)]))
    assert excinfo.value.returncode == 127
    assert "not executable" in excinfo.value.hint
    assert f"chmod +x {render_positional_arg(str(shim))}" in excinfo.value.hint


def test_subprocess_runner_interactive_inherits_the_terminal(tmp_path):
    """An editor needs the real stdin, stdout and stderr, not pipes.

    ``run`` deliberately captures and tees; this mode must not. Comparing the
    child's fd identities against the parent's is the only assertion that
    fails if any redirection is reintroduced.

    pytest leaves the parent's stdin on /dev/null, so without replacing fd 0
    a ``stdin=subprocess.DEVNULL`` regression would be indistinguishable.
    """
    probe = (
        "import os, sys, pathlib; "
        "pathlib.Path(sys.argv[1]).write_text("
        "repr([(os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (0, 1, 2)]))"
    )
    seen = tmp_path / "fds.txt"
    stand_in = tmp_path / "stdin.txt"
    stand_in.write_text("")
    saved = os.dup(0)
    try:
        with open(stand_in) as replacement:
            os.dup2(replacement.fileno(), 0)
            try:
                expected = repr([(os.fstat(fd).st_dev, os.fstat(fd).st_ino) for fd in (0, 1, 2)])
                status = SubprocessRunner().run_interactive(
                    Command([sys.executable, "-c", probe, str(seen)])
                )
            finally:
                os.dup2(saved, 0)
    finally:
        os.close(saved)
    assert status == 0
    assert seen.read_text() == expected


def _py(code: str) -> Command:
    return Command([sys.executable, "-c", code])


def test_run_with_input_survives_a_full_stderr_pipe(capsys: pytest.CaptureFixture[str]) -> None:
    code = ("import sys; sys.stderr.write('e' * 200000 + '\\nlast\\n'); sys.stderr.flush(); "
            "sys.exit(0 if len(sys.stdin.read()) == 300000 else 3)")
    status, tail = SubprocessRunner().run_with_input(_py(code), "x" * 300000)
    assert status == 0 and tail.endswith("last")


def test_run_with_input_child_exits_without_reading(capsys: pytest.CaptureFixture[str]) -> None:
    before = threading.active_count()
    status, _ = SubprocessRunner().run_with_input(_py("import sys; sys.exit(2)"), "x" * 1_000_000)
    assert status == 2
    assert threading.active_count() == before
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "Exception ignored" not in err


def test_run_with_input_keeps_the_last_20_stderr_lines(capsys: pytest.CaptureFixture[str]) -> None:
    code = "import sys\nfor i in range(30): print(f'line {i}', file=sys.stderr)"
    _, tail = SubprocessRunner().run_with_input(_py(code), "")
    assert tail.splitlines() == [f"line {i}" for i in range(10, 30)]
    assert "line 0" in capsys.readouterr().err


def test_run_with_input_spawn_failure_is_a_tool_error() -> None:
    with pytest.raises(ToolError):
        SubprocessRunner().run_with_input(Command(["/nonexistent/uv-stack-ssh"]), "")


class _ExplodingStream:
    def write(self, text: str) -> int:
        raise RuntimeError("stderr is gone")

    def flush(self) -> None:
        pass


def test_run_with_input_reaps_the_child_when_streaming_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def spy(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        spawned.append(real_popen(*args, **kwargs))
        return spawned[-1]

    before = threading.active_count()
    # The child never reads stdin, so the writer blocks on a full pipe until
    # the child is killed; without the kill this test hangs for 60 seconds.
    code = "import sys, time; sys.stderr.write('first\\n'); sys.stderr.flush(); time.sleep(60)"
    monkeypatch.setattr(subprocess, "Popen", spy)
    monkeypatch.setattr(sys, "stderr", _ExplodingStream())
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="stderr is gone"):
        SubprocessRunner().run_with_input(_py(code), "x" * 1_000_000)
    assert time.monotonic() - started < 30
    assert spawned[0].returncode is not None
    assert threading.active_count() == before


def test_run_with_input_forwards_unterminated_stderr_before_exit(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    flag = tmp_path / "seen"

    class RecordingStream:
        def __init__(self) -> None:
            self.text = ""

        def write(self, chunk: str) -> int:
            self.text += chunk
            if "partial" in self.text and not flag.exists():
                flag.write_text("")
            return len(chunk)

        def flush(self) -> None:
            pass

    recorder = RecordingStream()
    monkeypatch.setattr(sys, "stderr", recorder)
    code = f"""import sys, time, pathlib
sys.stderr.write('partial')
sys.stderr.flush()
flag = pathlib.Path(r'{flag}')
start = time.monotonic()
while not flag.exists() and time.monotonic() - start < 5:
    time.sleep(0.01)
sys.exit(0 if flag.exists() else 1)
"""
    status, _ = SubprocessRunner().run_with_input(_py(code), "")
    assert status == 0


def test_run_with_input_bounds_a_newline_free_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[str] = []

    class RecordingStream:
        def write(self, chunk: str) -> int:
            writes.append(chunk)
            return len(chunk)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, "stderr", RecordingStream())
    code = "import sys; sys.stderr.write('x' * 2_000_000); sys.exit(0)"
    _, tail = SubprocessRunner().run_with_input(_py(code), "")
    assert all(len(w) <= 65536 for w in writes)
    assert sum(len(w) for w in writes) == 2_000_000
    assert set(tail) == {"x"}
    assert len(tail) <= _STDERR_TAIL_LINES * 1024


def test_run_with_input_reaps_the_child_when_the_writer_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    def spy(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        spawned.append(real_popen(*args, **kwargs))
        return spawned[-1]

    def fail_to_start(self: Any) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(subprocess, "Popen", spy)
    monkeypatch.setattr(threading.Thread, "start", fail_to_start)
    code = "import time; time.sleep(60)"
    started = time.monotonic()
    with pytest.raises(UvStackError):
        SubprocessRunner().run_with_input(_py(code), "")
    assert time.monotonic() - started < 30
    assert spawned[0].returncode is not None
    assert spawned[0].stdin.closed
    assert spawned[0].stderr.closed


def test_run_with_input_reraises_a_writer_failure_after_reaping(
    capsys: pytest.CaptureFixture[str],
) -> None:
    spawned: list[subprocess.Popen[bytes]] = []
    real_popen = subprocess.Popen

    class FailingStdin:
        def __init__(self, real_stdin: Any) -> None:
            self.real_stdin = real_stdin

        def write(self, data: bytes) -> int:
            raise ValueError("stdin write failed")

        def close(self) -> None:
            self.real_stdin.close()

    def spy(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        proc = real_popen(*args, **kwargs)
        proc.stdin = FailingStdin(proc.stdin)
        spawned.append(proc)
        return proc

    before = threading.active_count()
    code = "import sys; sys.stdin.read(); sys.exit(0)"
    real_popen_ref = subprocess.Popen
    subprocess.Popen = spy
    try:
        with pytest.raises(ValueError, match="stdin write failed"):
            SubprocessRunner().run_with_input(_py(code), "x" * 1000)
    finally:
        subprocess.Popen = real_popen_ref
    assert spawned[0].returncode is not None
    assert threading.active_count() == before
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "Exception ignored" not in err


def test_recording_runner_records_input() -> None:
    runner = RecordingRunner(input_responder=lambda cmd, text: (5, "tail"))
    assert runner.run_with_input(Command(["ssh", "h", "stack import -"]), "doc") == (5, "tail")
    assert runner.inputs == ["doc"] and runner.commands[0].args[0] == "ssh"


def test_recording_runner_matches_the_real_signature() -> None:
    assert inspect.signature(RecordingRunner.run_with_input) == inspect.signature(
        SubprocessRunner.run_with_input)
