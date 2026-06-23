"""The command-execution seam.

Operations build :class:`Command` objects and hand them to a :class:`Runner`.
``SubprocessRunner`` actually executes (streaming output by default, or
capturing it); ``RecordingRunner`` records commands for tests and dry-runs. This
is the single mock point for the otherwise side-effect-free core.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from uv_stack.errors import ToolError


def _pty_supported() -> bool:
    """Whether this platform has the POSIX pty machinery uv needs for bars."""
    try:
        import fcntl  # noqa: F401
        import pty  # noqa: F401
        import termios  # noqa: F401
    except ImportError:  # pragma: no cover - non-Unix platforms
        return False
    return True


_PTY_AVAILABLE = _pty_supported()

# How many trailing stderr lines to retain as a failed command's ``detail``.
# Enough to capture a uv "requires X, but Y is installed" diagnostic without
# hauling an entire resolver dump into the error.
_STDERR_TAIL_LINES = 20


# Matches ANSI CSI escape sequences (colour, cursor moves, progress-bar
# redraws). uv emits these when writing to a terminal; we strip them so a
# captured failure ``detail`` is plain text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from ``text``."""
    return _ANSI_RE.sub("", text)


def _tail(text: str) -> str:
    """Return the last :data:`_STDERR_TAIL_LINES` non-empty lines of ``text``."""
    lines = [line for line in _strip_ansi(text).splitlines() if line.strip()]
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


@dataclass
class Command:
    """A single external command to run.

    :param args: The argv list.
    :param cwd: Optional working directory.
    """

    args: list[str]
    cwd: Path | None = None


@dataclass
class CommandResult:
    """The outcome of running a :class:`Command`.

    :param returncode: Process exit code.
    :param stdout: Captured standard output (empty unless ``capture=True``).
    """

    returncode: int
    stdout: str = ""


class Runner(Protocol):
    """Protocol implemented by command runners."""

    def run(
        self, command: Command, *, capture: bool = False, check: bool = True
    ) -> CommandResult:
        """Execute ``command``.

        :param command: The command to run.
        :param capture: Capture and return stdout instead of streaming it.
        :param check: Raise :class:`ToolError` on a non-zero exit.
        """
        ...


class SubprocessRunner:
    """Runs commands with :mod:`subprocess`."""

    def run(
        self, command: Command, *, capture: bool = False, check: bool = True
    ) -> CommandResult:
        if capture:
            completed = subprocess.run(
                command.args,
                cwd=command.cwd,
                capture_output=True,
                text=True,
            )
            returncode = completed.returncode
            stdout = completed.stdout or ""
            detail = _tail(completed.stderr or "")
        elif _PTY_AVAILABLE and sys.stderr.isatty():
            # uv only renders colour and progress bars when its stderr is a
            # terminal. A plain pipe would silence them, so give the child a
            # pseudo-terminal while still teeing the stream to capture a failure
            # tail. stdout is left inherited.
            returncode, detail = self._run_with_pty(command)
            stdout = ""
        else:
            # No TTY (CI, redirected output) or non-Unix: stream stderr through a
            # bounded buffer so a failing command can still report *why* it
            # failed (e.g. uv's "requires X, but Y is installed").
            proc = subprocess.Popen(
                command.args,
                cwd=command.cwd,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert proc.stderr is not None
            tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
            for line in proc.stderr:
                sys.stderr.write(line)
                if line.strip():
                    tail.append(line.rstrip())
            returncode = proc.wait()
            stdout = ""
            detail = "\n".join(tail)

        if check and returncode != 0:
            raise ToolError(
                f"Command failed ({returncode}): {' '.join(command.args)}",
                command=command.args,
                returncode=returncode,
                detail=detail or None,
            )
        return CommandResult(returncode=returncode, stdout=stdout)

    @staticmethod
    def _run_with_pty(command: Command) -> tuple[int, str]:
        """Run ``command`` with its stderr attached to a pseudo-terminal.

        uv detects the pty as a real terminal and keeps emitting colour and
        progress bars. The output is passed through to the real stderr verbatim
        while a bounded copy is retained so a failure can report its cause.

        :param command: The command to run (stdout is inherited).
        :returns: The exit code and a plain-text tail of stderr.
        """
        import fcntl
        import pty
        import termios

        master, slave = pty.openpty()
        # Match the child's terminal size to ours so uv sizes its bars correctly.
        try:
            winsize = fcntl.ioctl(
                sys.stderr.fileno(), termios.TIOCGWINSZ, b"\x00" * 8
            )
            fcntl.ioctl(slave, termios.TIOCSWINSZ, winsize)
        except OSError:  # pragma: no cover - depends on the host terminal
            pass

        proc = subprocess.Popen(command.args, cwd=command.cwd, stderr=slave)
        os.close(slave)
        # Retain only the trailing bytes: progress-bar redraws are voluminous but
        # transient, and uv prints the actionable diagnostic last.
        captured = bytearray()
        cap = _STDERR_TAIL_LINES * 1024
        try:
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError:  # EIO when the child closes the pty
                    break
                if not chunk:
                    break
                os.write(sys.stderr.fileno(), chunk)
                captured += chunk
                if len(captured) > cap:
                    del captured[:-cap]
        finally:
            os.close(master)
        returncode = proc.wait()
        detail = _tail(captured.decode("utf-8", errors="replace"))
        return returncode, detail


@dataclass
class RecordingRunner:
    """Records commands without executing them.

    :param responder: Optional callable returning a :class:`CommandResult` for a
        given command; defaults to a successful empty result.
    """

    responder: Callable[[Command], CommandResult] | None = None
    commands: list[Command] = field(default_factory=list)

    def run(
        self, command: Command, *, capture: bool = False, check: bool = True
    ) -> CommandResult:
        self.commands.append(command)
        if self.responder is not None:
            return self.responder(command)
        return CommandResult(returncode=0, stdout="")
