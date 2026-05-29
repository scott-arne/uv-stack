"""The command-execution seam.

Operations build :class:`Command` objects and hand them to a :class:`Runner`.
``SubprocessRunner`` actually executes (streaming output by default, or
capturing it); ``RecordingRunner`` records commands for tests and dry-runs. This
is the single mock point for the otherwise side-effect-free core.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from uvstack.errors import ToolError


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
        completed = subprocess.run(
            command.args,
            cwd=command.cwd,
            capture_output=capture,
            text=True,
        )
        if check and completed.returncode != 0:
            raise ToolError(
                f"Command failed ({completed.returncode}): {' '.join(command.args)}",
                command=command.args,
                returncode=completed.returncode,
            )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "" if capture else "",
        )


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
