"""Exception hierarchy for uv-stack.

All recoverable, user-facing failures are :class:`UvStackError` subclasses so
the CLI edge can render them as panels with a remediation hint. Anything else
propagates as an ordinary exception (a real bug) with a full traceback.
"""

from __future__ import annotations


class UvStackError(Exception):
    """Base class for user-facing uv-stack errors.

    :param message: Human-readable description of what went wrong.
    :param hint: Optional remediation hint shown alongside the message.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ConfigError(UvStackError):
    """A config file or directory is missing or invalid."""


class ResolutionError(UvStackError):
    """A stack token could not be resolved (bad token, missing profile/bundle, cycle)."""


class EnvError(UvStackError):
    """A micromamba environment is missing and creation was not requested."""


class ToolError(UvStackError):
    """An external ``uv`` or ``micromamba`` command exited non-zero.

    :param command: The argv list of the command that failed.
    :param returncode: The process exit code.
    """

    def __init__(
        self,
        message: str,
        command: list[str],
        returncode: int,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint)
        self.command = command
        self.returncode = returncode
