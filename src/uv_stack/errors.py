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
    :ivar resolution_warnings: Non-fatal advisories attached by the layer that
        raised, when an earlier step succeeded with caveats worth reporting
        alongside the failure. Named for the resolver, which was the first
        producer, but not limited to it — adoption notices, unverifiable-name
        warnings, and skipped-removal notices travel here too.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.resolution_warnings: list[str] = []


class ConfigError(UvStackError):
    """A config file or directory is missing or invalid."""


class NewerSchemaError(ConfigError):
    """A ``[tool.uv-stack]`` table declares a schema this version cannot read.

    Distinct from a plain :class:`ConfigError` because the ``--force`` guards
    tolerate unreadable tracking (it only decides which hint to show) but must
    never mask a forward-compatibility refusal.
    """


class ResolutionError(UvStackError):
    """A stack token could not be resolved (bad token, missing profile/bundle, cycle)."""


class EnvError(UvStackError):
    """A micromamba environment is missing and creation was not requested."""


class ToolError(UvStackError):
    """An external ``uv`` or ``micromamba`` command failed.

    Covers both shapes: a process that ran and exited non-zero, and one that
    could never be started at all — a missing or non-executable binary, which
    :func:`~uv_stack.runner._spawn_error` reports here with a synthetic status.

    :param command: The argv list of the command that failed.
    :param returncode: The process exit code, or 127 — the shell's conventional
        status for a command that could not be executed — when no process ran.
    :param detail: A concise tail of the command's stderr, when captured, so the
        CLI can report *why* it failed rather than only the exit code.
    """

    def __init__(
        self,
        message: str,
        command: list[str],
        returncode: int,
        hint: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message, hint)
        self.command = command
        self.returncode = returncode
        self.detail = detail
