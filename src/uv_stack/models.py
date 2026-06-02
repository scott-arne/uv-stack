"""Pydantic data models for uv-stack config objects.

Models are pure data: they hold no filesystem or subprocess knowledge. The
``from_lines`` constructors accept raw file lines and clean them, so callers may
pass either raw or pre-cleaned input.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from uv_stack.parse import clean_line


def _clean(lines: Iterable[str]) -> list[str]:
    cleaned = (clean_line(line) for line in lines)
    return [line for line in cleaned if line]


class Profile(BaseModel):
    """A reusable package group (``profiles/<name>.in``)."""

    name: str
    requirements: list[str] = Field(default_factory=list)

    @classmethod
    def from_lines(cls, name: str, lines: Iterable[str]) -> Profile:
        """Build a profile from raw file lines.

        :param name: Profile name (file stem).
        :param lines: Raw lines from the ``.in`` file.
        """
        return cls(name=name, requirements=_clean(lines))


class Bundle(BaseModel):
    """A composable recipe (``bundles/<name>.bundle``) of stack tokens."""

    name: str
    tokens: list[str] = Field(default_factory=list)

    @classmethod
    def from_lines(cls, name: str, lines: Iterable[str]) -> Bundle:
        """Build a bundle from raw file lines.

        :param name: Bundle name (file stem).
        :param lines: Raw lines from the ``.bundle`` file.
        """
        return cls(name=name, tokens=_clean(lines))


class EnvConfig(BaseModel):
    """Resolved view of a named environment's source files."""

    name: str
    python: str = "3.12"
    stack: list[str] = Field(default_factory=list)
    micromamba: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)


class ResolvedStack(BaseModel):
    """The result of resolving a list of stack tokens.

    ``profiles`` are emitted as ``-r profiles/<name>.in`` references; ``inline``
    holds literal packages, editable installs, and local archive paths.
    """

    profiles: list[str] = Field(default_factory=list)
    inline: list[str] = Field(default_factory=list)
