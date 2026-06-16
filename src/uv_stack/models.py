"""Pydantic data models for uv-stack config objects.

Models are pure data: they hold no filesystem or subprocess knowledge. Profiles
and bundles are validated from the mappings parsed out of their YAML files.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Profile(BaseModel):
    """A reusable package group (``profiles/<name>.yaml``).

    :param name: Profile name (the file stem; not stored in the YAML).
    :param description: Optional one-line human description.
    :param tags: Optional free-form categorization tags.
    :param includes: Literal package specifications.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)


class Bundle(BaseModel):
    """A composable recipe (``bundles/<name>.yaml``).

    :param name: Bundle name (the file stem; not stored in the YAML).
    :param description: Optional one-line human description.
    :param tags: Optional free-form categorization tags.
    :param includes: References to profiles, other bundles, or packages, which
        the resolver expands.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)


class EnvConfig(BaseModel):
    """Resolved view of a named environment's source files."""

    name: str
    python: str = "3.12"
    stack: list[str] = Field(default_factory=list)
    micromamba: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)


class ResolvedStack(BaseModel):
    """The result of resolving a list of stack tokens.

    Profiles are recorded by name and expanded inline at render time; ``inline``
    holds literal packages, editable installs, and local archive paths.
    """

    profiles: list[str] = Field(default_factory=list)
    inline: list[str] = Field(default_factory=list)
