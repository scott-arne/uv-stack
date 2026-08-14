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
    ``warnings`` carries non-fatal resolution advisories (near-miss typos,
    profile/bundle shadowing) for the CLI edge to print.
    """

    profiles: list[str] = Field(default_factory=list)
    inline: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ClassifiedTokens(BaseModel):
    """The result of classifying stack tokens without expansion.

    :param entries: Full specifiers (e.g. ``["bundle:standard", "package:x"]``).
    :param warnings: Non-fatal resolution advisories, as in
        :class:`ResolvedStack`.
    """

    entries: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProjectTracking(BaseModel):
    """The ``[tool.uv-stack]`` project tracking table.

    ``applied`` is the removal ledger: refresh may only ever remove packages
    recorded there, so user-added dependencies are never touched.

    :param version: Tracking schema version (always 1 for this release).
    :param stack: The create-time stack tokens, verbatim.
    :param python: The raw ``--python`` value when one was given; ``None``
        keeps the project portable (machine defaults apply).
    :param applied: The flattened requirements uv-stack last applied.
    :param pending: Intent record for an in-flight refresh/init: the target
        applied list, written before uv mutations and cleared by the final
        ledger write. Present on disk only between a crash and the next
        successful run.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    stack: list[str]
    python: str | None = None
    applied: list[str] = Field(default_factory=list)
    pending: list[str] | None = Field(default=None)
