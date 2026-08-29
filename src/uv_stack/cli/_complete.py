"""Shared pieces of the KIND/NAME argument pair, and NAME completion.

Every completion callback swallows all errors and returns an empty list:
completion must never crash the shell, even with a missing or broken config
root.
"""

from __future__ import annotations

from typing import Any

import rich_click as click

from uv_stack.config import ConfigRoot

#: KIND choices for the commands that accept a project as well as the three
#: named resources: ``stack show`` and ``stack edit``. Shared because the two
#: must offer the same set — a KIND one accepts and the other rejects sends
#: the user to a command that cannot help them. ``stack list`` deliberately
#: omits ``project``: there is no registry of projects to list.
KIND_CHOICES = ("env", "profile", "bundle", "project")


def _config_from_ctx(ctx: click.Context) -> ConfigRoot:
    root_ctx = ctx.find_root()
    return ConfigRoot.discover(root_ctx.params.get("root"))


def complete_env_names(
    ctx: click.Context, param: Any, incomplete: str
) -> list[str]:
    """Complete environment names for upgrade/status/show."""
    try:
        names = _config_from_ctx(ctx).list_envs()
    except Exception:
        return []
    return [name for name in names if name.startswith(incomplete)]


def complete_show_names(
    ctx: click.Context, param: Any, incomplete: str
) -> list[str]:
    """Complete NAME for the commands whose first argument is a KIND.

    Serves both ``stack show`` and ``stack edit``: each takes KIND then NAME,
    so the candidate list depends on the KIND already typed. ``project`` takes
    no NAME and yields no candidates.

    :param ctx: Click context, read for the KIND already on the command line.
    :param param: The parameter being completed; unused.
    :param incomplete: The partial NAME typed so far.
    :returns: Matching names, or an empty list when the KIND has none.
    """
    kind = ctx.params.get("kind")
    try:
        config = _config_from_ctx(ctx)
        if kind == "env":
            names = config.list_envs()
        elif kind == "profile":
            names = config.list_profiles()
        elif kind == "bundle":
            names = config.list_bundles()
        else:
            return []
    except Exception:
        return []
    return [name for name in names if name.startswith(incomplete)]
