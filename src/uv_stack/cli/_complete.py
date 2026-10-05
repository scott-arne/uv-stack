"""Shared pieces of the KIND/NAME argument pair, and NAME completion.

Every completion callback swallows all errors and returns an empty list:
completion must never crash the shell, even with a missing or broken config
root.
"""

from __future__ import annotations

from typing import Any

import rich_click as click

from uv_stack.config import ConfigRoot
from uv_stack.hints import escape_controls
from uv_stack.operations.remote import load_remotes

#: KIND choices for the commands that accept a project as well as the three
#: named resources: ``stack show`` and ``stack edit``. Shared because the two
#: must offer the same set — a KIND one accepts and the other rejects sends
#: the user to a command that cannot help them. The one exception is
#: ``remotes``, which only ``stack edit`` takes (:data:`EDIT_KIND_CHOICES`):
#: ``stack config remote list`` is its reader. ``stack list`` deliberately
#: omits ``project``: there is no registry of projects to list.
KIND_CHOICES = ("env", "profile", "bundle", "project")

#: ``stack edit``'s KIND choices: :data:`KIND_CHOICES` plus ``remotes``.
EDIT_KIND_CHOICES = (*KIND_CHOICES, "remotes")


def _config_from_ctx(ctx: click.Context) -> ConfigRoot:
    root_ctx = ctx.find_root()
    return ConfigRoot.discover(root_ctx.params.get("root"))


def complete_env_names(
    ctx: click.Context, param: Any, incomplete: str
) -> list[str]:
    """Complete environment names for upgrade/sync/status/show/delete."""
    try:
        names = _config_from_ctx(ctx).list_envs()
    except Exception:
        return []
    return [name for name in names if name.startswith(incomplete)]


def complete_profile_names(
    ctx: click.Context, param: Any, incomplete: str
) -> list[str]:
    """Complete profile names for ``stack delete profile``."""
    try:
        names = _config_from_ctx(ctx).list_profiles()
    except Exception:
        return []
    return [name for name in names if name.startswith(incomplete)]


def complete_bundle_names(
    ctx: click.Context, param: Any, incomplete: str
) -> list[str]:
    """Complete bundle names for ``stack delete bundle``."""
    try:
        names = _config_from_ctx(ctx).list_bundles()
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


def complete_remote_hosts(
    ctx: click.Context, param: Any, incomplete: str
) -> list[str]:
    """Complete HOST for ``stack config remote set`` and ``remove``."""
    try:
        hosts = load_remotes(_config_from_ctx(ctx))
    except Exception:
        return []
    # Click writes candidates unescaped and UTF-8-encodes them, so a control
    # character would corrupt or forge records and a lone surrogate would crash
    # the encode. Such a host can still be typed in full.
    return [host for host in hosts if host.startswith(incomplete) and escape_controls(host) == host]
