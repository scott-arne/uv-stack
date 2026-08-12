"""Shell-completion callbacks for NAME arguments.

Every callback swallows all errors and returns an empty list: completion must
never crash the shell, even with a missing or broken config root.
"""

from __future__ import annotations

from typing import Any

import rich_click as click

from uv_stack.config import ConfigRoot


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
    """Complete the NAME argument of ``show`` based on its KIND argument."""
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
