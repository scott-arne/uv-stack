"""The ``config init`` operation: create the config directory tree.

Creates ``profiles/``, ``bundles/``, and ``envs/`` under the config root if they
are absent. It seeds no profiles or bundles; those are authored by the user.
Existing directories are left untouched.
"""

from __future__ import annotations

from pathlib import Path

from uv_stack.config import ConfigRoot


def init_config_root(config: ConfigRoot) -> list[Path]:
    """Create any missing config directories under the root.

    :param config: The configuration root to initialize.
    :returns: The directories actually created (absent ones only).
    """
    created: list[Path] = []
    for directory in (config.profiles_dir, config.bundles_dir, config.envs_dir):
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created
