"""The ``config init`` operation: create the config directory tree.

Creates ``profiles/``, ``bundles/``, ``envs/``, and ``.locks/`` under the config
root if they are absent. It seeds no profiles or bundles; those are authored by
the user. ``.locks/`` would otherwise be created lazily by the first name lock
taken against the root; creating it here means a fresh root has it with the
initializing user's umask rather than whichever user happens to publish first.
Existing directories are left untouched.
"""

from __future__ import annotations

from pathlib import Path

from uv_stack.config import ConfigRoot


def init_config_root(config: ConfigRoot) -> list[Path]:
    """Create any missing config directories under the root.

    A root reached through a dangling link gets the link's target created
    first: asking to initialize the root is asking for it to exist, which no
    other writer may assume.

    :param config: The configuration root to initialize.
    :returns: The directories actually created (absent ones only), the link's
        target first when there was one to create.
    :raises ConfigError: When the root, or the nearest existing path above it,
        is not a directory.
    """
    created: list[Path] = []
    link = config.dangling_link()
    if link is None:
        config.refuse_broken_root()
    else:
        link.target.mkdir(parents=True, exist_ok=True)
        created.append(link.target)
    for directory in (
        config.profiles_dir,
        config.bundles_dir,
        config.envs_dir,
        config.locks_dir,
    ):
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    return created
