"""The ``config init`` operation: seed default profiles and bundles.

Writes the contents from :mod:`uvstack.defaults` into the config tree, creating
``profiles/``, ``bundles/``, and ``envs/`` as needed. Existing files are never
overwritten; only absent files are written.
"""

from __future__ import annotations

from pathlib import Path

from uvstack.config import ConfigRoot
from uvstack.defaults import DEFAULT_BUNDLES, DEFAULT_PROFILES


def seed_defaults(config: ConfigRoot) -> list[Path]:
    """Write default profiles and bundles into the config tree.

    :param config: The configuration root to seed.
    :returns: The list of files actually written (absent files only).
    """
    config.profiles_dir.mkdir(parents=True, exist_ok=True)
    config.bundles_dir.mkdir(parents=True, exist_ok=True)
    config.envs_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for name, content in DEFAULT_PROFILES.items():
        path = config.profile_path(name)
        if not path.exists():
            path.write_text(content)
            written.append(path)

    for name, content in DEFAULT_BUNDLES.items():
        path = config.bundle_path(name)
        if not path.exists():
            path.write_text(content)
            written.append(path)

    return written
