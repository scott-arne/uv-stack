"""``stack sync remote``: settings, the ssh command, and exit-status hints."""

from __future__ import annotations

import os

import yaml
from pydantic import ValidationError

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.fsutil import read_text_utf8, require_regular_file
from uv_stack.models import RemoteSettings


def load_remotes(config: ConfigRoot) -> dict[str, RemoteSettings]:
    """Load ``remotes.yaml``; an absent file has no entries.

    :raises ConfigError: When the file is not regular, is not valid UTF-8, or
        does not parse or validate.
    :raises OSError: When the file exists but cannot be read.
    """
    path = config.remotes_path()
    if not os.path.lexists(path):
        return {}
    require_regular_file(path)
    # Read outside the broad except: a decode failure is already a ConfigError
    # naming the file, and an unreadable file is an OSError, not bad YAML.
    text = read_text_utf8(path)
    try:
        data = yaml.safe_load(text)
    except Exception as exc:  # Same breadth as ConfigRoot._load_yaml_model.
        raise ConfigError(f"Invalid YAML in {path}: {type(exc).__name__}: {exc}",
                          hint="Fix the YAML syntax.", path=path) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}, got {type(data).__name__}.",
                          path=path)
    remotes: dict[str, RemoteSettings] = {}
    for host, entry in data.items():
        try:
            remotes[str(host)] = RemoteSettings.model_validate({} if entry is None else entry)
        except ValidationError as exc:
            raise ConfigError(f"Invalid remotes config in {path}: {exc}", path=path) from exc
    return remotes
