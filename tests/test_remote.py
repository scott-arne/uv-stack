from __future__ import annotations

import os

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.models import RemoteSettings
from uv_stack.operations.remote import load_remotes

_IS_ROOT = getattr(os, "geteuid", lambda: -1)() == 0


def test_absent_remotes_file_has_no_entries(config_tree: ConfigRoot) -> None:
    assert load_remotes(config_tree) == {}


@pytest.mark.parametrize("text", ["", "# no remotes yet\n", "---\n"])
def test_empty_remotes_document_has_no_entries(config_tree: ConfigRoot, text: str) -> None:
    config_tree.remotes_path().write_text(text)
    assert load_remotes(config_tree) == {}


def test_remotes_entries_load(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text(
        "gpu-box:\n  stack: ~/.local/bin/stack\n  root: /data/envs\nplain:\n")
    remotes = load_remotes(config_tree)
    assert remotes["gpu-box"] == RemoteSettings(stack="~/.local/bin/stack", root="/data/envs")
    assert remotes["plain"] == RemoteSettings()


@pytest.mark.parametrize("text", ["gpu-box: [1, 2]\n", "gpu-box:\n  host: x\n",
                                  "gpu-box:\n  stack: ''\n", "- a\n", "a: [\n",
                                  "null\n", "~\n", "!!null x\n"])
def test_invalid_remotes_file_is_refused(config_tree: ConfigRoot, text: str) -> None:
    config_tree.remotes_path().write_text(text)
    with pytest.raises(ConfigError, match="remotes.yaml"):
        load_remotes(config_tree)


def test_non_regular_remotes_file_is_refused(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().mkdir()
    with pytest.raises(ConfigError):
        load_remotes(config_tree)


def test_non_utf8_remotes_file_is_not_reported_as_bad_yaml(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_bytes(b"gpu-box:\n  root: /d\xff\n")
    with pytest.raises(ConfigError, match="not valid UTF-8") as caught:
        load_remotes(config_tree)
    assert "Invalid YAML" not in caught.value.message


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the file mode this relies on")
def test_unreadable_remotes_file_is_an_os_error(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("gpu-box:\n")
    config_tree.remotes_path().chmod(0)
    try:
        with pytest.raises(PermissionError):
            load_remotes(config_tree)
    finally:
        config_tree.remotes_path().chmod(0o644)


@pytest.mark.parametrize("text", ["yes:\n", "1:\n", "null:\n", "2026-09-29:\n"])
def test_non_string_remotes_host_key_is_refused(config_tree: ConfigRoot, text: str) -> None:
    config_tree.remotes_path().write_text(text)
    with pytest.raises(ConfigError, match="remotes.yaml") as caught:
        load_remotes(config_tree)
    assert "Quote the host name" in (caught.value.hint or "")


def test_quoted_remotes_host_keys_load_verbatim(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text('"yes":\n  root: /a\n"1":\n  root: /b\n')
    remotes = load_remotes(config_tree)
    assert remotes == {"yes": RemoteSettings(root="/a"), "1": RemoteSettings(root="/b")}
