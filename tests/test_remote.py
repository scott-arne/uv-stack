from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

import uv_stack.cli.sync_cmd
from uv_stack.cli import cli
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.models import RemoteSettings
from uv_stack.operations.remote import (
    ResolvedRemote,
    check_destination,
    explain_exit,
    load_remotes,
    remote_command,
    resolve_settings,
)
from uv_stack.runner import RecordingRunner

_IS_ROOT = getattr(os, "geteuid", lambda: -1)() == 0


def test_absent_remotes_file_has_no_entries(config_tree: ConfigRoot) -> None:
    assert load_remotes(config_tree) == {}


@pytest.mark.parametrize("text", ["", "# no remotes yet\n", "---\n", "---\n# no remotes yet\n"])
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
                                  "null\n", "~\n", "!!null x\n", "!!null\n",
                                  '!!null ""\n', "--- !!null\n"])
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


def test_settings_precedence(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("box:\n  stack: /opt/stack\n  root: /r\n")
    assert resolve_settings(config_tree, "box", stack_flag=None, root_flag=None) == \
        ResolvedRemote("/opt/stack", "/r")
    assert resolve_settings(config_tree, "box", stack_flag="s", root_flag="/f") == \
        ResolvedRemote("s", "/f")
    assert resolve_settings(config_tree, "other", stack_flag=None, root_flag=None) == \
        ResolvedRemote("stack", None)


def test_stack_is_verbatim_and_the_rest_quoted() -> None:
    command = remote_command("box", "uvx --from uv-stack stack", "/data/my envs",
                             ["--overwrite"])
    assert command.args == [
        "ssh", "box", "uvx --from uv-stack stack --root '/data/my envs' import - --overwrite"]


def test_destination_starting_with_dash_is_refused() -> None:
    with pytest.raises(UvStackError):
        check_destination("-oProxyCommand=x")


@pytest.mark.parametrize(("code", "tail", "needle"), [
    (255, "", "connect"),
    (127, "", "uv tool install uv-stack"),
    (2, "Error: No such command 'import'.", "Upgrade uv-stack"),
])
def test_exit_hints(code: int, tail: str, needle: str) -> None:
    error = explain_exit(code, tail, "box", "stack")
    assert error is not None
    assert needle in error.message + (error.hint or "")


def test_exit_127_names_the_remotes_entry() -> None:
    error = explain_exit(127, "", "box", "stack")
    assert error is not None and error.hint is not None
    assert "box:\n    stack: ~/.local/bin/stack" in error.hint


@pytest.mark.parametrize(("code", "tail"), [(2, "Usage: stack import"), (1, ""), (0, "")])
def test_other_exits_get_no_hint(code: int, tail: str) -> None:
    assert explain_exit(code, tail, "box", "stack") is None


def test_sync_remote_sends_the_document(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner(input_responder=lambda cmd, text: (0, ""))
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "main", "box", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(runner.inputs[0])["items"] == ["env:main"]
    assert runner.commands[0].args == ["ssh", "box", "stack import - --dry-run"]


def test_sync_remote_exit_status_is_the_remotes(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner(input_responder=lambda cmd, text: (255, ""))
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "main", "box"])
    assert result.exit_code == 255 and "connect" in result.output


def test_sync_remote_refuses_recreate_with_no_build_before_connecting(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "main", "box", "--recreate", "--no-build"])
    assert result.exit_code == 2 and runner.commands == []
    assert "cannot be combined" in result.output


def test_sync_remote_with_only_dest_sends_the_whole_root(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner(input_responder=lambda cmd, text: (0, ""))
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote", "box"])
    assert result.exit_code == 0, result.output
    assert runner.commands[0].args[1] == "box"
    expected = {"env:main", "profile:ds", "bundle:standard"}
    assert expected <= set(json.loads(runner.inputs[0])["items"])


def test_sync_remote_missing_item_never_connects(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "profile:ghost", "box"])
    assert result.exit_code == 1 and runner.commands == []


@pytest.mark.parametrize(("flag", "message"), [
    ("--remote-stack", "--remote-stack cannot be empty."),
    ("--remote-root", "--remote-root cannot be empty."),
])
def test_sync_remote_refuses_empty_flag_values(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch, flag: str, message: str
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "main", "box", flag, ""])
    assert result.exit_code == 2 and runner.commands == []
    assert message in result.output
