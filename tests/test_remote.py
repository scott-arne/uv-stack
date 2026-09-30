from __future__ import annotations

import json
import os
import shlex
import stat
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest
import rich_click as click
from click.testing import CliRunner

import uv_stack.cli.sync_cmd
from tests.conftest import _lock_held_by_another_process
from uv_stack import fsutil
from uv_stack.cli import cli
from uv_stack.cli.transfer_cmd import import_cmd
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.models import RemoteSettings
from uv_stack.operations import remote as remote_ops
from uv_stack.operations.remote import (
    Removal,
    ResolvedRemote,
    check_destination,
    explain_exit,
    has_comment,
    load_remotes,
    parse_remotes,
    remote_command,
    remove_remote,
    resolve_settings,
    set_remote,
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


def test_a_null_host_key_is_named_in_yaml_terms(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("null:\n")
    with pytest.raises(ConfigError) as caught:
        load_remotes(config_tree)
    assert "is a YAML null, not a name" in caught.value.message


def test_quoted_remotes_host_keys_load_verbatim(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text('"yes":\n  root: /a\n"1":\n  root: /b\n')
    remotes = load_remotes(config_tree)
    assert remotes == {"yes": RemoteSettings(root="/a"), "1": RemoteSettings(root="/b")}


def test_parse_remotes_names_the_given_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="elsewhere.yaml"):
        parse_remotes("a: [\n", tmp_path / "elsewhere.yaml")


def test_parse_remotes_keeps_file_order(tmp_path: Path) -> None:
    remotes = parse_remotes("b: {}\na:\n  root: /a\n", tmp_path / "remotes.yaml")
    assert list(remotes) == ["b", "a"]
    assert remotes["a"] == RemoteSettings(root="/a")


@pytest.mark.parametrize("text", [
    "# c\na: {}\n",                              # full line
    "a:\n  stack: x # c\n",                      # trailing
    "a: {stack: x, # c\n  root: y}\n",           # inside a flow mapping
    "a:\n  stack: | # c\n    x\n",               # literal block header
    "a:\n  stack: >- # c\n    x\n",              # folded block header with chomping
    "a:\n  stack: |2 # c\n     x\n",             # block header with an indent indicator
    "a:\r\n  stack: | # c\r\n    x\r\n",         # block header, CRLF
    "a:\n  stack: |\n    x\n# c\nb: {}\n",       # after a block scalar
    "a:\n  stack: |+\n    x\n\n# c\nb: {}\n",    # after a keep-chomped block scalar
    "# no remotes yet\n",                        # the whole file
    "a:\n  stack: x\n  # c\n  root: y\n",        # between keys
    "a: &x # c\n  stack: s\n",                   # after an anchor
    "a: !!map # c\n  stack: s\n",                # after a tag
    "a: # c\n",                                  # an empty value
    "a:\n  stack: x\n    y # c\n",               # a multi-line plain scalar
    "--- # c\na: {}\n",                          # a document start
    "a: {}\n... # c\n",                          # a document end
    "%YAML 1.1 # c\n---\na: {}\n",               # a directive
])
def test_has_comment_finds_a_comment(text: str) -> None:
    assert has_comment(text)


@pytest.mark.parametrize("text", [
    "a:\n  stack: a#b\n",                        # inside a plain scalar
    "a:\n  root: /x/#/y\n",                      # a path
    "a:\n  stack: x\n    y#z\n",                 # a multi-line plain scalar
    "a#: {}\n",                                  # a plain key
    "a:\n  stack: 'x # y'\n",                    # single-quoted
    'a:\n  stack: "x # y"\n',                    # double-quoted
    "'a # b':\n  stack: x\n",                    # a quoted key
    "a:\n  stack: |\n    x # y\n",               # block scalar content
    "a:\n  stack: |\n    # y\n",                 # block content that starts with #
    "a:\r\n  stack: |\r\n    x # y\r\n",         # block content, CRLF
    "a:\n  stack: |\n    x # y",                 # block content at EOF, no newline
    "a:\n  stack: |-\n    x\n",                  # a header with no comment
    "",                                          # an empty file
])
def test_has_comment_ignores_a_hash_inside_a_token(text: str) -> None:
    assert not has_comment(text)


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


@pytest.mark.parametrize("dest", ["box", "yes"])
def test_exit_127_suggests_a_remotes_entry_that_loads(config_tree: ConfigRoot, dest: str) -> None:
    # The entry is there to be pasted into remotes.yaml, so it must load: a
    # DEST that YAML reads as a bool, such as yes, has to come out quoted.
    error = explain_exit(127, "", dest, "stack")
    assert error is not None and error.hint is not None
    entry = [line for line in error.hint.splitlines() if line.startswith("  ")]
    config_tree.remotes_path().write_text(textwrap.dedent("\n".join(entry)) + "\n")
    assert load_remotes(config_tree) == {
        dest: RemoteSettings(stack="PATH=$HOME/.local/bin:$PATH stack")}


def test_exit_255_allows_for_a_dropped_session() -> None:
    error = explain_exit(255, "", "box", "stack")
    assert error is not None and error.hint is not None
    assert "failed or dropped" in error.message
    assert "re-running the same command is safe" in error.hint


def _options(command: click.Command) -> dict[str | None, click.Option]:
    return {p.name: p for p in command.params if isinstance(p, click.Option)}


def test_sync_remote_help_matches_import_and_names_its_metavars() -> None:
    remote = _options(uv_stack.cli.sync_cmd.sync_remote)
    assert remote["overwrite"].help == _options(import_cmd)["overwrite"].help
    assert remote["remote_root"].metavar == "PATH"
    assert remote["remote_stack"].metavar == "CMD"
    group = uv_stack.cli.sync_cmd.sync
    assert "remote" in (group.help or "") and "remote" in (uv_stack.cli.sync_cmd.__doc__ or "")
    # rich-click reads help as markup, so a bracketed word would vanish.
    helps = [uv_stack.cli.sync_cmd.sync_remote.help, *(o.help for o in remote.values())]
    assert not any("[" in (text or "") for text in helps)


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


@pytest.mark.parametrize(("flags", "sent"), [
    (["--strict"], "--strict"),
    (["--recreate"], "--recreate"),
    (["--no-build"], "--no-build"),
    (["--strict", "--dry-run", "--no-build", "--overwrite"],
     "--overwrite --no-build --dry-run --strict"),
])
def test_sync_remote_forwards_its_import_flags(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch, flags: list[str], sent: str
) -> None:
    runner = RecordingRunner(input_responder=lambda cmd, text: (0, ""))
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "main", "box", *flags])
    assert result.exit_code == 0, result.output
    assert runner.commands[0].args == ["ssh", "box", f"stack import - {sent}"]


def test_sync_remote_explains_a_remote_too_old_for_import(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    tail = "Usage: stack [OPTIONS] COMMAND\nError: No such command 'import'.\n"
    runner = RecordingRunner(input_responder=lambda cmd, text: (2, tail))
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "main", "box"], env={"COLUMNS": "200"})
    assert result.exit_code == 2
    assert "Upgrade uv-stack on box: 'uv tool upgrade uv-stack'." in result.output


def test_sync_remote_takes_its_command_and_root_from_remotes_yaml(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_tree.remotes_path().write_text(
        "box:\n  stack: PATH=$HOME/.local/bin:$PATH stack\n  root: ~/envs\n")
    runner = RecordingRunner(input_responder=lambda cmd, text: (0, ""))
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "main", "box"])
    assert result.exit_code == 0, result.output
    # The stack value arrives verbatim for the remote shell to expand; the
    # root arrives quoted, so only the remote's own expanduser reads its '~'.
    assert runner.commands[0].args == [
        "ssh", "box", "PATH=$HOME/.local/bin:$PATH stack --root '~/envs' import -"]


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


def test_sync_remote_prints_an_invalid_remotes_yaml_without_a_control_character(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A YAML "\e" puts ESC in the key, and pydantic quotes an extra key in the
    # error that load_remotes embeds in its message.
    config_tree.remotes_path().write_text('box:\n  "evil\\e[2J": 1\n')
    runner = RecordingRunner()
    monkeypatch.setattr(uv_stack.cli.sync_cmd, "SubprocessRunner", lambda: runner)
    result = CliRunner().invoke(cli, ["--root", str(config_tree.root), "sync", "remote",
                                      "main", "box"], env={"COLUMNS": "200"})
    assert result.exit_code == 1 and runner.commands == []
    text = result.output + result.stderr
    assert "Invalid remotes config" in text
    assert "\x1b" not in text
    assert "evil\\x1b[2J" in text


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


_FAKE_STACK = '''
import json, sys
from pathlib import Path
from uv_stack.commands import micromamba_python_info, micromamba_python_path
from uv_stack.runner import CommandResult, RecordingRunner
import uv_stack.cli.transfer_cmd, uv_stack.cli.upgrade

log = Path(sys.argv.pop(1))
# The env is absent until micromamba creates it; after that the interpreter
# probe must report a path, or upgrade_env cannot find one to sync into.
created = False

def respond(cmd):
    global created
    with log.open("a") as handle:
        handle.write(json.dumps(cmd.args) + "\\n")
    if cmd.args[1:2] == ["create"]:
        created = True
    if cmd.args == micromamba_python_path("main").args:
        return CommandResult(0, "/envs/main/bin/python\\n") if created else CommandResult(1, "")
    if cmd.args[:3] == ["uv", "pip", "compile"]:
        out = cmd.args[cmd.args.index("-o") + 1]
        Path(out).write_text("numpy==1.26.4\\n")
    if cmd.args == micromamba_python_info("main").args:
        return CommandResult(1, "")
    return CommandResult(0, "")

runner = RecordingRunner(responder=respond)
uv_stack.cli.transfer_cmd.SubprocessRunner = lambda: runner
uv_stack.cli.upgrade.SubprocessRunner = lambda: runner
from uv_stack.cli import main
main()
'''


def _fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text('#!/bin/sh\nshift\nexec /bin/sh -c "$1"\n')
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    script = tmp_path / "fake_stack.py"
    script.write_text(_FAKE_STACK)
    log = tmp_path / "commands.log"
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {shlex.quote(str(log))}", log


def main_exit(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exited:
        cli.main(args=argv, prog_name="stack")
    code = exited.value.code
    return code if isinstance(code, int) else 0


def test_sync_remote_round_trip(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    stack, log = _fake_ssh(tmp_path, monkeypatch)
    remote = ConfigRoot(tmp_path / "remote")
    remote.root.mkdir()
    base = ["--root", str(config_tree.root), "sync", "remote", "main", "box",
            "--remote-stack", stack, "--remote-root", str(remote.root)]

    assert main_exit(base) == 0
    assert remote.env_stack_path("main").read_text() == "@standard\n"
    assert any('"compile"' in line for line in log.read_text().splitlines())

    remote.profile_path("ds").write_text("includes:\n  - scipy\n")
    assert main_exit(base) == 1
    captured = capfd.readouterr()
    assert "profiles/ds.yaml (this machine)" in captured.out + captured.err

    assert remote.profile_path("ds").read_text() == "includes:\n  - scipy\n"

    logged = len(log.read_text().splitlines())
    assert main_exit([*base, "--overwrite", "--no-build"]) == 0
    assert remote.profile_path("ds").read_text() == config_tree.profile_path("ds").read_text()
    # Each run is a fresh remote process that finds no environment, so a
    # dropped --no-build would show up here as a create and a compile.
    assert not any('"compile"' in line for line in log.read_text().splitlines()[logged:])


def _umask() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return umask


def _racing(monkeypatch: pytest.MonkeyPatch, race: Callable[[], None]) -> None:
    """Run ``race`` after the writer's read and before its publish."""
    serialize = remote_ops._serialize_remotes

    def racing(remotes: dict[str, remote_ops.RemoteSettings]) -> str:
        race()
        return serialize(remotes)

    monkeypatch.setattr(remote_ops, "_serialize_remotes", racing)


def test_set_remote_creates_the_file(config_tree: ConfigRoot) -> None:
    entry = set_remote(config_tree, "gpu-box", stack="/opt/stack")
    assert entry == RemoteSettings(stack="/opt/stack")
    assert config_tree.remotes_path().read_text() == "gpu-box:\n  stack: /opt/stack\n"


def test_set_remote_merges_and_keeps_host_order(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a:\n  root: /a\nb:\n  stack: s\n")
    assert set_remote(config_tree, "a", stack="/s") == RemoteSettings(stack="/s", root="/a")
    set_remote(config_tree, "c", root="/c")
    assert config_tree.remotes_path().read_text() == (
        "a:\n  stack: /s\n  root: /a\nb:\n  stack: s\nc:\n  root: /c\n")


def test_set_remote_twice_with_a_hash_in_a_value(config_tree: ConfigRoot) -> None:
    # The first write quotes the value, so the second must not read it as a
    # comment and refuse.
    set_remote(config_tree, "a", stack="run # not a comment")
    set_remote(config_tree, "a", root="/r")
    assert load_remotes(config_tree) == {
        "a": RemoteSettings(stack="run # not a comment", root="/r")}


def test_set_remote_quotes_hosts_yaml_would_retype(config_tree: ConfigRoot) -> None:
    for host in ("yes", "null", "1"):
        set_remote(config_tree, host, root=f"/{host}")
    assert load_remotes(config_tree) == {
        "yes": RemoteSettings(root="/yes"),
        "null": RemoteSettings(root="/null"),
        "1": RemoteSettings(root="/1"),
    }


def test_set_remote_rewrites_a_null_entry_as_empty(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("laptop:\n")
    set_remote(config_tree, "gpu-box", stack="/opt/stack")
    assert config_tree.remotes_path().read_text() == (
        "laptop: {}\ngpu-box:\n  stack: /opt/stack\n")


def test_set_remote_refuses_a_commented_file(config_tree: ConfigRoot) -> None:
    original = b"# my hosts\na:\n  root: /a\n"
    config_tree.remotes_path().write_bytes(original)
    with pytest.raises(ConfigError, match="contains comments") as caught:
        set_remote(config_tree, "a", stack="/s")
    assert caught.value.hint == "Edit it with 'stack edit remotes', or remove the comments first."
    assert config_tree.remotes_path().read_bytes() == original


@pytest.mark.parametrize("text", ["gpu-box: [1, 2]\n", "a: [\n", "yes:\n"])
def test_set_remote_refuses_an_invalid_file(config_tree: ConfigRoot, text: str) -> None:
    config_tree.remotes_path().write_text(text)
    with pytest.raises(ConfigError, match="remotes.yaml"):
        set_remote(config_tree, "a", stack="/s")
    assert config_tree.remotes_path().read_text() == text


def test_set_remote_refuses_a_non_regular_file(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().mkdir()
    with pytest.raises(ConfigError, match="Not a regular file"):
        set_remote(config_tree, "a", stack="/s")


def test_set_remote_refuses_an_empty_value(config_tree: ConfigRoot) -> None:
    with pytest.raises(ConfigError, match="Invalid remotes config") as caught:
        set_remote(config_tree, "a", stack="")
    assert "at least 1 character" in caught.value.message
    assert not config_tree.remotes_path().exists()


def test_set_remote_refuses_a_value_that_would_not_read_back(config_tree: ConfigRoot) -> None:
    # safe_dump writes U+0085 as a line break, which loads back as a space.
    with pytest.raises(ConfigError, match="would not read back"):
        set_remote(config_tree, "a", root="a\x85b")
    assert not config_tree.remotes_path().exists()


@pytest.mark.parametrize("host,message", [
    ("", "empty host name"),
    ("-oProxyCommand=x", "begins with '-'"),
    ("a\x1b[31mb", "control character"),
    ("a\nb", "control character"),
])
def test_set_remote_refuses_a_bad_host_before_any_io(
    config_tree: ConfigRoot, host: str, message: str
) -> None:
    with pytest.raises(UvStackError, match=message):
        set_remote(config_tree, host, stack="/s")
    assert not config_tree.remotes_path().exists()
    assert not config_tree.remotes_lock_path().exists()


def test_remove_remote_deletes_the_entry(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a:\n  root: /a\nb: {}\n")
    assert remove_remote(config_tree, "a") == Removal(None, ())
    assert config_tree.remotes_path().read_text() == "b: {}\n"


def test_remove_remote_of_the_last_entry_leaves_an_empty_mapping(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a: {}\n")
    remove_remote(config_tree, "a")
    assert config_tree.remotes_path().read_text() == "{}\n"
    assert load_remotes(config_tree) == {}


def test_remove_remote_clears_a_field(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a:\n  stack: s\n  root: /a\n")
    assert remove_remote(config_tree, "a", ["root"]) == Removal(RemoteSettings(stack="s"), ())
    assert config_tree.remotes_path().read_text() == "a:\n  stack: s\n"


def test_remove_remote_of_the_last_field_keeps_the_entry(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a:\n  root: /a\n")
    assert remove_remote(config_tree, "a", ["root"]) == Removal(RemoteSettings(), ())
    assert config_tree.remotes_path().read_text() == "a: {}\n"


def test_remove_remote_refuses_a_missing_host(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a: {}\n")
    with pytest.raises(ConfigError, match="No remote named 'b'") as caught:
        remove_remote(config_tree, "b")
    assert caught.value.hint == "Run 'stack config remote list' to see the configured hosts."
    assert config_tree.remotes_path().read_text() == "a: {}\n"


def test_remove_remote_from_an_absent_file_names_the_host(config_tree: ConfigRoot) -> None:
    with pytest.raises(ConfigError, match="No remote named 'a'"):
        remove_remote(config_tree, "a")
    assert not config_tree.remotes_path().exists()


def test_remove_remote_of_an_unset_field_writes_nothing(config_tree: ConfigRoot) -> None:
    # Not in normalized form, so any rewrite would show.
    original = "a:\n    root:   /a\n"
    config_tree.remotes_path().write_text(original)
    assert remove_remote(config_tree, "a", ["stack"]) == Removal(
        RemoteSettings(root="/a"), ("stack",))
    assert config_tree.remotes_path().read_text() == original


def test_remove_remote_notes_only_the_unset_fields(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a:\n  stack: s\n")
    assert remove_remote(config_tree, "a", ["root", "stack"]) == Removal(
        RemoteSettings(), ("root",))
    assert config_tree.remotes_path().read_text() == "a: {}\n"


def test_remove_remote_handles_a_repeated_field_once(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a:\n  stack: s\n")
    assert remove_remote(config_tree, "a", ["root", "root"]).not_set == ("root",)
    assert remove_remote(config_tree, "a", ["stack", "stack"]) == Removal(RemoteSettings(), ())


def test_remove_remote_takes_hosts_set_would_refuse(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text('"-bad": {}\n"a\\eb": {}\nok: {}\n')
    remove_remote(config_tree, "-bad")
    remove_remote(config_tree, "a\x1bb")
    assert config_tree.remotes_path().read_text() == "ok: {}\n"


def test_remove_remote_refuses_a_commented_file(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a: {} # c\n")
    with pytest.raises(ConfigError, match="contains comments"):
        remove_remote(config_tree, "a")
    assert config_tree.remotes_path().read_text() == "a: {} # c\n"


def test_set_remote_writes_through_a_symlink(config_tree: ConfigRoot, tmp_path: Path) -> None:
    real = tmp_path / "dotfiles" / "remotes.yaml"
    real.parent.mkdir()
    real.write_text("a: {}\n")
    config_tree.remotes_path().symlink_to(real)
    set_remote(config_tree, "a", root="/r")
    assert config_tree.remotes_path().is_symlink()
    assert os.readlink(config_tree.remotes_path()) == str(real)
    assert real.read_text() == "a:\n  root: /r\n"


def test_set_remote_keeps_a_private_mode(config_tree: ConfigRoot) -> None:
    config_tree.remotes_path().write_text("a: {}\n")
    config_tree.remotes_path().chmod(0o600)
    set_remote(config_tree, "a", root="/r")
    assert stat.S_IMODE(config_tree.remotes_path().stat().st_mode) == 0o600


def test_set_remote_keeps_a_private_mode_through_a_symlink(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    real = tmp_path / "remotes-real.yaml"
    real.write_text("a: {}\n")
    real.chmod(0o600)
    config_tree.remotes_path().symlink_to(real)
    set_remote(config_tree, "a", root="/r")
    assert stat.S_IMODE(real.stat().st_mode) == 0o600
    assert config_tree.remotes_path().is_symlink()


def test_set_remote_gives_a_new_file_the_conventional_mode(config_tree: ConfigRoot) -> None:
    set_remote(config_tree, "a", root="/r")
    assert stat.S_IMODE(config_tree.remotes_path().stat().st_mode) == 0o666 & ~_umask()


def test_set_remote_refuses_while_the_lock_is_held(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fsutil, "_LOCK_TIMEOUT", 0.05)
    with _lock_held_by_another_process(config_tree.remotes_lock_path()), \
            pytest.raises(ConfigError, match="updating 'remotes.yaml'"):
        set_remote(config_tree, "a", root="/r")
    assert not config_tree.remotes_path().exists()


def test_a_rewrite_during_the_update_is_not_overwritten(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fsutil, "_LOCK_AVAILABLE", False)
    path = config_tree.remotes_path()
    path.write_text("a: {}\n")
    _racing(monkeypatch, lambda: path.write_text("b: {}\n"))
    with pytest.raises(ConfigError, match="changed while it was being updated") as caught:
        set_remote(config_tree, "a", root="/r")
    assert caught.value.hint == "Run the command again."
    assert path.read_text() == "b: {}\n"


def test_a_file_created_during_the_update_is_not_overwritten(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = config_tree.remotes_path()
    _racing(monkeypatch, lambda: path.write_text("b: {}\n"))
    with pytest.raises(ConfigError, match="changed while it was being updated"):
        set_remote(config_tree, "a", root="/r")
    assert path.read_text() == "b: {}\n"


def test_a_link_retargeted_during_the_update_is_refused(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Identical text in both files, so only the resolution check can see it.
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("a: {}\n")
    second.write_text("a: {}\n")
    path = config_tree.remotes_path()
    path.symlink_to(first)

    def retarget() -> None:
        path.unlink()
        path.symlink_to(second)

    _racing(monkeypatch, retarget)
    with pytest.raises(ConfigError, match="changed while it was being updated"):
        set_remote(config_tree, "a", root="/r")
    assert os.readlink(path) == str(second)
    assert first.read_text() == "a: {}\n"
    assert second.read_text() == "a: {}\n"


def test_a_root_link_retargeted_during_the_update_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The config root is the link and remotes.yaml is a regular file in both
    # trees, with identical text, so only resolving the whole path can see it.
    first = tmp_path / "first"
    second = tmp_path / "second"
    for tree in (first, second):
        tree.mkdir()
        (tree / "remotes.yaml").write_text("a: {}\n")
    link = tmp_path / "root"
    link.symlink_to(first)

    def retarget() -> None:
        link.unlink()
        link.symlink_to(second)

    _racing(monkeypatch, retarget)
    with pytest.raises(ConfigError, match="changed while it was being updated"):
        set_remote(ConfigRoot(link), "a", root="/r")
    assert (first / "remotes.yaml").read_text() == "a: {}\n"
    assert (second / "remotes.yaml").read_text() == "a: {}\n"


def test_a_root_link_retargeted_before_the_first_write_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Neither tree has the file, so both reads see the same absence.
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    link = tmp_path / "root"
    link.symlink_to(first)

    def retarget() -> None:
        link.unlink()
        link.symlink_to(second)

    _racing(monkeypatch, retarget)
    with pytest.raises(ConfigError, match="changed while it was being updated"):
        set_remote(ConfigRoot(link), "a", root="/r")
    assert not (first / "remotes.yaml").exists()
    assert not (second / "remotes.yaml").exists()
