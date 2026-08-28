from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.editor import EditorCommand, editor_argv, resolve_editor
from uv_stack.errors import ConfigError


@pytest.fixture(autouse=True)
def _no_ambient_editor(monkeypatch):
    """The machine's own editor settings must not decide these tests."""
    for var in ("UV_STACK_EDITOR", "VISUAL", "EDITOR"):
        monkeypatch.delenv(var, raising=False)


def test_flag_wins_over_everything(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("UV_STACK_EDITOR", "from-uv-stack")
    monkeypatch.setenv("VISUAL", "from-visual")
    monkeypatch.setenv("EDITOR", "from-editor")
    (tmp_path / "editor.txt").write_text("from-file\n", encoding="utf-8")
    resolved = resolve_editor(ConfigRoot(tmp_path), "from-flag")
    assert resolved.command == "from-flag"
    assert resolved.source == "--editor"
    assert resolved.from_flag is True


def test_uv_stack_editor_wins_over_file_and_generic_vars(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("UV_STACK_EDITOR", "from-uv-stack")
    monkeypatch.setenv("VISUAL", "from-visual")
    (tmp_path / "editor.txt").write_text("from-file\n", encoding="utf-8")
    resolved = resolve_editor(ConfigRoot(tmp_path), None)
    assert resolved.command == "from-uv-stack"
    assert resolved.source == "$UV_STACK_EDITOR"
    assert resolved.from_flag is False


def test_editor_txt_wins_over_visual_and_editor(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VISUAL", "from-visual")
    monkeypatch.setenv("EDITOR", "from-editor")
    (tmp_path / "editor.txt").write_text("from-file\n", encoding="utf-8")
    resolved = resolve_editor(ConfigRoot(tmp_path), None)
    assert resolved.command == "from-file"
    assert resolved.source == str(tmp_path / "editor.txt")


def test_visual_wins_over_editor(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VISUAL", "from-visual")
    monkeypatch.setenv("EDITOR", "from-editor")
    resolved = resolve_editor(ConfigRoot(tmp_path), None)
    assert resolved.command == "from-visual"
    assert resolved.source == "$VISUAL"


def test_editor_is_the_last_rung(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EDITOR", "from-editor")
    resolved = resolve_editor(ConfigRoot(tmp_path), None)
    assert resolved.command == "from-editor"
    assert resolved.source == "$EDITOR"


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
@pytest.mark.parametrize("rung", ["UV_STACK_EDITOR", "editor.txt", "VISUAL"])
def test_each_blank_rung_is_skipped(tmp_path: Path, monkeypatch, rung, blank):
    """An exported-but-empty setting must not shadow the rung below it.

    ``UV_STACK_EDITOR=`` in a shell profile is the common shape; a blank
    ``editor.txt`` is what ``touch`` leaves behind. Every rung above the last
    is checked, because the skip is a per-rung decision and a chain that only
    skips one of them would be a silent dead end for the others.
    """
    if rung == "editor.txt":
        (tmp_path / "editor.txt").write_text(blank, encoding="utf-8")
    else:
        monkeypatch.setenv(rung, blank)
    monkeypatch.setenv("EDITOR", "from-editor")
    assert resolve_editor(ConfigRoot(tmp_path), None).command == "from-editor"


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_last_rung_exhausts_the_chain(tmp_path: Path, monkeypatch, blank):
    monkeypatch.setenv("EDITOR", blank)
    with pytest.raises(ConfigError) as excinfo:
        resolve_editor(ConfigRoot(tmp_path), None)
    assert excinfo.value.message == "No editor configured."


def test_editor_txt_skips_a_leading_comment_line(tmp_path: Path):
    (tmp_path / "editor.txt").write_text(
        "# the editor stack edit should use\nvim\n", encoding="utf-8"
    )
    assert resolve_editor(ConfigRoot(tmp_path), None).command == "vim"


def test_editor_txt_is_truncated_at_a_trailing_comment(tmp_path: Path):
    """The documented consequence of reading this file with first_clean_line.

    ``first_clean_line`` strips ``#`` comments, so an editor command containing
    one loses everything from the ``#`` onward. The spec accepts this rather
    than forking the parser for one file; this test pins the behaviour so the
    README note and the code cannot drift apart.
    """
    (tmp_path / "editor.txt").write_text("vim  # my editor\n", encoding="utf-8")
    assert resolve_editor(ConfigRoot(tmp_path), None).command == "vim"


def test_invalid_utf8_in_editor_txt_is_a_config_error(tmp_path: Path):
    """Undecodable editor.txt must fail resolution, not print a traceback.

    ``UnicodeDecodeError`` is a ``ValueError`` — neither ``UvStackError`` nor
    ``OSError`` — so it would escape ``UvStackGroup.invoke`` uncaught. This
    happens during resolution, before anything is launched.
    """
    (tmp_path / "editor.txt").write_bytes(b"\xff\xfe not utf-8\n")
    with pytest.raises(ConfigError) as excinfo:
        resolve_editor(ConfigRoot(tmp_path), None)
    assert "editor.txt" in excinfo.value.message


def test_no_editor_anywhere_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError) as excinfo:
        resolve_editor(ConfigRoot(tmp_path), None)
    assert excinfo.value.message == "No editor configured."
    assert "--editor" in (excinfo.value.hint or "")
    assert "$UV_STACK_EDITOR" in (excinfo.value.hint or "")
    assert "editor.txt" in (excinfo.value.hint or "")
    assert "$VISUAL" in (excinfo.value.hint or "")
    assert "$EDITOR" in (excinfo.value.hint or "")


def test_argv_splits_a_command_with_flags(tmp_path: Path):
    editor = EditorCommand("code -w --new-window", "$EDITOR", from_flag=False)
    assert editor_argv(editor, tmp_path / "stack.txt") == [
        "code",
        "-w",
        "--new-window",
        str(tmp_path / "stack.txt"),
    ]


def test_argv_keeps_an_existing_path_with_spaces_intact(tmp_path: Path):
    exe = tmp_path / "My Editor.app"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    editor = EditorCommand(str(exe), "$EDITOR", from_flag=False)
    # shlex would shred this into two arguments; an existing file is verbatim.
    assert editor_argv(editor, tmp_path / "t.txt") == [str(exe), str(tmp_path / "t.txt")]


@pytest.mark.parametrize(
    "rung", ["UV_STACK_EDITOR", "editor.txt", "VISUAL", "EDITOR"]
)
def test_unbalanced_quoting_names_each_stored_source(tmp_path: Path, monkeypatch, rung):
    """Every stored rung must name itself in the parse failure.

    The user has to know *which* of four places holds the broken value.
    Resolution runs end to end here rather than constructing an
    ``EditorCommand`` by hand, so a rung that reports the wrong ``source``
    fails this test.
    """
    bad = 'code "-w'
    if rung == "editor.txt":
        (tmp_path / "editor.txt").write_text(f"{bad}\n", encoding="utf-8")
        expected = str(tmp_path / "editor.txt")
    else:
        monkeypatch.setenv(rung, bad)
        expected = f"${rung}"
    resolved = resolve_editor(ConfigRoot(tmp_path), None)
    assert resolved.from_flag is False
    with pytest.raises(ConfigError) as excinfo:
        editor_argv(resolved, tmp_path / "t.txt")
    assert f"Cannot parse the editor command from {expected}" in excinfo.value.message


def test_unbalanced_quoting_from_the_flag_is_reported_against_the_flag(tmp_path: Path):
    # The CLI turns this into a UsageError; the module only owes the source.
    resolved = resolve_editor(ConfigRoot(tmp_path), 'code "-w')
    assert resolved.from_flag is True
    with pytest.raises(ConfigError) as excinfo:
        editor_argv(resolved, tmp_path / "t.txt")
    assert "Cannot parse the editor command from --editor" in excinfo.value.message


def test_argv_reports_a_command_that_splits_to_nothing(tmp_path: Path):
    editor = EditorCommand("''", "$EDITOR", from_flag=False)
    with pytest.raises(ConfigError) as excinfo:
        editor_argv(editor, tmp_path / "t.txt")
    assert excinfo.value.message == "No editor configured."


def test_interior_empty_argument_is_preserved(tmp_path: Path):
    """An empty argument can be a load-bearing $0 sentinel in a wrapper command.

    ``sh -c 'exec vim "$@"' ''`` passes an empty ``$0`` to the inner command so
    ``$@`` starts at the first real argument. Dropping it shifts the appended
    target into ``$0``, so the editor opens no file.
    """
    editor = EditorCommand('vim "" f', "$EDITOR", from_flag=False)
    assert editor_argv(editor, tmp_path / "t.txt") == [
        "vim",
        "",
        "f",
        str(tmp_path / "t.txt"),
    ]


def test_leading_empty_executable_is_rejected(tmp_path: Path):
    """Empty executable must fail, not silently promote the second argument.

    ``"" vim`` should report no editor rather than resolving to executable
    ``vim`` — the latter would hide a misconfiguration that should surface.
    """
    editor = EditorCommand('"" vim', "$EDITOR", from_flag=False)
    with pytest.raises(ConfigError) as excinfo:
        editor_argv(editor, tmp_path / "t.txt")
    assert excinfo.value.message == "No editor configured."
