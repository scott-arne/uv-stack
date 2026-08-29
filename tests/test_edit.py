from __future__ import annotations

from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.editor import EditorCommand, editor_argv, resolve_editor
from uv_stack.errors import ConfigError, NewerSchemaError, ResolutionError
from uv_stack.operations.edit import missing_project_error, validate


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


def test_argv_keeps_a_relative_path_with_a_separator_intact(tmp_path: Path, monkeypatch):
    """A separator makes the string path-like, so the space is not a split."""
    exe = tmp_path / "my editor"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    editor = EditorCommand("./my editor", "$EDITOR", from_flag=False)
    assert editor_argv(editor, tmp_path / "t.txt") == [
        "./my editor",
        str(tmp_path / "t.txt"),
    ]


def test_argv_ignores_a_cwd_file_named_like_the_whole_command(
    tmp_path: Path, monkeypatch
):
    """An unrelated cwd file must not decide how the command is spelled.

    ``Path("code -w").is_file()`` resolves against the current directory, so
    without the path-like test a file that happens to be named ``code -w``
    would swallow the flag and hand ``execvp`` a program that does not exist.
    """
    (tmp_path / "code -w").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    editor = EditorCommand("code -w", "$EDITOR", from_flag=False)
    assert editor_argv(editor, tmp_path / "t.txt") == [
        "code",
        "-w",
        str(tmp_path / "t.txt"),
    ]


def test_argv_expands_a_tilde_path_with_a_space(tmp_path: Path, monkeypatch):
    """A ``~`` path with a space is one argument *and* is expanded.

    Both halves matter. ``shlex`` would shred the path into arguments that
    name nothing, and a surviving literal ``~`` would too: the runner hands
    argv to ``execvp``, which has no shell behind it to expand the prefix.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = tmp_path / "My Editor" / "code"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    editor = EditorCommand("~/My Editor/code", "$EDITOR", from_flag=False)
    assert editor_argv(editor, tmp_path / "t.txt") == [
        str(exe),
        str(tmp_path / "t.txt"),
    ]


def test_argv_expands_a_tilde_path_without_a_space(tmp_path: Path, monkeypatch):
    """Expansion is not a spaces-only concern; ``execvp`` needs it either way."""
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = tmp_path / "bin" / "ed"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    editor = EditorCommand("~/bin/ed", "$EDITOR", from_flag=False)
    assert editor_argv(editor, tmp_path / "t.txt") == [
        str(exe),
        str(tmp_path / "t.txt"),
    ]


def test_argv_splits_a_tilde_path_that_names_nothing(tmp_path: Path, monkeypatch):
    """A ``~`` path naming no file gets no error of its own.

    It falls through to the split exactly as a nonexistent absolute path
    does, so the launch fails in the runner and reports the command that was
    actually tried.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    editor = EditorCommand("~/nope", "$EDITOR", from_flag=False)
    assert editor_argv(editor, tmp_path / "t.txt") == [
        "~/nope",
        str(tmp_path / "t.txt"),
    ]


def test_argv_survives_a_home_directory_that_cannot_be_determined(
    tmp_path: Path, monkeypatch
):
    """``Path.expanduser`` raises ``RuntimeError`` with no home to expand to.

    ``UvStackGroup.invoke`` handles ``UvStackError`` and ``OSError`` only, so
    that would escape as a traceback. Treating it as "not a usable path"
    degrades to the split — today's behaviour — instead of crashing.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    exe = tmp_path / "My Editor" / "code"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n", encoding="utf-8")

    def _no_home(self: Path) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "expanduser", _no_home)
    editor = EditorCommand("~/My Editor/code", "$EDITOR", from_flag=False)
    assert editor_argv(editor, tmp_path / "t.txt") == [
        "~/My",
        "Editor/code",
        str(tmp_path / "t.txt"),
    ]


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


def test_validate_profile_accepts_a_good_profile(config_tree: ConfigRoot):
    assert validate(config_tree, "profile", "ds", config_tree.root) == []


def test_validate_profile_rejects_a_broken_schema(config_tree: ConfigRoot):
    config_tree.profile_path("ds").write_text("includes: not-a-list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        validate(config_tree, "profile", "ds", config_tree.root)


def test_validate_profile_rejects_undecodable_bytes(config_tree: ConfigRoot):
    config_tree.profile_path("ds").write_bytes(b"includes:\n  - \xff\xfe\n")
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "profile", "ds", config_tree.root)
    assert "not valid UTF-8" in excinfo.value.message


def test_validate_bundle_rejects_a_self_reference(config_tree: ConfigRoot):
    # At edit time the bundle exists, so the resolver's recursion guard makes a
    # self-reference expand silently to nothing. Only an explicit check sees it.
    config_tree.bundle_path("standard").write_text(
        "includes:\n  - standard\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "bundle", "standard", config_tree.root)
    assert "cannot include itself" in excinfo.value.message


def test_validate_bundle_rejects_a_missing_include(config_tree: ConfigRoot):
    config_tree.bundle_path("standard").write_text(
        "includes:\n  - profile:ghost\n", encoding="utf-8"
    )
    with pytest.raises(ResolutionError):
        validate(config_tree, "bundle", "standard", config_tree.root)


def test_validate_env_accepts_the_seeded_env(config_tree: ConfigRoot):
    assert validate(config_tree, "env", "main", config_tree.root) == []


def test_validate_env_rejects_a_missing_profile(config_tree: ConfigRoot):
    config_tree.env_stack_path("main").write_text("profile:ghost\n", encoding="utf-8")
    with pytest.raises(ResolutionError):
        validate(config_tree, "env", "main", config_tree.root)


def test_validate_env_rejects_an_undecodable_local_requirements_file(
    config_tree: ConfigRoot,
):
    # render_requirements_in only emits '-r <path>' for this file and never
    # opens it, so nothing downstream would notice the bad bytes.
    config_tree.env_local_path("main").write_bytes(b"-e \xff\xfe/pkg\n")
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "env", "main", config_tree.root)
    assert "not valid UTF-8" in excinfo.value.message


def test_validate_env_names_the_root_when_an_included_profile_is_undecodable(
    config_tree: ConfigRoot,
):
    # The seeded env 'main' has stack '@standard', and bundle 'standard'
    # includes profile 'ds'. Corrupting ds.yaml should report the root, not the
    # env directory — the resolver reads profiles under <root>/profiles/.
    config_tree.profile_path("ds").write_bytes(b"includes:\n  - \xff\xfe\n")
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "env", "main", config_tree.root)
    assert "not valid UTF-8" in excinfo.value.message
    assert str(config_tree.env_dir("main")) not in excinfo.value.message


def test_validate_project_warns_when_untracked(config_tree: ConfigRoot, tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    warnings = validate(config_tree, "project", "", project)
    assert len(warnings) == 1
    assert "[tool.uv-stack]" in warnings[0]


def test_validate_project_reports_a_deleted_pyproject(
    config_tree: ConfigRoot, tmp_path: Path
):
    # An editor that deleted the file must not read back as "untracked".
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "project", "", tmp_path)
    assert "No pyproject.toml" in excinfo.value.message


def test_validate_project_rejects_undecodable_bytes(
    config_tree: ConfigRoot, tmp_path: Path
):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_bytes(b'[project]\nname = "\xff\xfe"\n')
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "project", "", project)
    assert "not valid UTF-8" in excinfo.value.message


def test_validate_project_resolves_tracked_stack_tokens(
    config_tree: ConfigRoot, tmp_path: Path
):
    # `stack refresh` resolves and flattens this same list, so a shape-only
    # check would accept a file the very next command rejects.
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n'
        "[tool.uv-stack]\nversion = 1\nstack = [\"profile:ghost\"]\n",
        encoding="utf-8",
    )
    with pytest.raises(ResolutionError):
        validate(config_tree, "project", "", project)


def test_validate_project_propagates_a_newer_schema(
    config_tree: ConfigRoot, tmp_path: Path
):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n'
        "[tool.uv-stack]\nversion = 2\nstack = []\n",
        encoding="utf-8",
    )
    with pytest.raises(NewerSchemaError):
        validate(config_tree, "project", "", project)


def test_validate_project_reuses_the_pre_launch_missing_hint(
    config_tree: ConfigRoot, tmp_path: Path
):
    """A deleted pyproject must give the same way forward as an absent one.

    The pre-launch check in cli/edit.py and this post-edit re-check are the
    same condition seen at two moments, so the spec requires one hint. Both
    call missing_project_error; this asserts the validator's error is
    indistinguishable from it.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    shared = missing_project_error(empty)
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "project", "", empty)
    assert excinfo.value.message == shared.message
    assert excinfo.value.hint == shared.hint
    assert "stack create project" in (shared.hint or "")


def test_validate_project_refuses_a_malformed_dependencies_value(
    config_tree: ConfigRoot, tmp_path: Path
):
    # refresh reads [project.dependencies] for ownership before it mutates
    # anything; a non-list crashes it with a bare TypeError, so accepting the
    # file here would send the user back out to a traceback.
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = 1\n\n'
        '[tool.uv-stack]\nversion = 1\nstack = ["ds"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "project", "", project)
    assert "must be an array" in excinfo.value.message


def test_validate_project_runs_the_refresh_write_preflight(
    config_tree: ConfigRoot, tmp_path: Path
):
    # read_tracking reads with universal newlines and accepts lone carriage
    # returns; refresh's pre-mutation preflight reads with newline="" and
    # refuses them. Without the preflight here the loop reports success on a
    # file the next refresh will not touch.
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_bytes(
        b'[project]\rname = "x"\rversion = "0.1.0"\rdependencies = []\r\r'
        b'[tool.uv-stack]\rversion = 1\rstack = ["ds"]\r'
    )
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "project", "", project)
    assert "Refusing to modify" in excinfo.value.message


def test_validate_project_refuses_a_non_table_project_value(
    config_tree: ConfigRoot, tmp_path: Path
):
    # The ownership read reaches [project] through a nested .get; a scalar
    # there raised AttributeError, which is not a UvStackError and so left the
    # re-offer loop with a traceback instead of something to show the user.
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        'project = 1\n\n[tool.uv-stack]\nversion = 1\nstack = ["ds"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        validate(config_tree, "project", "", project)
    assert "must be a table" in excinfo.value.message
