from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.operations.portable import (
    BEGIN_MARKER,
    END_MARKER,
    _newline,
    ignore_patterns,
    next_steps,
    render_block,
    write_portable_ignore,
)

# The managed block's patterns, spelled out. Every other expectation in this
# module is built by calling ignore_patterns() or render_block(), so the suite
# agrees with whatever the renderer currently emits; only a literal notices a
# pattern quietly dropping out of the block and its file getting committed.
GOLDEN_PATTERNS = [
    ".locks/",
    "editor.txt",
    "variables.local.txt",
    "envs/*/requirements.in",
    "envs/*/environment.yml",
    "envs/*/requirements.lock.txt",
    "envs/*/requirements.local.in",
    ".DS_Store",
]


def _record_writes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Swap the module's atomic_write for a recorder of the paths it is given.

    A write that must not happen leaves nothing on disk to assert on:
    atomic_write skips an identical rewrite by itself, so the only proof that
    this module's own guard held is that the writer was never reached.
    """
    written: list[Path] = []
    monkeypatch.setattr(
        "uv_stack.operations.portable.atomic_write",
        lambda path, text: written.append(path),
    )
    return written


def test_patterns_are_derived_from_the_path_accessors(config_tree: ConfigRoot):
    assert ignore_patterns(config_tree) == GOLDEN_PATTERNS


def test_every_uv_stack_pattern_comes_from_an_accessor(config_tree: ConfigRoot):
    # The literal list above is the readable assertion; this one is the
    # regression guard. Rename a generated file and the pattern must move with
    # it, so only '.DS_Store' — which names no uv-stack path — may be a literal.
    derived = {
        f"{config_tree.locks_dir.relative_to(config_tree.root).as_posix()}/",
        config_tree.editor_path().name,
        config_tree.variables_local_path().name,
        config_tree.env_requirements_in("*").relative_to(config_tree.root).as_posix(),
        config_tree.env_environment_yml("*").relative_to(config_tree.root).as_posix(),
        config_tree.env_requirements_lock("*").relative_to(config_tree.root).as_posix(),
        config_tree.env_local_path("*").relative_to(config_tree.root).as_posix(),
    }
    assert set(ignore_patterns(config_tree)) - {".DS_Store"} == derived


def test_block_is_delimited_by_the_markers(config_tree: ConfigRoot):
    lines = render_block(config_tree).split("\n")
    assert lines[0] == BEGIN_MARKER
    assert lines[-1] == END_MARKER


def test_render_block_joins_with_the_requested_newline(config_tree: ConfigRoot):
    assert "\r\n" not in render_block(config_tree)
    crlf = render_block(config_tree, "\r\n")
    assert crlf.startswith(BEGIN_MARKER + "\r\n")
    assert crlf.endswith("\r\n" + END_MARKER)
    assert "\n" not in crlf.replace("\r\n", "")


def test_the_rendered_block_matches_its_golden_text(config_tree: ConfigRoot):
    # The marker lines are spelled out rather than imported for the same
    # reason: a renamed marker would move every constant-derived expectation
    # with it and orphan the blocks already written to users' files.
    lines = [
        "# BEGIN uv-stack — managed block, do not edit by hand.",
        *GOLDEN_PATTERNS,
        "# END uv-stack",
    ]
    assert render_block(config_tree) == "\n".join(lines)
    assert render_block(config_tree, "\r\n") == "\r\n".join(lines)


def test_writing_into_a_root_without_a_gitignore_creates_it(config_tree: ConfigRoot):
    result = write_portable_ignore(config_tree)
    assert result.outcome == "created"
    assert result.path == config_tree.root / ".gitignore"
    assert result.path.read_text() == render_block(config_tree) + "\n"


def test_writing_twice_is_unchanged(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
):
    write_portable_ignore(config_tree)
    before = (config_tree.root / ".gitignore").read_text()
    written = _record_writes(monkeypatch)
    result = write_portable_ignore(config_tree)
    assert result.outcome == "unchanged"
    assert written == []
    assert (config_tree.root / ".gitignore").read_text() == before


def test_an_existing_gitignore_without_markers_keeps_its_content(
    config_tree: ConfigRoot,
):
    path = config_tree.root / ".gitignore"
    path.write_text("lib/\n*.swp\n")
    result = write_portable_ignore(config_tree)
    assert result.outcome == "updated"
    text = path.read_text()
    assert text.startswith("lib/\n*.swp\n")
    assert BEGIN_MARKER in text


def test_an_existing_gitignore_without_a_trailing_newline_is_not_glued(
    config_tree: ConfigRoot,
):
    path = config_tree.root / ".gitignore"
    path.write_text("lib/")
    write_portable_ignore(config_tree)
    assert path.read_text().startswith("lib/\n" + BEGIN_MARKER)


def test_a_stale_block_is_replaced_in_place(config_tree: ConfigRoot):
    path = config_tree.root / ".gitignore"
    path.write_text(f"before\n{BEGIN_MARKER}\nstale-entry\n{END_MARKER}\nafter\n")
    result = write_portable_ignore(config_tree)
    assert result.outcome == "updated"
    # The whole file is compared rather than its prefix and its suffix: a
    # splice that duplicated everything after the block would satisfy both.
    expected = f"before\n{render_block(config_tree)}\nafter\n"
    assert path.read_bytes() == expected.encode("utf-8")


@pytest.mark.parametrize(
    "content",
    [
        # BEGIN alone, END alone, END before BEGIN, two BEGINs, two ENDs,
        # nested — one row per topology the spec enumerates.
        f"{BEGIN_MARKER}\na\n",
        f"{END_MARKER}\n",
        f"{END_MARKER}\na\n{BEGIN_MARKER}\n",
        f"{BEGIN_MARKER}\na\n{BEGIN_MARKER}\nb\n{END_MARKER}\n",
        f"{BEGIN_MARKER}\na\n{END_MARKER}\nb\n{END_MARKER}\n",
        f"{BEGIN_MARKER}\na\n{BEGIN_MARKER}\nb\n{END_MARKER}\n{END_MARKER}\n",
    ],
)
def test_a_malformed_topology_is_refused_and_writes_nothing(
    config_tree: ConfigRoot, content, monkeypatch: pytest.MonkeyPatch
):
    path = config_tree.root / ".gitignore"
    path.write_text(content)
    written = _record_writes(monkeypatch)
    with pytest.raises(ConfigError) as excinfo:
        write_portable_ignore(config_tree)
    assert str(path) in excinfo.value.message
    assert path.read_text() == content
    assert written == []


def test_a_malformed_topology_is_refused_under_dry_run_too(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
):
    path = config_tree.root / ".gitignore"
    path.write_text(f"{BEGIN_MARKER}\na\n")
    written = _record_writes(monkeypatch)
    with pytest.raises(ConfigError):
        write_portable_ignore(config_tree, dry_run=True)
    assert path.read_text() == f"{BEGIN_MARKER}\na\n"
    assert written == []


def test_dry_run_writes_nothing(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
):
    written = _record_writes(monkeypatch)
    result = write_portable_ignore(config_tree, dry_run=True)
    assert result.outcome == "created"
    assert not (config_tree.root / ".gitignore").exists()
    assert written == []


def test_next_steps_for_a_root_that_is_not_a_repository(config_tree: ConfigRoot):
    result = write_portable_ignore(config_tree, dry_run=True)
    assert result.is_repository is False
    steps = next_steps(config_tree, result)
    assert any(step.endswith(" init") for step in steps)
    assert not any("rm -r --cached" in step for step in steps)
    assert sum("push" in step for step in steps) == 1


def test_next_steps_for_an_existing_repository_untrack_first(config_tree: ConfigRoot):
    (config_tree.root / ".git").mkdir()
    result = write_portable_ignore(config_tree, dry_run=True)
    assert result.is_repository is True
    steps = next_steps(config_tree, result)
    untrack = next(i for i, s in enumerate(steps) if "rm -r --cached" in s)
    add = next(i for i, s in enumerate(steps) if s.endswith(" add ."))
    assert untrack < add
    assert not any(" init" in step for step in steps)
    assert sum("push" in step for step in steps) == 1
    # The pathspec is compared whole and in order: a truncated one still
    # names a pattern or two while leaving every environment artifact tracked,
    # which is precisely the state this step exists to get the root out of.
    tokens = shlex.split(steps[untrack])
    assert tokens[tokens.index("--") + 1 :] == GOLDEN_PATTERNS


def test_the_repository_branch_does_not_depend_on_the_write_outcome(
    config_tree: ConfigRoot,
):
    (config_tree.root / ".git").mkdir()
    write_portable_ignore(config_tree)
    result = write_portable_ignore(config_tree)
    assert result.outcome == "unchanged"
    assert any("rm -r --cached" in step for step in next_steps(config_tree, result))


def test_a_root_path_with_a_space_is_quoted(tmp_path):
    root = tmp_path / "my configs"
    root.mkdir()
    config = ConfigRoot(root)
    result = write_portable_ignore(config, dry_run=True)
    assert all("'" in step or '"' in step for step in next_steps(config, result))


def test_the_push_step_names_head_not_a_branch(config_tree: ConfigRoot):
    # The bootstrap sequence must work on a machine whose git defaults to
    # 'main' and on one that defaults to 'master', so it never names either.
    result = write_portable_ignore(config_tree, dry_run=True)
    push = next(step for step in next_steps(config_tree, result) if " push" in step)
    assert push.endswith("HEAD")
    assert "main" not in push
    assert "master" not in push


CRLF = "\r\n"


def _write_bytes(config: ConfigRoot, text: str) -> Path:
    """Put exact bytes in the root's .gitignore, bypassing newline translation."""
    path = config.root / ".gitignore"
    path.write_bytes(text.encode("utf-8"))
    return path


def test_a_crlf_file_without_a_block_keeps_crlf_throughout(config_tree: ConfigRoot):
    path = _write_bytes(config_tree, "lib/\r\n*.swp\r\n")
    assert write_portable_ignore(config_tree).outcome == "updated"
    text = path.read_bytes().decode("utf-8")
    assert text.startswith("lib/\r\n*.swp\r\n")
    assert "\n" not in text.replace("\r\n", "")
    assert text.endswith(END_MARKER + "\r\n")


def test_a_crlf_file_with_a_stale_block_keeps_the_bytes_around_it(
    config_tree: ConfigRoot,
):
    path = _write_bytes(
        config_tree,
        f"before\r\n{BEGIN_MARKER}\r\nstale-entry\r\n{END_MARKER}\r\nafter\r\n",
    )
    assert write_portable_ignore(config_tree).outcome == "updated"
    # Bytes, not decoded text: a translated read would hide a terminator the
    # splice failed to restore, and a tail is only pinned when compared whole.
    expected = f"before\r\n{render_block(config_tree, CRLF)}\r\nafter\r\n"
    assert path.read_bytes() == expected.encode("utf-8")


def test_a_crlf_file_already_current_is_byte_identical(config_tree: ConfigRoot):
    path = _write_bytes(
        config_tree, f"before\r\n{render_block(config_tree, CRLF)}\r\n"
    )
    before = path.read_bytes()
    assert write_portable_ignore(config_tree).outcome == "unchanged"
    assert path.read_bytes() == before


def test_a_bare_cr_file_keeps_bare_cr(config_tree: ConfigRoot):
    path = _write_bytes(config_tree, "lib/\r")
    write_portable_ignore(config_tree)
    text = path.read_bytes().decode("utf-8")
    assert "\n" not in text
    assert text.startswith("lib/\r" + BEGIN_MARKER)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain\n", "\n"),
        ("crlf\r\n", "\r\n"),
        ("cronly\r", "\r"),
        ("", "\n"),
        ("no terminator", "\n"),
        # A file whose endings are mixed is decided by its first terminator,
        # so an earlier bare CR wins over the LF that follows it later on.
        ("a\rb\n", "\r"),
        ("\rx\n", "\r"),
        ("a\rb\r\n", "\r"),
        # A CRLF counts as one terminator rather than as a CR plus an LF.
        ("a\r\nb\n", "\r\n"),
        ("a\nb\r", "\n"),
    ],
)
def test_the_first_terminator_decides_the_newline(text, expected):
    assert _newline(text) == expected


def test_a_block_at_the_end_without_a_terminator_stays_unterminated(
    config_tree: ConfigRoot,
):
    path = _write_bytes(
        config_tree, f"before\n{BEGIN_MARKER}\nstale-entry\n{END_MARKER}"
    )
    assert write_portable_ignore(config_tree).outcome == "updated"
    text = path.read_bytes().decode("utf-8")
    assert text.endswith(END_MARKER)
    assert not text.endswith(END_MARKER + "\n")


def test_a_non_utf8_ignore_file_is_refused(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
):
    path = config_tree.root / ".gitignore"
    before_bytes = b"\xff\xfe not utf-8\n"
    path.write_bytes(before_bytes)
    written = _record_writes(monkeypatch)
    with pytest.raises(ConfigError) as excinfo:
        write_portable_ignore(config_tree)
    assert "UTF-8" in excinfo.value.message
    # Unchanged bytes are weaker than the contract: republishing identical
    # content satisfies them while still replacing the inode. A refusal must
    # not reach the writer at all.
    assert path.read_bytes() == before_bytes
    assert written == []


# The eight separators str.splitlines() breaks on that _terminator cannot
# put back. The three it can — LF, CR, and CRLF — are the only ones the
# marker comprehensions in _find_span are allowed to strip.
UNRESTORABLE_SEPARATORS = [
    "\v",  # line tabulation
    "\x0c",  # form feed
    "\x1c",  # file separator
    "\x1d",  # group separator
    "\x1e",  # record separator
    "\x85",  # next line
    "\u2028",  # line separator
    "\u2029",  # paragraph separator
]


@pytest.mark.parametrize("separator", UNRESTORABLE_SEPARATORS)
def test_an_end_marker_ended_by_an_unrestorable_separator_is_refused(
    config_tree: ConfigRoot, separator, monkeypatch: pytest.MonkeyPatch
):
    # splitlines() breaks on eleven separators and _terminator can put back
    # only three, so matching an END marker ended by one of the other eight
    # would delete that separator and glue the next line onto the marker.
    path = _write_bytes(
        config_tree, f"{BEGIN_MARKER}\nstale-entry\n{END_MARKER}{separator}after\n"
    )
    before = path.read_bytes()
    written = _record_writes(monkeypatch)
    with pytest.raises(ConfigError) as excinfo:
        write_portable_ignore(config_tree)
    assert str(path) in excinfo.value.message
    assert "END on line(s) none" in excinfo.value.message
    assert path.read_bytes() == before
    # Identical bytes would also survive a rewrite that republished them, so
    # the refusal is only pinned once the writer is shown to be unreached.
    assert written == []


@pytest.mark.parametrize("separator", UNRESTORABLE_SEPARATORS)
def test_a_begin_marker_ended_by_an_unrestorable_separator_is_refused(
    config_tree: ConfigRoot, separator, monkeypatch: pytest.MonkeyPatch
):
    # The END-side sibling cannot pin this half. The splice replaces the whole
    # span, so whatever follows the BEGIN marker on its line is discarded
    # either way and a BEGIN matched through an exotic separator corrupts
    # nothing. Both comprehensions are held to the same strip set regardless,
    # so that "marker" means one thing in each, and this is what pins it.
    path = _write_bytes(
        config_tree, f"{BEGIN_MARKER}{separator}stale-entry\n{END_MARKER}\n"
    )
    before = path.read_bytes()
    written = _record_writes(monkeypatch)
    with pytest.raises(ConfigError) as excinfo:
        write_portable_ignore(config_tree)
    assert str(path) in excinfo.value.message
    assert "BEGIN on line(s) none" in excinfo.value.message
    assert path.read_bytes() == before
    assert written == []


def test_a_marker_line_with_trailing_blanks_is_still_matched(config_tree: ConfigRoot):
    path = _write_bytes(
        config_tree,
        f"before\n{BEGIN_MARKER} \nstale-entry\n{END_MARKER}\t \nafter\n",
    )
    assert write_portable_ignore(config_tree).outcome == "updated"
    text = path.read_bytes().decode("utf-8")
    assert text.startswith("before\n")
    assert text.endswith("after\n")
    assert "stale-entry" not in text
    assert BEGIN_MARKER in text


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda config: None, id="absent"),
        pytest.param(lambda config: "lib/\n*.swp\n", id="no-markers"),
        pytest.param(
            lambda config: f"keep\n{BEGIN_MARKER}\nstale\n{END_MARKER}\ntail\n",
            id="stale-block",
        ),
        pytest.param(lambda config: render_block(config) + "\n", id="current-lf"),
        pytest.param(
            lambda config: render_block(config, CRLF) + CRLF, id="current-crlf"
        ),
    ],
)
def test_dry_run_writes_nothing_whatever_the_file_already_holds(
    config_tree: ConfigRoot, build, monkeypatch: pytest.MonkeyPatch
):
    # An already-current file is spared by the 'updated == original' test, so
    # it cannot show the dry-run half of the guard working. Only the states a
    # real run would rewrite do that, and a guard keyed on whether the file
    # exists rather than on dry_run passes the absent case regardless.
    path = config_tree.root / ".gitignore"
    content = build(config_tree)
    before = None if content is None else _write_bytes(config_tree, content).read_bytes()
    written = _record_writes(monkeypatch)
    write_portable_ignore(config_tree, dry_run=True)
    assert written == []
    # Bytes, not text: the CRLF row is only compared honestly untranslated.
    if before is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == before
