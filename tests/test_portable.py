from __future__ import annotations

import os
import shlex
import shutil
import subprocess
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
    # GOLDEN_PATTERNS is the readable assertion; this one is the
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
    # Nothing at <root>/.git and nothing above it either: the walk has to
    # reach the filesystem root and stop there, which it does because pytest's
    # temp directory is not itself inside a working tree.
    result = write_portable_ignore(config_tree, dry_run=True)
    assert result.repository_root is None
    assert result.is_repository is False
    steps = next_steps(config_tree, result)
    assert any(step.endswith(" init") for step in steps)
    assert not any("rm -r --cached" in step for step in steps)
    assert sum("push" in step for step in steps) == 1
    # Bootstrapping stages the whole root on purpose, and only this branch may:
    # the repository it stages into is the one 'git init' just created one line
    # above, so there is nothing else in it to sweep up and no remote to push
    # a surprise to. The repository branch narrows its own add to the ignore
    # file; this assertion is what keeps that narrowing from spreading here and
    # leaving a fresh root with an initial commit holding one file.
    add = next(step for step in steps if " add " in step and "remote" not in step)
    assert shlex.split(add) == ["git", "-C", str(config_tree.root), "add", "."]


def test_next_steps_for_an_existing_repository_untrack_first(config_tree: ConfigRoot):
    (config_tree.root / ".git").mkdir()
    result = write_portable_ignore(config_tree, dry_run=True)
    assert result.is_repository is True
    steps = next_steps(config_tree, result)
    untrack = next(i for i, s in enumerate(steps) if "rm -r --cached" in s)
    add = next(i for i, s in enumerate(steps) if s.endswith(" add -f .gitignore"))
    assert untrack < add
    assert not any(" init" in step for step in steps)
    assert sum("push" in step for step in steps) == 1
    # The pathspec is compared whole and in order: a truncated one still
    # names a pattern or two while leaving every environment artifact tracked,
    # which is precisely the state this step exists to get the root out of.
    tokens = shlex.split(steps[untrack])
    assert tokens[tokens.index("--") + 1 :] == GOLDEN_PATTERNS


def test_next_steps_for_a_root_nested_in_a_repository_address_the_root(
    config_tree: ConfigRoot,
):
    # The fixture's root is tmp_path/'python-envs', so a .git beside it makes
    # the root a subdirectory of a working tree rather than the working tree —
    # a dotfiles repo with python-envs/ inside it, and the layout a lone
    # <root>/.git probe read as no repository at all, printing 'git init'
    # inside an existing repository and never offering to untrack anything.
    top = config_tree.root.parent
    (top / ".git").mkdir()
    result = write_portable_ignore(config_tree, dry_run=True)
    assert result.repository_root == top
    assert result.is_repository is True
    steps = next_steps(config_tree, result)
    assert not any(" init" in step for step in steps)
    untrack = next(step for step in steps if "rm -r --cached" in step)
    tokens = shlex.split(untrack)
    # Addressed to the root, not to the top level the walk found: git
    # discovers the repository from the directory -C names, so this still
    # reaches the dotfiles repo, and every pathspec then resolves against the
    # root — which is what the block's patterns are already relative to. The
    # list is compared whole because a truncated one still names a pattern or
    # two while leaving every environment artifact tracked.
    assert tokens[:3] == ["git", "-C", str(config_tree.root)]
    assert tokens[tokens.index("--") + 1 :] == GOLDEN_PATTERNS


def test_the_nested_add_step_stages_only_the_ignore_file(config_tree: ConfigRoot):
    # '.' is bounded by the -C directory, so it cannot reach the enclosing
    # dotfiles repository — but the root itself is where this machine's
    # private values live, and 'add .' stages any untracked file there that no
    # pattern happens to cover, and the sequence ends in a push. The
    # untracking step has already staged its deletions, so naming the ignore
    # file stages everything the sequence's purpose requires and nothing else.
    (config_tree.root.parent / ".git").mkdir()
    result = write_portable_ignore(config_tree, dry_run=True)
    add = next(step for step in next_steps(config_tree, result) if " add " in step)
    assert shlex.split(add) == ["git", "-C", str(config_tree.root), "add", "-f", ".gitignore"]


# The user's own git configuration must not decide what the executed test
# below observes: a global core.excludesFile, a commit template or a hooks
# path all change what 'add' stages or what 'status' reports. The identity
# variables are what let its setup commit run against no configured user.
_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "uv-stack tests",
    "GIT_AUTHOR_EMAIL": "tests@example.invalid",
    "GIT_COMMITTER_NAME": "uv-stack tests",
    "GIT_COMMITTER_EMAIL": "tests@example.invalid",
}


def _git(*args: str) -> str:
    """Run one git command to completion, failing the test on a non-zero exit.

    :param args: The command's arguments, without the leading ``git``.
    :returns: Its standard output.
    """
    done = subprocess.run(["git", *args], capture_output=True, text=True, env=_GIT_ENV)
    assert done.returncode == 0, f"git {' '.join(args)}\n{done.stdout}{done.stderr}"
    return done.stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="requires a git executable")
def test_the_ignore_file_is_staged_even_when_the_repository_ignores_it(tmp_path: Path):
    """The add is forced, because the enclosing repository may ignore this path.

    A parent .gitignore holding ``python-envs/`` is enough for git to refuse
    an unforced ``add`` of the file this command just wrote. The refusal does
    not stop the sequence: the commit and push two lines below still run, and
    publish the untracking without the rules that justify it, so the next
    clone tracks the generated files again.
    """
    top = tmp_path / "dotfiles"
    root = top / "python-envs"
    root.mkdir(parents=True)
    _git("init", "-q", str(top))
    (top / ".gitignore").write_text("python-envs/\n")
    (root / "variables.local.txt").write_text("DEV=/srv/src\n")
    _git("-C", str(top), "add", "-f", ".gitignore", "python-envs/variables.local.txt")
    _git("-C", str(top), "commit", "-q", "-m", "initial")

    config = ConfigRoot(root)
    steps = next_steps(config, write_portable_ignore(config))
    _git(*shlex.split(next(s for s in steps if "rm -r --cached" in s))[1:])
    _git(*shlex.split(next(s for s in steps if " add " in s))[1:])

    staged = {
        line[3:]
        for line in _git("-C", str(top), "status", "--porcelain").splitlines()
        if line[0] not in " ?"
    }
    assert staged == {"python-envs/.gitignore", "python-envs/variables.local.txt"}


@pytest.mark.skipif(shutil.which("git") is None, reason="requires a git executable")
def test_the_printed_commands_stage_nothing_the_user_did_not_ask_for(tmp_path: Path):
    """Run the printed sequence against a real repository and read the index.

    Comparing command strings pins their spelling; only git can say what the
    user's repository is left holding, which is what was wrong. Under
    ``add .`` a file sitting untracked in the config root — the one place this
    machine's private values live — was staged, committed under a message
    about generated files, and pushed by the last line of the same sequence.
    """
    top = tmp_path / "dotfiles"
    root = top / "python-envs"
    root.mkdir(parents=True)
    _git("init", "-q", str(top))
    (top / "TRACKED.md").write_text("dotfiles\n")
    (root / "variables.txt").write_text("DEV\n")
    (root / "variables.local.txt").write_text("DEV=/srv/src\n")
    _git("-C", str(top), "add", "-A")
    _git("-C", str(top), "commit", "-q", "-m", "initial")

    # The two things the sequence must not pick up: a file in the root that no
    # pattern covers and that was never tracked, and an edit outside the root
    # the user had already staged for a commit of their own.
    (root / ".env.secret").write_text("TOKEN=hunter2\n")
    (top / "TRACKED.md").write_text("dotfiles, edited\n")
    _git("-C", str(top), "add", "TRACKED.md")

    config = ConfigRoot(root)
    steps = next_steps(config, write_portable_ignore(config))
    # The sequence is run as far as the add. Its commit would take the staged
    # edit outside the root — the residual its own note line warns about, and
    # not something this step can narrow away — and its push has no remote.
    _git(*shlex.split(next(s for s in steps if "rm -r --cached" in s))[1:])
    _git(*shlex.split(next(s for s in steps if " add " in s))[1:])

    status = set(_git("-C", str(top), "status", "--porcelain").splitlines())
    # A staged entry is one whose index column is neither blank nor '?'.
    staged = {line[3:] for line in status if line[0] not in " ?"}
    assert "?? python-envs/.env.secret" in status
    assert staged == {
        "python-envs/.gitignore",
        "python-envs/variables.local.txt",
        # Staged before any of this ran and left exactly as it was found. The
        # commit will still take it, which is why the sequence says so.
        "TRACKED.md",
    }
    # The untracking step reaches the generated files and stops there: a
    # declared file that must keep travelling with the root stays in the index.
    assert "python-envs/variables.txt" in _git("-C", str(top), "ls-files").splitlines()


def test_the_walk_climbs_past_more_than_one_level(tmp_path: Path):
    # One level up is the easy case to get right by accident. Nothing in the
    # printed commands depends on the distance any more, but doctor's decision
    # to judge the ignore block at all still rides on this walk, and so does
    # the choice between 'git init' and untracking.
    root = tmp_path / "repo" / "config" / "python-envs"
    root.mkdir(parents=True)
    (tmp_path / "repo" / ".git").mkdir()
    config = ConfigRoot(root)
    result = write_portable_ignore(config, dry_run=True)
    assert result.repository_root == tmp_path / "repo"
    steps = next_steps(config, result)
    assert not any(" init" in step for step in steps)
    untrack = next(s for s in steps if "rm -r --cached" in s)
    tokens = shlex.split(untrack)
    assert tokens[:3] == ["git", "-C", str(root)]
    assert tokens[tokens.index("--") + 1 :] == GOLDEN_PATTERNS


# Directory names git would read as pathspec magic, an option, or a glob if
# any of them ever reached a pathspec position. All are creatable on APFS and
# ext4, and '--root' admits every one of them.
ADVERSARIAL_ROOT_NAMES = [
    ":(exclude)python-envs",
    ":!python-envs",
    ":python-envs",
    "-envs",
    "*",
    "?envs",
]


@pytest.mark.parametrize("name", ADVERSARIAL_ROOT_NAMES)
def test_no_printed_pathspec_carries_the_roots_own_name(tmp_path: Path, name: str):
    # The defect this pins: addressing the enclosing repository means
    # prefixing every pattern with the root's path relative to it, and git
    # reads ':(exclude)' at the head of a pathspec as magic. Every generated
    # pathspec became an exclusion, so 'rm -r --cached' matched everything
    # else instead — it untracked the entire repository, including the two
    # files that must stay tracked and files outside the root altogether.
    # shlex.quote cannot help, because git receives the argument intact and
    # git is what interprets it.
    top = tmp_path / "dotfiles"
    (top / ".git").mkdir(parents=True)
    root = top / name
    root.mkdir()
    config = ConfigRoot(root)

    steps = next_steps(config, write_portable_ignore(config, dry_run=True))

    untrack = next(s for s in steps if "rm -r --cached" in s)
    tokens = shlex.split(untrack)
    # The root's name appears once, as -C's operand, which git takes as a
    # directory to chdir to and never parses as a pathspec or an option.
    assert tokens[:3] == ["git", "-C", str(root)]
    assert tokens[tokens.index("--") + 1 :] == GOLDEN_PATTERNS
    add = next(s for s in steps if " add " in s)
    assert shlex.split(add) == ["git", "-C", str(root), "add", "-f", ".gitignore"]


def test_a_git_file_counts_as_a_repository(config_tree: ConfigRoot):
    # A worktree or submodule spells .git as a file holding 'gitdir: <path>'.
    # Its files are tracked exactly as a .git directory's are, so the probe
    # asks whether the entry exists and never whether it is a directory.
    (config_tree.root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    result = write_portable_ignore(config_tree, dry_run=True)
    assert result.repository_root == config_tree.root
    assert any("rm -r --cached" in step for step in next_steps(config_tree, result))


def test_a_dot_dot_in_the_root_does_not_hand_it_to_a_repository_it_passes_through(
    tmp_path: Path,
):
    # Path.parent is lexical and does not collapse '..', so an unnormalised
    # walk steps through 'repo/sub/..', finds repo's .git, and names a top
    # level the root is not under. The printed commands then carry an escaping
    # '../' pathspec that git refuses outright ("is outside repository"), and
    # doctor judges the ignore block of a root that is in no repository.
    (tmp_path / "repo" / "sub").mkdir(parents=True)
    (tmp_path / "repo" / ".git").mkdir()
    (tmp_path / "outside" / "python-envs").mkdir(parents=True)

    config = ConfigRoot(tmp_path / "repo" / "sub" / ".." / ".." / "outside" / "python-envs")
    result = write_portable_ignore(config, dry_run=True)

    assert result.repository_root is None
    assert any(step.endswith(" init") for step in next_steps(config, result))


def test_a_relative_root_is_anchored_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # '--root' is documented with no constraint and ConfigRoot does not
    # absolutise, so a relative root is a supported spelling. Walking it
    # unnormalised bottoms out at '.' -- whose parts are empty, which
    # relative_to then accepts as a prefix of anything -- and the root inherits
    # the working directory's repository rather than its own.
    (tmp_path / "dotfiles" / ".git").mkdir(parents=True)
    (tmp_path / "python-envs").mkdir()
    monkeypatch.chdir(tmp_path / "dotfiles")

    result = write_portable_ignore(ConfigRoot(Path("../python-envs")), dry_run=True)

    assert result.repository_root is None


def test_a_relative_root_inside_a_repository_reports_an_absolute_top_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The other half: normalising must not cost a relative root the repository
    # it really is in, or the untracking advice is replaced by 'git init'
    # inside a working tree. The walk reports the top level absolute because
    # it normalised before climbing; the printed commands name the root as the
    # user spelled it, which is the directory that spelling resolves against
    # in the shell they just ran uv-stack in.
    (tmp_path / ".git").mkdir()
    (tmp_path / "python-envs").mkdir()
    monkeypatch.chdir(tmp_path)

    config = ConfigRoot(Path("python-envs"))
    result = write_portable_ignore(config, dry_run=True)

    assert result.repository_root == tmp_path
    untrack = next(s for s in next_steps(config, result) if "rm -r --cached" in s)
    tokens = shlex.split(untrack)
    assert tokens[:3] == ["git", "-C", "python-envs"]
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
