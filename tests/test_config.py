from __future__ import annotations

import os
from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError


def test_discover_precedence_explicit_over_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("UV_STACK_ROOT", raising=False)
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "from-env"))
    cfg = ConfigRoot.discover(root=tmp_path / "explicit")
    assert cfg.root == (tmp_path / "explicit")


def test_discover_explicit_root_wins_over_both_env_vars(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("UV_STACK_ROOT", str(tmp_path / "stack-root"))
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "env-root"))
    cfg = ConfigRoot.discover(root=tmp_path / "explicit")
    assert cfg.root == (tmp_path / "explicit")


def test_discover_uses_env_when_no_explicit(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("UV_STACK_ROOT", raising=False)
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "from-env"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "from-env")


def test_discover_default(monkeypatch):
    monkeypatch.delenv("UV_STACK_ROOT", raising=False)
    monkeypatch.delenv("UV_ENV_ROOT", raising=False)
    cfg = ConfigRoot.discover()
    assert cfg.root == (Path.home() / ".config" / "python-envs")


def test_discover_prefers_uv_stack_root(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("UV_ENV_ROOT", raising=False)
    monkeypatch.setenv("UV_STACK_ROOT", str(tmp_path / "stack-root"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "stack-root")


def test_discover_uv_stack_root_wins_over_legacy(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("UV_STACK_ROOT", str(tmp_path / "stack-root"))
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "legacy-root"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "stack-root")


def test_discover_falls_back_to_legacy_uv_env_root(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("UV_STACK_ROOT", raising=False)
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "legacy-root"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "legacy-root")


def test_discover_empty_uv_stack_root_falls_through_to_legacy(monkeypatch, tmp_path: Path):
    """An empty UV_STACK_ROOT is treated as unset, matching the existing
    ``if env:`` guard's handling of an empty UV_ENV_ROOT."""
    monkeypatch.setenv("UV_STACK_ROOT", "")
    monkeypatch.setenv("UV_ENV_ROOT", str(tmp_path / "legacy-root"))
    cfg = ConfigRoot.discover()
    assert cfg.root == (tmp_path / "legacy-root")


def test_path_helpers(config_tree: ConfigRoot):
    root = config_tree.root
    assert config_tree.profile_path("ds") == root / "profiles" / "ds.yaml"
    assert config_tree.bundle_path("qsar") == root / "bundles" / "qsar.yaml"
    assert config_tree.env_stack_path("main") == root / "envs" / "main" / "stack.txt"
    assert (
        config_tree.env_requirements_lock("main")
        == root / "envs" / "main" / "requirements.lock.txt"
    )
    assert config_tree.locks_dir == root / ".locks"
    assert config_tree.stem_lock_path("x") == root / ".locks" / "stem-x.lock"
    assert config_tree.env_lock_path("myenv") == root / ".locks" / "env-myenv.lock"


def test_existence_checks(config_tree: ConfigRoot):
    assert config_tree.profile_exists("ds")
    assert not config_tree.profile_exists("nope")
    assert config_tree.bundle_exists("standard")
    assert not config_tree.bundle_exists("nope")
    assert config_tree.env_exists("main")
    assert not config_tree.env_exists("ghost")


def test_listing(config_tree: ConfigRoot):
    assert config_tree.list_profiles() == ["chem", "ds", "utils"]
    assert config_tree.list_bundles() == ["qsar", "standard"]
    assert config_tree.list_envs() == ["main"]


def test_load_profile(config_tree: ConfigRoot):
    p = config_tree.load_profile("ds")
    assert p.includes == ["numpy", "pandas"]
    assert p.description == "Core data-science stack"
    assert p.tags == ["data", "core"]


def test_load_profile_missing_raises(config_tree: ConfigRoot):
    with pytest.raises(ConfigError):
        config_tree.load_profile("nope")


def test_load_bundle(config_tree: ConfigRoot):
    b = config_tree.load_bundle("standard")
    assert b.includes == ["ds", "chem", "utils"]


def test_load_profile_malformed_yaml_raises(config_tree: ConfigRoot):
    config_tree.profile_path("ds").write_text("includes: [unterminated\n")
    with pytest.raises(ConfigError):
        config_tree.load_profile("ds")


# One document per exception family. PyYAML's contract is that a loader raises
# YAMLError; its constructors break that contract, and not in one way.
_CONSTRUCTOR_FAILURES = [
    pytest.param("description: 2020-99-99\nincludes: []\n", id="value-error"),
    pytest.param('description: !!bool "nope"\nincludes: []\n', id="key-error"),
    pytest.param('description: !!timestamp "nope"\nincludes: []\n', id="attribute-error"),
]


@pytest.mark.parametrize("document", _CONSTRUCTOR_FAILURES)
def test_load_profile_constructor_failure_raises_config_error(
    config_tree: ConfigRoot, document: str
):
    config_tree.profile_path("ds").write_text(document)
    with pytest.raises(ConfigError) as caught:
        config_tree.load_profile("ds")
    assert str(config_tree.profile_path("ds")) in str(caught.value)


@pytest.mark.parametrize("document", _CONSTRUCTOR_FAILURES)
def test_load_bundle_constructor_failure_raises_config_error(
    config_tree: ConfigRoot, document: str
):
    config_tree.bundle_path("standard").write_text(document)
    with pytest.raises(ConfigError) as caught:
        config_tree.load_bundle("standard")
    assert str(config_tree.bundle_path("standard")) in str(caught.value)


def test_a_constructor_failure_names_the_exception_type(config_tree: ConfigRoot):
    # Two of the three families stringify to a bare operand: KeyError('nope')
    # renders as "'nope'", which on its own tells a reader nothing about what
    # the loader objected to. The type is what makes the message legible.
    config_tree.profile_path("ds").write_text('description: !!bool "nope"\nincludes: []\n')

    with pytest.raises(ConfigError) as caught:
        config_tree.load_profile("ds")

    assert "KeyError" in str(caught.value)


def test_load_profile_bad_utf8_is_not_reported_as_invalid_yaml(config_tree: ConfigRoot):
    # The YAML guard is broad, so the read has to stay outside it: a file that
    # is not UTF-8 never reached the parser, and "Invalid YAML" would send the
    # reader hunting for a syntax error that is not there.
    config_tree.profile_path("ds").write_bytes(b"includes: [\xff\xfe]\n")
    with pytest.raises(ConfigError) as caught:
        config_tree.load_profile("ds")
    assert "Cannot read" in str(caught.value)
    assert "Invalid YAML" not in str(caught.value)


def test_load_profile_empty_file_raises(config_tree: ConfigRoot):
    config_tree.profile_path("ds").write_text("")
    with pytest.raises(ConfigError):
        config_tree.load_profile("ds")


def test_load_profile_unknown_key_raises(config_tree: ConfigRoot):
    config_tree.profile_path("ds").write_text("includes: [numpy]\nbogus: true\n")
    with pytest.raises(ConfigError):
        config_tree.load_profile("ds")


def test_load_bundle_malformed_yaml_raises(config_tree: ConfigRoot):
    config_tree.bundle_path("standard").write_text("includes: [unterminated\n")
    with pytest.raises(ConfigError):
        config_tree.load_bundle("standard")


def test_load_env(config_tree: ConfigRoot):
    env = config_tree.load_env("main")
    assert env.python == "3.12"
    assert env.stack == ["@standard"]
    assert env.micromamba == ["graphviz"]
    assert env.channels == ["bioconda"]


def test_load_env_without_channels(config_tree: ConfigRoot):
    config_tree.env_channels_path("main").unlink()
    env = config_tree.load_env("main")
    assert env.channels == []


def test_load_env_missing_stack_raises(config_tree: ConfigRoot):
    (config_tree.root / "envs" / "broken").mkdir()
    with pytest.raises(ConfigError):
        config_tree.load_env("broken")


def test_load_env_missing_hint_mentions_create_env_tokens(tmp_path):
    from uv_stack.config import ConfigRoot
    from uv_stack.errors import ConfigError

    cfg = ConfigRoot(tmp_path)
    try:
        cfg.load_env("ghost")
    except ConfigError as error:
        assert error.hint is not None
        assert "pass TOKENS: stack create env ghost TOKENS..." in error.hint
    else:
        raise AssertionError("expected ConfigError")


def test_load_env_hint_shell_quotes_the_name(tmp_path: Path):
    """The hint is a runnable command, so the name must survive a paste intact."""
    root = ConfigRoot(tmp_path / "python-envs")
    hostile = "a b; rm -rf /"
    with pytest.raises(ConfigError) as excinfo:
        root.load_env(hostile)
    assert "stack create env 'a b; rm -rf /' TOKENS..." in excinfo.value.hint


def test_load_env_hint_leaves_a_plain_name_unquoted(tmp_path: Path):
    root = ConfigRoot(tmp_path / "python-envs")
    with pytest.raises(ConfigError) as excinfo:
        root.load_env("ghost")
    assert "stack create env ghost TOKENS..." in excinfo.value.hint


def test_load_env_hint_prefixes_leading_dash_name(tmp_path: Path):
    """A name starting with - must be prefixed with -- to remain positional."""
    root = ConfigRoot(tmp_path / "python-envs")
    with pytest.raises(ConfigError) as excinfo:
        root.load_env("--recreate")
    assert "stack create env -- --recreate TOKENS..." in excinfo.value.hint


def test_probe_lock_path_cannot_collide_with_a_user_name(config_tree):
    """The stem and env locks are prefixed, so a fixed bare name is safe."""
    assert config_tree.probe_lock_path() == config_tree.locks_dir / "probe.lock"
    assert config_tree.probe_lock_path() != config_tree.stem_lock_path("probe")
    assert config_tree.probe_lock_path() != config_tree.env_lock_path("probe")


def test_project_lock_path_names_the_resolved_project(config_tree, tmp_path):
    """A project reached through a symlink takes the same lock as its target."""
    project = tmp_path / "proj"
    project.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(project)
    other = tmp_path / "other"
    other.mkdir()

    lock = config_tree.project_lock_path(project)

    assert lock.parent == config_tree.locks_dir
    assert lock.name.startswith("project-") and lock.suffix == ".lock"
    assert config_tree.project_lock_path(alias) == lock
    assert config_tree.project_lock_path(other) != lock


def test_editor_path_is_root_scoped(tmp_path: Path):
    # editor.txt is a property of the config root, not of any one env.
    assert ConfigRoot(tmp_path).editor_path() == tmp_path / "editor.txt"


def test_default_editor_reads_first_clean_line(tmp_path: Path):
    (tmp_path / "editor.txt").write_text("# my editor\nvim -f\n", encoding="utf-8")
    assert ConfigRoot(tmp_path).default_editor() == "vim -f"


def test_default_editor_is_none_when_absent(tmp_path: Path):
    assert ConfigRoot(tmp_path).default_editor() is None


def test_default_editor_is_none_when_blank(tmp_path: Path):
    (tmp_path / "editor.txt").write_text("   \n# nothing here\n", encoding="utf-8")
    assert ConfigRoot(tmp_path).default_editor() is None


def test_default_editor_reports_undecodable_file_as_config_error(tmp_path: Path):
    # UnicodeDecodeError is a ValueError, so unwrapped it escapes the CLI edge,
    # which renders only UvStackError and OSError.
    (tmp_path / "editor.txt").write_bytes(b"\xff\xfe vim")
    with pytest.raises(ConfigError) as excinfo:
        ConfigRoot(tmp_path).default_editor()
    assert "not valid UTF-8" in excinfo.value.message
    assert str(tmp_path / "editor.txt") in excinfo.value.message


def test_require_env_accepts_an_env_with_a_stack_file(config_tree: ConfigRoot):
    config_tree.require_env("main")


def test_require_env_reports_a_missing_stack_file(tmp_path: Path):
    with pytest.raises(ConfigError) as excinfo:
        ConfigRoot(tmp_path).require_env("ghost")
    assert "Missing stack file for env 'ghost'" in excinfo.value.message
    assert "stack create env ghost TOKENS..." in (excinfo.value.hint or "")


@pytest.mark.parametrize(
    "break_it,expected,hint",
    [
        (lambda path: path.mkdir(), "Not a regular file", "Remove or rename"),
        (
            lambda path: path.symlink_to(path.parent / "nowhere"),
            "Broken symlink",
            "Point it at a real file",
        ),
    ],
    ids=["directory", "dangling-symlink"],
)
def test_require_env_names_a_non_regular_stack_path(
    tmp_path: Path, break_it, expected, hint
):
    """Something occupying stack.txt is not the same as nothing being there.

    ``env_exists`` answers with ``is_file()``, which is False either way, so
    without the guard both report "Missing stack file" and hint at ``stack
    create env`` — which opens ``O_EXCL``, sees the entry, and refuses. Every
    caller of ``require_env`` inherits this, ``show env`` and ``upgrade``
    included.
    """
    config = ConfigRoot(tmp_path)
    path = config.env_stack_path("main")
    path.parent.mkdir(parents=True)
    break_it(path)
    with pytest.raises(ConfigError) as excinfo:
        config.require_env("main")
    assert expected in excinfo.value.message
    assert str(path) in excinfo.value.message
    assert hint in (excinfo.value.hint or "")


def test_variables_paths(config_tree: ConfigRoot):
    assert config_tree.variables_path() == config_tree.root / "variables.txt"
    assert (
        config_tree.variables_local_path() == config_tree.root / "variables.local.txt"
    )


def test_load_variables_on_a_root_with_no_variable_files(config_tree: ConfigRoot):
    variables = config_tree.load_variables()
    assert variables.declared == ()
    assert dict(variables.values) == {}


def test_load_variables_reads_declarations_and_values(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("DEV\n# a comment\nWORK\n")
    config_tree.variables_local_path().write_text("DEV=/home/me/dev\n")
    variables = config_tree.load_variables()
    assert variables.declared == ("DEV", "WORK")
    assert dict(variables.values) == {"DEV": "/home/me/dev"}
    assert variables.undefined() == ["WORK"]


def test_a_value_may_contain_an_equals_sign(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("HOST\n")
    config_tree.variables_local_path().write_text("HOST=https://h/simple?a=b\n")
    assert config_tree.load_variables().values["HOST"] == "https://h/simple?a=b"


def test_spaces_around_the_separator_are_tolerated(config_tree: ConfigRoot):
    # 'DEV = /x' is what a user writes by habit. The name is stripped before
    # the declared-name lookup and the value before the whitespace refusal, so
    # neither strip is cosmetic.
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV = /home/me/dev\n")
    assert config_tree.load_variables().values["DEV"] == "/home/me/dev"


def test_a_value_is_tilde_expanded(config_tree: ConfigRoot, monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=~/code\n")
    assert config_tree.load_variables().values["DEV"] == "/home/tester/code"


def test_an_environment_variable_wins_over_the_file(config_tree: ConfigRoot, monkeypatch):
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=/from/file\n")
    monkeypatch.setenv("DEV", "/from/env")
    assert config_tree.load_variables().values["DEV"] == "/from/env"


def test_an_empty_environment_variable_does_not_count(config_tree: ConfigRoot, monkeypatch):
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=/from/file\n")
    monkeypatch.setenv("DEV", "   ")
    assert config_tree.load_variables().values["DEV"] == "/from/file"


def test_an_environment_value_may_contain_a_hash(config_tree: ConfigRoot, monkeypatch):
    # The file grammar strips '#' as a comment; the environment has no grammar.
    config_tree.variables_path().write_text("DEV\n")
    monkeypatch.setenv("DEV", "/a#b")
    assert config_tree.load_variables().values["DEV"] == "/a#b"


def test_an_environment_variable_for_an_undeclared_name_is_ignored(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.setenv("NOPE", "/x")
    assert dict(config_tree.load_variables().values) == {}


_NON_REGULAR_SHAPES = [
    "directory",
    "dangling-symlink",
    pytest.param(
        "fifo",
        marks=pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform lacks mkfifo"),
    ),
]


def _make_non_regular(path: Path, shape: str) -> None:
    """Create ``path`` as something that is present but not a regular file."""
    if shape == "directory":
        path.mkdir()
    elif shape == "dangling-symlink":
        path.symlink_to(path.parent / "nowhere")
    else:
        os.mkfifo(path)


@pytest.mark.parametrize("shape", _NON_REGULAR_SHAPES)
def test_a_non_regular_variables_file_is_refused(config_tree: ConfigRoot, shape: str):
    # is_file() cannot tell absent from broken, so without the guard a
    # directory named variables.txt reads as a root that declares nothing and
    # every command runs on with no variables at all.
    _make_non_regular(config_tree.variables_path(), shape)
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "variables.txt" in excinfo.value.message


@pytest.mark.parametrize("shape", _NON_REGULAR_SHAPES)
def test_a_non_regular_local_variables_file_is_refused(config_tree: ConfigRoot, shape: str):
    config_tree.variables_path().write_text("DEV\n")
    _make_non_regular(config_tree.variables_local_path(), shape)
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "variables.local.txt" in excinfo.value.message


@pytest.mark.parametrize("shape", _NON_REGULAR_SHAPES)
def test_a_non_regular_project_python_file_is_refused(config_tree: ConfigRoot, shape: str):
    # first_clean_line reads through the same is_file() test, so each of these
    # shapes answered "no default is configured" -- the answer an absent file
    # gives -- for a file that is present and unreadable.
    _make_non_regular(config_tree.project_python_path(), shape)
    with pytest.raises(ConfigError) as excinfo:
        config_tree.default_project_python()
    assert "project-python.txt" in excinfo.value.message


@pytest.mark.parametrize("shape", _NON_REGULAR_SHAPES)
def test_a_non_regular_editor_file_is_refused(config_tree: ConfigRoot, shape: str):
    # The same argument as the method above, and as the five targets
    # cli/edit.py guards: without this, a directory at editor.txt reads as
    # "no editor configured" and 'stack edit' quietly opens $VISUAL instead of
    # the editor the root names.
    _make_non_regular(config_tree.editor_path(), shape)
    with pytest.raises(ConfigError) as excinfo:
        config_tree.default_editor()
    assert "editor.txt" in excinfo.value.message


@pytest.mark.parametrize("shape", _NON_REGULAR_SHAPES)
@pytest.mark.parametrize(
    "accessor", ["env_python_path", "env_micromamba_path", "env_channels_path"]
)
def test_a_non_regular_env_source_is_refused(
    config_tree: ConfigRoot, accessor: str, shape: str
):
    # All three are optional, and both readers answer for an unreadable path
    # what they answer for an absent one -- so the configured value was not
    # failing to load, it was being replaced by a default: 3.12 for the
    # interpreter, nothing at all for the conda packages and channels.
    # stack.txt is absent from this list because require_env, one call
    # earlier, already holds it to the same rule.
    path = getattr(config_tree, accessor)("main")
    path.unlink()
    _make_non_regular(path, shape)
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_env("main")
    assert str(path) in excinfo.value.message


def test_load_env_still_admits_absent_optional_sources(config_tree: ConfigRoot):
    # The guard above must not turn an optional file into a required one;
    # require_regular_file is a no-op on a genuinely absent path.
    for accessor in ("env_python_path", "env_micromamba_path", "env_channels_path"):
        getattr(config_tree, accessor)("main").unlink()
    env = config_tree.load_env("main")
    assert (env.python, env.micromamba, env.channels) == ("3.12", [], [])


def test_a_malformed_declaration_names_the_file_and_line(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("DEV\n2BAD\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "variables.txt" in excinfo.value.message
    assert "line 2" in excinfo.value.message


def test_a_duplicate_declaration_names_both_lines(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("DEV\nWORK\nDEV\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "lines 1 and 3" in excinfo.value.message


def test_a_value_line_without_an_equals_sign_is_refused(config_tree: ConfigRoot):
    # 'Line', capitalized: this message opens with the position, unlike the
    # declaration message, which puts it mid-sentence.
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV /home/me\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "Line 1" in excinfo.value.message
    assert "not an assignment" in excinfo.value.message


def test_an_undeclared_assignment_is_refused(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("NOPE=/x\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "NOPE" in excinfo.value.message


def test_a_duplicate_assignment_names_both_lines(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=/a\nDEV=/b\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "lines 1 and 2" in excinfo.value.message


def test_an_empty_value_is_refused(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "empty" in excinfo.value.message


def test_a_value_containing_whitespace_is_refused(config_tree: ConfigRoot):
    # 'DEV=--requirement /tmp' would turn the admitted '${DEV}/deps.txt' into a
    # recursive include, which is what condition 3 exists to prevent.
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=--requirement /tmp\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "whitespace" in excinfo.value.message


def test_whitespace_is_refused_after_tilde_expansion(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.setenv("HOME", "/home/my tester")
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=~/code\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "whitespace" in excinfo.value.message


def test_a_value_whose_expansion_fails_is_refused(config_tree: ConfigRoot):
    # os.path.expanduser is not total: the '~user' form goes to the password
    # database, and a NUL in the name raises ValueError. A ValueError is
    # neither a UvStackError nor an OSError, so unconverted this one leaves
    # 'stack doctor', 'stack sync', and 'stack status' printing a
    # traceback for a file the user can fix in one edit.
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=~a\x00b/x\n")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "'DEV'" in excinfo.value.message
    assert "cannot be expanded" in excinfo.value.message
    assert "variables.local.txt" in excinfo.value.message


def test_an_environment_value_containing_whitespace_is_refused(
    config_tree: ConfigRoot, monkeypatch
):
    config_tree.variables_path().write_text("DEV\n")
    monkeypatch.setenv("DEV", "/a b")
    with pytest.raises(ConfigError) as excinfo:
        config_tree.load_variables()
    assert "whitespace" in excinfo.value.message


def test_load_env_from_dir_reads_a_copied_directory(tmp_path):
    from uv_stack.config import load_env_from_dir

    source = tmp_path / "copied"
    source.mkdir()
    (source / "stack.txt").write_text("standard\n")
    (source / "python.txt").write_text("3.13\n")
    (source / "micromamba.txt").write_text("gdal\n")
    (source / "channels.txt").write_text("bioconda\n")

    env = load_env_from_dir(source, "copied")

    assert env.name == "copied"
    assert env.python == "3.13"
    assert env.stack == ["standard"]
    assert env.micromamba == ["gdal"]
    assert env.channels == ["bioconda"]


def test_load_env_from_dir_defaults_a_missing_python(tmp_path):
    from uv_stack.config import load_env_from_dir

    source = tmp_path / "copied"
    source.mkdir()
    (source / "stack.txt").write_text("standard\n")

    assert load_env_from_dir(source, "copied").python == "3.12"


def test_load_env_from_dir_rejects_a_directory_named_channels_txt(tmp_path):
    from uv_stack.config import load_env_from_dir

    source = tmp_path / "copied"
    source.mkdir()
    (source / "stack.txt").write_text("standard\n")
    (source / "channels.txt").mkdir()

    with pytest.raises(ConfigError) as excinfo:
        load_env_from_dir(source, "copied")
    assert "channels.txt" in str(excinfo.value)


def test_load_env_from_dir_rejects_a_directory_named_stack_txt(tmp_path):
    # load_env never reaches this: require_env has already refused the shape.
    # A copied directory has no such caller, so the helper holds stack.txt to
    # the rule itself rather than reading the directory as an empty stack.
    from uv_stack.config import load_env_from_dir

    source = tmp_path / "copied"
    source.mkdir()
    (source / "stack.txt").mkdir()

    with pytest.raises(ConfigError) as excinfo:
        load_env_from_dir(source, "copied")
    assert "stack.txt" in str(excinfo.value)
