"""Tests for the export closure and document builder."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.hints import has_control
from uv_stack.operations.export import (
    AmbiguousItemError,
    absolute_path_warnings,
    build_document,
    closure,
    normalize_items,
    parse_file_key,
    read_seed,
    serialize_document,
)
from uv_stack.variables import UNDECLARED_HINT


def test_no_items_means_the_whole_root(config_tree: ConfigRoot) -> None:
    assert normalize_items(config_tree, []) == [
        "bundle:qsar", "bundle:standard", "env:main",
        "profile:chem", "profile:ds", "profile:utils",
    ]


@pytest.mark.parametrize("stray", ["profiles/-x.yaml", "bundles/a b.yaml", "envs/-e/stack.txt"])
def test_whole_root_refuses_an_invalid_name_on_disk(config_tree: ConfigRoot, stray: str) -> None:
    path = config_tree.root / stray
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("includes:\n  - numpy\n" if stray.endswith(".yaml") else "numpy\n")
    with pytest.raises(ConfigError):
        normalize_items(config_tree, [])


def test_items_are_qualified_sorted_and_deduplicated(config_tree: ConfigRoot) -> None:
    assert normalize_items(config_tree, ["@qsar", "main", "profile:ds", "bundle:qsar"]) == [
        "bundle:qsar", "env:main", "profile:ds",
    ]


def test_a_bare_name_naming_two_kinds_is_ambiguous(config_tree: ConfigRoot) -> None:
    config_tree.env_dir("ds").mkdir()
    config_tree.env_stack_path("ds").write_text("@standard\n")
    with pytest.raises(AmbiguousItemError) as caught:
        normalize_items(config_tree, ["ds"])
    assert "env:ds" in caught.value.message and "profile:ds" in caught.value.message
    assert caught.value.hint == "Qualify it: env:ds or profile:ds."


def test_profile_and_bundle_of_one_name_are_ambiguous(config_tree: ConfigRoot) -> None:
    (config_tree.bundles_dir / "chem.yaml").write_text("includes:\n  - rdkit\n")
    with pytest.raises(AmbiguousItemError):
        normalize_items(config_tree, ["chem"])


@pytest.mark.parametrize("raw", ["ghost", "env:ghost", "profile:ghost", "@ghost"])
def test_a_missing_item_is_refused(config_tree: ConfigRoot, raw: str) -> None:
    with pytest.raises(ConfigError):
        normalize_items(config_tree, [raw])


def test_an_unknown_kind_is_refused(config_tree: ConfigRoot) -> None:
    with pytest.raises(ConfigError, match="Unknown item kind"):
        normalize_items(config_tree, ["widget:ds"])


def test_env_closure_ships_its_files_and_everything_it_reaches(config_tree: ConfigRoot) -> None:
    assert closure(config_tree, ["env:main"]) == {
        "envs/main/stack.txt", "envs/main/python.txt", "envs/main/micromamba.txt",
        "envs/main/channels.txt", "bundles/standard.yaml",
        "profiles/ds.yaml", "profiles/chem.yaml", "profiles/utils.yaml",
    }


def test_bundle_closure_package_tokens_bring_no_file(config_tree: ConfigRoot) -> None:
    assert closure(config_tree, ["bundle:qsar"]) == {
        "bundles/qsar.yaml", "bundles/standard.yaml",
        "profiles/ds.yaml", "profiles/chem.yaml", "profiles/utils.yaml",
    }


def test_closure_of_a_bundle_cycle_terminates(config_tree: ConfigRoot) -> None:
    (config_tree.bundles_dir / "a.yaml").write_text("includes:\n  - '@b'\n")
    (config_tree.bundles_dir / "b.yaml").write_text("includes:\n  - '@a'\n  - profile:ds\n")
    assert closure(config_tree, ["bundle:a"]) == {
        "bundles/a.yaml", "bundles/b.yaml", "profiles/ds.yaml",
    }


def test_explicit_reference_to_a_missing_profile_is_refused(config_tree: ConfigRoot) -> None:
    (config_tree.bundles_dir / "odd.yaml").write_text("includes:\n  - profile:ghost\n")
    with pytest.raises(UvStackError, match="ghost"):
        closure(config_tree, ["bundle:odd"])


def test_seed_absent_is_none(config_tree: ConfigRoot) -> None:
    assert read_seed(config_tree, "main") is None


def test_seed_present_is_read_verbatim(config_tree: ConfigRoot) -> None:
    config_tree.env_requirements_lock("main").write_text("numpy==1.26.4\n")
    assert read_seed(config_tree, "main") == "numpy==1.26.4\n"


def test_seed_directory_is_refused(config_tree: ConfigRoot) -> None:
    config_tree.env_requirements_lock("main").mkdir()
    with pytest.raises(ConfigError):
        read_seed(config_tree, "main")


def test_seed_unparseable_is_refused(config_tree: ConfigRoot) -> None:
    config_tree.env_requirements_lock("main").write_text("not a requirement\n")
    with pytest.raises(ConfigError, match="line 1"):
        read_seed(config_tree, "main")


def test_an_undeclared_variable_is_refused(config_tree: ConfigRoot) -> None:
    config_tree.profile_path("ds").write_text("includes:\n  - -e ${WORK}/pkg\n")
    with pytest.raises(ConfigError) as caught:
        build_document(config_tree, ["profile:ds"])
    assert "profiles/ds.yaml" in caught.value.message and "WORK" in caught.value.message
    assert caught.value.hint == UNDECLARED_HINT


def test_a_declared_variable_is_accepted(config_tree: ConfigRoot) -> None:
    config_tree.variables_path().write_text("WORK\n")
    config_tree.profile_path("ds").write_text("includes:\n  - -e ${WORK}/pkg\n")
    result = build_document(config_tree, ["profile:ds"])
    assert "${WORK}" in result.document.files["profiles/ds.yaml"]


@pytest.mark.parametrize("source", ["file", "environment"])
def test_a_malformed_local_value_does_not_stop_an_export(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    # The document never ships values, so neither this machine's value file
    # nor an environment override of a declared name is read.
    config_tree.variables_path().write_text("WORK\n")
    if source == "file":
        config_tree.variables_local_path().write_text("not a value line\n")
    else:
        monkeypatch.setenv("WORK", "has space")
    config_tree.profile_path("ds").write_text("includes:\n  - -e ${WORK}/pkg\n")
    result = build_document(config_tree, ["profile:ds"])
    assert "${WORK}" in result.document.files["profiles/ds.yaml"]


def test_absolute_paths_warn_with_a_count(config_tree: ConfigRoot) -> None:
    config_tree.profile_path("ds").write_text(
        "includes:\n  - -e /src/a\n  - /wheels/b.whl\n  - -e ./rel\n  - numpy\n"
    )
    result = build_document(config_tree, ["profile:ds"])
    assert result.warnings == [
        "profiles/ds.yaml: 2 absolute editable or local path(s); these build only where "
        "the same paths exist. Use ${NAME} to make them portable."
    ]


def test_file_urls_warn_as_absolute_paths(config_tree: ConfigRoot) -> None:
    config_tree.variables_path().write_text("WHEELS\n")
    config_tree.profile_path("ds").write_text(
        "includes:\n"
        "  - file:///wheels/a.whl\n"
        "  - pkg-b @ file:///wheels/b.whl\n"
        "  - pkg-c@file:///wheels/c.whl ; python_version >= '3.10'\n"
        "  - -e file:///src/d\n"
        "  - -e=file:///src/e\n"
        "  - pkg-f @ file://${WHEELS}/f.whl\n"
        "  - pkg-g @ https://example.invalid/g.whl\n"
        # Quoted, or the YAML parser eats the comment before the check sees it.
        "  - \"pkg-h  # see file:///docs/h\"\n"
    )
    result = build_document(config_tree, ["profile:ds"])
    assert result.warnings == [
        "profiles/ds.yaml: 5 absolute editable or local path(s); these build only where "
        "the same paths exist. Use ${NAME} to make them portable."
    ]


@pytest.mark.parametrize("entry", ["pkg @ FILE:///wheels/a.whl", "-e File:///src/pkg"])
def test_file_urls_warn_whatever_their_case(config_tree: ConfigRoot, entry: str) -> None:
    # A URL scheme is case-insensitive, and pip and uv install from either spelling.
    config_tree.profile_path("ds").write_text(f"includes:\n  - {entry}\n")
    assert absolute_path_warnings(config_tree, ["profiles/ds.yaml"]) == [
        "profiles/ds.yaml: 1 absolute editable or local path(s); these build only where "
        "the same paths exist. Use ${NAME} to make them portable."
    ]


def test_document_is_deterministic_and_carries_seeds(config_tree: ConfigRoot) -> None:
    config_tree.env_requirements_lock("main").write_text("numpy==1.26.4\n")
    first = serialize_document(build_document(config_tree, ["main"]).document)
    second = serialize_document(build_document(config_tree, ["main"]).document)
    assert first == second
    data = json.loads(first)
    assert data["items"] == ["env:main"]
    assert data["seeds"] == {"main": "numpy==1.26.4\n"}
    assert data["format"] == "uv-stack-export" and data["version"] == 1


def test_crlf_is_preserved(config_tree: ConfigRoot) -> None:
    config_tree.profile_path("ds").write_bytes(b"includes:\r\n  - numpy\r\n")
    document = build_document(config_tree, ["profile:ds"]).document
    assert document.files["profiles/ds.yaml"] == "includes:\r\n  - numpy\r\n"


@pytest.mark.parametrize("key", ["../x.yaml", "/abs/profiles/x.yaml", "profiles/a/b.yaml",
                                 "envs/main/requirements.lock.txt", "profiles/-x.yaml",
                                 "profiles/a\x00b.yaml", "bundles/a\x00b.yaml",
                                 "envs/a\x00b/stack.txt", "profiles/a\x1bb.yaml",
                                 "envs/a\x9bb/stack.txt", "profiles/ds.yaml\n"])
def test_parse_file_key_rejects_unsafe_keys(key: str) -> None:
    assert parse_file_key(key) is None


def test_whole_root_export_refuses_a_control_character_in_a_name_on_disk(
    config_tree: ConfigRoot,
) -> None:
    # Every import refuses such a name, so shipping it would only move the
    # refusal to the other machine.
    try:
        (config_tree.profiles_dir / "a\x1bb.yaml").write_text("includes:\n  - numpy\n")
    except OSError:
        pytest.skip("this filesystem refuses an escape character in a file name")
    with pytest.raises(ConfigError) as caught:
        build_document(config_tree, [])
    assert "a\\x1bb" in caught.value.message
    assert not has_control(caught.value.message)


def test_whole_root_export_refuses_a_name_on_disk_that_is_not_valid_utf8(
    config_tree: ConfigRoot,
) -> None:
    # Python reads such a name as lone surrogates, which no UTF-8 terminal
    # can print, so the name is refused like any other unusable file stem.
    path = config_tree.profiles_dir / os.fsdecode(b"a\xffb.yaml")
    try:
        path.write_text("includes:\n  - numpy\n")
    except OSError:
        pytest.skip("this filesystem refuses a file name that is not valid UTF-8")
    with pytest.raises(ConfigError) as caught:
        build_document(config_tree, [])
    assert "a\\udcffb" in caught.value.message


@pytest.mark.parametrize("name", ["a\x1bb", "../a\x1bb"])
def test_a_reference_holding_a_control_character_is_named_escaped(
    config_tree: ConfigRoot, name: str
) -> None:
    # The file exists, so only the reference's own check can refuse it.
    try:
        config_tree.profile_path(name).write_text("includes:\n  - numpy\n")
    except OSError:
        pytest.skip("this filesystem refuses an escape character in a file name")
    config_tree.env_dir("bad").mkdir()
    config_tree.env_stack_path("bad").write_text(f"profile:{name}\n")
    with pytest.raises(ConfigError) as caught:
        build_document(config_tree, ["env:bad"])
    assert name.replace("\x1b", "\\x1b") in caught.value.message
    assert not has_control(caught.value.message)


def _folded_root(tmp_path: Path) -> ConfigRoot:
    """Return a root whose env ``main`` says ``foo`` beside ``profiles/Foo.yaml``.

    Skips unless ``tmp_path`` is on a filesystem that folds letter case.
    """
    (tmp_path / "Probe").write_text("")
    if not (tmp_path / "probe").exists():
        pytest.skip("this filesystem does not fold letter case")
    root = tmp_path / "root"
    (root / "profiles").mkdir(parents=True)
    (root / "bundles").mkdir()
    (root / "envs" / "main").mkdir(parents=True)
    (root / "profiles" / "Foo.yaml").write_text("includes:\n  - numpy\n")
    (root / "envs" / "main" / "stack.txt").write_text("foo\n")
    return ConfigRoot(root)


def test_a_file_reached_under_two_spellings_is_refused(tmp_path: Path) -> None:
    # The listing gives Foo and the env's reference gives foo. Shipping both
    # makes a folding target refuse the document and gives a case-sensitive
    # one two definitions where this root has one.
    config = _folded_root(tmp_path)
    with pytest.raises(ConfigError) as caught:
        build_document(config, [])
    assert caught.value.message == (
        "profiles/Foo.yaml and profiles/foo.yaml name the same file on this machine."
    )
    assert "letter case" in caught.value.hint


def test_an_env_named_under_two_spellings_is_refused(tmp_path: Path) -> None:
    config = _folded_root(tmp_path)
    with pytest.raises(ConfigError) as caught:
        build_document(config, ["main", "env:Main"])
    assert caught.value.message == (
        "envs/Main/stack.txt and envs/main/stack.txt name the same file on this machine."
    )


def test_a_lone_reference_spelled_unlike_its_file_ships_as_spelled(tmp_path: Path) -> None:
    # Respelling it Foo would leave the env's foo unmatched on a
    # case-sensitive target.
    config = _folded_root(tmp_path)
    files = build_document(config, ["main"]).document.files
    assert "profiles/foo.yaml" in files
    assert "profiles/Foo.yaml" not in files


def test_hard_linked_profiles_both_ship(config_tree: ConfigRoot) -> None:
    # Both are names the directory stores, so neither reached the file by
    # folding; import decides what the link means on the target.
    config_tree.profile_path("a").write_text("includes:\n  - numpy\n")
    os.link(config_tree.profile_path("a"), config_tree.profile_path("b"))
    config_tree.env_dir("linked").mkdir()
    config_tree.env_stack_path("linked").write_text("profile:a\nprofile:b\n")
    files = build_document(config_tree, ["env:linked"]).document.files
    assert {"profiles/a.yaml", "profiles/b.yaml"} <= files.keys()


def test_a_symlinked_env_directory_ships_under_both_names(config_tree: ConfigRoot) -> None:
    os.symlink("main", config_tree.env_dir("alt"))
    files = build_document(config_tree, ["env:alt", "env:main"]).document.files
    assert {"envs/alt/stack.txt", "envs/main/stack.txt"} <= files.keys()


def test_closure_refuses_profile_reference_that_escapes_root(config_tree: ConfigRoot) -> None:
    (config_tree.root / "outside.yaml").write_text("includes:\n  - numpy\n")
    (config_tree.bundles_dir / "bad.yaml").write_text("includes:\n  - profile:../outside\n")
    with pytest.raises(ConfigError) as caught:
        closure(config_tree, ["bundle:bad"])
    assert "../outside" in caught.value.message
    assert "plain name" in caught.value.hint


def test_bundle_include_with_escaping_profile_is_refused(config_tree: ConfigRoot) -> None:
    (config_tree.root / "outside.yaml").write_text("includes:\n  - numpy\n")
    (config_tree.bundles_dir / "bad.yaml").write_text("includes:\n  - profile:../outside\n")
    with pytest.raises(ConfigError) as caught:
        build_document(config_tree, ["bundle:bad"])
    assert "../outside" in caught.value.message


def test_bundle_include_with_escaping_bundle_is_refused(config_tree: ConfigRoot) -> None:
    (config_tree.root / "outside2.yaml").write_text("includes:\n  - numpy\n")
    (config_tree.bundles_dir / "bad.yaml").write_text("includes:\n  - bundle:../outside2\n")
    with pytest.raises(ConfigError) as caught:
        build_document(config_tree, ["bundle:bad"])
    assert "../outside2" in caught.value.message


def test_env_stack_with_escaping_profile_is_refused(config_tree: ConfigRoot) -> None:
    (config_tree.root / "outside.yaml").write_text("includes:\n  - numpy\n")
    config_tree.env_dir("bad").mkdir()
    config_tree.env_stack_path("bad").write_text("profile:../outside\n")
    with pytest.raises(ConfigError) as caught:
        build_document(config_tree, ["env:bad"])
    assert "../outside" in caught.value.message
