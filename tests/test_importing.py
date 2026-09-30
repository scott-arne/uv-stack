"""Tests for stack import's phases."""

from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import _lock_held_by_another_process
from uv_stack import fsutil
from uv_stack.commands import micromamba_python_info
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.operations import importing
from uv_stack.operations.export import build_document, serialize_document
from uv_stack.operations.importing import (
    BuildRequest,
    BuildStep,
    ConflictError,
    FileChange,
    ImportOptions,
    ImportPlan,
    change_diff,
    check_document,
    check_meanings,
    classify_changes,
    document_root,
    load_document,
    pin_report,
    prepared_import,
    read_document,
    refuse_shadowing,
    staged_root,
    used_by,
    validate_staged,
)
from uv_stack.runner import Command, CommandResult, RecordingRunner


def _raw(config: ConfigRoot, *items: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(serialize_document(build_document(config, items).document))
    return data


def _load(data: dict[str, Any]) -> Any:
    return load_document(json.dumps(data))


def test_round_trip(config_tree: ConfigRoot) -> None:
    text = serialize_document(build_document(config_tree, ["main"]).document)
    assert serialize_document(load_document(text)) == text


def test_read_document_from_stdin_and_rejects_non_utf8() -> None:
    assert read_document("-", io.BytesIO(b"{}")) == "{}"
    with pytest.raises(ConfigError, match="standard input"):
        read_document("-", io.BytesIO(b"\xff"))


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda d: d.update(format="other"), "format"),
        (lambda d: d.update(version=2, created_by="uv-stack 9.0"), "uv-stack 9.0"),
        (lambda d: d.update(version=0), "version 0"),
        (lambda d: d.update(bonus=1), "bonus"),
        (lambda d: d["files"].update({"../x.yaml": ""}), "../x.yaml"),
        (lambda d: d["files"].update({"/etc/profiles/x.yaml": ""}), "/etc/profiles/x.yaml"),
        (lambda d: d["files"].pop("envs/main/stack.txt"), "stack.txt"),
        (lambda d: d["seeds"].update(ghost="numpy==1\n"), "ghost"),
        (lambda d: d["items"].append("profile:nope"), "profile:nope"),
        (lambda d: d["items"].reverse(), "sorted"),
        (lambda d: d["seeds"].update(main="not a requirement\n"), "seeds/main, line 1"),
    ],
)
def test_document_refusals(config_tree: ConfigRoot, mutate: Any, needle: str) -> None:
    data = _raw(config_tree, "main", "profile:utils")
    data["seeds"].setdefault("main", "numpy==1.26.4\n")
    mutate(data)
    with pytest.raises(ConfigError, match=None) as caught:
        _load(data)
    assert needle in caught.value.message


def _ship_item(data: dict[str, Any], item: str) -> None:
    # The item's file ships too, so only the control-character check can
    # refuse it.
    data["items"] = sorted([*data["items"], item])
    data["files"][f"profiles/{item.partition(':')[2]}.yaml"] = "includes:\n  - rich\n"


def _has_control(text: str) -> bool:
    return any(unicodedata.category(char) == "Cc" for char in text)


@pytest.mark.parametrize(
    ("field", "value", "mutate"),
    [
        ("created_by", "uv-stack 0.7.0\x1b]0;owned\x07", lambda d, v: d.update(created_by=v)),
        ("source_platform", "darwin-arm64\x1b[2K", lambda d, v: d.update(source_platform=v)),
        ("item", "profile:a\x1bb", _ship_item),
        ("file key", "profiles/a\x9bb.yaml",
         lambda d, v: d["files"].update({v: "includes:\n  - rich\n"})),
        ("seed", "ma\x1bin", lambda d, v: d["seeds"].update({v: "numpy==1\n"})),
    ],
)
def test_a_control_character_in_a_document_name_is_refused(
    config_tree: ConfigRoot, field: str, value: str, mutate: Any
) -> None:
    data = _raw(config_tree, "main", "profile:utils")
    mutate(data, value)
    with pytest.raises(ConfigError) as caught:
        _load(data)
    assert field in caught.value.message and repr(value) in caught.value.message
    assert not _has_control(caught.value.message)
    assert caught.value.hint is not None and "stack export" in caught.value.hint


@pytest.mark.parametrize("writer", ["uv-stack 9.0\x1b[2K", 5])
def test_a_newer_version_names_only_a_writer_it_can_print(
    config_tree: ConfigRoot, writer: object
) -> None:
    # The version refusal runs before schema validation, so created_by may be
    # any JSON value.
    data = _raw(config_tree, "profile:utils")
    data.update(version=2, created_by=writer)
    with pytest.raises(ConfigError) as caught:
        _load(data)
    assert "written by an unknown writer;" in caught.value.message
    assert not _has_control(caught.value.message)


def test_bad_json_is_refused() -> None:
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_document("{")


def test_a_nul_in_a_document_name_is_refused(config_tree: ConfigRoot) -> None:
    # JSON carries \u0000 and validate_name allows it, but document_root would
    # raise ValueError building a path from it. The item and file agree, and
    # NUL is a control character, so the control-character check stops it.
    data = _raw(config_tree, "profile:ds")
    data["items"] = sorted([*data["items"], "profile:a\0b"])
    data["files"]["profiles/a\0b.yaml"] = "includes:\n  - rich\n"
    with pytest.raises(ConfigError) as caught:
        _load(data)
    assert "control character: 'profile:a\\x00b'" in caught.value.message


def test_unreachable_extra_file_is_refused(config_tree: ConfigRoot) -> None:
    data = _raw(config_tree, "profile:ds")
    data["files"]["profiles/extra.yaml"] = "includes:\n  - rich\n"
    document = _load(data)
    with document_root(document) as root, pytest.raises(ConfigError) as caught:
        check_document(document, root)
    assert "unreachable" in caught.value.message and "profiles/extra.yaml" in caught.value.message


def test_qualified_reference_to_an_omitted_profile_names_no_temp_path(
    config_tree: ConfigRoot,
) -> None:
    data = _raw(config_tree, "bundle:standard")
    # The fixture's standard.yaml names ds bare, which would fall through to
    # a package once ds.yaml is gone; the qualified spelling must not.
    data["files"]["bundles/standard.yaml"] = "includes:\n  - profile:ds\n  - chem\n  - utils\n"
    del data["files"]["profiles/ds.yaml"]
    document = _load(data)
    with document_root(document) as root, pytest.raises(UvStackError) as caught:
        check_document(document, root)
    assert "uv-stack-document-" not in caught.value.message
    assert "ds" in caught.value.message


def test_bare_token_whose_profile_is_omitted_is_a_package(config_tree: ConfigRoot) -> None:
    data = _raw(config_tree, "main")
    data["files"]["envs/main/stack.txt"] = "@standard\nextra\n"
    document = _load(data)
    with document_root(document) as root:
        assert check_document(document, root) == []


def test_an_escaping_reference_is_refused_before_anything_outside_is_read(
    tmp_path: Path, config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    # The outside file is three parents above the tempdir's uv-stack-document-*
    outside = tmp_path / "outside.yaml"
    # Write deliberately malformed YAML so reading it would surface a parse error
    outside.write_text("includes: [unclosed\n")
    data = _raw(config_tree, "bundle:standard")
    data["files"]["bundles/standard.yaml"] = "includes:\n  - bundle:../../../outside\n"
    document = _load(data)
    with document_root(document) as root, pytest.raises(ConfigError) as caught:
        check_document(document, root)
    assert "does not map to a valid file key" in caught.value.message


def test_a_path_shaped_package_token_is_not_refused(config_tree: ConfigRoot) -> None:
    data = _raw(config_tree, "main")
    data["files"]["envs/main/stack.txt"] = "@standard\n./wheels/local.whl\n"
    document = _load(data)
    with document_root(document) as root:
        assert check_document(document, root) == []


def _case_insensitive(root: Path) -> bool:
    """Whether this filesystem resolves two spellings to one directory."""
    probe = root / "CaseProbe"
    probe.mkdir()
    try:
        return (root / "caseprobe").is_dir()
    finally:
        probe.rmdir()


@pytest.mark.parametrize("absolute", [False, True])
def test_bare_traversal_tokens_are_packages_not_probes(
    tmp_path: Path, config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch, absolute: bool
) -> None:
    """Bare tokens with path separators or absolute paths are packages, not probes."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    outside = tmp_path / "outside.yaml"
    outside.write_text("includes:\n  - rich\n")
    data = _raw(config_tree, "main")
    token = str(tmp_path / "outside") if absolute else "../../../outside"
    data["files"]["envs/main/stack.txt"] = f"@standard\n{token}\n"
    document = _load(data)
    with document_root(document) as root:
        check_document(document, root)


def test_case_aliased_keys_are_refused_at_materialization(
    config_tree: ConfigRoot,
) -> None:
    """Case-aliasing keys refuse at materialization time."""
    if not _case_insensitive(Path(tempfile.gettempdir())):
        pytest.skip("filesystem is case-sensitive")
    data = _raw(config_tree, "profile:utils")
    data["files"]["profiles/Utils.yaml"] = "includes:\n  - rich\n"
    document = _load(data)
    with pytest.raises(ConfigError) as caught:
        with document_root(document):
            pass
    assert "profiles/Utils.yaml" in caught.value.message


def test_a_bare_case_variant_resolves_to_an_unshipped_key(
    config_tree: ConfigRoot,
) -> None:
    """A bare reference that aliases a shipped name is refused."""
    if not _case_insensitive(Path(tempfile.gettempdir())):
        pytest.skip("filesystem is case-sensitive")
    data = _raw(config_tree, "main")
    data["files"]["envs/main/stack.txt"] = "@standard\nUTILS\n"
    document = _load(data)
    with document_root(document) as root, pytest.raises(ConfigError) as caught:
        check_document(document, root)
    assert "does not ship" in caught.value.message


@pytest.mark.parametrize(
    ("profile", "bundle", "refused"),
    [("utils", "UTILS", True), ("caf\u00e9", "cafe\u0301", True), ("utils", "utils", False)],
)
def test_a_profile_and_bundle_that_fold_together_are_refused(
    config_tree: ConfigRoot, tmp_path: Path, profile: str, bundle: str, refused: bool
) -> None:
    """A shipped pair that folds together is refused whatever this filesystem folds."""
    data = _raw(config_tree, "profile:utils")
    data["files"] = {
        f"profiles/{profile}.yaml": "includes:\n  - rich\n",
        f"bundles/{bundle}.yaml": f"includes:\n  - profile:{profile}\n",
    }
    data["items"] = sorted([f"profile:{profile}", f"bundle:{bundle}"])
    target = ConfigRoot(tmp_path / "target")
    target.root.mkdir()
    if not refused:
        refuse_shadowing(target, _load(data))
        return
    with pytest.raises(ConfigError) as caught:
        refuse_shadowing(target, _load(data))
    assert caught.value.message == (
        f"Profile '{profile}' would shadow the shipped bundle '{bundle}': "
        "the names differ only in letter case or Unicode normalization."
    )


def test_a_huge_integer_is_refused_as_invalid_json() -> None:
    """Huge integers that exceed the digit limit are refused."""
    data = '{"format": "uv-stack-export", "version": ' + "1" * 5000 + "}"
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_document(data)


def test_deeply_nested_json_is_refused() -> None:
    """Deeply nested JSON that triggers RecursionError is refused."""
    data = "[" * 100_000 + "]" * 100_000
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_document(data)


@pytest.mark.parametrize(
    "location",
    ["file_value", "file_key", "created_by"],
)
def test_lone_surrogates_are_refused(config_tree: ConfigRoot, location: str) -> None:
    """Lone surrogates in JSON escapes are detected and refused."""
    data = _raw(config_tree, "main")
    if location == "file_value":
        data["files"]["envs/main/stack.txt"] = "@standard\n\ud800\n"
    elif location == "file_key":
        data["files"]["profiles/\ud800.yaml"] = "includes:\n  - rich\n"
        data["items"].append("profile:\ud800")
    else:  # created_by
        data["created_by"] = "uv-stack \ud800 1.0"
    text = json.dumps(data)
    with pytest.raises(ConfigError, match="not valid Unicode"):
        load_document(text)


def test_deeply_nested_bundle_chain_is_refused() -> None:
    """A bundle chain too deep to resolve is refused."""
    depth = 1500
    files = {}
    for i in range(depth - 1):
        files[f"bundles/b{i}.yaml"] = f"includes:\n  - bundle:b{i+1}\n"
    files[f"bundles/b{depth-1}.yaml"] = "includes:\n  - numpy\n"

    data = {
        "format": "uv-stack-export",
        "version": 1,
        "created_by": "uv-stack 0.6.0",
        "source_platform": "test",
        "items": ["bundle:b0"],
        "files": files,
        "seeds": {},
    }
    document = load_document(json.dumps(data))
    with document_root(document) as root, pytest.raises(ConfigError, match="nest too deeply"):
        check_document(document, root)


def test_a_bundle_named_recursionerror_with_yaml_errors_is_not_misreported() -> None:
    """A bundle named RecursionError with YAML errors reports the YAML error."""
    data = {
        "format": "uv-stack-export",
        "version": 1,
        "created_by": "uv-stack 0.6.0",
        "source_platform": "test",
        "items": ["bundle:RecursionError"],
        "files": {"bundles/RecursionError.yaml": "includes: [\n"},
        "seeds": {},
    }
    document = load_document(json.dumps(data))
    with document_root(document) as root, pytest.raises(ConfigError) as caught:
        check_document(document, root)
    assert "Invalid YAML" in caught.value.message
    assert "nest too deeply" not in caught.value.message


@pytest.mark.parametrize("kind", ["profile", "env"])
def test_overlong_names_are_refused(kind: str) -> None:
    """Names too long for the filesystem are refused at materialization."""
    # 251 chars is too long for a filename; 300 is too long for a directory name
    long_name = "p" * 251 if kind == "profile" else "p" * 300
    if kind == "profile":
        data = {
            "format": "uv-stack-export",
            "version": 1,
            "created_by": "uv-stack 0.6.0",
            "source_platform": "test",
            "items": [f"profile:{long_name}"],
            "files": {f"profiles/{long_name}.yaml": "includes:\n  - numpy\n"},
            "seeds": {},
        }
    else:  # env
        data = {
            "format": "uv-stack-export",
            "version": 1,
            "created_by": "uv-stack 0.6.0",
            "source_platform": "test",
            "items": [f"env:{long_name}"],
            "files": {f"envs/{long_name}/stack.txt": "numpy\n"},
            "seeds": {},
        }
    document = load_document(json.dumps(data))
    with pytest.raises(ConfigError, match="cannot store") as caught:
        with document_root(document):
            pass
    assert "uv-stack-document-" not in caught.value.message


def test_overlong_bare_token_does_not_crash_on_stat(config_tree: ConfigRoot) -> None:
    """A bare token too long to stat is treated as a package, not OSError."""
    data = _raw(config_tree, "profile:ds", "env:main")
    long_token = "x" * 300
    data["files"]["envs/main/stack.txt"] = f"@standard\n{long_token}\n"
    document = _load(data)
    with document_root(document) as root:
        # Should not raise OSError; the long token is a package
        check_document(document, root)


def _stage(config: ConfigRoot, data: dict[str, Any], strict: bool = False) -> list[str]:
    document = _load(data)
    with document_root(document) as root:
        names = check_document(document, root)
    refuse_shadowing(config, document)
    with staged_root(config, document, names) as staged:
        return validate_staged(config, staged, document, ImportOptions(strict=strict))


def test_validator_refusal_names_the_target_path(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    (target.root / "profiles").mkdir(parents=True)
    data = _raw(config_tree, "profile:ds")
    data["files"]["profiles/ds.yaml"] = "includes: 3\n"
    with pytest.raises(ConfigError) as caught:
        _stage(target, data)
    assert str(target.root) in caught.value.message
    assert "uv-stack-" not in caught.value.message


def test_bundle_self_reference_is_refused(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    data = _raw(config_tree, "bundle:standard")
    data["files"]["bundles/standard.yaml"] = "includes:\n  - '@standard'\n  - profile:ds\n"
    # The edited bundle no longer reaches these, and phase 1 would refuse them
    # as unreachable before the validator ever saw the self-reference.
    del data["files"]["profiles/chem.yaml"]
    del data["files"]["profiles/utils.yaml"]
    with pytest.raises(UvStackError, match="cannot include itself"):
        _stage(target, data)


def test_trailing_backslash_entry_is_refused(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    data = _raw(config_tree, "main")
    data["files"]["envs/main/stack.txt"] = "@standard\nnumpy \\\n"
    with pytest.raises(ConfigError):
        _stage(target, data)


def test_strict_reaches_the_validators(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    data = _raw(config_tree, "bundle:qsar")
    _stage(target, data)
    with pytest.raises(UvStackError):
        _stage(target, data, strict=True)


def test_cross_kind_shadowing_is_refused(config_tree: ConfigRoot) -> None:
    data = _raw(config_tree, "profile:utils")
    data["files"] = {"profiles/standard.yaml": "includes:\n  - rich\n"}
    data["items"] = ["profile:standard"]
    with pytest.raises(ConfigError, match="would shadow the existing bundle"):
        refuse_shadowing(config_tree, _load(data))


def test_a_same_name_pair_arriving_together_is_allowed(config_tree: ConfigRoot) -> None:
    source = config_tree
    (source.bundles_dir / "utils.yaml").write_text("includes:\n  - profile:utils\n")
    data = _raw(source, "profile:utils", "bundle:utils")
    # Checked against the source, which holds both files: without the
    # arriving-together exemption each would shadow the other and raise.
    refuse_shadowing(source, _load(data))


def test_variables_are_appended_preserving_existing_lines(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.root.mkdir(parents=True)
    target.variables_path().write_text("# mine\nHOME_DIR")
    config_tree.variables_path().write_text("WORK\n")
    config_tree.profile_path("ds").write_text("includes:\n  - -e ${WORK}/pkg\n")
    document = _load(_raw(config_tree, "profile:ds"))
    with document_root(document) as root:
        names = check_document(document, root)
    with staged_root(target, document, names) as staged:
        assert staged.missing_variables == ["WORK"]
        assert staged.variables_text == "# mine\nHOME_DIR\nWORK\n"


def test_a_local_value_for_an_incoming_name_is_accepted(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.root.mkdir(parents=True)
    target.variables_path().write_text("HOME_DIR\n")
    target.variables_local_path().write_text("WORK=/w\n")
    config_tree.variables_path().write_text("WORK\n")
    config_tree.profile_path("ds").write_text("includes:\n  - -e ${WORK}/pkg\n")
    document = _load(_raw(config_tree, "profile:ds"))
    with document_root(document) as root:
        names = check_document(document, root)
    with staged_root(target, document, names) as staged:
        assert staged.missing_variables == ["WORK"]
        assert staged.root.load_variables().values["WORK"] == "/w"


def test_a_local_value_nobody_declares_names_the_target(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.root.mkdir(parents=True)
    target.variables_local_path().write_text("STRAY=/s\n")
    document = _load(_raw(config_tree, "profile:utils"))
    with pytest.raises(ConfigError) as caught:
        with staged_root(target, document, []):
            pass
    assert str(target.root) in caught.value.message
    assert "uv-stack-" not in caught.value.message


@pytest.mark.parametrize(("blocker", "item"), [("profiles", "profile:ds"), ("envs/main", "main")])
def test_a_file_where_a_directory_belongs_is_refused(
    config_tree: ConfigRoot, tmp_path: Path, blocker: str, item: str
) -> None:
    target = ConfigRoot(tmp_path / "target")
    (target.root / blocker).parent.mkdir(parents=True, exist_ok=True)
    (target.root / blocker).write_text("")
    pattern = f"^Not a directory: {re.escape(str(target.root / blocker))}$"
    with pytest.raises(ConfigError, match=pattern):
        _stage(target, _raw(config_tree, item))
    assert (target.root / blocker).is_file()


@pytest.mark.parametrize(
    ("links", "pattern"),
    [
        (
            ("envs/main", "envs/other"),
            r"^The document ships envs/main/(\S+) and envs/other/\1, "
            r"which are the same file on this machine\.$",
        ),
        (
            ("bundles", "profiles"),
            r"^The document ships bundles/utils\.yaml and profiles/utils\.yaml, "
            r"which are the same file on this machine\.$",
        ),
    ],
)
def test_shipped_keys_the_target_links_together_are_refused(
    config_tree: ConfigRoot, tmp_path: Path, links: tuple[str, str], pattern: str
) -> None:
    shutil.copytree(config_tree.env_dir("main"), config_tree.env_dir("other"))
    config_tree.bundle_path("utils").write_text("includes:\n  - profile:utils\n")
    data = _raw(config_tree, "main", "other", "profile:utils", "bundle:utils")
    target = ConfigRoot(tmp_path / "target")
    shared = tmp_path / "shared"
    shared.mkdir()
    for link in links:
        (target.root / link).parent.mkdir(parents=True, exist_ok=True)
        (target.root / link).symlink_to(shared, target_is_directory=True)
    with pytest.raises(ConfigError, match=pattern):
        _stage(target, data)


@pytest.mark.parametrize(("shipped", "linked"), [("other", "main"), ("main", "other")])
def test_a_shipped_env_linked_with_an_existing_env_is_refused(
    config_tree: ConfigRoot, tmp_path: Path, shipped: str, linked: str
) -> None:
    """Writing either name of two linked environments would change the other."""
    shutil.copytree(config_tree.env_dir("main"), config_tree.env_dir("other"))
    target = ConfigRoot(tmp_path / "target")
    shutil.copytree(config_tree.env_dir("main"), target.env_dir("main"))
    target.env_dir("other").symlink_to(target.env_dir("main"), target_is_directory=True)
    pattern = (
        rf"^The document ships envs/{shipped}/(\S+), which on this machine is the "
        rf"same file as envs/{linked}/\1\.$"
    )
    with pytest.raises(ConfigError, match=pattern):
        _stage(target, _raw(config_tree, shipped))


def test_an_existing_file_of_a_folded_name_is_not_a_link(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    """A folding filesystem's other spelling of a shipped name is that item, not a link."""
    target = ConfigRoot(tmp_path / "target")
    target.profiles_dir.mkdir(parents=True)
    target.profile_path("Utils").write_text("includes:\n  - rich\n")
    if not target.profile_path("utils").exists():
        pytest.skip("this filesystem does not fold letter case")
    _stage(target, _raw(config_tree, "profile:utils"))


def test_a_single_linked_directory_still_stages(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    shared = tmp_path / "shared"
    shared.mkdir()
    target.root.mkdir()
    (target.root / "profiles").symlink_to(shared, target_is_directory=True)
    _stage(target, _raw(config_tree, "profile:utils"))


def test_classify_new_identical_different(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    (target.root / "profiles").mkdir(parents=True)
    target.profile_path("ds").write_text(config_tree.profile_path("ds").read_text())
    target.profile_path("chem").write_text("includes:\n  - openeye\n")
    document = _load(_raw(config_tree, "profile:ds", "profile:chem", "profile:utils"))
    statuses = {c.key: c.status for c in classify_changes(target, document)}
    assert statuses == {"profiles/chem.yaml": "different", "profiles/ds.yaml": "identical",
                        "profiles/utils.yaml": "new"}


@pytest.mark.parametrize("filename", ["python.txt", "micromamba.txt", "channels.txt"])
@pytest.mark.parametrize("orphan", [False, True])
def test_target_only_optional_file_is_a_removal(
    config_tree: ConfigRoot, tmp_path: Path, filename: str, orphan: bool
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.env_dir("main").mkdir(parents=True)
    if not orphan:
        target.env_stack_path("main").write_text("@standard\n")
    (target.env_dir("main") / filename).write_text("x\n")
    (config_tree.env_dir("main") / filename).unlink()
    changes = classify_changes(target, _load(_raw(config_tree, "main")))
    removal = next(c for c in changes if c.key == f"envs/main/{filename}")
    assert (removal.status, removal.current, removal.incoming) == ("target-only", "x\n", None)


def test_non_regular_target_path_is_refused(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.profile_path("ds").mkdir(parents=True)
    with pytest.raises(ConfigError, match="Not a regular file"):
        classify_changes(target, _load(_raw(config_tree, "profile:ds")))


def test_change_diff_keeps_brackets_and_marks_missing_newline() -> None:
    change = FileChange("profiles/x.yaml", "different", "includes:\n  - a[b]", "includes:\n  - a\n")
    text = change_diff(change)
    assert "--- profiles/x.yaml (this machine)" in text
    assert "+++ profiles/x.yaml (incoming)" in text
    assert "-  - a[b]\n\\ No newline at end of file\n" in text


def test_used_by_follows_bundles(config_tree: ConfigRoot) -> None:
    keys = ["profiles/ds.yaml", "profiles/ghost.yaml", "bundles/standard.yaml", "bundles/qsar.yaml"]
    found, warnings = used_by(config_tree, keys)
    assert found == {"bundles/standard.yaml": ["main"], "profiles/ds.yaml": ["main"]}
    assert warnings == []


def test_used_by_warns_about_an_environment_it_cannot_read(config_tree: ConfigRoot) -> None:
    # A component longer than any filesystem allows makes the resolver's
    # profile probe raise OSError, as an unreadable file would.
    config_tree.env_dir("work").mkdir()
    config_tree.env_stack_path("work").write_text("x" * 300 + "\n")
    found, warnings = used_by(config_tree, ["profiles/ds.yaml"])
    assert found == {"profiles/ds.yaml": ["main"]}
    assert len(warnings) == 1
    assert warnings[0].startswith("Cannot tell what environment 'work' uses: ")
    assert str(config_tree.profiles_dir) in warnings[0]


def _folds_case(directory: Path) -> bool:
    """Whether this filesystem opens a file under another letter case."""
    probe = directory / "FoldProbe"
    probe.write_text("")
    try:
        return (directory / "foldprobe").exists()
    finally:
        probe.unlink()


def test_a_case_variant_reports_the_users_of_the_definition_it_replaces(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    if not _folds_case(tmp_path):
        pytest.skip("this filesystem does not fold letter case")
    # 'work' spells the profile differently again, so the match cannot be by
    # the target file's spelling either.
    config_tree.env_dir("work").mkdir()
    config_tree.env_stack_path("work").write_text("UTILS\n")
    data = _raw(config_tree, "profile:utils")
    data["items"] = ["profile:Utils"]
    data["files"] = {"profiles/Utils.yaml": "includes:\n  - rich\n  - typer\n"}
    expected = {"profiles/Utils.yaml": ["main", "work"]}
    with pytest.raises(ConflictError) as caught:
        _plan(config_tree, data)
    assert caught.value.used_by == expected
    plan = _plan(config_tree, data, overwrite=True, dry_run=True)
    assert plan.used_by == expected
    assert plan.dependents == ["main", "work"]


def _meanings(target: ConfigRoot, data: dict[str, Any]) -> None:
    document = _load(data)
    with document_root(document) as root:
        names = check_document(document, root)
        with staged_root(target, document, names) as staged:
            check_meanings(target, document, root, staged)


def test_direction_1_shipped_package_captured_by_target_profile(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    (target.root / "profiles").mkdir(parents=True)
    target.profile_path("requests[security]").write_text("includes:\n  - requests\n")
    data = _raw(config_tree, "main")
    data["files"]["envs/main/stack.txt"] = "@standard\nrequests[security]\n"
    with pytest.raises(ConfigError) as caught:
        _meanings(target, data)
    message = caught.value.message
    assert "'requests[security]'" in message and "the package 'requests[security]'" in message
    assert "profile 'requests[security]'" in message
    assert caught.value.hint is not None and "pkg:requests[security]" in caught.value.hint


def test_direction_2_target_package_captured_by_incoming_profile(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.env_dir("work").mkdir(parents=True)
    target.env_stack_path("work").write_text("utils\n")
    with pytest.raises(ConfigError, match="on this machine but would mean profile 'utils'"):
        _meanings(target, _raw(config_tree, "profile:utils"))


def test_a_replaced_env_old_token_is_not_checked(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.env_dir("main").mkdir(parents=True)
    target.env_stack_path("main").write_text("utils\n")
    _meanings(target, _raw(config_tree, "main", "profile:utils"))


def test_a_folded_replaced_env_old_token_is_not_checked(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.env_dir("Main").mkdir(parents=True)
    target.env_stack_path("Main").write_text("utils\n")
    if not target.env_dir("main").exists():
        pytest.skip("this filesystem does not fold letter case")
    _meanings(target, _raw(config_tree, "main", "profile:utils"))


def test_a_folded_replaced_bundle_old_token_is_not_checked(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.bundles_dir.mkdir(parents=True)
    target.bundle_path("STANDARD").write_text("includes:\n  - utils\n")
    if not target.bundle_path("standard").exists():
        pytest.skip("this filesystem does not fold letter case")
    _meanings(target, _raw(config_tree, "bundle:standard", "profile:utils"))


def test_blank_bundle_includes_are_skipped(config_tree: ConfigRoot, tmp_path: Path) -> None:
    config_tree.bundle_path("standard").write_text(
        "includes:\n  - '   '\n  - ds\n  - chem\n  - utils\n")
    target = ConfigRoot(tmp_path / "target")
    target.bundles_dir.mkdir(parents=True)
    target.bundle_path("spare").write_text("includes:\n  - ''\n  - rich\n")
    _meanings(target, _raw(config_tree, "bundle:standard"))


def test_an_unreadable_target_bundle_is_refused(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.bundles_dir.mkdir(parents=True)
    target.bundle_path("bad").write_text("includes: 3\n")
    with pytest.raises(ConfigError) as caught:
        _meanings(target, _raw(config_tree, "profile:utils"))
    assert str(target.bundle_path("bad")) in caught.value.message


def test_a_target_bundle_with_an_invalid_stem_is_still_checked(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.bundles_dir.mkdir(parents=True)
    target.bundle_path("-x").write_text("includes:\n  - utils\n")
    with pytest.raises(ConfigError, match="would mean profile 'utils'"):
        _meanings(target, _raw(config_tree, "profile:utils"))


def test_a_staged_os_error_names_the_target_path(config_tree: ConfigRoot, tmp_path: Path) -> None:
    # The target has no profiles/ to probe, so only the staged view, which the
    # shipped profile gives one, meets the overlong name.
    target = ConfigRoot(tmp_path / "target")
    target.env_dir("work").mkdir(parents=True)
    target.env_stack_path("work").write_text("x" * 300 + "\n")
    with pytest.raises(OSError) as caught:
        _meanings(target, _raw(config_tree, "profile:utils"))
    assert caught.value.filename == str(target.profiles_dir / ("x" * 300 + ".yaml"))
    assert "uv-stack-" not in str(caught.value)


def _plan(target: ConfigRoot, data: dict[str, Any], *, overwrite: bool = False,
          dry_run: bool = False) -> ImportPlan:
    with prepared_import(target, _load(data), ImportOptions(overwrite=overwrite),
                         dry_run=dry_run) as plan:
        return plan


def test_writes_go_in_kind_order_with_stack_txt_last(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []
    real = importing.atomic_write
    target = ConfigRoot(tmp_path / "target")

    def record(path: Path, text: str) -> None:
        written.append(path.relative_to(target.root).as_posix())
        real(path, text)

    monkeypatch.setattr(importing, "atomic_write", record)
    config_tree.variables_path().write_text("WORK\n")
    config_tree.profile_path("utils").write_text("includes:\n  - -e ${WORK}/x\n")
    _plan(target, _raw(config_tree, "main"))
    assert written[:3] == ["profiles/chem.yaml", "profiles/ds.yaml", "profiles/utils.yaml"]
    assert written[3:5] == ["bundles/standard.yaml", "variables.txt"]
    assert written[-1] == "envs/main/stack.txt"


def test_conflict_writes_nothing_and_overwrite_replaces(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.profile_path("ds").parent.mkdir(parents=True)
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    with pytest.raises(ConflictError) as caught:
        _plan(target, _raw(config_tree, "profile:ds"))
    assert [c.key for c in caught.value.conflicts] == ["profiles/ds.yaml"]
    assert target.profile_path("ds").read_text() == "includes:\n  - scipy\n"
    _plan(target, _raw(config_tree, "profile:ds"), overwrite=True)
    assert target.profile_path("ds").read_text() == config_tree.profile_path("ds").read_text()


def test_a_file_conflict_is_reported_before_a_meaning_change(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.profile_path("ds").parent.mkdir(parents=True)
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    target.env_dir("work").mkdir(parents=True)
    target.env_stack_path("work").write_text("utils\n")
    data = _raw(config_tree, "profile:ds", "profile:utils")
    with pytest.raises(ConflictError):
        _plan(target, data)
    with pytest.raises(ConfigError, match="would mean profile 'utils'"):
        _plan(target, data, overwrite=True)
    assert target.profile_path("ds").read_text() == "includes:\n  - scipy\n"


def test_a_conflict_carries_the_validation_and_used_by_warnings(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.profile_path("ds").parent.mkdir(parents=True)
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    target.env_dir("broken").mkdir(parents=True)
    target.env_stack_path("broken").write_text("profile:ghost\n")
    data = _raw(config_tree, "main")
    # 'util' is one letter from the shipped profile 'utils', which the
    # validator reports as a likely typo.
    data["files"]["envs/main/stack.txt"] = "@standard\nutil\n"
    with pytest.raises(ConflictError) as caught:
        _plan(target, data)
    warnings = caught.value.resolution_warnings
    assert any("'util' resolved to a literal package" in w for w in warnings)
    assert any(w.startswith("Cannot tell what environment 'broken' uses: ") for w in warnings)


def test_an_identical_bundle_whose_bare_token_a_new_profile_captures_has_dependents(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    config_tree.bundle_path("b").write_text("includes:\n  - x\n  - rich\n")
    config_tree.profile_path("x").write_text("includes:\n  - numpy\n")
    data = _raw(config_tree, "bundle:b")
    assert sorted(data["files"]) == ["bundles/b.yaml", "profiles/x.yaml"]
    target = ConfigRoot(tmp_path / "target")
    target.bundles_dir.mkdir(parents=True)
    target.bundle_path("b").write_text(config_tree.bundle_path("b").read_text())
    target.env_dir("main").mkdir(parents=True)
    target.env_stack_path("main").write_text("@b\n")
    plan = _plan(target, data, dry_run=True)
    assert {c.key: c.status for c in plan.changes} == {
        "bundles/b.yaml": "identical", "profiles/x.yaml": "new"}
    assert plan.dependents == ["main"]
    assert plan.used_by == {}


@pytest.mark.parametrize("outcome", ["refused", "dry-run", "overwrite"])
@pytest.mark.parametrize("orphan", [False, True])
@pytest.mark.parametrize("filename", ["python.txt", "micromamba.txt", "channels.txt"])
def test_a_target_only_env_file_through_the_whole_import(
    config_tree: ConfigRoot, tmp_path: Path, filename: str, orphan: bool, outcome: str
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.env_dir("main").mkdir(parents=True)
    if not orphan:
        target.env_stack_path("main").write_text("@standard\n")
    extra = target.env_dir("main") / filename
    extra.write_text("x\n")
    (config_tree.env_dir("main") / filename).unlink()
    data = _raw(config_tree, "main")
    key = f"envs/main/{filename}"
    if outcome == "refused":
        with pytest.raises(ConflictError) as caught:
            _plan(target, data)
        assert [(c.key, c.status) for c in caught.value.conflicts] == [(key, "target-only")]
        assert extra.read_text() == "x\n"
    elif outcome == "dry-run":
        plan = _plan(target, data, overwrite=True, dry_run=True)
        assert {c.key: c.status for c in plan.changes}[key] == "target-only"
        assert extra.read_text() == "x\n"
    else:
        _plan(target, data, overwrite=True)
        assert not extra.exists()
        assert target.env_stack_path("main").is_file()


def test_rerun_after_interruption_completes(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.env_dir("main").mkdir(parents=True)
    (target.env_dir("main") / "channels.txt").write_text("old\n")
    (config_tree.env_dir("main") / "channels.txt").unlink()
    calls = {"n": 0}
    real = importing.atomic_write

    def flaky(path: Path, text: str) -> None:
        calls["n"] += 1
        if path.name == "stack.txt":
            raise OSError(28, "No space left on device")
        real(path, text)

    monkeypatch.setattr(importing, "atomic_write", flaky)
    with pytest.raises(ConfigError, match=r"^Import interrupted after writing \d+ of \d+ file"):
        _plan(target, _raw(config_tree, "main"), overwrite=True)
    assert not (target.env_dir("main") / "channels.txt").exists()  # removal landed first
    monkeypatch.setattr(importing, "atomic_write", real)
    plan = _plan(target, _raw(config_tree, "main"), overwrite=True)
    statuses = {c.key: c.status for c in plan.changes}
    assert statuses["profiles/ds.yaml"] == "identical"
    assert statuses["envs/main/stack.txt"] == "new"
    assert "envs/main/channels.txt" not in statuses


def test_dry_run_writes_nothing_and_takes_no_locks(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    target.root.mkdir()
    target.locks_dir.mkdir(parents=True, exist_ok=True)
    with _lock_held_by_another_process(target.import_lock_path()):
        plan = _plan(target, _raw(config_tree, "main"), dry_run=True)
    assert {c.status for c in plan.changes} == {"new"}
    assert not target.env_stack_path("main").exists()


@pytest.mark.parametrize("which", ["import", "stem", "env"])
def test_a_held_lock_refuses(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    monkeypatch.setattr(fsutil, "_LOCK_TIMEOUT", 0.05)
    target = ConfigRoot(tmp_path / "target")
    target.locks_dir.mkdir(parents=True)
    path = {"import": target.import_lock_path(), "stem": target.stem_lock_path("ds"),
            "env": target.env_lock_path("main")}[which]
    with _lock_held_by_another_process(path), pytest.raises(UvStackError) as caught:
        _plan(target, _raw(config_tree, "main"))
    if which == "import":
        assert f"importing into '{target.root}'" in caught.value.message
    assert not target.env_stack_path("main").exists()


def test_profile_and_bundle_of_one_name_share_one_stem_lock(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fsutil, "_LOCK_TIMEOUT", 0.05)
    (config_tree.bundles_dir / "utils.yaml").write_text("includes:\n  - profile:utils\n")
    target = ConfigRoot(tmp_path / "target")
    _plan(target, _raw(config_tree, "profile:utils", "bundle:utils"))
    assert target.bundle_path("utils").is_file()


def test_locks_are_taken_in_one_order_and_held_through_the_block(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stems interleave profiles (chem, ds, utils) with bundles (qsar,
    # standard), so a profiles-then-bundles order would not pass as sorted.
    config_tree.env_dir("alpha").mkdir()
    config_tree.env_stack_path("alpha").write_text("profile:ds\n")
    target = ConfigRoot(tmp_path / "target")
    events: list[tuple[str, Path]] = []
    real = importing.name_lock

    @contextmanager
    def recording(path: Path, name: str, **kwargs: Any) -> Iterator[None]:
        with real(path, name, **kwargs):
            events.append(("acquire", path))
            yield
            events.append(("release", path))

    monkeypatch.setattr(importing, "name_lock", recording)
    with prepared_import(target, _load(_raw(config_tree)), ImportOptions(), dry_run=False):
        events.append(("block", target.root))
    stems = ["chem", "ds", "qsar", "standard", "utils"]
    order = [target.import_lock_path(), *map(target.stem_lock_path, stems),
             *map(target.env_lock_path, ["alpha", "main"])]
    assert events == [*(("acquire", p) for p in order), ("block", target.root),
                      *(("release", p) for p in reversed(order))]


def _probe(python: str | None) -> Callable[[Command], CommandResult]:
    def respond(cmd: Command) -> CommandResult:
        if cmd.args == micromamba_python_info("main").args:
            if python is None:
                return CommandResult(1, "")
            return CommandResult(0, f"/envs/main/bin/python\n{python}\n")
        return CommandResult(0, "")
    return respond


def _build_plan(
    target: ConfigRoot, data: dict[str, Any], *, recreate: bool = False,
    python: str | None = None, dry_run: bool = True
) -> ImportPlan:
    build = BuildRequest(RecordingRunner(responder=_probe(python)), recreate=recreate)
    with prepared_import(target, _load(data), ImportOptions(), dry_run=dry_run,
                         build=build) as plan:
        return plan


def test_preflight_refuses_a_missing_variable_value(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    config_tree.variables_path().write_text("WORK\n")
    config_tree.variables_local_path().write_text(f"WORK={tmp_path}\n")
    config_tree.profile_path("ds").write_text("includes:\n  - -e ${WORK}/pkg\n")
    target = ConfigRoot(tmp_path / "target")
    with pytest.raises(UvStackError) as caught:
        _build_plan(target, _raw(config_tree, "main"), dry_run=False)
    assert "WORK" in caught.value.message
    assert caught.value.hint is not None and "--no-build" in caught.value.hint
    assert not target.env_stack_path("main").exists()


def test_preflight_checks_a_relative_editable_against_the_working_directory(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # uv resolves a relative -e against its working directory, which uv-stack
    # leaves as the one stack runs in, so a checkout under the target root
    # does not satisfy the build. The brackets sit mid-path: editable_target
    # strips only a trailing extras suffix, so this also pins that the path is
    # reported intact.
    config_tree.profile_path("ds").write_text("includes:\n  - -e ./check[1]/x\n")
    target = ConfigRoot(tmp_path / "target")
    (target.root / "check[1]" / "x").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    with pytest.raises(ConfigError) as caught:
        _build_plan(target, _raw(config_tree, "main"))
    assert "./check[1]/x" in caught.value.message
    assert caught.value.hint is not None and "--no-build" in caught.value.hint
    (work / "check[1]" / "x").mkdir(parents=True)
    _build_plan(target, _raw(config_tree, "main"))


@pytest.mark.parametrize("flag", ["-e ", "-e="])
def test_preflight_refuses_a_missing_file_url_editable_before_writing(
    config_tree: ConfigRoot, tmp_path: Path, flag: str
) -> None:
    # A local file URL names a checkout as surely as a path does; passing over
    # it as remote let the definitions be written before uv failed on it.
    checkout = tmp_path / "co" / "pkg"
    config_tree.profile_path("ds").write_text(f"includes:\n  - {flag}{checkout.as_uri()}\n")
    target = ConfigRoot(tmp_path / "target")
    with pytest.raises(ConfigError) as caught:
        _build_plan(target, _raw(config_tree, "main"), dry_run=False)
    assert str(checkout) in caught.value.message
    assert caught.value.hint is not None and "--no-build" in caught.value.hint
    assert not target.profile_path("ds").exists()
    assert not target.env_stack_path("main").exists()
    checkout.mkdir(parents=True)
    assert _build_plan(target, _raw(config_tree, "main")).builds == [BuildStep("main", "create")]


def test_preflight_keeps_an_encoded_bracket_in_a_file_url_editable_path(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    # as_uri() percent-encodes the brackets, so the checkout is 'pkg[dev]';
    # reading them as extras would probe the present 'pkg' and let it pass.
    (tmp_path / "pkg").mkdir()
    checkout = tmp_path / "pkg[dev]"
    config_tree.profile_path("ds").write_text(f"includes:\n  - -e {checkout.as_uri()}\n")
    target = ConfigRoot(tmp_path / "target")
    with pytest.raises(ConfigError) as caught:
        _build_plan(target, _raw(config_tree, "main"), dry_run=False)
    assert str(checkout) in caught.value.message
    assert not target.profile_path("ds").exists()
    assert not target.env_stack_path("main").exists()


def test_preflight_checks_a_file_url_editable_after_expanding_its_variable(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    config_tree.variables_path().write_text("DEV\n")
    config_tree.profile_path("ds").write_text("includes:\n  - -e file://${DEV}/pkg\n")
    target = ConfigRoot(tmp_path / "target")
    target.root.mkdir(parents=True)
    target.variables_local_path().write_text(f"DEV={tmp_path / 'dev'}\n")
    with pytest.raises(ConfigError) as caught:
        _build_plan(target, _raw(config_tree, "main"))
    assert str(tmp_path / "dev" / "pkg") in caught.value.message


def test_preflight_reports_a_nul_in_a_tilde_user_editable(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    # os.path.expanduser resolves '~user' through pwd.getpwnam, which rejects
    # an embedded NUL with ValueError; test_doctor.py's
    # test_a_nul_in_a_tilde_user_entry_is_reported_not_raised pins the same
    # input for doctor's missing-checkout finding.
    config_tree.env_stack_path("main").write_text("@standard\n-e ~ab\0cd/widget\n")
    with pytest.raises(ConfigError) as caught:
        _build_plan(ConfigRoot(tmp_path / "target"), _raw(config_tree, "main"))
    assert "~ab\0cd/widget" in caught.value.message


def test_preflight_refuses_python_drift_unless_recreate(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = ConfigRoot(tmp_path / "target")
    with pytest.raises(ConfigError) as caught:
        _build_plan(target, _raw(config_tree, "main"), python="3.11.9")
    assert "runs Python 3.11.9, but the incoming python.txt requests 3.12" in caught.value.message
    assert caught.value.hint is not None and "--recreate" in caught.value.hint
    plan = _build_plan(target, _raw(config_tree, "main"), python="3.11.9", recreate=True)
    assert plan.builds == [BuildStep("main", "recreate")]


def test_python_drift_without_a_shipped_python_txt_names_the_default(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    config_tree.env_python_path("main").unlink()
    with pytest.raises(ConfigError) as caught:
        _build_plan(ConfigRoot(tmp_path / "target"), _raw(config_tree, "main"), python="3.11.9")
    assert ("runs Python 3.11.9, but the document ships no python.txt for it, so it "
            "defaults to 3.12.") in caught.value.message


def test_preflight_actions(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = ConfigRoot(tmp_path / "target")
    assert _build_plan(target, _raw(config_tree, "main")).builds == [BuildStep("main", "create")]
    plan = _build_plan(target, _raw(config_tree, "main"), python="3.12.4")
    assert plan.builds == [BuildStep("main", "sync")]


@pytest.mark.parametrize("dry_run", [True, False])
def test_recreate_refuses_a_non_plain_python(
    config_tree: ConfigRoot, tmp_path: Path, dry_run: bool
) -> None:
    (config_tree.env_dir("main") / "python.txt").write_text("3.12.*\n")
    target = ConfigRoot(tmp_path / "target")
    with pytest.raises(ConfigError):
        _build_plan(target, _raw(config_tree, "main"), recreate=True, dry_run=dry_run)
    assert not target.env_stack_path("main").exists()


def test_no_build_skips_preflight(config_tree: ConfigRoot, tmp_path: Path) -> None:
    config_tree.profile_path("ds").write_text("includes:\n  - -e /nowhere/pkg\n")
    assert _plan(ConfigRoot(tmp_path / "target"), _raw(config_tree, "main")).builds == []


def test_pin_report_counts_and_names(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text("numpy==1.26.4\nrich==14.0.0\nscipy==1.13.0\n")
    assert pin_report("main", "numpy==1.26.4\npandas==2.2.0\nrich==13.7.0\n", lock) == [
        "main: 1 pins kept, 1 changed, 1 dropped, 1 added",
        "  changed: rich 13.7.0 -> 14.0.0",
        "  dropped: pandas 2.2.0",
        "  added: scipy 1.13.0",
    ]
