"""Tests for stack import's phases."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError, UvStackError
from uv_stack.operations.export import build_document, serialize_document
from uv_stack.operations.importing import (
    ImportOptions,
    check_document,
    document_root,
    load_document,
    read_document,
    refuse_shadowing,
    staged_root,
    validate_staged,
)


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


def test_bad_json_is_refused() -> None:
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_document("{")


def test_a_nul_in_a_document_name_is_refused(config_tree: ConfigRoot) -> None:
    # JSON carries \u0000 and validate_name allows it, but document_root would
    # raise ValueError building a path from it; the item and file agree, so
    # only the key check can stop it.
    data = _raw(config_tree, "profile:ds")
    data["items"] = sorted([*data["items"], "profile:a\0b"])
    data["files"]["profiles/a\0b.yaml"] = "includes:\n  - rich\n"
    with pytest.raises(ConfigError) as caught:
        _load(data)
    assert "invalid file key" in caught.value.message


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
