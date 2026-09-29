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
    check_document,
    document_root,
    load_document,
    read_document,
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
