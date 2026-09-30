"""Tests for stack export and stack import."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner, Result

from uv_stack.cli import cli
from uv_stack.config import ConfigRoot


def _invoke(root: ConfigRoot, *args: str, env: dict[str, str] | None = None,
            input: str | None = None) -> Result:
    return CliRunner().invoke(cli, ["--root", str(root.root), *args], env=env, input=input)


def test_export_writes_json_to_stdout(config_tree: ConfigRoot) -> None:
    result = _invoke(config_tree, "export", "env:main")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["items"] == ["env:main"]


def test_export_warning_goes_to_stderr_only(config_tree: ConfigRoot) -> None:
    config_tree.profile_path("ds").write_text("includes:\n  - -e /src/a\n")
    result = _invoke(config_tree, "export", "profile:ds")
    assert result.exit_code == 0
    json.loads(result.stdout)
    assert "profiles/ds.yaml: 1 absolute editable or local path(s)" in result.stderr


def test_export_output_option_writes_the_file(config_tree: ConfigRoot, tmp_path: Path) -> None:
    out = tmp_path / "doc.json"
    result = _invoke(config_tree, "export", "profile:ds", "-o", str(out))
    assert result.exit_code == 0 and result.stdout == ""
    assert json.loads(out.read_text())["items"] == ["profile:ds"]


def test_export_ambiguous_item_exits_2(config_tree: ConfigRoot) -> None:
    config_tree.env_dir("ds").mkdir()
    config_tree.env_stack_path("ds").write_text("@standard\n")
    result = _invoke(config_tree, "export", "ds")
    assert result.exit_code == 2
    assert "env:ds" in result.stderr and "profile:ds" in result.stderr


def test_export_bracketed_ambiguous_name_is_printed_intact(config_tree: ConfigRoot) -> None:
    config_tree.profile_path("x[1]").write_text("includes:\n  - rich\n")
    (config_tree.bundles_dir / "x[1].yaml").write_text("includes:\n  - rich\n")
    result = _invoke(config_tree, "export", "x[1]", env={"COLUMNS": "200"})
    assert result.exit_code == 2 and "profile:x[1]" in result.stderr


def test_export_missing_item_exits_1(config_tree: ConfigRoot) -> None:
    result = _invoke(config_tree, "export", "profile:ghost")
    assert result.exit_code == 1 and "ghost" in result.stderr


def _export(root: ConfigRoot, *items: str) -> str:
    result = _invoke(root, "export", *items)
    assert result.exit_code == 0, result.output
    return result.stdout


def _run_import(root: ConfigRoot, doc: str, *args: str) -> Result:
    return _invoke(root, "import", "-", *args, input=doc, env={"COLUMNS": "200"})


def _target(tmp_path: Path) -> ConfigRoot:
    target = ConfigRoot(tmp_path / "target")
    target.root.mkdir()
    return target


def _flat_import(result: Result) -> str:
    """Output with rich's error-panel borders removed and whitespace collapsed."""
    return " ".join(result.output.replace("│", " ").split())


def test_import_installs_definitions(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = _target(tmp_path)
    result = _run_import(target, _export(config_tree, "main"))
    assert result.exit_code == 0, result.output
    assert "Importing 1 item(s) exported by uv-stack 0.6.0 on " in result.output
    assert target.env_stack_path("main").read_text() == "@standard\n"
    assert "new: profiles/ds.yaml" in result.output


def test_import_conflict_prints_diff_and_users(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    target.profile_path("ds").write_text("includes:\n  - requests[security]\n")
    result = _run_import(target, _export(config_tree, "profile:ds"))
    assert result.exit_code == 1
    assert "-  - requests[security]" in result.output
    assert "profiles/ds.yaml is used by: main" in result.output
    assert "nothing was written" in result.output
    assert target.profile_path("ds").read_text() == "includes:\n  - requests[security]\n"


def test_import_overwrite_replaces(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    result = _run_import(target, _export(config_tree, "profile:ds"), "--overwrite")
    assert result.exit_code == 0, result.output
    assert "replace: profiles/ds.yaml" in result.output
    assert target.profile_path("ds").read_text() == config_tree.profile_path("ds").read_text()


def test_import_dry_run_lists_a_removal_and_writes_nothing(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    (config_tree.env_dir("main") / "python.txt").unlink()
    result = _run_import(target, _export(config_tree, "main"), "--overwrite", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "remove: envs/main/python.txt" in result.output
    assert "Dry run: nothing written." in result.output
    assert (target.env_dir("main") / "python.txt").exists()


def test_import_meaning_change_is_refused_even_with_overwrite(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = _target(tmp_path)
    target.env_dir("work").mkdir(parents=True)
    target.env_stack_path("work").write_text("utils\n")
    result = _run_import(target, _export(config_tree, "profile:utils"), "--overwrite")
    assert result.exit_code == 1 and "would mean profile 'utils'" in _flat_import(result)
    assert not target.profile_path("utils").exists()


def test_import_reports_a_platform_difference(config_tree: ConfigRoot, tmp_path: Path) -> None:
    data = json.loads(_export(config_tree, "profile:ds"))
    data["source_platform"] = "plan9-mips"
    result = _run_import(_target(tmp_path), json.dumps(data))
    assert "exported by uv-stack 0.6.0 on plan9-mips." in result.output
    assert "pins re-resolved where this platform differs." in result.output


def test_import_bad_document_exits_1(tmp_path: Path) -> None:
    result = _run_import(_target(tmp_path), "{")
    assert result.exit_code == 1 and "not valid JSON" in result.output


def test_import_refuses_a_nul_in_a_name_without_writing(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    data = json.loads(_export(config_tree, "profile:ds"))
    data["items"] = sorted([*data["items"], "profile:a\0b"])
    data["files"]["profiles/a\0b.yaml"] = "includes:\n  - rich\n"
    target = _target(tmp_path)
    result = _run_import(target, json.dumps(data))
    assert result.exit_code == 1 and "invalid file key" in result.output
    # CliRunner also exits 1 on an uncaught ValueError; SystemExit proves the
    # refusal went through the error renderer rather than a traceback.
    assert isinstance(result.exception, SystemExit)
    assert list(target.root.iterdir()) == []
