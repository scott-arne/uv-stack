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
