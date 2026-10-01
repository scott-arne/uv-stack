"""Tests for stack export and stack import."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

import uv_stack.cli.transfer_cmd
import uv_stack.cli.upgrade
import uv_stack.operations.importing
from tests.test_operations import _compile_output
from uv_stack import __version__, fsutil
from uv_stack.cli import cli
from uv_stack.commands import micromamba_create, micromamba_python_info, micromamba_python_path
from uv_stack.config import ConfigRoot
from uv_stack.errors import ToolError, UvStackError
from uv_stack.fsutil import name_lock
from uv_stack.operations.upgrade import UpgradeOptions
from uv_stack.runner import Command, CommandResult, RecordingRunner


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


def _run_import(root: ConfigRoot, doc: str, *args: str, build: bool = False) -> Result:
    extra = [] if build else ["--no-build"]
    return _invoke(root, "import", "-", *args, *extra, input=doc, env={"COLUMNS": "200"})


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
    assert f"Importing 1 item(s) exported by uv-stack {__version__} on " in result.output
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


def test_import_conflict_prints_why_used_by_may_be_incomplete(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    target.env_dir("broken[1]").mkdir()
    target.env_stack_path("broken[1]").write_text("profile:ghost\n")
    result = _run_import(target, _export(config_tree, "profile:ds"))
    assert result.exit_code == 1
    assert "profiles/ds.yaml is used by: main" in result.output
    assert "warning: Cannot tell what environment 'broken[1]' uses: " in result.stderr


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
    assert f"exported by uv-stack {__version__} on plan9-mips." in result.output
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
    assert result.exit_code == 1 and "holding a control character" in result.output
    # CliRunner also exits 1 on an uncaught ValueError; SystemExit proves the
    # refusal went through the error renderer rather than a traceback.
    assert isinstance(result.exception, SystemExit)
    assert list(target.root.iterdir()) == []


def test_import_refuses_a_control_character_in_the_header_without_printing_it(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    # The header echoes created_by, and click passes an escape sequence
    # through to a terminal, where it could rewrite what the dry run shows.
    data = json.loads(_export(config_tree, "profile:ds"))
    data["created_by"] = f"uv-stack {__version__}\x1b[2K"
    result = _run_import(_target(tmp_path), json.dumps(data), "--dry-run")
    assert result.exit_code != 0
    assert "\x1b" not in result.output and "\x1b" not in result.stderr


def test_import_refuses_a_schema_error_without_printing_a_control_character(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    # Schema validation fails before the control-character check, so its own
    # message is what reaches the terminal.
    data = json.loads(_export(config_tree, "profile:ds"))
    data["files"]["profiles/a\x1b[2K.yaml"] = 5
    result = _run_import(_target(tmp_path), json.dumps(data), "--dry-run")
    assert result.exit_code != 0
    assert "\x1b" not in result.output and "\x1b" not in result.stderr
    assert "profiles/a\\x1b[2K.yaml" in result.output


def _fake_build(monkeypatch: pytest.MonkeyPatch, target: ConfigRoot, *,
                envs: tuple[str, ...] = ("main",), fail: tuple[str, ...] = (),
                lock_text: str = "numpy==1.26.4\nrich==14.0.0\n") -> dict[str, bool | str]:
    # Each env is absent until micromamba creates it, so pre-flight plans a
    # create and upgrade_env's interpreter probe then finds a real path.
    state: dict[str, bool | str] = {"locked": False}
    created: set[str] = set()

    def respond(cmd: Command) -> CommandResult:
        for name in envs:
            if cmd.args == micromamba_create(target.env_environment_yml(name)).args:
                created.add(name)
            if cmd.args == micromamba_python_path(name).args:
                if name in created:
                    return CommandResult(0, f"/envs/{name}/bin/python\n")
                return CommandResult(1, "")
            if cmd.args == micromamba_python_info(name).args:
                return CommandResult(1, "")
        if cmd.args[:3] == ["uv", "pip", "compile"]:
            if _compile_output(cmd).parent.name in fail:
                raise ToolError("uv pip compile failed.", command=cmd.args, returncode=1)
            monkeypatch.setattr(fsutil, "_LOCK_TIMEOUT", 0.05)
            try:
                with name_lock(target.import_lock_path(), "probe"):
                    pass
            except UvStackError:
                state["locked"] = True
            candidate = _compile_output(cmd)
            state["candidate"] = candidate.read_text() if candidate.exists() else ""
            candidate.write_text(lock_text)
        return CommandResult(0, "")

    runner = RecordingRunner(responder=respond)
    monkeypatch.setattr(uv_stack.cli.upgrade, "SubprocessRunner", lambda: runner)
    monkeypatch.setattr(uv_stack.cli.transfer_cmd, "SubprocessRunner", lambda: runner)
    return state


def test_import_builds_under_the_locks_and_reports_pins(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_tree.env_requirements_lock("main").write_text(
        "numpy==1.26.4\npandas==2.2.0\nrich==13.7.0\n")
    target = _target(tmp_path)
    state = _fake_build(monkeypatch, target)
    result = _run_import(target, _export(config_tree, "main"), build=True)
    assert result.exit_code == 0, result.output
    assert state["locked"] is True
    assert state["candidate"] == "numpy==1.26.4\npandas==2.2.0\nrich==13.7.0\n"
    assert "main: 1 pins kept, 1 changed, 1 dropped, 0 added" in result.output


@pytest.mark.skipif(not fsutil._LOCK_AVAILABLE, reason="requires fcntl")
def test_import_builds_in_the_tree_it_resolved(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The link is retargeted as the import locks are taken and stays that way,
    # so a build or pin report through the unresolved root would find no
    # environment.
    config_tree.env_requirements_lock("main").write_text(
        "numpy==1.26.4\npandas==2.2.0\nrich==13.7.0\n")
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    link = tmp_path / "root"
    link.symlink_to(first)
    real = uv_stack.operations.importing.name_lock

    @contextmanager
    def retargeting(*args: Any, **kwargs: Any) -> Iterator[None]:
        link.unlink()
        link.symlink_to(second)
        with real(*args, **kwargs):
            yield

    monkeypatch.setattr(uv_stack.operations.importing, "name_lock", retargeting)
    resolved = ConfigRoot(first)
    _fake_build(monkeypatch, resolved)
    doc = _export(config_tree, "main")
    result = _run_import(ConfigRoot(link), doc, build=True)
    assert result.exit_code == 0, result.output
    assert resolved.env_requirements_lock("main").read_text() == "numpy==1.26.4\nrich==14.0.0\n"
    assert "main: 1 pins kept, 1 changed, 1 dropped, 0 added" in result.output
    assert list(second.iterdir()) == []


def test_import_without_a_seed_prints_no_pin_report(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path)
    state = _fake_build(monkeypatch, target)
    result = _run_import(target, _export(config_tree, "main"), build=True)
    assert result.exit_code == 0, result.output
    assert state["candidate"] == ""
    assert "pins kept" not in result.output


def test_build_failure_keeps_definitions_and_prints_rerun_and_dependents(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    target.env_dir("work").mkdir(parents=True)
    target.env_stack_path("work").write_text("profile:ds\n")
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    _fake_build(monkeypatch, target, fail=("main",))
    result = _run_import(target, _export(config_tree, "main"), "--overwrite", build=True)
    assert result.exit_code == 1
    assert "stack sync env main" in result.output
    assert "Not rebuilt, but using changed definitions: work." in result.output
    assert target.profile_path("ds").read_text() == config_tree.profile_path("ds").read_text()


def test_dependents_line_prints_with_no_build(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    target.env_dir("work").mkdir(parents=True)
    target.env_stack_path("work").write_text("profile:ds\n")
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    result = _run_import(target, _export(config_tree, "main"), "--overwrite")
    assert result.exit_code == 0, result.output
    assert "Not rebuilt, but using changed definitions: work." in result.output


def test_recreate_with_no_build_is_a_usage_error(config_tree: ConfigRoot, tmp_path: Path) -> None:
    result = _run_import(_target(tmp_path), _export(config_tree, "main"), "--recreate")
    # Click also exits 2 on an unknown option; the text proves the refusal.
    assert result.exit_code == 2 and "cannot be combined" in result.output


def test_missing_bracketed_editable_is_printed_intact(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_tree.profile_path("ds").write_text("includes:\n  - -e ./check[1]/x\n")
    target = _target(tmp_path)
    _fake_build(monkeypatch, target)
    result = _run_import(target, _export(config_tree, "main"), build=True)
    assert result.exit_code == 1 and "./check[1]/x" in result.output
    assert not target.env_stack_path("main").exists()


def test_import_strict_refuses_a_name_that_falls_through_to_a_package(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    # qsar's bare umap-learn names no profile or bundle; a profile's own
    # entries are never checked, as they reference nothing.
    target = _target(tmp_path)
    doc = _export(config_tree, "bundle:qsar")
    result = _run_import(target, doc, "--strict")
    assert result.exit_code == 1
    assert "Unqualified token 'umap-learn' resolved to a literal package." \
        in _flat_import(result)
    assert not target.bundle_path("qsar").exists()
    assert _run_import(target, doc).exit_code == 0


def test_import_strict_reaches_the_build(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, package in (("ds", "numpy"), ("chem", "rdkit"), ("utils", "rich")):
        config_tree.profile_path(name).write_text(f"includes:\n  - pkg:{package}\n")
    target = _target(tmp_path)
    _fake_build(monkeypatch, target)
    seen: list[UpgradeOptions] = []
    real = uv_stack.cli.transfer_cmd._run_upgrade

    def spy(config: ConfigRoot, names: list[str], options: UpgradeOptions, **kwargs: Any) -> None:
        seen.append(options)
        real(config, names, options, **kwargs)

    monkeypatch.setattr(uv_stack.cli.transfer_cmd, "_run_upgrade", spy)
    result = _run_import(target, _export(config_tree, "main"), "--strict", build=True)
    assert result.exit_code == 0, result.output
    assert [options.strict for options in seen] == [True]


def test_import_prints_the_variable_names_it_declares(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    config_tree.variables_path().write_text("WORK\n")
    config_tree.profile_path("ds").write_text("includes:\n  - -e ${WORK}/pkg\n")
    target = _target(tmp_path)
    result = _run_import(target, _export(config_tree, "profile:ds"))
    assert result.exit_code == 0, result.output
    assert "  declare in variables.txt: WORK\n" in result.output
    assert "WORK" in target.variables_path().read_text().splitlines()


def test_import_lists_an_identical_file(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = _target(tmp_path)
    doc = _export(config_tree, "profile:ds")
    _run_import(target, doc)
    result = _run_import(target, doc)
    assert result.exit_code == 0, result.output
    assert "  identical: profiles/ds.yaml\n" in result.output


def test_dependents_line_prints_on_a_dry_run(config_tree: ConfigRoot, tmp_path: Path) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    target.env_dir("work").mkdir(parents=True)
    target.env_stack_path("work").write_text("profile:ds\n")
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    result = _run_import(target, _export(config_tree, "main"), "--overwrite", "--dry-run")
    assert result.exit_code == 0, result.output
    assert ("Not rebuilt, but using changed definitions: work. "
            "Rebuild them with 'stack sync env work'.") in result.output


@pytest.mark.parametrize(("flags", "action"), [((), "create"), (("--recreate",), "recreate")])
def test_dry_run_lists_each_build_step(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    flags: tuple[str, ...], action: str,
) -> None:
    target = _target(tmp_path)
    state = _fake_build(monkeypatch, target)
    result = _run_import(target, _export(config_tree, "main"), "--dry-run", *flags, build=True)
    assert result.exit_code == 0, result.output
    assert f"  build: main ({action})\n" in result.output
    assert "candidate" not in state and not target.env_stack_path("main").exists()


# Rich reads '[x]' as a style tag and drops it, while '[1]' is not a tag and
# survives either way, so these use letters to prove the text is not markup.


def test_import_header_and_plan_keep_bracketed_text(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    config_tree.profile_path("p[x]").write_text("includes:\n  - rich\n")
    data = json.loads(_export(config_tree, "profile:p[x]"))
    data["created_by"] = "uv-stack [dev]"
    data["source_platform"] = "plan[x]-mips"
    result = _run_import(_target(tmp_path), json.dumps(data))
    assert result.exit_code == 0, result.output
    assert "exported by uv-stack [dev] on plan[x]-mips.\n" in result.output
    assert "  new: profiles/p[x].yaml\n" in result.output


def test_import_used_by_line_keeps_bracketed_names(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    config_tree.profile_path("p[x]").write_text("includes:\n  - rich\n")
    target = _target(tmp_path)
    target.profiles_dir.mkdir()
    target.profile_path("p[x]").write_text("includes:\n  - scipy\n")
    target.env_dir("e[x]").mkdir(parents=True)
    target.env_stack_path("e[x]").write_text("profile:p[x]\n")
    result = _run_import(target, _export(config_tree, "profile:p[x]"))
    assert result.exit_code == 1
    assert "profiles/p[x].yaml is used by: e[x]\n" in result.output


def test_pin_report_keeps_bracketed_text(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_tree.env_requirements_lock("main").write_text(
        "numpy==1.26.4\nwidget @ file:///w/widget[x].whl\n")
    target = _target(tmp_path)
    _fake_build(monkeypatch, target, lock_text=(
        "numpy==1.26.4\n-e ./src[x]/w\nwidget @ file:///w/widget[y].whl\n"))
    result = _run_import(target, _export(config_tree, "main"), build=True)
    assert result.exit_code == 0, result.output
    assert "  changed: widget file:///w/widget[x].whl -> file:///w/widget[y].whl\n" \
        in result.output
    assert "  added: -e ./src[x]/w\n" in result.output


def test_a_mixed_build_reruns_only_the_failure_and_reports_the_success(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("alpha", "beta"):
        config_tree.env_dir(name).mkdir()
        config_tree.env_stack_path(name).write_text("profile:ds\n")
        config_tree.env_requirements_lock(name).write_text("numpy==1.26.4\n")
    target = _target(tmp_path)
    _fake_build(monkeypatch, target, envs=("alpha", "beta"), fail=("beta",),
                lock_text="numpy==1.26.4\n")
    result = _run_import(target, _export(config_tree, "env:alpha", "env:beta"), build=True)
    assert result.exit_code == 1
    assert "alpha: 1 pins kept, 0 changed, 0 dropped, 0 added\n" in result.output
    assert not any(line.startswith("beta: ") for line in result.output.splitlines())
    assert "Re-run the failed build(s) once the cause is fixed: stack sync env beta\n" \
        in result.output


# Everything import prints comes from another machine's document or from
# names on this machine's disk, and a terminal acts on ESC and C1 characters.
# A YAML double-quoted "\e" is how an escape reaches a parsed value, since
# PyYAML refuses a raw ESC anywhere in the stream.


def _assert_escaped(result: Result, escaped: str) -> None:
    """Assert no raw ESC or CSI reached either stream, and the escape did."""
    text = result.output + result.stderr
    assert "\x1b" not in text and "\x9b" not in text
    assert escaped in text


def test_missing_editable_with_an_escape_is_refused_cleanly(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_tree.profile_path("ds").write_text('includes:\n  - "-e /missing\\e[2Kpkg"\n')
    target = _target(tmp_path)
    _fake_build(monkeypatch, target)
    result = _run_import(target, _export(config_tree, "main"), build=True)
    assert result.exit_code == 1 and "missing on this machine" in result.output
    _assert_escaped(result, "/missing\\x1b[2Kpkg")


def test_pin_report_escapes_a_seed_entry(
    config_tree: ConfigRoot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_tree.env_requirements_lock("main").write_text(
        "numpy==1.26.4\nwidget @ file:///w/widget\x1b[2K.whl\n")
    target = _target(tmp_path)
    _fake_build(monkeypatch, target, lock_text="numpy==1.26.4\nwidget @ file:///w/w2.whl\n")
    result = _run_import(target, _export(config_tree, "main"), build=True)
    assert result.exit_code == 0, result.output
    _assert_escaped(result, "  changed: widget file:///w/widget\\x1b[2K.whl -> file:///w/w2.whl\n")


def test_used_by_and_dependents_lines_escape_an_environment_name(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    try:
        target.env_dir("a\x1bb").mkdir()
    except OSError:
        pytest.skip("this filesystem refuses an escape in a file name")
    target.env_stack_path("a\x1bb").write_text("profile:ds\n")
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    doc = _export(config_tree, "main")
    conflict = _run_import(target, doc)
    assert conflict.exit_code == 1
    _assert_escaped(conflict, "profiles/ds.yaml is used by: a\\x1bb")
    replaced = _run_import(target, doc, "--overwrite")
    assert replaced.exit_code == 0, replaced.output
    _assert_escaped(replaced, "Not rebuilt, but using changed definitions: a\\x1bb.")


# Import keeps newlines and tabs for its diff's layout, so a name from this
# machine's disk holding one would otherwise start a line of its own choosing.
_LAYOUT_NAMES = pytest.mark.parametrize(
    ("name", "escaped"), [("a\nb", "a\\nb"), ("a\tb", "a\\tb")], ids=["newline", "tab"]
)


@_LAYOUT_NAMES
def test_used_by_and_dependents_lines_escape_a_layout_character_in_a_name(
    config_tree: ConfigRoot, tmp_path: Path, name: str, escaped: str
) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    target.env_dir(name).mkdir()
    target.env_stack_path(name).write_text("profile:ds\n")
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    doc = _export(config_tree, "main")
    conflict = _run_import(target, doc)
    assert conflict.exit_code == 1
    assert f"profiles/ds.yaml is used by: {escaped}, main\n" in conflict.output
    replaced = _run_import(target, doc, "--overwrite")
    assert replaced.exit_code == 0, replaced.output
    assert (f"Not rebuilt, but using changed definitions: {escaped}. "
            f"Rebuild them with 'stack sync env '{escaped}''.\n") in replaced.output


@_LAYOUT_NAMES
def test_used_by_warning_escapes_a_layout_character_in_a_name(
    config_tree: ConfigRoot, tmp_path: Path, name: str, escaped: str
) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    target.profile_path("ds").write_text("includes:\n  - scipy\n")
    target.env_dir(name).mkdir()
    target.env_stack_path(name).write_text("profile:ghost\n")
    result = _run_import(target, _export(config_tree, "profile:ds"))
    assert result.exit_code == 1
    assert f"warning: Cannot tell what environment '{escaped}' uses: " in result.stderr


def test_shipped_profile_with_an_escape_in_a_key_is_refused_cleanly(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    data = json.loads(_export(config_tree, "profile:ds"))
    data["files"]["profiles/ds.yaml"] = '"evil\\e[2J": 1\n'
    target = _target(tmp_path)
    result = _run_import(target, json.dumps(data))
    assert result.exit_code == 1 and "Invalid profile config" in _flat_import(result)
    _assert_escaped(result, "evil\\x1b[2J")


def test_conflict_diff_escapes_content_but_keeps_its_lines(
    config_tree: ConfigRoot, tmp_path: Path
) -> None:
    target = _target(tmp_path)
    _run_import(target, _export(config_tree, "main"))
    config_tree.env_dir("main").joinpath("micromamba.txt").write_text("graphviz\x1b[2Kx\n")
    result = _run_import(target, _export(config_tree, "main"))
    assert result.exit_code == 1
    _assert_escaped(result, "\n-graphviz\n+graphviz\\x1b[2Kx\n")
