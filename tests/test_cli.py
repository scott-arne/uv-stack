from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest
import rich_click
from click.testing import CliRunner

from uv_stack.cli import cli


def _row_cells(output: str, name: str) -> list[str]:
    """Return the trimmed cell values of the rich table row containing ``name``.

    Splits the matching line on the box-drawing column separator so a count
    assertion targets the intended column rather than matching a digit that
    appears incidentally elsewhere in the row (e.g. the ``3.12`` Python cell).
    """
    line = next(line for line in output.splitlines() if name in line)
    return [cell.strip() for cell in line.strip().strip("│").split("│")]


def _combined_output(result) -> str:
    """stdout plus stderr, tolerant of click versions that separate them."""
    try:
        return result.output + result.stderr
    except (ValueError, AttributeError):
        return result.output


def _flat_panel(result) -> str:
    """Output with rich's error-panel borders removed and whitespace collapsed.

    ``render_error`` prints a :class:`~rich.panel.Panel`, so a long message is
    folded at the console width and every row is padded out to the border.
    Stripping newlines alone is not enough — the ``│`` and the padding remain
    between the halves of a split message. Callers that assert a contiguous
    string containing an absolute path also need a width wide enough that the
    path is not broken mid-token, since no normalisation can rejoin that
    without also collapsing a genuine space.
    """
    return " ".join(_combined_output(result).replace("│", " ").split())


def _seeded_root(tmp_path: Path) -> Path:
    """A config root with the profiles and bundles the CLI tests reference.

    Mirrors the ``config_tree`` fixture: profiles ds/chem/utils and a
    ``standard`` bundle (ds chem utils). ``config init`` no longer seeds any
    profiles or bundles, so the tests author the ones they need.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    cfg.profile_path("ds").write_text(
        "description: Core data-science stack with numpy scipy pandas polars and friends\n"
        "tags: [data, core]\n"
        "includes:\n  - numpy\n  - pandas\n"
    )
    cfg.profile_path("chem").write_text(
        "description: Cheminformatics\ntags: [chem, bio]\nincludes:\n  - rdkit\n"
    )
    cfg.profile_path("utils").write_text("includes:\n  - rich\n")
    cfg.bundle_path("standard").write_text(
        "description: Everything for daily work\n"
        "tags: [core]\n"
        "includes:\n  - ds\n  - chem\n  - utils\n"
    )
    return root


def _env_root(tmp_path: Path) -> Path:
    from uv_stack.config import ConfigRoot

    root = _seeded_root(tmp_path)
    cfg = ConfigRoot(root)
    env_dir = cfg.env_dir("main")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    (env_dir / "stack.txt").write_text("@standard\n")
    (env_dir / "micromamba.txt").write_text("graphviz\n")
    (env_dir / "channels.txt").write_text("bioconda\n")
    return root


def _two_failing_envs_root(tmp_path: Path) -> Path:
    """A config root with two envs whose stacks fail at resolution.

    Each stack references an explicit, missing profile, so ``upgrade`` raises a
    ResolutionError during the pure resolve step before any subprocess call —
    keeping the batch-behavior tests hermetic.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    for name in ("alpha", "beta"):
        env_dir = cfg.env_dir(name)
        env_dir.mkdir(parents=True)
        (env_dir / "python.txt").write_text("3.12\n")
        (env_dir / "stack.txt").write_text("profile:ghost\n")
    return root


# ---------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "stack, version 0.4.4" in result.output


def test_config_init_reports_the_locks_directory(tmp_path: Path):
    """The printed list is init_config_root's return value, so .locks appears in it."""
    root = tmp_path / "fresh-root"
    result = CliRunner().invoke(cli, ["--root", str(root), "config", "init"])
    assert result.exit_code == 0
    assert str(root / ".locks") in _combined_output(result)
    assert (root / ".locks").is_dir()


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------


def test_upgrade_dry_run(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "upgrade", "--dry-run", "main"]
    )
    assert result.exit_code == 0
    assert "compile" in result.output
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    assert cfg.env_requirements_in("main").is_file()
    assert not cfg.env_requirements_lock("main").is_file()


def test_upgrade_all_dry_run_targets_every_env(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "--all", "--dry-run"])
    # Both envs fail at resolution, so dry-run exits nonzero.
    assert result.exit_code == 1
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" in result.output


def test_upgrade_all_does_not_prompt(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    # No stdin supplied: an unconditional confirm() prompt would abort the batch.
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "--all"])
    assert result.exit_code == 1
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" in result.output


def test_upgrade_all_rejects_explicit_names(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "--all", "alpha"])
    assert result.exit_code == 2
    assert "--all cannot be combined" in result.output


def test_upgrade_all_no_envs(tmp_path: Path):
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    init_config_root(ConfigRoot(root))
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "--all"])
    assert result.exit_code == 0
    assert "No environments discovered." in result.output


def test_upgrade_batch_continues_on_failure_and_summarizes(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "alpha", "beta"])
    assert result.exit_code == 1
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" in result.output
    assert "Summary" in result.output
    assert "2 of 2 environment(s) failed." in result.output
    # Each ✗ row is annotated with a one-line reason drawn from the error.
    summary = result.output.split("Summary", 1)[1]
    assert "Missing profile" in summary


def test_upgrade_stop_on_error_aborts_after_first(tmp_path: Path):
    root = _two_failing_envs_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "upgrade", "--stop-on-error", "alpha", "beta"]
    )
    assert result.exit_code == 1
    assert "Upgrading alpha" in result.output
    assert "Upgrading beta" not in result.output


def test_upgrade_dry_run_strict_exits_nonzero_on_failure(tmp_path: Path):
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    env_dir = cfg.env_dir("test")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    # "numpyy" is a near-miss typo that will fail strict mode.
    (env_dir / "stack.txt").write_text("numpyy\n")
    result = CliRunner().invoke(
        cli, ["--root", str(root), "upgrade", "--dry-run", "--strict", "test"]
    )
    assert result.exit_code == 1


def test_upgrade_refuses_python_version_drift_cli(tmp_path: Path, monkeypatch):
    """CLI prints the hint when upgrade refuses on Python version drift."""
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.upgrade.SubprocessRunner",
        lambda: _FakeProbeRunner(stdout="/envs/main/bin/python\n3.13.1\n"),
    )
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "main"])
    assert result.exit_code == 1
    # Error message mentions both versions.
    output = _combined_output(result)
    assert "3.13.1" in output
    assert "3.12" in output
    # Hint text reaches the user.
    assert "--recreate" in output


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_env_passes_create_option(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    captured: dict = {}

    def fake_run_upgrade(config, names, options, *, stop_on_error=False):
        captured["names"] = names
        captured["options"] = options

    monkeypatch.setattr("uv_stack.cli.create._run_upgrade", fake_run_upgrade)
    result = CliRunner().invoke(cli, ["--root", str(root), "create", "env", "main"])
    assert result.exit_code == 0
    assert captured["names"] == ["main"]
    assert captured["options"].create is True
    assert captured["options"].recreate is False


def test_create_env_recreate_passes_recreate_option(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    captured: dict = {}

    def fake_run_upgrade(config, names, options, *, stop_on_error=False):
        captured["options"] = options

    monkeypatch.setattr("uv_stack.cli.create._run_upgrade", fake_run_upgrade)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "create", "env", "main", "--recreate"]
    )
    assert result.exit_code == 0
    assert captured["options"].recreate is True
    assert captured["options"].create is False


def test_create_no_subcommand_shows_help():
    result = CliRunner().invoke(cli, ["create"])
    assert result.exit_code == 2
    assert "env" in result.output
    assert "project" in result.output


def test_create_project_help(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["create", "project", "--help"])
    assert result.exit_code == 0
    assert "--python" in result.output
    assert "--no-sync" in result.output
    # The --python help advertises micromamba env-name support.
    assert "micromamba" in result.output
    # rich-click parses option help as markup; the bracketed table name must
    # survive rather than being eaten as an unknown style tag.
    assert "[tool.uv-stack]" in result.output


@pytest.mark.parametrize(
    ("name", "tokens"),
    [("daily", ["pkg:numpy", "daily"]), ("app", ["app"])],
)
def test_create_bundle_refuses_self_reference(tmp_path: Path, name: str, tokens: list[str]):
    """The bug this fix exists for: `stack create bundle app app` wrote a dead bundle."""
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "create", "bundle", name, *tokens])
    assert result.exit_code == 1
    assert "cannot include itself" in _flat_panel(result)
    assert not (root / "bundles" / f"{name}.yaml").exists()


def test_enumerated_kind_help_qualifies_shared_environment():
    """'environment' must be qualified wherever 'project' shares the sentence.

    Adjudicated during the 0.4.0 terminology work: the vocabulary rule governs
    these enumerations, so a bare 'environment' beside 'project' is a defect.
    Asserted against the module docstrings and the command's help attribute
    rather than rendered output, which rich wraps at the console width.
    """
    from uv_stack.cli import create as create_mod
    from uv_stack.cli import show as show_mod

    assert "shared environment" in (create_mod.__doc__ or "")
    assert "shared environment" in (show_mod.__doc__ or "")
    assert "shared environment" in (cli.commands["create"].help or "")


def test_create_bundle_surfaces_cycle_warnings_from_existing_bundles(tmp_path: Path):
    """A cycle inside an already-existing referenced bundle must reach the user."""
    from uv_stack.config import ConfigRoot

    root = _seeded_root(tmp_path)
    cfg = ConfigRoot(root)
    cfg.bundle_path("loop").write_text("includes:\n  - '@loop'\n  - pkg:rich\n")
    result = CliRunner().invoke(
        cli, ["--root", str(root), "create", "bundle", "app", "@loop"]
    )
    assert result.exit_code == 0
    assert "Bundle cycle skipped: loop -> loop" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_env(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "env"])
    assert result.exit_code == 0
    assert "main" in result.output
    assert "Python" in result.output
    assert "Stack" in result.output
    assert "3.12" in result.output
    cells = _row_cells(result.output, "main")
    assert cells[-1] == "1"  # @standard -> one stack token


def test_list_profile_columns(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "list", "profile"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0
    assert "Packages" in result.output
    assert "Tags" in result.output
    assert "Description" in result.output
    cells = _row_cells(result.output, "chem")
    assert cells[0] == "chem"
    assert cells[1] == "1"
    assert "chem" in cells[2] and "bio" in cells[2]
    assert cells[3] == "Cheminformatics"


def test_list_profile_truncates_long_description(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "list", "profile"], env={"COLUMNS": "200"}
    )
    cells = _row_cells(result.output, "ds")
    assert cells[-1].endswith("…")
    assert len(cells[-1]) <= 40


def test_list_bundle_renames_count_to_entries(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "list", "bundle"], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0
    assert "Entries" in result.output
    assert "Tokens" not in result.output
    cells = _row_cells(result.output, "standard")
    assert cells[0] == "standard"
    assert cells[1] == "3"


def test_list_profile_tag_filter(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "list", "profile", "--tag", "data"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "ds" in result.output
    assert "rdkit" not in result.output
    assert "chem" not in result.output
    assert "utils" not in result.output


def test_list_profile_tag_filter_or(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli,
        ["--root", str(root), "list", "profile", "--tag", "data", "--tag", "bio"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    assert "ds" in result.output
    assert "chem" in result.output
    assert "utils" not in result.output


def test_list_env_tag_is_usage_error(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "env", "--tag", "x"])
    assert result.exit_code == 2
    assert "tag" in result.output.lower()


def test_list_bad_kind(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "widget"])
    assert result.exit_code == 2
    assert "widget" in result.output


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def test_show_env(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "env", "main"])
    assert result.exit_code == 0
    assert "Environment: main" in result.output
    assert "Channels:" in result.output
    assert "conda-forge" in result.output
    assert "bioconda" in result.output


def test_show_env_defaults_to_main(tmp_path: Path):
    root = _env_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "env"])
    assert result.exit_code == 0
    assert "Environment: main" in result.output


def test_show_profile(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "profile", "ds"])
    assert result.exit_code == 0
    assert "Profile: ds" in result.output


def test_show_bundle(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "bundle", "standard"])
    assert result.exit_code == 0
    assert "Bundle: standard" in result.output


def test_show_profile_requires_name(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "profile"])
    assert result.exit_code == 2
    assert "NAME" in result.output


def test_show_bad_kind(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "widget", "x"])
    assert result.exit_code == 2
    assert "widget" in result.output


def test_show_missing_env_errors(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "env", "ghost"])
    assert result.exit_code == 1
    assert "ghost" in result.output


# ---------------------------------------------------------------------------
# resolve / doctor
# ---------------------------------------------------------------------------


def test_resolve_classifies_without_expanding(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "resolve", "standard", "ds", "numpy"]
    )
    assert result.exit_code == 0
    lines = result.output.split()
    assert lines == ["bundle:standard", "profile:ds", "package:numpy"]


def test_resolve_full_expands_to_flat_packages(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(
        cli, ["--root", str(root), "resolve", "--full", "standard"]
    )
    assert result.exit_code == 0
    # standard -> ds (numpy, pandas), chem (rdkit), utils (rich); no prefixes.
    lines = result.output.split()
    assert lines == ["numpy", "pandas", "rdkit", "rich"]


def test_resolve_prints_near_miss_warning(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "resolve", "standrd"])
    assert result.exit_code == 0
    assert "package:standrd" in result.output
    assert "did you mean 'standard'" in _combined_output(result)


def test_resolve_strict_fails_on_bare_literal(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "resolve", "--strict", "numpyy"]
    )
    assert result.exit_code == 1


def test_resolve_full_strict_and_warnings(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "resolve", "--full", "standrd"]
    )
    assert result.exit_code == 0
    assert "did you mean 'standard'" in _combined_output(result)


def test_doctor_clean(tmp_path: Path):
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "doctor"])
    assert result.exit_code == 0
    assert "No problems detected" in result.output


def test_doctor_reports_missing_dirs(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    result = CliRunner().invoke(cli, ["--root", str(root), "doctor"])
    assert result.exit_code == 0
    assert "Missing" in result.output


# ---------------------------------------------------------------------------
# error arms
# ---------------------------------------------------------------------------


def test_bare_oserror_renders_a_panel_instead_of_a_traceback(tmp_path: Path, monkeypatch):
    """Every uncaught OSError below the CLI edge becomes the same panel shape."""
    import errno

    from uv_stack.cli import doctor as cli_doctor

    bracketed = str(tmp_path / "[tool.uv-stack]")

    def boom(config):
        raise OSError(errno.ENOSPC, "No space left on device", bracketed)

    monkeypatch.setattr(cli_doctor, "diagnose", boom)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(tmp_path), "doctor"])
    assert result.exit_code == 1
    panel = _flat_panel(result)
    assert "No space left on device" in panel
    assert "ENOSPC" in panel
    # Assembled as Text, so the bracketed path is not parsed as rich markup
    # and appears on the Path: line. Long paths wrap, so check that the
    # bracketed filename follows the Path: label in the flattened output.
    assert "Path:" in panel
    assert "[tool.uv-stack]" in panel
    assert panel.index("Path:") < panel.index("[tool.uv-stack]")
    assert "Traceback" not in panel


def test_broken_pipe_exits_quietly_with_the_signal_status(tmp_path: Path, monkeypatch):
    """`stack list | head` is ordinary shell usage, not an error to render."""
    from uv_stack.cli import doctor as cli_doctor

    def boom(config):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(cli_doctor, "diagnose", boom)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(tmp_path), "doctor"])
    assert result.exit_code == 128 + int(signal.SIGPIPE)
    assert "uv-stack error" not in _flat_panel(result)


def test_broken_pipe_falls_back_to_status_1_without_sigpipe(tmp_path: Path, monkeypatch):
    """signal.SIGPIPE is POSIX-only; reaching for it unguarded would raise.

    CliRunner reports exit_code == 1 for both a clean sys.exit(1) and an
    uncaught AttributeError, so the exception type is checked to separate them.
    """
    from uv_stack.cli import doctor as cli_doctor

    def boom(config):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(cli_doctor, "diagnose", boom)
    monkeypatch.delattr(signal, "SIGPIPE", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(tmp_path), "doctor"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "uv-stack error" not in _flat_panel(result)


def test_uvstackerror_arm_is_unaffected_by_the_new_arms(tmp_path: Path):
    """The pre-existing arm still renders its own panel, ahead of the OSError one.

    UvStackError is not an OSError, so ordering cannot break this — but the arm
    is now one of three, and a regression here would be silent otherwise.
    """
    root = _seeded_root(tmp_path)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "env", "ghost"])
    assert result.exit_code == 1
    panel = _flat_panel(result)
    assert "uv-stack error" in panel
    assert "ghost" in panel


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_list_into_a_closed_pipe_exits_without_a_panel(tmp_path: Path):
    """The real `stack list | head`, not a simulated one.

    ``CliRunner`` cannot produce this: it hands the command an in-memory buffer,
    so no write ever meets a closed pipe and the ``dup2`` has no descriptor to
    redirect. Only a child process writing down a real pipe exercises the arm
    end to end — and the shutdown flush the ``dup2`` exists to protect happens
    after the arm returns, so nothing short of a separate interpreter can
    observe whether it was protected.
    """
    import subprocess
    import sys

    root = _seeded_root(tmp_path)
    # Close the read end *before* the child starts: a pipe with no reader fails
    # every write immediately, which makes this deterministic. Handing the child
    # a live reader and closing it afterwards would race — `list profile` output
    # fits in the pipe buffer, so the child would finish before the close.
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from uv_stack.cli import main; main()",
                "--root",
                str(root),
                "list",
                "profile",
            ],
            stdout=write_fd,
            stderr=subprocess.PIPE,
        )
    finally:
        os.close(write_fd)

    stderr = proc.stderr.decode()
    assert proc.returncode == 128 + int(signal.SIGPIPE)
    assert "uv-stack error" not in stderr
    assert "Traceback" not in stderr
    # The dup2's whole job: without it the interpreter's shutdown flush meets
    # the same dead pipe and CPython prints this to stderr, exit status 120.
    assert "Exception ignored" not in stderr


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_broken_pipe_from_stderr_exits_with_signal_status(tmp_path: Path):
    """A broken pipe originating on stderr also exits 141, not 120.

    When a BrokenPipeError is raised by error_console writing to a closed
    stderr, the arm must redirect both stdout and stderr. Without redirecting
    stderr, the interpreter's exit flush meets the same dead pipe and turns
    the arm's intended exit code 141 into 120 with an "Exception ignored"
    message on stderr. This test pins that both streams are redirected.
    """
    import subprocess
    import sys

    root = _seeded_root(tmp_path)
    # Close the read end of the stderr pipe before the child starts, but pass
    # stdout to a normal PIPE — we want the break to originate from stderr.
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        # Drive render_warnings through a throwaway command registered on the
        # real group to exercise the real chain: render_warnings ->
        # error_console -> _ConsoleWithBrokenPipePropagation -> arm.
        driver = (
            "from uv_stack.cli import cli, main\n"
            "from uv_stack.cli._render import render_warnings\n"
            "@cli.command('emit-warning')\n"
            "def _emit_warning():\n"
            "    render_warnings(['pipe probe'])\n"
            "main()\n"
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                driver,
                "--root",
                str(root),
                "emit-warning",
            ],
            stdout=subprocess.PIPE,
            stderr=write_fd,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode == 128 + int(signal.SIGPIPE)


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_broken_pipe_from_stderr_survives_stdout_closed_at_startup(tmp_path: Path):
    """The redirect handles a stream CPython replaced with None, not a stream.

    Closing fd 1 before the interpreter starts leaves sys.stdout as None, so the
    redirect's stream.fileno() has nothing to call. AttributeError is not one of
    the shapes the redirect suppresses, so unguarded it escapes the arm entirely
    and the shutdown flush of the still-dead stderr turns 141 into 120. Both
    conditions are needed: with a live stderr nothing raises BrokenPipeError, so
    the arm never runs and the redirect is never reached (rc 0).
    """
    import subprocess
    import sys

    root = _seeded_root(tmp_path)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        driver = (
            "import sys\n"
            "from uv_stack.cli import cli, main\n"
            "from uv_stack.cli._render import render_warnings\n"
            "@cli.command('emit-warning')\n"
            "def _emit_warning():\n"
            "    render_warnings(['pipe probe'])\n"
            # Guard against silent vacuity: if `>&-` ever stopped producing a
            # None stdout, the ordinary two-stream redirect would still exit
            # 141 and this test would pass while covering nothing.
            "assert sys.stdout is None\n"
            "main()\n"
        )
        # subprocess cannot hand a child a *closed* fd 1 — passing None or
        # DEVNULL both leave it open — so go through the shell, whose `>&-`
        # closes it before exec. That is what makes CPython set sys.stdout None.
        proc = subprocess.run(
            [
                "/bin/sh",
                "-c",
                'exec "$1" -c "$2" --root "$3" emit-warning >&-',
                "sh",
                sys.executable,
                driver,
                str(root),
            ],
            stderr=write_fd,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode == 128 + int(signal.SIGPIPE)


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_broken_pipe_while_rendering_the_error_panel_exits_with_signal_status(tmp_path: Path):
    """A break met while rendering the UvStackError panel exits 141, not 1.

    render_error writes through error_console, whose on_broken_pipe re-raises so
    the edge can apply the signal status — but it runs *inside* the
    `except UvStackError` arm, and Python does not dispatch an exception raised
    in an except block to a sibling arm of the same try. Without the arm's own
    inner catch the break escapes invoke() and rich-click's EPIPE arm exits 1.

    The status is the whole assertion because stderr is the dead pipe here:
    a traceback or an "Exception ignored" flush failure has nowhere to be
    printed. Both are still visible in it — an escaped exception exits 1, a
    failed shutdown flush exits 120 — and a command that stopped raising at all
    would exit 0.
    """
    import subprocess
    import sys

    root = _seeded_root(tmp_path)
    # Dead before the child starts, as in the sibling tests: a pipe with no
    # reader fails every write immediately, which makes this deterministic.
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from uv_stack.cli import main; main()",
                "--root",
                str(root),
                "show",
                "env",
                "ghost",
            ],
            stdout=subprocess.PIPE,
            stderr=write_fd,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode == 128 + int(signal.SIGPIPE)


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_broken_pipe_while_rendering_the_os_error_panel_exits_with_signal_status(tmp_path: Path):
    """The bare-OSError arm nests its renderer for the same reason.

    render_os_error runs inside `except OSError`, so a break met while printing
    that panel escapes the same way the UvStackError one does. The raised errno
    is EACCES, which Python maps to PermissionError — an OSError that is not a
    BrokenPipeError, so it reaches the arm under test rather than the pipe arm
    above it.
    """
    import subprocess
    import sys

    root = _seeded_root(tmp_path)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        # A throwaway command on the real group, as in the render_warnings test
        # above, so the whole chain runs: renderer -> error_console ->
        # _ConsoleWithBrokenPipePropagation -> arm.
        driver = (
            "from uv_stack.cli import cli, main\n"
            "@cli.command('explode')\n"
            "def _explode():\n"
            "    raise OSError(13, 'Permission denied', '/nope')\n"
            "main()\n"
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                driver,
                "--root",
                str(root),
                "explode",
            ],
            stdout=subprocess.PIPE,
            stderr=write_fd,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode == 128 + int(signal.SIGPIPE)


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_help_into_a_closed_pipe_exits_without_exception_ignored():
    """Help output into a closed pipe exits cleanly, not with status 120.

    rich-click emits --help from an eager parameter callback, before
    Group.invoke runs — so the BrokenPipeError arm in invoke never sees it — and
    it writes with the builtin print(), which does not flush. Without the guard
    in main(), the rendered help text sits in stdout's buffer at interpreter
    shutdown, where CPython's final flush meets the dead pipe and prints
    "Exception ignored on flushing sys.stdout" to stderr, exit status 120. The
    guard flushes before shutdown so the BrokenPipeError can be caught.

    CliRunner cannot reproduce this: it hands the command an in-memory buffer,
    so no write ever meets a closed pipe and there is no shutdown flush to
    protect. Only a child process writing down a real pipe exercises the guard.
    """
    import subprocess
    import sys

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    # Remove PYTHONUNBUFFERED so the test pins the buffered-shutdown path the
    # guard was written for. With stdout unbuffered, the pipe break surfaces
    # during rendering and Click's own EPIPE arm handles it (rc 1, clean stderr).
    # The product behavior is still correct there, but this test is verifying
    # the guard's behavior specifically, so hand the child a sanitized env.
    env = os.environ.copy()
    env.pop("PYTHONUNBUFFERED", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "from uv_stack.cli import main; main()", "--help"],
            stdout=write_fd,
            stderr=subprocess.PIPE,
            env=env,
        )
    finally:
        os.close(write_fd)

    stderr = proc.stderr.decode()
    assert proc.returncode == 128 + int(signal.SIGPIPE)
    assert "Exception ignored" not in stderr
    assert "Traceback" not in stderr


def test_help_into_a_closed_pipe_falls_back_to_status_1_without_sigpipe():
    """The main() guard's no-SIGPIPE fallback works for help output.

    signal.SIGPIPE is POSIX-only; reaching for it unguarded would replace a
    clean exit with an AttributeError on the non-POSIX platforms the fallback
    exists for. The invoke arm's no-SIGPIPE fallback has a dedicated test
    (test_broken_pipe_falls_back_to_status_1_without_sigpipe); this pins the
    main() arm's equivalent.

    Reading the status before redirecting stderr to devnull is what makes a
    broken fallback loud. Reverse that order and an unguarded status expression
    also exits 1 in silence, indistinguishable from the intended fallback and
    passing every assertion below. In the order the arm actually uses, the
    AttributeError reaches the live stderr and the shutdown flush of the
    still-dead stdout turns the status into 120, so all three assertions fire.

    Popping PYTHONUNBUFFERED is what makes the assertions mean anything. Status
    1 with clean stderr is also the signature of rich-click's own EPIPE arm,
    which is what handles help output when stdout is unbuffered: the write
    reaches the fd immediately, so the break surfaces inside the help callback
    and the guard's flush finds nothing left to fail on. Only with stdout
    buffered does the help text survive to the guard, so only then does this
    test observe the arm it names. The child pops rather than deletes SIGPIPE
    so it does not raise on a platform that never had it.
    """
    import subprocess
    import sys

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    env = os.environ.copy()
    env.pop("PYTHONUNBUFFERED", None)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                'import signal; signal.__dict__.pop("SIGPIPE", None); '
                "from uv_stack.cli import main; main()",
                "--help",
            ],
            stdout=write_fd,
            stderr=subprocess.PIPE,
            env=env,
        )
    finally:
        os.close(write_fd)

    stderr = proc.stderr.decode()
    assert proc.returncode == 1
    assert "Exception ignored" not in stderr
    assert "Traceback" not in stderr


def test_a_broken_pipe_at_exit_does_not_swallow_an_internal_exception(tmp_path: Path):
    """The flush guard yields to a genuine bug instead of taking its status.

    A command that leaves bytes in stdout's buffer and then raises a
    non-SystemExit exception, with stdout dead but stderr live, must still
    report itself. Replacing an in-flight *SystemExit* with the signal status is
    the guard's job — `stack list | head` depends on it — but doing the same to
    a RuntimeError exits 141 with the traceback destroyed on a stderr that
    would have shown it.

    All three assertions are load-bearing, because each failure mode has a
    distinct signature. Taking over the status gives 141 and empty stderr. No
    guard at all leaves the buffer for the interpreter's shutdown flush, which
    meets the dead pipe, prints "Exception ignored on flushing sys.stdout" and
    exits 120 — so the surviving redirect of the broken stream is pinned too.
    Only yielding to the exception while redirecting the stream that actually
    broke gives 1 with the traceback on stderr.

    Buffered stdout is the precondition, hence the PYTHONUNBUFFERED pop: written
    straight through, the print() itself would break inside the command and the
    invoke arm would exit 141, which these assertions also reject rather than
    passing vacuously on.
    """
    import subprocess
    import sys

    root = _seeded_root(tmp_path)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    env = os.environ.copy()
    env.pop("PYTHONUNBUFFERED", None)
    try:
        # builtin print(), not click.echo: echo flushes, so the break would
        # surface inside the command rather than at the guard.
        driver = (
            "from uv_stack.cli import cli, main\n"
            "@cli.command('explode')\n"
            "def _explode():\n"
            "    print('buffered output')\n"
            "    raise RuntimeError('a real internal bug')\n"
            "main()\n"
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                driver,
                "--root",
                str(root),
                "explode",
            ],
            stdout=write_fd,
            stderr=subprocess.PIPE,
            env=env,
        )
    finally:
        os.close(write_fd)

    stderr = proc.stderr.decode()
    assert proc.returncode == 1
    assert "RuntimeError: a real internal bug" in stderr
    assert "Exception ignored" not in stderr


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_a_broken_pipe_at_exit_takes_over_when_stderr_cannot_report(tmp_path: Path):
    """`stack explode 2>&1 | head`: with nowhere to report, the pipe wins again.

    The test above yields to the RuntimeError because stderr is live and can
    print its traceback. Here both streams are the same dead pipe, and stderr
    flushing cleanly proves nothing: it is line-buffered, so its buffer is empty
    by the time the guard runs, and an empty buffer flushes cleanly down a pipe
    with no reader. It therefore never joins the broken list, and the guard has
    to ask the descriptor itself whether it could still report.

    Skip that question and the guard yields as it does above, the interpreter
    writes the traceback to the dead pipe, and the shutdown flush that follows
    fails: status 120, which this assertion rejects. Status 1 would mean the
    exception kept a status nobody could read the traceback for, and 0 that the
    command stopped raising at all.
    """
    import subprocess
    import sys

    root = _seeded_root(tmp_path)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    env = os.environ.copy()
    env.pop("PYTHONUNBUFFERED", None)
    try:
        driver = (
            "import sys\n"
            "from uv_stack.cli import cli, main\n"
            "@cli.command('explode')\n"
            "def _explode():\n"
            "    print('buffered output')\n"
            "    raise RuntimeError('a real internal bug')\n"
            # Guard against silent vacuity: unbuffered, the print() breaks
            # inside the command and the invoke arm exits 141 too, so the
            # assertion below would pass without the guard running at all.
            "assert not sys.stdout.write_through\n"
            "main()\n"
        )
        # One dead pipe for both streams, as `2>&1 | head` produces.
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                driver,
                "--root",
                str(root),
                "explode",
            ],
            stdout=write_fd,
            stderr=write_fd,
            env=env,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode == 128 + int(signal.SIGPIPE)


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_a_broken_pipe_at_exit_takes_over_when_stderr_is_gone(tmp_path: Path):
    """The same takeover, for the stderr CPython replaces with None.

    Closing fd 2 before the interpreter starts leaves sys.stderr as None. That
    is not a stream that happens to be silent but no stream at all, so the
    can-report probe answers False without a descriptor to ask and the pipe
    status wins.

    Have it answer True instead and the guard yields to the RuntimeError, whose
    traceback CPython then has nowhere to print: status 1, which this assertion
    rejects.
    """
    import subprocess
    import sys

    root = _seeded_root(tmp_path)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    env = os.environ.copy()
    env.pop("PYTHONUNBUFFERED", None)
    try:
        driver = (
            "import sys\n"
            "from uv_stack.cli import cli, main\n"
            "@cli.command('explode')\n"
            "def _explode():\n"
            "    print('buffered output')\n"
            "    raise RuntimeError('a real internal bug')\n"
            # Both guard against silent vacuity: unbuffered, the print() breaks
            # inside the command and the invoke arm exits 141 without the guard
            # running, and a stderr that was still a stream would reach the
            # descriptor probe rather than the None branch this test covers.
            "assert not sys.stdout.write_through\n"
            "assert sys.stderr is None\n"
            "main()\n"
        )
        # subprocess cannot hand a child a *closed* fd 2, so `2>&-` in the shell
        # closes it before exec, the way the stdout sibling above uses `>&-`.
        proc = subprocess.run(
            [
                "/bin/sh",
                "-c",
                'exec "$1" -c "$2" --root "$3" explode 2>&-',
                "sh",
                sys.executable,
                driver,
                str(root),
            ],
            stdout=write_fd,
            env=env,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode == 128 + int(signal.SIGPIPE)


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_a_broken_pipe_printing_a_usage_error_exits_with_the_signal_status(tmp_path: Path):
    """A break met while printing a usage error is still just a broken pipe.

    rich-click renders a usage error with print(..., file=sys.stderr) from
    inside its own `except ClickException` arm, so a break there cannot reach
    that arm's sibling EPIPE arm — the same shape the inner catches in
    UvStackGroup.invoke exist for — and a bare BrokenPipeError reaches the flush
    guard in main(). Treated there as a real exception it keeps its own status
    and exits 1, which this assertion rejects: the break is the pipe closing,
    not a bug to report.

    Two clauses of the takeover condition each cover this case, so removing
    either one alone still exits 141: the in-flight BrokenPipeError is admitted
    outright, and this stderr also fails the can-report probe. The test below
    takes the probe away, as Windows does, to pin the first clause on its own.

    Popping PYTHONUNBUFFERED is again what makes the assertion mean anything.
    Written straight through, the failed write leaves stderr's buffer empty, the
    guard's flush of it succeeds, and the BrokenPipeError propagates out of
    main() untouched for status 1 — the same behavior as before this guard
    existed, and not what this test is about.
    """
    import subprocess
    import sys

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    env = os.environ.copy()
    env.pop("PYTHONUNBUFFERED", None)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from uv_stack.cli import main; main()",
                "--root",
                str(tmp_path),
                "nosuchcommand",
            ],
            stdout=subprocess.PIPE,
            stderr=write_fd,
            env=env,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode == 128 + int(signal.SIGPIPE)


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_a_broken_pipe_printing_a_usage_error_exits_141_without_poll(tmp_path: Path):
    """Without the descriptor probe, the in-flight break alone must still decide.

    select.poll is POSIX-only, so on Windows _stream_can_report fails open and
    calls even a dead stderr reportable — deliberately, since a probe that
    cannot run may not subtract confidence. The other clause is what
    keeps the status right there: a BrokenPipeError in flight is the break
    itself. Take it away and the guard preserves that BrokenPipeError instead,
    redirecting the stderr it just found broken to devnull and letting the
    break print its own traceback there, for status 1.

    The child pops poll rather than deleting it so it does not raise on a
    platform that never had it, the same way the no-SIGPIPE test above does.
    """
    import subprocess
    import sys

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    env = os.environ.copy()
    env.pop("PYTHONUNBUFFERED", None)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                'import select; select.__dict__.pop("poll", None); '
                "from uv_stack.cli import main; main()",
                "--root",
                str(tmp_path),
                "nosuchcommand",
            ],
            stdout=subprocess.PIPE,
            stderr=write_fd,
            env=env,
        )
    finally:
        os.close(write_fd)

    assert proc.returncode == 128 + int(signal.SIGPIPE)


@pytest.mark.skipif(
    not hasattr(signal, "SIGPIPE"), reason="the shell status this pins is POSIX-only"
)
def test_a_broken_pipe_at_exit_ignores_an_enclosing_handlers_exception(tmp_path: Path):
    """Only what cli() unwinds counts, not what the caller happens to be handling.

    The guard binds the exception cli() raised. Reading sys.exc_info() in the
    finally instead would answer with the *enclosing* handler's exception
    whenever nothing is in flight, so an embedding harness that calls main()
    from inside its own `except` block would have its own error mistaken for
    ours: the guard would take the preserve branch and return to the harness —
    status 7 below — instead of exiting 141 for the pipe.

    Click's standalone mode always leaves through SystemExit, so the console
    script cannot reach this; a cli() that returns cleanly is what an embedding
    harness looks like, and the marker file keeps the substitution honest —
    without it, a main() that stopped calling this module's `cli` would run the
    real one and exit 141 for the wrong reason.
    """
    import subprocess
    import sys

    marker = tmp_path / "cli-was-called"
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    env = os.environ.copy()
    env.pop("PYTHONUNBUFFERED", None)
    try:
        driver = (
            "import sys\n"
            "from pathlib import Path\n"
            "import uv_stack.cli as cli_module\n"
            "def _returns_cleanly():\n"
            f"    Path({str(marker)!r}).write_text('called')\n"
            "cli_module.cli = _returns_cleanly\n"
            # Buffered bytes on the dead stdout are what wake the guard at all.
            "print('buffered output')\n"
            "try:\n"
            "    raise RuntimeError('the harness own failure')\n"
            "except RuntimeError:\n"
            "    cli_module.main()\n"
            "sys.exit(7)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", driver],
            stdout=write_fd,
            stderr=subprocess.PIPE,
            env=env,
        )
    finally:
        os.close(write_fd)

    assert marker.exists()
    assert proc.returncode == 128 + int(signal.SIGPIPE)


def test_the_devnull_redirect_closes_the_descriptor_it_opened(tmp_path: Path):
    """The redirect leaves no descriptor behind for the branch that keeps running.

    Every other caller exits within a few statements, so the open devnull went
    unnoticed; the guard's preserve branch redirects and then returns to an
    unwinding exception. POSIX hands out the lowest free descriptor, so opening
    one before and after the redirects reads the leak directly: unclosed, each
    call consumes another number and the second probe lands higher.

    A throwaway file stands in for the real stream. Handing this pytest process
    its own sys.stdout would dup2 devnull onto fd 1 for the rest of the session.
    """
    from uv_stack.cli import _redirect_stream_to_devnull

    with (tmp_path / "target").open("w") as stream:
        before = os.open(os.devnull, os.O_WRONLY)
        os.close(before)
        for _ in range(5):
            _redirect_stream_to_devnull(stream)
        after = os.open(os.devnull, os.O_WRONLY)
        os.close(after)

    assert after == before


def test_the_can_report_probe_calls_an_unprobable_stream_reportable():
    """A stream the probe cannot reach counts as able to report, and never raises.

    The predicate may only ever subtract confidence: answering False for a
    stream it merely failed to measure would send tracebacks to devnull. Both
    shapes below reach it through an embedding harness that replaced sys.stderr
    — io.StringIO raises UnsupportedOperation from fileno(), and a bare writer
    has no fileno() to call at all. The second is also why AttributeError is
    caught: raised from inside main()'s finally it would escape the guard and
    replace the very exception the guard had just chosen to preserve.
    """
    import io

    from uv_stack.cli import _stream_can_report

    class _BareWriter:
        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            pass

    assert _stream_can_report(io.StringIO()) is True
    assert _stream_can_report(_BareWriter()) is True  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# clean-break guards: the old noun groups must no longer exist
# ---------------------------------------------------------------------------


def test_old_env_group_is_gone():
    result = CliRunner().invoke(cli, ["env", "upgrade", "main"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_old_profile_group_is_gone():
    result = CliRunner().invoke(cli, ["profile", "list"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_old_project_group_is_gone():
    result = CliRunner().invoke(cli, ["project", "init", "ds"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_create_profile_writes_yaml(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "create", "profile", "viz",
         "matplotlib", "seaborn", "--description", "Plotting", "--tag", "viz"],
    )
    assert result.exit_code == 0
    assert "Wrote" in result.output
    from uv_stack.config import ConfigRoot

    prof = ConfigRoot(root).load_profile("viz")
    assert prof.includes == ["matplotlib", "seaborn"]
    assert prof.description == "Plotting"
    assert prof.tags == ["viz"]


def test_create_profile_requires_packages(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "create", "profile", "viz"])
    assert result.exit_code == 2


def test_create_profile_refuses_overwrite(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "ds", "numpy"]
    )
    assert result.exit_code == 1


def test_create_profile_refuses_collision_with_bundle(tmp_path: Path):
    """Creating a profile named 'standard' (existing bundle) exits with code 1."""
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "standard", "numpy"]
    )
    assert result.exit_code == 1
    assert "would shadow the existing bundle" in _combined_output(result)
    # Verify the file was not created.
    from uv_stack.config import ConfigRoot
    assert not ConfigRoot(root).profile_path("standard").exists()


def test_create_bundle_writes_yaml_and_warns(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "create", "bundle", "daily", "ds", "standrd"],
    )
    assert result.exit_code == 0
    assert "did you mean 'standard'" in _combined_output(result)
    from uv_stack.config import ConfigRoot

    assert ConfigRoot(root).load_bundle("daily").includes == ["ds", "standrd"]


def test_create_bundle_strict_rejects_typo(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "create", "bundle", "daily", "--strict", "numpyy"],
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot

    assert not ConfigRoot(root).bundle_exists("daily")


def test_create_bundle_rejects_missing_explicit_profile(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "daily", "profile:ghost"]
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot

    assert not ConfigRoot(root).bundle_exists("daily")


def test_create_bundle_rejects_missing_explicit_bundle(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "daily", "@ghost"]
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot

    assert not ConfigRoot(root).bundle_exists("daily")


def test_create_bundle_strict_accepts_existing_bundle_with_bare_package(tmp_path: Path):
    """Strict mode applies to direct tokens only; existing bundle with bare package is OK."""
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    # Create bundle 'web' containing bare 'httpx'.
    cfg.bundle_path("web").write_text("includes:\n  - httpx\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "app", "--strict", "web"]
    )
    assert result.exit_code == 0
    assert cfg.bundle_exists("app")
    assert cfg.load_bundle("app").includes == ["web"]


def test_create_bundle_strict_rejects_direct_bare_near_miss(tmp_path: Path):
    """Strict mode rejects direct bare near-miss even if existing bundles are fine."""
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.bundle_path("web").write_text("includes:\n  - httpx\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "app", "--strict", "numpyy"]
    )
    assert result.exit_code == 1
    assert not cfg.bundle_exists("app")


def test_create_env_with_tokens_scaffolds_and_builds(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    calls: list[tuple[list[str], object]] = []
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade",
        lambda config, names, options, **kw: calls.append((names, options)),
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "create", "env", "fresh",
         "@standard", "httpx", "--python", "3.13"],
    )
    assert result.exit_code == 0
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    assert cfg.env_stack_path("fresh").read_text() == "@standard\nhttpx\n"
    assert cfg.env_python_path("fresh").read_text() == "3.13\n"
    assert calls and calls[0][0] == ["fresh"]
    assert calls[0][1].create is True
    assert "micromamba activate fresh" in result.output


def test_create_env_python_without_tokens_is_usage_error(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "--python", "3.13"]
    )
    assert result.exit_code == 2


def test_create_env_tokens_refuse_existing_stack(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade", lambda *a, **kw: None
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "main", "ds"]
    )
    assert result.exit_code == 1


def test_create_env_without_tokens_still_works(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade",
        lambda config, names, options, **kw: calls.append(names),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "create", "env", "main"])
    assert result.exit_code == 0
    assert calls == [["main"]]
    assert "micromamba activate main" in result.output


def test_create_env_strict_typo_exits_before_scaffolding(tmp_path: Path, monkeypatch):
    """Strict failure pre-validates so stack.txt is never written."""
    root = _seeded_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade", lambda *a, **kw: None
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "--strict", "numpyy"]
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot
    # stack.txt must not exist; the pre-validation failed before any write.
    assert not ConfigRoot(root).env_stack_path("fresh").exists()


def test_create_env_missing_explicit_profile_exits_before_scaffolding(tmp_path: Path, monkeypatch):
    """Missing explicit reference pre-validates so stack.txt is never written."""
    root = _seeded_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade", lambda *a, **kw: None
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "profile:ghost"]
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot
    assert not ConfigRoot(root).env_stack_path("fresh").exists()


def test_create_env_python_empty_string_is_usage_error(tmp_path: Path):
    """--python '' with tokens exits 2 (UsageError) before writing anything."""
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "ds", "--python", ""]
    )
    assert result.exit_code == 2
    from uv_stack.config import ConfigRoot
    assert not ConfigRoot(root).env_stack_path("fresh").exists()


def test_create_env_python_empty_string_without_tokens_is_usage_error(tmp_path: Path):
    """--python '' without tokens also exits 2."""
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "--python", ""]
    )
    assert result.exit_code == 2


def test_create_env_rejects_malformed_profile_before_scaffolding(
    tmp_path: Path, monkeypatch
):
    """Malformed profile YAML pre-validates so stack.txt is never written."""
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.profile_path("broken").write_text("includes: {not: [valid\n")
    monkeypatch.setattr(
        "uv_stack.cli.create._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "env", "fresh", "broken"]
    )
    assert result.exit_code == 1
    assert not cfg.env_stack_path("fresh").exists()


# ---------------------------------------------------------------------------
# activation hints
# ---------------------------------------------------------------------------


def test_activation_hint_shell_quotes_names(capsys):
    from uv_stack.cli._render import print_activation_hint

    print_activation_hint("safe; touch pwn")
    out = capsys.readouterr().out
    assert "micromamba activate 'safe; touch pwn'" in out
    assert "-n 'safe; touch pwn'" in out


def test_activation_hint_leaves_safe_names_unquoted(capsys):
    from uv_stack.cli._render import print_activation_hint

    print_activation_hint("main")
    out = capsys.readouterr().out
    assert "micromamba activate main" in out
    assert "-n main" in out


# ---------------------------------------------------------------------------
# doctor --fix and JSON
# ---------------------------------------------------------------------------


def test_doctor_fix_repairs_and_reports(tmp_path: Path):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    (cfg.profiles_dir / "old.in").write_text("numpy\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix"])
    assert result.exit_code == 0
    assert "fixed:" in result.output
    assert "No problems detected." in result.output
    assert cfg.profile_exists("old")


def test_doctor_fix_iterates_to_fixed_point(tmp_path: Path):
    # A completely absent root: fixing the root reveals the missing subdirectories,
    # which are then fixed in subsequent rounds.
    root = tmp_path / "python-envs"
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix"])
    assert result.exit_code == 0
    # Expect multiple "fixed:" lines (one for root, one for each subdir).
    assert result.output.count("fixed:") >= 4  # root + profiles + bundles + envs
    assert "No problems detected." in result.output
    # Verify all directories were created.
    from uv_stack.config import ConfigRoot
    cfg = ConfigRoot(root)
    assert cfg.root.is_dir()
    assert cfg.profiles_dir.is_dir()
    assert cfg.bundles_dir.is_dir()
    assert cfg.envs_dir.is_dir()


def test_doctor_fix_json_iterates_to_fixed_point(tmp_path: Path):
    import json
    # Same as above, but verify JSON output structure.
    root = tmp_path / "python-envs"
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    # Expect multiple actions: missing-root, then missing-dir for each subdir.
    action_kinds = [a["kind"] for a in payload["actions"]]
    assert "missing-root" in action_kinds
    assert action_kinds.count("missing-dir") >= 3  # profiles, bundles, envs
    assert all(a["applied"] for a in payload["actions"])
    assert payload["remaining"] == []


def test_doctor_fix_exit_1_when_errors_remain(tmp_path: Path, monkeypatch):
    import json
    # A root whose parent is read-only cannot be created; simulate by pointing
    # repair at a path under a file (mkdir raises, so the error finding stays).
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(blocker / "python-envs"), "doctor", "--fix", "--json"]
    )
    assert result.exit_code == 1
    # Exit 1 must be from graceful failure reporting, not an unhandled OSError.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    payload = json.loads(result.output)
    # Verify the repair handler converted the mkdir OSError into a skipped action.
    assert payload["actions"][0]["applied"] is False
    assert payload["actions"][0]["reason"]  # non-empty reason
    assert payload["remaining"][0]["kind"] == "missing-root"


def test_doctor_json_lists_findings(tmp_path: Path):
    import json

    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(tmp_path / "nope"), "doctor", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["kind"] == "missing-root"
    assert payload[0]["level"] == "error"


def test_doctor_fix_json_shape(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    (cfg.bundles_dir / "old.bundle").write_text("ds\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload.keys()) == {"actions", "remaining"}
    action = payload["actions"][0]
    assert action["kind"] == "legacy-bundle"
    assert action["applied"] is True
    assert action["reason"] is None
    assert payload["remaining"] == []


def test_doctor_fix_terminates_with_a_permanently_unfixable_finding(tmp_path: Path, monkeypatch):
    """The --fix loop exits when no action in a round reports applied."""
    from uv_stack.operations import doctor

    monkeypatch.setattr(doctor, "probe_locking", lambda path: False)
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "doctor", "--fix"])
    assert result.exit_code in (0, 1)
    assert "Name locking is unavailable" in _flat_panel(result)


# ---------------------------------------------------------------------------
# show env config and interpreter
# ---------------------------------------------------------------------------


class _FakeProbeRunner:
    """Stands in for SubprocessRunner in probe-only CLI paths."""

    def __init__(self, stdout: str = "/envs/main/bin/python\n3.12.7\n", returncode: int = 0):
        self._stdout = stdout
        self._returncode = returncode

    def run(self, command, *, capture=False, check=True):
        from uv_stack.runner import CommandResult

        return CommandResult(returncode=self._returncode, stdout=self._stdout)


def test_show_env_prints_config_and_interpreter(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.show.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "show", "env", "main"])
    assert result.exit_code == 0
    assert f"Config: {root}/envs/main" in result.output
    assert "Interpreter: /envs/main/bin/python" in result.output


def test_show_env_interpreter_not_created(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.show.SubprocessRunner",
        lambda: _FakeProbeRunner(stdout="", returncode=1),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "show", "env", "main"])
    assert result.exit_code == 0
    assert "Interpreter: not created (run 'stack create env main')" in result.output


def test_show_env_shell_quotes_name_in_hint(tmp_path: Path, monkeypatch):
    from uv_stack.config import ConfigRoot

    root = _env_root(tmp_path)
    cfg = ConfigRoot(root)
    env_dir = cfg.env_dir("bad;touch")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    (env_dir / "stack.txt").write_text("ds\n")
    monkeypatch.setattr(
        "uv_stack.cli.show.SubprocessRunner",
        lambda: _FakeProbeRunner(stdout="", returncode=1),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "show", "env", "bad;touch"])
    assert result.exit_code == 0
    assert "stack create env 'bad;touch'" in result.output


def test_show_env_prefixes_leading_dash_name_in_hint(tmp_path: Path, monkeypatch):
    from uv_stack.config import ConfigRoot

    root = _env_root(tmp_path)
    cfg = ConfigRoot(root)
    env_dir = cfg.env_dir("--recreate")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    (env_dir / "stack.txt").write_text("ds\n")
    monkeypatch.setattr(
        "uv_stack.cli.show.SubprocessRunner",
        lambda: _FakeProbeRunner(stdout="", returncode=1),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "show", "env", "--", "--recreate"])
    assert result.exit_code == 0
    assert "stack create env -- --recreate" in result.output


def test_status_table(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.status_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "status"])
    assert result.exit_code == 0
    cells = _row_cells(result.output, "main")
    assert cells[0] == "main"
    assert cells[1] == "3.12"
    assert cells[2] == "yes"          # Created
    assert cells[3] == "no"           # Lock (never upgraded here)
    assert cells[4] == "never built"  # State


def test_status_config_error_row_prints_message(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)
    from uv_stack.config import ConfigRoot

    ConfigRoot(root).env_stack_path("main").unlink()
    monkeypatch.setattr(
        "uv_stack.cli.status_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "status", "main"])
    assert result.exit_code == 0
    assert "config error" in result.output
    assert "main: Missing stack file" in result.output


def test_status_json(tmp_path: Path, monkeypatch):
    import json

    root = _env_root(tmp_path)
    monkeypatch.setattr(
        "uv_stack.cli.status_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "status", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == [
        {
            "name": "main",
            "python": "3.12",
            "actual_python": "3.12.7",
            "created": True,
            "lock": False,
            "state": "never built",
            "message": None,
        }
    ]


def test_status_python_changed_renders_both_versions(tmp_path: Path, monkeypatch):
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.upgrade import UpgradeOptions, upgrade_env
    from uv_stack.runner import RecordingRunner

    root = _env_root(tmp_path)
    # Build the env so state is not "never built".
    def _ok_responder(cmd):
        from uv_stack.runner import CommandResult
        if "run" in cmd.args:
            return CommandResult(returncode=0, stdout="/envs/main/bin/python\n3.12.7\n")
        return CommandResult(returncode=0, stdout="")
    upgrade_env(
        ConfigRoot(root),
        RecordingRunner(responder=_ok_responder),
        "main",
        UpgradeOptions(),
    )
    # Now probe with a mismatched version.
    monkeypatch.setattr(
        "uv_stack.cli.status_cmd.SubprocessRunner",
        lambda: _FakeProbeRunner(stdout="/envs/main/bin/python\n3.13.1\n"),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "status"])
    assert result.exit_code == 0
    cells = _row_cells(result.output, "main")
    assert cells[1] == "3.12 (env 3.13.1)"
    assert cells[4] == "python changed"


def test_status_json_includes_actual_python(tmp_path: Path, monkeypatch):
    import json

    from uv_stack.config import ConfigRoot
    from uv_stack.operations.upgrade import UpgradeOptions, upgrade_env
    from uv_stack.runner import RecordingRunner

    root = _env_root(tmp_path)
    # Build the env.
    def _ok_responder(cmd):
        from uv_stack.runner import CommandResult
        if "run" in cmd.args:
            return CommandResult(returncode=0, stdout="/envs/main/bin/python\n3.12.7\n")
        return CommandResult(returncode=0, stdout="")
    upgrade_env(
        ConfigRoot(root),
        RecordingRunner(responder=_ok_responder),
        "main",
        UpgradeOptions(),
    )
    # Probe with mismatched version.
    monkeypatch.setattr(
        "uv_stack.cli.status_cmd.SubprocessRunner",
        lambda: _FakeProbeRunner(stdout="/envs/main/bin/python\n3.13.1\n"),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "status", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload) == 1
    assert payload[0]["actual_python"] == "3.13.1"
    assert payload[0]["state"] == "python changed"


def test_list_env_json(tmp_path: Path):
    import json

    root = _env_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "env", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {"name": "main", "python": "3.12", "stack": ["@standard"]}
    ]


def test_list_profile_json_full_lists(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "list", "profile", "--json", "--tag", "chem"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {
            "name": "chem",
            "packages": ["rdkit"],
            "tags": ["chem", "bio"],
            "description": "Cheminformatics",
        }
    ]


def test_show_env_json_has_no_interpreter_probe(tmp_path: Path, monkeypatch):
    import json

    root = _env_root(tmp_path)

    def _boom():
        raise AssertionError("JSON mode must not construct a runner")

    monkeypatch.setattr("uv_stack.cli.show.SubprocessRunner", _boom)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "show", "env", "main", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "main"
    assert payload["config_dir"] == str(root / "envs" / "main")
    assert payload["channels"][0] == "conda-forge"
    assert payload["profiles"] == ["ds", "chem", "utils"]


def test_show_env_surfaces_resolver_warnings_json(tmp_path: Path, monkeypatch):
    import json

    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    env_dir = cfg.env_dir("main")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    # Use a near-miss typo (standrd -> standard) to trigger a warning.
    (env_dir / "stack.txt").write_text("standrd\n")

    def _boom():
        raise AssertionError("JSON mode must not construct a runner")

    monkeypatch.setattr("uv_stack.cli.show.SubprocessRunner", _boom)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "show", "env", "main", "--json"]
    )
    assert result.exit_code == 0
    # Extract JSON from output (warnings may appear before/after JSON in combined output).
    # Try parsing lines until we find a valid JSON object.
    combined = _combined_output(result)
    lines = combined.split('\n')
    json_lines = []
    in_json = False
    for line in lines:
        if line.startswith('{'):
            in_json = True
        if in_json:
            json_lines.append(line)
        if in_json and line.strip() == '}':
            break
    json_text = '\n'.join(json_lines)
    payload = json.loads(json_text)
    assert payload["name"] == "main"
    # Verify the warning is in the combined output.
    assert "did you mean 'standard'" in combined


def test_show_profile_json(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "show", "profile", "utils", "--json"]
    )
    assert json.loads(result.output) == {
        "name": "utils",
        "description": None,
        "tags": [],
        "includes": ["rich"],
    }


def test_resolve_json(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "resolve", "--json", "standard", "ds", "numpy"]
    )
    assert json.loads(result.output) == [
        {"input": "standard", "kind": "bundle", "name": "standard"},
        {"input": "ds", "kind": "profile", "name": "ds"},
        {"input": "numpy", "kind": "package", "name": "numpy"},
    ]


def test_resolve_full_json(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "resolve", "--full", "--json", "standard"]
    )
    assert json.loads(result.output) == {
        "packages": ["numpy", "pandas", "rdkit", "rich"]
    }


def test_list_bundle_json_full_lists(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "bundle", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == [
        {
            "name": "standard",
            "entries": ["ds", "chem", "utils"],
            "tags": ["core"],
            "description": "Everything for daily work",
        }
    ]


def test_show_bundle_json(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "show", "bundle", "standard", "--json"]
    )
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "name": "standard",
        "description": "Everything for daily work",
        "tags": ["core"],
        "includes": ["ds", "chem", "utils"],
    }


def test_list_env_empty_hint(tmp_path: Path):
    root = _seeded_root(tmp_path)  # profiles/bundles but no envs
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "env"])
    assert result.exit_code == 0
    assert "No environments yet" in result.output
    assert "stack create env" in result.output


def test_list_profile_empty_hint(tmp_path: Path):
    root = tmp_path / "python-envs"
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    init_config_root(ConfigRoot(root))
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "profile"])
    assert result.exit_code == 0
    assert "No profiles yet" in result.output


def test_list_tag_filter_no_match_hint(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "list", "profile", "--tag", "nonexistent"]
    )
    assert result.exit_code == 0
    assert "No profiles match tags: nonexistent." in result.output


def test_list_env_empty_json_is_empty_list(tmp_path: Path):
    import json

    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "list", "env", "--json"])
    assert json.loads(result.output) == []


def test_init_yes_on_empty_root_seeds_and_builds(tmp_path: Path, monkeypatch):
    root = tmp_path / "python-envs"
    calls: list[tuple[list[str], object]] = []
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda config, names, options, **kw: calls.append((names, options)),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "init", "--yes"])
    assert result.exit_code == 0
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    assert cfg.profile_exists("starter")
    starter_text = cfg.profile_path("starter").read_text()
    assert starter_text.startswith("# A profile is a reusable")
    assert cfg.env_stack_path("main").read_text() == "starter\n"
    assert cfg.env_python_path("main").read_text() == "3.12\n"
    assert calls and calls[0][0] == ["main"] and calls[0][1].create is True
    assert "micromamba activate main" in result.output


def test_init_yes_is_idempotent(tmp_path: Path, monkeypatch):
    root = _env_root(tmp_path)  # profiles and env 'main' already exist
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "init", "--yes"])
    assert result.exit_code == 0
    from uv_stack.config import ConfigRoot

    # Nothing new was seeded: profiles existed, so no starter profile.
    assert not ConfigRoot(root).profile_exists("starter")
    assert "Config root:" in result.output


def test_init_yes_reports_the_locks_directory(tmp_path: Path, monkeypatch):
    """The guided surface is a second command body, not a wrapper around the first."""
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda config, names, options, **kw: None,
    )
    root = tmp_path / "python-envs"
    result = CliRunner().invoke(cli, ["--root", str(root), "init", "--yes"])
    assert result.exit_code == 0
    assert str(root / ".locks") in _combined_output(result)
    assert (root / ".locks").is_dir()


def test_init_interactive_decline_build_prints_next_step(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "python-envs"
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    # Prompts: seed starter? y | create env? y | name [main] | tokens [starter]
    # | python [3.12] | build now? n
    result = runner.invoke(
        cli,
        ["--root", str(root), "init"],
        input="y\ny\n\n\n\nn\n",
    )
    assert result.exit_code == 0
    assert "Build it with: stack create env main" in result.output
    assert "micromamba activate" not in result.output


def test_init_rejects_bad_tokens_before_writing(tmp_path: Path, monkeypatch):
    root = tmp_path / "python-envs"
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    # Prompts: seed starter? y | create env? y | name [main] | tokens: profile:ghost
    # | python [3.12] — command errors at pre-validation before writing.
    result = runner.invoke(
        cli,
        ["--root", str(root), "init"],
        input="y\ny\n\nprofile:ghost\n\n",
    )
    assert result.exit_code == 1
    from uv_stack.config import ConfigRoot

    assert not ConfigRoot(root).env_stack_path("main").exists()


def test_init_decline_build_still_surfaces_warnings(tmp_path: Path, monkeypatch):
    root = tmp_path / "python-envs"
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    # Prompts: seed starter? y | create env? y | name [main] | tokens: startr
    # (near-miss of the seeded 'starter' profile) | python [3.12] | build now? n
    result = runner.invoke(
        cli,
        ["--root", str(root), "init"],
        input="y\ny\n\nstartr\n\nn\n",
    )
    assert result.exit_code == 0
    assert result.output.count("did you mean 'starter'") == 1
    assert "Build it with: stack create env main" in result.output


# No leading-dash counterpart: init's name reaches write_env_sources, whose
# _validate_name refuses a leading '-', so this site cannot render that case.
def test_init_shell_quotes_env_name_in_build_hint(tmp_path: Path, monkeypatch):
    root = tmp_path / "python-envs"
    monkeypatch.setattr(
        "uv_stack.cli.init_cmd._run_upgrade",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    runner = CliRunner()
    # Prompts: seed starter? y | create env? y | name: bad;touch | tokens: ds |
    # python [3.12] | build now? n
    result = runner.invoke(
        cli,
        ["--root", str(root), "init"],
        input="y\ny\nbad;touch\nds\n\nn\n",
    )
    assert result.exit_code == 0
    assert "Build it with: stack create env 'bad;touch'" in result.output


# ---------------------------------------------------------------------------
# completion
# ---------------------------------------------------------------------------


def test_completion_zsh_prints_script(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "zsh"])
    assert result.exit_code == 0
    assert result.output.startswith('# Add to ~/.zshrc: eval "$(stack completion zsh)"')
    assert "_STACK_COMPLETE" in result.output


def test_completion_bash_prints_script():
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "bash"])
    assert result.exit_code == 0
    assert result.output.startswith("# Requires bash >= 4.4.")
    assert "_STACK_COMPLETE" in result.output


def test_completion_fish_prints_script():
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "fish"])
    assert result.exit_code == 0
    assert result.output.startswith("# Add to ~/.config/fish/config.fish")
    assert "_STACK_COMPLETE" in result.output


def test_completion_requires_shell_argument():
    runner = CliRunner()
    result = runner.invoke(cli, ["completion"])
    assert result.exit_code == 2


def test_complete_env_names(tmp_path: Path):
    import click as _click

    from uv_stack.cli._complete import complete_env_names

    root = _env_root(tmp_path)
    ctx = _click.Context(cli)
    ctx.params = {"root": str(root)}
    assert complete_env_names(ctx, None, "") == ["main"]
    assert complete_env_names(ctx, None, "ma") == ["main"]
    assert complete_env_names(ctx, None, "zz") == []


def test_complete_env_names_survives_bad_root():
    import click as _click

    from uv_stack.cli._complete import complete_env_names

    ctx = _click.Context(cli)
    ctx.params = {}
    # No root param at all: fall back to discovery, never raise.
    assert isinstance(complete_env_names(ctx, None, "zzz-no-such"), list)


def test_complete_show_names_dispatches_on_kind(tmp_path: Path):
    import click as _click

    from uv_stack.cli._complete import complete_show_names

    root = _seeded_root(tmp_path)
    ctx = _click.Context(cli)
    ctx.params = {"root": str(root), "kind": "profile"}
    assert complete_show_names(ctx, None, "d") == ["ds"]


def test_help_contains_no_rest_double_backticks():
    runner = CliRunner()
    for args in (
        ["--help"],
        ["upgrade", "--help"],
        ["create", "--help"],
        ["create", "env", "--help"],
        ["create", "project", "--help"],
        ["create", "profile", "--help"],
        ["create", "bundle", "--help"],
        ["list", "--help"],
        ["show", "--help"],
        ["resolve", "--help"],
        ["refresh", "--help"],
        ["status", "--help"],
        ["doctor", "--help"],
        ["config", "--help"],
        ["init", "--help"],
        ["completion", "--help"],
    ):
        result = runner.invoke(cli, args)
        assert result.exit_code == 0
        assert "``" not in result.output, f"double backticks in: {args}"


def test_create_project_no_track_flag(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    captured: dict[str, object] = {}

    def _fake_init(config, runner, tokens, options, *, cwd):
        captured["track"] = options.track
        return []

    monkeypatch.setattr("uv_stack.cli.create.init_project", _fake_init)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "project", "ds", "--no-track"]
    )
    assert result.exit_code == 0
    assert captured["track"] is False


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


def _tracked_project_dir(tmp_path: Path, *, extra_entry: str | None = None) -> Path:
    """Write a project tracked against the seeded ``standard`` bundle.

    :param extra_entry: A requirement placed in both ``[project.dependencies]``
        and the applied ledger, for scenarios needing an entry the stack does
        not provide (which refresh then reports as a dropped entry).
    """
    extra = f', "{extra_entry}"' if extra_entry else ""
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        f'dependencies = ["numpy", "pandas", "rdkit", "rich"{extra}]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["standard"]\n'
        f'applied = ["numpy", "pandas", "rdkit", "rich"{extra}]\n'
    )
    return project_dir


def test_refresh_outside_project_fails_with_hint(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    # The width pin keeps the tmp path from being broken mid-token; flattening
    # the panel handles the fold that still lands between the message's words.
    monkeypatch.setenv("COLUMNS", "1000")
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "refresh"])
    assert result.exit_code == 1
    assert f"No tracked project in {empty}." in _flat_panel(result)


def test_refresh_happy_path_prints_summary(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    ConfigRoot(root).profile_path("chem").write_text("includes:\n  - chemprop\n")
    project_dir = _tracked_project_dir(tmp_path)
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        "uv_stack.cli.refresh_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "refresh", "--python", "3.12"])
    assert result.exit_code == 0
    assert "Removed (1): rdkit" in result.output
    assert "chemprop" in result.output  # listed under Added
    assert "Project refreshed." in result.output


def test_refresh_success_prints_the_skipped_removal_notice(tmp_path: Path, monkeypatch):
    """A successful refresh reports entries it will not remove itself.

    Asserted against ``SKIPPED_REMOVAL_NOTICE`` rather than a copy of its text,
    so this pins the wording parity with the failure path (which attaches the
    same constant to the raised error) and not merely that some line was
    printed.
    """
    from uv_stack.operations.project import SKIPPED_REMOVAL_NOTICE

    root = _seeded_root(tmp_path)
    # A direct reference the stack no longer provides: dropped from the ledger,
    # never auto-removed from [project.dependencies].
    direct_ref = "torch @ https://example.invalid/torch-2.0-py3-none-any.whl"
    project_dir = _tracked_project_dir(tmp_path, extra_entry=direct_ref)
    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        "uv_stack.cli.refresh_cmd.SubprocessRunner", lambda: _FakeProbeRunner()
    )
    result = CliRunner().invoke(cli, ["--root", str(root), "refresh", "--python", "3.12"])
    assert result.exit_code == 0
    # echo() is plain click output, so the line is unwrapped on stdout.
    assert SKIPPED_REMOVAL_NOTICE.format(entry=direct_ref) in result.output
    assert "Project refreshed." in result.output


def test_refresh_dry_run_prints_plan(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    project_dir = _tracked_project_dir(tmp_path)
    monkeypatch.chdir(project_dir)
    runner = CliRunner()
    result = runner.invoke(cli, ["--root", str(root), "refresh", "--dry-run"])
    assert result.exit_code == 0
    assert "Planned commands:" in result.output
    assert "uv add" in result.output


def test_refresh_flags_pass_through_to_refresh_project(tmp_path: Path, monkeypatch):
    from uv_stack.operations.project import RefreshOptions, RefreshResult
    root = _seeded_root(tmp_path)
    project_dir = _tracked_project_dir(tmp_path)
    monkeypatch.chdir(project_dir)
    captured: dict = {}

    def fake_refresh(config, runner, options, *, cwd):
        captured["options"] = options
        captured["cwd"] = cwd
        return RefreshResult()

    monkeypatch.setattr("uv_stack.cli.refresh_cmd.refresh_project", fake_refresh)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--root", str(root), "refresh", "--python", "3.13", "--strict", "--no-sync"]
    )
    assert result.exit_code == 0
    opts: RefreshOptions = captured["options"]
    assert opts.python == "3.13"
    assert opts.strict is True
    assert opts.no_sync is True
    assert opts.dry_run is False
    assert captured["cwd"] == project_dir

    result2 = runner.invoke(cli, ["--root", str(root), "refresh", "--dry-run"])
    assert result2.exit_code == 0
    opts2: RefreshOptions = captured["options"]
    assert opts2.dry_run is True


def test_refresh_renders_advisories_from_a_failed_run(tmp_path: Path, monkeypatch):
    """A failed refresh must still print the advisories the run computed.

    They reach the CLI only on the raised error — the RefreshResult that
    normally carries them is never produced — so the command renders
    error.resolution_warnings and re-raises into the group-level handler that
    prints the panel.
    """
    from uv_stack.errors import ToolError

    root = _seeded_root(tmp_path)
    project_dir = _tracked_project_dir(tmp_path)
    monkeypatch.chdir(project_dir)
    # Keep the advisory off the fold so it can be matched whole.
    monkeypatch.setenv("COLUMNS", "1000")
    notice = (
        "Not auto-removed (edit pyproject.toml manually): "
        "torch @ https://example.invalid/torch-2.0-py3-none-any.whl"
    )

    def fail_refresh(config, runner, options, *, cwd):
        error = ToolError("uv sync failed", command=["uv", "sync"], returncode=1)
        error.resolution_warnings = [notice]
        raise error

    monkeypatch.setattr("uv_stack.cli.refresh_cmd.refresh_project", fail_refresh)
    result = CliRunner().invoke(cli, ["--root", str(root), "refresh"])
    assert result.exit_code == 1
    flat = _flat_panel(result)
    assert f"warning: {notice}" in flat
    # Re-raised, so the group-level handler still rendered the error panel.
    assert "uv sync failed" in flat


def test_command_panels_separate_create_env_and_project_work():
    """Top-level help must not file project work under Environments.

    Asserts structurally against COMMAND_GROUPS: 'refresh' only ever operates
    on projects and gets its own panel; 'create' is cross-cutting (it makes
    environments, projects, profiles, and bundles) and gets its own panel;
    'upgrade' is the only genuinely environment-only command.
    """
    command_groups = rich_click.rich_click.COMMAND_GROUPS.get("stack", [])
    panels = {group["name"]: group["commands"] for group in command_groups}
    assert panels.get("Create") == ["create"], panels
    assert panels.get("Environments") == ["upgrade"], panels
    assert panels.get("Projects") == ["refresh"], panels

    # Completeness: the per-panel assertions above pin what each panel holds,
    # but a newly registered command filed in no panel would still pass them.
    # rich-click silently drops such a command from the help screen.
    placed = [name for group in command_groups for name in group["commands"]]
    assert sorted(placed) == sorted(cli.commands), (placed, sorted(cli.commands))

    # Smoke assertion: help renders and the Projects panel is visible, so
    # "project" appears as a heading on the top-level help screen.
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"], prog_name="stack")
    assert result.exit_code == 0
    assert "Projects" in result.output
    assert "refresh" in result.output


def test_create_profile_warns_on_bare_token_retarget(tmp_path: Path):
    root = _env_root(tmp_path)  # env 'main' stack.txt contains '@standard'
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.env_stack_path("main").write_text("@standard\nhttpx\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "httpx", "httpx>=0.27"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)
    # Styled rendering may wrap; assert distinctive unwrappable fragments.
    assert "'httpx' is used as a bare token" in combined
    assert "envs/main/stack.txt" in combined
    assert "use pkg:httpx there" in combined


def test_create_bundle_warns_on_bare_token_in_other_bundle(tmp_path: Path):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.bundle_path("web").write_text("includes:\n  - httpx\n")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "bundle", "httpx", "pkg:httpx"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)
    # Styled rendering may wrap; assert distinctive unwrappable fragments.
    assert "'httpx' is used as a bare token" in combined
    assert "bundles/web.yaml" in combined
    assert "this bundle" in combined


def test_create_profile_no_warning_without_bare_usage(tmp_path: Path):
    root = _seeded_root(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "viz2", "matplotlib"]
    )
    assert result.exit_code == 0
    assert "used as a bare token" not in _combined_output(result)


def test_create_profile_warns_on_whitespace_padded_bundle_token(tmp_path: Path):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    cfg.bundle_path("web").write_text('includes:\n  - " httpx "\n')
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "httpx", "httpx>=0.27"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)
    assert "'httpx' is used as a bare token" in combined
    assert "bundles/web.yaml" in combined


def test_create_profile_warns_on_case_variant_retarget(tmp_path: Path):
    """Case-variant retargeting warning on case-insensitive filesystems."""
    root = _env_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    # Env stack contains bare "httpx" (lowercase).
    cfg.env_stack_path("main").write_text("@standard\nhttpx\n")

    # Detect whether the filesystem is case-insensitive.
    # Write a lowercase profile and check if uppercase path resolves to it.
    sentinel = cfg.profile_path("sentinel_lowercase")
    sentinel.write_text("includes: []\n")
    case_insensitive = cfg.profile_path("SENTINEL_LOWERCASE").is_file()
    sentinel.unlink()

    runner = CliRunner()
    # Create profile "HTTPX" (uppercase).
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "HTTPX", "httpx>=0.27"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)

    if case_insensitive:
        # On case-insensitive filesystems, the bare "httpx" now retargets to "HTTPX".
        assert "'HTTPX' is used as a bare token" in combined
        assert "envs/main/stack.txt" in combined
    else:
        # On case-sensitive filesystems, no retarget occurs.
        assert "used as a bare token" not in combined


def test_create_profile_tolerates_corrupt_env(tmp_path: Path):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    cfg = ConfigRoot(root)
    env_dir = cfg.env_dir("broken")
    env_dir.mkdir(parents=True)
    (env_dir / "stack.txt").write_bytes(b"\xff\xfe")
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "something", "pkg:something"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)
    assert "Wrote" in combined
    assert "Traceback" not in combined


def test_create_profile_tolerates_unreadable_bundles_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    def _raise_permission_error(*args, **kwargs):
        raise PermissionError("mock unreadable bundles directory")

    monkeypatch.setattr(ConfigRoot, "list_bundles", _raise_permission_error)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "something", "pkg:something"]
    )
    assert result.exit_code == 0
    combined = _combined_output(result)
    assert "Wrote" in combined
    assert "Traceback" not in combined


def test_create_profile_tolerates_unreadable_envs_dir(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    def exploding_list_envs(self):
        raise PermissionError("envs dir unreadable")

    monkeypatch.setattr(ConfigRoot, "list_envs", exploding_list_envs)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--root", str(root), "create", "profile", "newprof", "pkg:something"]
    )
    assert result.exit_code == 0
    assert "Wrote" in _combined_output(result)


def test_newer_schema_with_extra_keys_friendly_via_cli(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n'
        "\n[tool.uv-stack]\nversion = 2\n"
        'stack = ["ds"]\n'
        'applied = []\n'
        'future_key = "whatever"\n'
    )
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()
    refresh_result = runner.invoke(cli, ["--root", str(root), "refresh"])
    assert refresh_result.exit_code == 1
    refresh_output = _combined_output(refresh_result)
    assert "newer uv-stack (schema 2)" in refresh_output
    # The hint names the table the user must edit; rich must not eat it.
    assert "edit [tool.uv-stack] manually" in refresh_output
    create_result = runner.invoke(
        cli, ["--root", str(root), "create", "project", "ds", "--force"]
    )
    assert create_result.exit_code == 1
    assert "newer uv-stack (schema 2)" in _combined_output(create_result)


def test_error_panel_preserves_bracketed_text(tmp_path: Path, monkeypatch):
    """An error message and its hint are data, not rich markup.

    ``[tool.uv-stack]`` is the single most likely bracketed literal to appear
    in a uv-stack error, and it is also valid rich markup — unescaped, rich
    parses it as a style tag and renders it as nothing, deleting the subject
    of the sentence.
    """
    root = _seeded_root(tmp_path)
    project_dir = tmp_path / "scalarproj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n[tool]\nuv-stack = "nope"\n'
    )
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["--root", str(root), "refresh"])
    assert result.exit_code == 1
    output = _combined_output(result)
    assert "[tool.uv-stack] in" in output
    assert "Replace the scalar/array value with a [tool.uv-stack] table." in output


def test_table_cells_preserve_bracketed_text(tmp_path: Path, monkeypatch):
    """Table cells carry user text (descriptions, tags) and are not markup."""
    root = _seeded_root(tmp_path)
    from uv_stack.config import ConfigRoot

    ConfigRoot(root).profile_path("web").write_text(
        "description: Installs requests[security]\nincludes:\n  - requests\n"
    )
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "profile"])
    assert result.exit_code == 0
    assert "requests[security]" in result.output


def test_warning_preserves_bracketed_text(tmp_path: Path, monkeypatch):
    """Warnings emitted by operations carry bracketed tokens like [project.dependencies].

    Those tokens are user data, not rich markup — unescaped, rich parses them
    as style tags and renders them as nothing.
    """
    root = _seeded_root(tmp_path)
    project_dir = tmp_path / "warnproj"
    project_dir.mkdir()
    # The stack asks for requests[security], but the user already owns that
    # requirement in [project.dependencies], so refresh skips it and warns —
    # quoting the extras-bearing requirement back at the user.
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["requests[security]"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["requests[security]"]\n'
        "applied = []\n"
    )
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("COLUMNS", "200")
    # --dry-run keeps this hermetic: warnings are rendered either way, but no
    # uv subprocess runs.
    result = CliRunner().invoke(cli, ["--root", str(root), "refresh", "--dry-run"])
    assert result.exit_code == 0
    output = _combined_output(result)
    assert "requests[security]" in output
    assert "is user-owned" in output


def test_upgrade_summary_failure_reason_preserves_bracketed_text(tmp_path: Path, monkeypatch):
    """The rule, the row name, and the failure reason are all user data.

    The bracketed token in the reason is a stand-in: any error message quoting
    a requirement extra or a TOML table name reaches that line the same way.
    The environment name is user data too — it is a CLI argument, and it is
    rendered separately in the batch rule and in the summary row.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    env_dir = cfg.env_dir("al[p]ha")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    # A missing profile fails during the pure resolve step, before any
    # subprocess call, and the reason quotes the token verbatim.
    (env_dir / "stack.txt").write_text("profile:ghost[security]\n")
    # The reason embeds an absolute tmp_path; without a pinned width rich folds
    # it at whatever column the path length happens to land on, which can split
    # the very token under test. Strip newlines on top so the assertion tests
    # escaping, not wrapping.
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "al[p]ha"])
    assert result.exit_code == 1
    flat = result.output.replace("\n", "")
    assert "Upgrading al[p]ha" in flat
    summary = flat.split("Summary", 1)[1]
    assert "al[p]ha" in summary
    assert "ghost[security]" in summary


def test_upgrade_summary_failure_reason_preserves_emoji_code(tmp_path: Path, monkeypatch):
    """A ``:name:`` run in the reason must not be substituted for an emoji.

    ``rich.markup.escape`` neutralises style tags but not emoji codes, so this
    line survived the escaping sweep and still turned ``:100:`` into 💯. Only
    rendering the whole line as ``Text`` closes it. Colons are ordinary in
    filenames and in tool diagnostics, so the shape is reachable.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    env_dir = cfg.env_dir("alpha")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    # Resolution fails before any subprocess call and quotes the missing
    # profile's path, carrying the emoji code into the summary reason.
    (env_dir / "stack.txt").write_text("profile:ghost:100:\n")
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "alpha"])
    assert result.exit_code == 1
    flat = result.output.replace("\n", "")
    assert "ghost:100:.yaml" in flat, flat
    assert "💯" not in flat


def test_upgrade_success_summary_preserves_bracketed_env_name(tmp_path: Path, monkeypatch):
    """The ✓ row names the environment, and that name is user data.

    Stubbing ``upgrade_env`` is what makes a successful batch reachable without
    micromamba or uv; the ✓ branch of the summary has no other CLI-level route.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root
    from uv_stack.operations.upgrade import UpgradeResult

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    env_dir = cfg.env_dir("al[p]ha")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    (env_dir / "stack.txt").write_text("numpy\n")
    monkeypatch.setattr(
        "uv_stack.cli.upgrade.upgrade_env",
        lambda config, runner, name, options: UpgradeResult(env_name=name),
    )
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["--root", str(root), "upgrade", "al[p]ha"])
    assert result.exit_code == 0
    flat = result.output.replace("\n", "")
    assert "✓ al[p]ha" in flat
    assert "All requested environments upgraded." in flat


def test_status_message_preserves_bracketed_text(tmp_path: Path, monkeypatch):
    """The ``config error`` note under the status table is user data, not markup.

    That note is a :class:`ConfigError` wrapping a pydantic ``ValidationError``,
    whose text is bracketed by construction (``[type=list_type, ...]``) — so
    this render site cannot be dismissed as carrying only safe constants. The
    environment name prefixing it is a second, independent interpolation.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    # A scalar where the schema wants a list: pydantic reports the mismatch with
    # a bracketed type/input suffix.
    cfg.profile_path("web").write_text("includes: 123\n")
    env_dir = cfg.env_dir("al[p]ha")
    env_dir.mkdir(parents=True)
    (env_dir / "python.txt").write_text("3.12\n")
    (env_dir / "stack.txt").write_text("profile:web\n")
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["--root", str(root), "status"])
    assert result.exit_code == 0
    output = _combined_output(result).replace("\n", "")
    assert "config error" in output
    assert "al[p]ha: Invalid profile config" in output
    assert "[type=list_type" in output


def test_table_directory_line_preserves_bracketed_path(tmp_path: Path, monkeypatch):
    """The ``<title> in <directory>`` line above a table carries the config root.

    That path comes from ``--root``/``UV_STACK_ROOT``, so it is user-controlled
    and must survive verbatim. A swallowed segment here is worse than a visible
    hole: it prints a plausible but wrong absolute path.
    """
    from uv_stack.config import ConfigRoot
    from uv_stack.operations.init import init_config_root

    root = tmp_path / "root[x]" / "python-envs"
    cfg = ConfigRoot(root)
    init_config_root(cfg)
    cfg.profile_path("demo").write_text("includes:\n  - numpy\n")
    monkeypatch.setenv("COLUMNS", "250")
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "profile"])
    assert result.exit_code == 0
    assert "root[x]" in result.output.replace("\n", "")


def test_doctor_output_preserves_bracketed_paths(tmp_path: Path, monkeypatch):
    """Every ``doctor`` line quotes paths derived from the config root.

    Findings, their fixes, and both repair outcomes are separate render sites,
    so each is asserted on its own line rather than on the output as a whole.
    """
    from uv_stack.config import ConfigRoot

    root = tmp_path / "root[x]" / "python-envs"
    # Doctor lines quote two absolute paths apiece; the width must exceed that
    # so the assertions below test escaping rather than where rich folds.
    monkeypatch.setenv("COLUMNS", "1000")
    runner = CliRunner()

    # An absent root is the one finding that needs no seeding, and repairing it
    # reports the directories it created.
    fixed = _combined_output(runner.invoke(cli, ["--root", str(root), "doctor", "--fix"]))
    assert [line for line in fixed.splitlines() if line.startswith("fixed:") and "root[x]" in line]

    cfg = ConfigRoot(root)
    # A legacy .in profile whose .bak backup already exists cannot be converted,
    # which is the only path that reports a skipped repair and its reason.
    cfg.profiles_dir.joinpath("old[y].in").write_text("numpy\n")
    cfg.profiles_dir.joinpath("old[y].in.bak").write_text("stale\n")
    output = _combined_output(runner.invoke(cli, ["--root", str(root), "doctor", "--fix"]))
    lines = output.splitlines()
    skipped = [line for line in lines if line.startswith("skipped:")]
    assert skipped and "root[x]" in skipped[0]
    # The parenthesised reason is a second interpolation on the same line.
    assert "(old[y].in.bak already exists)" in skipped[0]
    assert [line for line in lines if line.startswith("WARN") and "old[y].in" in line]
    assert [line for line in lines if line.strip().startswith("fix:") and "old[y].yaml" in line]


def test_list_profile_no_match_preserves_bracketed_tag(tmp_path: Path, monkeypatch):
    """The no-match message in ``list profile --tag`` carries the tag verbatim.

    Tags are user-controlled, so a bracketed tag must survive.
    """
    root = _seeded_root(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(cli, ["--root", str(root), "list", "profile", "--tag", "[nope]"])
    assert result.exit_code == 0
    assert "No profiles match tags: [nope]." in result.output.replace("\n", "")


def test_refresh_spawn_failure_past_pending_write_prints_adoption_warning(
    tmp_path: Path, monkeypatch
):
    """An adopting refresh that fails after the pending write must still print the warning.

    Regression for a spawn failure that escaped UvStackError handlers at the CLI
    edge, discarding every advisory computed before the crash — including the
    adoption warning the user is promised will appear before any remove.

    The adoption warning contains `[tool.uv-stack].applied`. Rich would parse
    those brackets as a style tag and render the phrase as nothing, but
    `render_warnings` assembles a `rich.text.Text`, which does not parse markup.
    The assertion on that phrase below pins the mitigation on this path.
    """
    root = _seeded_root(tmp_path)
    # Build the project fixture inline: chemprop present in dependencies and in
    # pending, absent from applied and from the stack, so _adopt_orphans adopts it.
    project_dir = tmp_path / "proj_spawn_fail"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n'
        'dependencies = ["numpy", "pandas", "rdkit", "rich", "chemprop"]\n'
        "\n[tool.uv-stack]\nversion = 1\n"
        'stack = ["standard"]\n'
        'applied = ["numpy", "pandas", "rdkit", "rich"]\n'
        'pending = ["numpy", "pandas", "rdkit", "rich", "chemprop"]\n'
    )
    monkeypatch.chdir(project_dir)
    # --python 3.12 so resolve_project_python short-circuits on the passthrough
    # version and never spawns micromamba.
    # PATH=/nowhere so the uv spawn genuinely fails (not a fake runner).
    nowhere = tmp_path / "nowhere"
    monkeypatch.setenv("PATH", str(nowhere))
    monkeypatch.setenv("COLUMNS", "1000")
    result = CliRunner().invoke(
        cli, ["--root", str(root), "refresh", "--python", "3.12"]
    )
    assert result.exit_code == 1
    flat = _flat_panel(result)
    # The adoption warning reached stderr.
    assert "was applied by an interrupted run" in flat
    assert "chemprop" in flat
    # Rendered through Text.assemble, so the brackets survive verbatim; a markup
    # string would swallow the phrase and leave the sentence nonsensical.
    assert "[tool.uv-stack].applied" in flat
    # The panel names the binary that could not be started, proving a rendered
    # ToolError rather than a traceback. ('uv' alone would match the panel's
    # own 'uv-stack error' title.)
    assert "Could not run uv" in flat
    assert "Is uv installed and on PATH?" in flat
    # Pin the boundary this test is named for. The pending write landed —
    # 'chemprop' is durably in applied, so no later run re-derives the adoption
    # warning — while pending is still set, so the final ledger write did not.
    # This is exactly the window in which losing the advisory is unrecoverable.
    import tomllib
    ledger = tomllib.loads((project_dir / "pyproject.toml").read_text())["tool"]["uv-stack"]
    assert "chemprop" in ledger["applied"]
    assert ledger.get("pending") is not None


# ---------------------------------------------------------------------------
# show project
# ---------------------------------------------------------------------------


def _project_with_tracking(tmp_path: Path, table: str, *, name: str = "showproj") -> Path:
    """A project directory whose pyproject.toml carries the given tracking table."""
    project_dir = tmp_path / name
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n\n' + table
    )
    return project_dir


def test_show_project_prints_path_stack_and_applied(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    project_dir = _project_with_tracking(
        tmp_path,
        "[tool.uv-stack]\nversion = 1\n"
        'stack = ["@standard", "pkg:httpx"]\n'
        'python = "3.12"\n'
        'applied = ["httpx", "numpy", "pandas"]\n',
    )
    monkeypatch.chdir(project_dir)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "project"])
    assert result.exit_code == 0
    assert f"Project: {project_dir}" in result.output
    assert "Python: 3.12" in result.output
    assert "Stack:" in result.output
    assert "  @standard" in result.output
    assert "  pkg:httpx" in result.output
    assert "Applied packages:" in result.output
    assert "  httpx" in result.output
    assert "  pandas" in result.output


def test_show_project_python_not_recorded(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    project_dir = _project_with_tracking(
        tmp_path,
        "[tool.uv-stack]\nversion = 1\nstack = [\"ds\"]\napplied = []\n",
    )
    monkeypatch.chdir(project_dir)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "project"])
    assert result.exit_code == 0
    assert "Python: (not recorded)" in result.output
    # Empty lists still print their headings, matching 'show env'.
    assert "Applied packages:" in result.output


def test_show_project_prints_pending_when_present(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    project_dir = _project_with_tracking(
        tmp_path,
        "[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["numpy"]\n'
        'pending = ["scipy"]\n',
    )
    monkeypatch.chdir(project_dir)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "project"])
    assert result.exit_code == 0
    expected_pending = (
        "Pending (interrupted run — the next successful 'stack refresh' clears it):"
    )
    assert expected_pending in result.output
    assert "  scipy" in result.output


def test_show_project_marks_an_empty_pending_record(tmp_path: Path, monkeypatch):
    """``pending = []`` is a real state and must stay distinguishable.

    An empty list still records that a run was interrupted, so the heading has
    to appear — but a heading with nothing under it reads as a rendering bug,
    so the empty case says so explicitly.
    """
    root = _seeded_root(tmp_path)
    project_dir = _project_with_tracking(
        tmp_path,
        "[tool.uv-stack]\nversion = 1\n"
        'stack = ["ds"]\n'
        'applied = ["numpy"]\n'
        "pending = []\n",
    )
    monkeypatch.chdir(project_dir)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "project"])
    assert result.exit_code == 0
    assert "Pending (interrupted run" in result.output
    assert "  (none — the interrupted run had no packages to apply)" in result.output


def test_show_project_omits_pending_when_absent(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    project_dir = _project_with_tracking(
        tmp_path,
        "[tool.uv-stack]\nversion = 1\nstack = [\"ds\"]\napplied = [\"numpy\"]\n",
    )
    monkeypatch.chdir(project_dir)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "project"])
    assert result.exit_code == 0
    assert "Pending" not in result.output


def test_show_project_outside_project_fails_with_hint(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    # The width pin keeps the tmp path from being broken mid-token; flattening
    # the panel handles the fold that still lands between the message's words.
    monkeypatch.setenv("COLUMNS", "1000")
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "project"])
    assert result.exit_code == 1
    output = _flat_panel(result)
    assert f"No tracked project in {empty}." in output
    assert "stack create project" in output


def test_show_project_rejects_a_name(tmp_path: Path, monkeypatch):
    root = _seeded_root(tmp_path)
    project_dir = _project_with_tracking(
        tmp_path,
        "[tool.uv-stack]\nversion = 1\nstack = [\"ds\"]\napplied = []\n",
    )
    monkeypatch.chdir(project_dir)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "project", "myproj"])
    assert result.exit_code == 2
    assert "takes no NAME" in _combined_output(result)


@pytest.mark.parametrize("pending_value", [None, []])
def test_show_project_json_shape(tmp_path: Path, monkeypatch, pending_value):
    import json

    root = _seeded_root(tmp_path)
    pending_line = (
        'pending = []\n' if pending_value == [] else ""
    )
    project_dir = _project_with_tracking(
        tmp_path,
        "[tool.uv-stack]\nversion = 1\n"
        'stack = ["@standard"]\n'
        f'applied = ["httpx"]\n{pending_line}',
    )
    monkeypatch.chdir(project_dir)
    result = CliRunner().invoke(cli, ["--root", str(root), "show", "project", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    # 'pending' is always present so consumers never branch on key existence.
    assert payload == {
        "path": str(project_dir),
        "version": 1,
        "stack": ["@standard"],
        "python": None,
        "applied": ["httpx"],
        "pending": pending_value,
    }


@pytest.mark.parametrize("broken", ["missing", "malformed"])
def test_show_project_never_resolves_tokens(tmp_path: Path, monkeypatch, broken: str):
    """'show project' is a pure read of the tracking table.

    It must still succeed when the recorded stack references a profile that
    does not exist or one whose YAML is malformed — the case where a user most
    needs to inspect the project. 'show env' resolves tokens, so this pins the
    guarantee against a future refactor that shares code between the two.
    """
    import json

    from uv_stack.config import ConfigRoot

    root = _seeded_root(tmp_path)
    if broken == "missing":
        # The 'profile:' prefix is load-bearing. A bare unknown token is a
        # valid literal package to the resolver (resolver.py:204-206), so
        # 'ghost' alone would resolve cleanly and the case would pass even
        # against an implementation that wrongly resolves. 'profile:ghost'
        # forces explicit=True, which raises ResolutionError.
        token = "profile:ghost"  # no profiles/ghost.yaml in the seeded root
    else:
        token = "ds"
        ConfigRoot(root).profile_path("ds").write_text("includes: [unterminated\n")
    project_dir = _project_with_tracking(
        tmp_path,
        "[tool.uv-stack]\nversion = 1\n"
        f'stack = ["{token}"]\n'
        'applied = ["numpy"]\n',
    )
    monkeypatch.chdir(project_dir)
    runner = CliRunner()

    plain = runner.invoke(cli, ["--root", str(root), "show", "project"])
    assert plain.exit_code == 0, _combined_output(plain)
    assert f"  {token}" in plain.output

    as_json = runner.invoke(cli, ["--root", str(root), "show", "project", "--json"])
    assert as_json.exit_code == 0, _combined_output(as_json)
    assert json.loads(as_json.output)["stack"] == [token]
