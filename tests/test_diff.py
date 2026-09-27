"""Tests for stack diff: source classification, lock grammar, comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.conftest import _deadline
from uv_stack.cli import cli
from uv_stack.errors import ConfigError

_IS_ROOT = os.geteuid() == 0


def _lock(tmp_path: Path, text: str, name: str = "a.lock.txt") -> Path:
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    return target


def test_parse_lock_accepts_pins_editables_and_direct_references(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(
        tmp_path,
        "# via nothing\n"
        "\n"
        "numpy==2.1.0\n"
        "rich==13.7.0  # via textual\n"
        "-e /src/tool\n"
        "mypkg @ https://example.invalid/mypkg.whl#sha256=abc\n"
        "--index-url https://example.invalid/simple\n",
    )

    assert parse_lock(path) == {
        "numpy": "2.1.0",
        "rich": "13.7.0",
        "-e /src/tool": None,
        "mypkg": "https://example.invalid/mypkg.whl#sha256=abc",
    }


def test_parse_lock_normalizes_distribution_names(tmp_path):
    from uv_stack.operations.diff import parse_lock

    assert parse_lock(_lock(tmp_path, "Foo_Bar==1.0\n")) == {"foo-bar": "1.0"}


@pytest.mark.parametrize(
    "line",
    ["-e /src/tool\n", "--editable /src/tool\n", "--editable=/src/tool\n"],
)
def test_parse_lock_normalizes_every_editable_spelling(tmp_path, line):
    from uv_stack.operations.diff import parse_lock

    assert parse_lock(_lock(tmp_path, line)) == {"-e /src/tool": None}


def test_parse_lock_discards_a_genuine_option_line(tmp_path):
    from uv_stack.operations.diff import parse_lock

    assert parse_lock(_lock(tmp_path, "--find-links /wheels\n")) == {}


def test_parse_lock_discards_indented_via_continuations(tmp_path):
    from uv_stack.operations.diff import parse_lock

    assert parse_lock(_lock(tmp_path, "numpy==2.1.0\n    #   via pandas\n")) == {
        "numpy": "2.1.0"
    }


def test_parse_lock_rejects_an_unparseable_line(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(tmp_path, "numpy==2.1.0\nthis is not a requirement\n")
    with pytest.raises(ConfigError) as excinfo:
        parse_lock(path)
    assert str(path) in str(excinfo.value)
    assert "line 2" in str(excinfo.value)


def test_parse_lock_rejects_a_hashed_continuation(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(tmp_path, "numpy==2.1.0 \\\n    --hash=sha256:abc\n")
    with pytest.raises(ConfigError) as excinfo:
        parse_lock(path)
    assert "line 1" in str(excinfo.value)
    # str() of a UvStackError is the message alone; the remedy is the hint.
    assert "hash" in (excinfo.value.hint or "")


def test_parse_lock_rejects_a_duplicate_identity(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(tmp_path, "numpy==2.1.0\nNumPy==2.2.0\n")
    with pytest.raises(ConfigError) as excinfo:
        parse_lock(path)
    assert "numpy" in str(excinfo.value)
    assert "line 2" in str(excinfo.value)


def test_parse_lock_reports_an_unreadable_file_as_a_config_error(tmp_path):
    from uv_stack.operations.diff import parse_lock

    if _IS_ROOT:
        pytest.skip("root reads regardless of mode")
    path = _lock(tmp_path, "numpy==2.1.0\n")
    path.chmod(0o000)
    try:
        with pytest.raises(ConfigError):
            parse_lock(path)
    finally:
        path.chmod(0o644)


def test_load_source_reads_an_environment_by_name(config_tree):
    from uv_stack.operations.diff import load_source

    (config_tree.env_requirements_lock("main")).write_text("numpy==2.1.0\n")

    source = load_source(config_tree, "main")

    assert source.label == "main"
    assert source.pins == {"numpy": "2.1.0"}
    assert source.python == "3.12"
    # python and pip are reported by the interpreter layer, not here.
    assert source.micromamba == ["graphviz"]
    assert source.channels == ["conda-forge", "bioconda"]


def test_load_source_reads_a_copied_directory(config_tree, tmp_path):
    import shutil

    from uv_stack.operations.diff import load_source

    config_tree.env_requirements_lock("main").write_text("numpy==2.1.0\n")
    copied = tmp_path / "elsewhere" / "main"
    copied.parent.mkdir()
    shutil.copytree(config_tree.env_dir("main"), copied)

    source = load_source(config_tree, str(copied))

    assert source.label == str(copied)
    assert source.pins == {"numpy": "2.1.0"}
    assert source.python == load_source(config_tree, "main").python


def test_load_source_reads_a_bare_lock_with_no_other_layers(config_tree, tmp_path):
    from uv_stack.operations.diff import load_source

    path = _lock(tmp_path, "numpy==2.1.0\n")
    source = load_source(config_tree, str(path))

    assert source.pins == {"numpy": "2.1.0"}
    assert source.python is None
    assert source.micromamba is None
    assert source.channels is None


def test_load_source_ignores_a_stray_envs_directory(config_tree, tmp_path, monkeypatch):
    from uv_stack.operations.diff import load_source

    # The fallback cases resolve a relative argument against the working
    # directory, so pin it to one known to hold no 'junk'.
    monkeypatch.chdir(tmp_path)
    (config_tree.envs_dir / "junk").mkdir()
    with pytest.raises(ConfigError) as excinfo:
        load_source(config_tree, "junk")
    assert excinfo.value.message == "No source named junk."
    assert "lock file" in (excinfo.value.hint or "")


def test_load_source_refuses_to_escape_the_envs_directory(config_tree, tmp_path, monkeypatch):
    from uv_stack.operations.diff import load_source

    # A complete, loadable environment that is reachable from envs/ only by
    # a '..' join. Without the name check running first, env_dir's unchecked
    # join would find its stack.txt and read it as an environment of this root.
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "stack.txt").write_text("pkg:numpy\n")
    (victim / "requirements.lock.txt").write_text("numpy==2.1.0\n")
    escape = os.path.relpath(victim, config_tree.envs_dir)
    assert (config_tree.envs_dir / escape / "stack.txt").is_file()
    # From here the same relative string names nothing, so the directory and
    # lock-file fallbacks cannot find the victim either.
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    with pytest.raises(ConfigError) as excinfo:
        load_source(config_tree, escape)
    assert excinfo.value.message == f"No source named {escape}."


def test_load_source_rejects_a_fifo_without_blocking(config_tree, tmp_path):
    from uv_stack.operations.diff import load_source

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    # The defect this covers does not fail; it blocks forever in open().
    with _deadline(5.0), pytest.raises(ConfigError) as excinfo:
        load_source(config_tree, str(fifo))
    assert str(fifo) in str(excinfo.value)
    assert "named pipe" in str(excinfo.value)


def test_load_source_rejects_a_directory_without_stack_txt(config_tree, tmp_path):
    from uv_stack.operations.diff import load_source

    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(ConfigError):
        load_source(config_tree, str(plain))


def test_load_source_rejects_a_dangling_symlink(config_tree, tmp_path):
    from uv_stack.operations.diff import load_source

    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(ConfigError):
        load_source(config_tree, str(link))


def test_load_source_names_the_missing_lock_for_a_named_environment(config_tree):
    from uv_stack.operations.diff import load_source

    with pytest.raises(ConfigError) as excinfo:
        load_source(config_tree, "main")
    message = f"{excinfo.value.message} {excinfo.value.hint}"
    assert str(config_tree.env_requirements_lock("main")) in message
    assert "stack sync env" in message


def test_load_source_names_the_missing_lock_for_a_copied_directory(config_tree, tmp_path):
    import shutil

    from uv_stack.operations.diff import load_source

    copied = tmp_path / "elsewhere" / "main"
    copied.parent.mkdir()
    shutil.copytree(config_tree.env_dir("main"), copied)

    with pytest.raises(ConfigError) as excinfo:
        load_source(config_tree, str(copied))
    message = f"{excinfo.value.message} {excinfo.value.hint}"
    assert str(copied / "requirements.lock.txt") in message
    assert "stack sync env" not in message


@pytest.mark.skipif(_IS_ROOT, reason="root searches regardless of mode")
def test_load_source_reports_an_unreadable_envs_directory(config_tree):
    from uv_stack.operations.diff import load_source

    config_tree.envs_dir.chmod(0o000)
    try:
        with pytest.raises(ConfigError) as excinfo:
            load_source(config_tree, "main")
        assert str(config_tree.envs_dir) in str(excinfo.value)
    finally:
        config_tree.envs_dir.chmod(0o755)


def _source(label, pins, python=None, micromamba=None, channels=None):
    from uv_stack.operations.diff import DiffSource

    return DiffSource(
        label=label, pins=pins, python=python, micromamba=micromamba, channels=channels
    )


def _env_source(label, pins, python="3.12", micromamba=None, channels=None):
    return _source(
        label,
        pins,
        python=python,
        micromamba=micromamba or [],
        channels=channels or ["conda-forge"],
    )


def test_diff_reports_identical_when_every_layer_matches():
    from uv_stack.operations.diff import VERDICT_IDENTICAL, diff_environments

    result = diff_environments(
        _env_source("a", {"numpy": "2.1.0"}), _env_source("b", {"numpy": "2.1.0"})
    )

    assert result.verdict == VERDICT_IDENTICAL
    assert result.pins.is_empty()
    assert result.micromamba is not None and result.micromamba.is_empty()


def test_diff_reports_identical_where_comparable_for_a_bare_lock():
    from uv_stack.operations.diff import VERDICT_WHERE_COMPARABLE, diff_environments

    result = diff_environments(
        _env_source("a", {"numpy": "2.1.0"}), _source("a.lock.txt", {"numpy": "2.1.0"})
    )

    assert result.verdict == VERDICT_WHERE_COMPARABLE
    assert result.python is None
    assert result.micromamba is None
    assert result.channels is None


def test_diff_reports_a_python_only_difference():
    from uv_stack.operations.diff import VERDICT_DIFFERENT, diff_environments

    result = diff_environments(
        _env_source("a", {"numpy": "2.1.0"}, python="3.12"),
        _env_source("b", {"numpy": "2.1.0"}, python="3.13"),
    )

    assert result.verdict == VERDICT_DIFFERENT
    assert result.python == ("3.12", "3.13")
    assert result.pins.is_empty()


def test_diff_reports_a_micromamba_only_difference():
    from uv_stack.operations.diff import VERDICT_DIFFERENT, diff_environments

    result = diff_environments(
        _env_source("a", {}, micromamba=["gdal"]), _env_source("b", {}, micromamba=[])
    )

    assert result.verdict == VERDICT_DIFFERENT
    assert result.micromamba is not None
    assert result.micromamba.only_in_a == ["gdal"]
    assert result.micromamba.only_in_b == []


def test_diff_reports_a_channel_reorder_as_a_difference():
    from uv_stack.operations.diff import VERDICT_DIFFERENT, diff_environments

    result = diff_environments(
        _env_source("a", {}, channels=["conda-forge", "bioconda"]),
        _env_source("b", {}, channels=["bioconda", "conda-forge"]),
    )

    assert result.verdict == VERDICT_DIFFERENT
    assert result.channels == (["conda-forge", "bioconda"], ["bioconda", "conda-forge"])


def test_diff_populates_all_three_pin_groups():
    from uv_stack.operations.diff import PinChange, PinEntry, diff_environments

    result = diff_environments(
        _env_source("a", {"numpy": "2.1.0", "rich": "13.7.0", "-e /src/tool": None}),
        _env_source("b", {"pandas": "2.0.0", "rich": "14.0.0"}),
    )

    assert result.pins.only_in_a == [PinEntry("-e /src/tool", None), PinEntry("numpy", "2.1.0")]
    assert result.pins.only_in_b == [PinEntry("pandas", "2.0.0")]
    assert result.pins.version_differs == [PinChange("rich", "13.7.0", "14.0.0")]


def test_diff_never_puts_an_editable_in_version_differs():
    from uv_stack.operations.diff import diff_environments

    result = diff_environments(
        _env_source("a", {"-e /src/one": None}), _env_source("b", {"-e /src/two": None})
    )

    assert result.pins.version_differs == []
    assert [entry.name for entry in result.pins.only_in_a] == ["-e /src/one"]
    assert [entry.name for entry in result.pins.only_in_b] == ["-e /src/two"]


def test_name_spellings_of_one_distribution_compare_identical(tmp_path):
    from uv_stack.operations.diff import diff_environments, parse_lock

    a = _source("a", parse_lock(_lock(tmp_path, "Foo_Bar==1.0\n", "a.lock.txt")))
    b = _source("b", parse_lock(_lock(tmp_path, "foo-bar==1.0\n", "b.lock.txt")))

    assert diff_environments(a, b).pins.is_empty()


def _declare_env(
    config, name, *, python="3.12", micromamba=(), channels=(), pins="numpy==2.1.0\n"
):
    directory = config.env_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stack.txt").write_text("standard\n")
    (directory / "python.txt").write_text(f"{python}\n")
    (directory / "micromamba.txt").write_text("".join(f"{entry}\n" for entry in micromamba))
    (directory / "channels.txt").write_text("".join(f"{entry}\n" for entry in channels))
    (directory / "requirements.lock.txt").write_text(pins)
    return directory


def test_an_empty_channels_file_matches_an_explicit_conda_forge(config_tree):
    from uv_stack.operations.diff import VERDICT_IDENTICAL, diff_environments, load_source

    _declare_env(config_tree, "left", channels=())
    _declare_env(config_tree, "right", channels=("conda-forge",))

    result = diff_environments(
        load_source(config_tree, "left"), load_source(config_tree, "right")
    )
    assert result.verdict == VERDICT_IDENTICAL


def test_first_wins_deduplication_makes_reordered_specs_differ(config_tree):
    from uv_stack.operations.diff import VERDICT_DIFFERENT, diff_environments, load_source

    _declare_env(config_tree, "left", micromamba=("numpy=1.26", "numpy=2.0"))
    _declare_env(config_tree, "right", micromamba=("numpy=2.0", "numpy=1.26"))

    result = diff_environments(
        load_source(config_tree, "left"), load_source(config_tree, "right")
    )
    assert result.verdict == VERDICT_DIFFERENT
    assert result.micromamba is not None
    assert result.micromamba.only_in_a == ["numpy=1.26"]
    assert result.micromamba.only_in_b == ["numpy=2.0"]


def test_a_stray_pip_or_python_entry_is_not_drift(config_tree):
    from uv_stack.operations.diff import VERDICT_IDENTICAL, diff_environments, load_source

    _declare_env(config_tree, "left", micromamba=("pip", "python=3.12"))
    _declare_env(config_tree, "right", micromamba=())

    result = diff_environments(
        load_source(config_tree, "left"), load_source(config_tree, "right")
    )
    assert result.verdict == VERDICT_IDENTICAL
    assert result.micromamba is not None
    assert result.micromamba.only_in_a == []


def test_an_environment_matches_a_copy_of_its_own_directory(config_tree, tmp_path):
    import shutil

    from uv_stack.operations.diff import VERDICT_IDENTICAL, diff_environments, load_source

    _declare_env(config_tree, "left", micromamba=("gdal",), channels=("bioconda",))
    copied = tmp_path / "elsewhere" / "left"
    copied.parent.mkdir()
    shutil.copytree(config_tree.env_dir("left"), copied)

    result = diff_environments(
        load_source(config_tree, "left"), load_source(config_tree, str(copied))
    )
    assert result.verdict == VERDICT_IDENTICAL


def test_an_environment_matches_its_own_lock_file_where_comparable(config_tree):
    from uv_stack.operations.diff import (
        VERDICT_WHERE_COMPARABLE,
        diff_environments,
        load_source,
    )

    _declare_env(config_tree, "left")
    lock = config_tree.env_requirements_lock("left")

    result = diff_environments(
        load_source(config_tree, "left"), load_source(config_tree, str(lock))
    )
    assert result.verdict == VERDICT_WHERE_COMPARABLE
    assert result.pins.is_empty()


def test_parse_lock_admits_an_unnamed_archive_path(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(tmp_path, "numpy==2.1.0\n/Users/me/ldclient-2024.1.4.tar.gz\n")
    result = parse_lock(path)
    
    assert result == {
        "numpy": "2.1.0",
        "/Users/me/ldclient-2024.1.4.tar.gz": None,
    }


def test_parse_lock_rejects_an_unnamed_line_with_whitespace(tmp_path):
    from uv_stack.operations.diff import parse_lock

    path = _lock(tmp_path, "numpy==2.1.0\nsee docs/README for details\n")
    with pytest.raises(ConfigError) as excinfo:
        parse_lock(path)
    assert "line 2" in str(excinfo.value)

    # Also reject a tab between slash-bearing parts
    path_tab = _lock(tmp_path, "numpy==2.1.0\na/b\tc/d\n", "tab.lock")
    with pytest.raises(ConfigError) as excinfo:
        parse_lock(path_tab)
    assert "line 2" in str(excinfo.value)


def test_parse_lock_refuses_a_fifo_without_blocking(tmp_path):
    from uv_stack.operations.diff import parse_lock

    fifo = tmp_path / "pipe.lock"
    os.mkfifo(fifo)
    # parse_lock called directly on a FIFO must refuse it, not block.
    with _deadline(5.0), pytest.raises(ConfigError) as excinfo:
        parse_lock(fifo)
    assert "named pipe" in str(excinfo.value)


def test_load_source_refuses_a_fifo_env_lock_without_blocking(config_tree):
    from uv_stack.operations.diff import load_source

    lock_path = config_tree.env_requirements_lock("main")
    os.mkfifo(lock_path)
    # A FIFO at the env's lock path must refuse it, not block.
    with _deadline(5.0), pytest.raises(ConfigError) as excinfo:
        load_source(config_tree, "main")
    assert "named pipe" in str(excinfo.value)


def test_load_source_names_a_non_regular_env_lock(config_tree):
    from uv_stack.operations.diff import load_source

    lock_path = config_tree.env_requirements_lock("main")
    lock_path.mkdir(parents=True, exist_ok=True)
    # A directory at the lock path should be named as "a directory", not "has no lock".
    with pytest.raises(ConfigError) as excinfo:
        load_source(config_tree, "main")
    assert "a directory" in str(excinfo.value)
    assert "has no lock" not in str(excinfo.value)


def test_diff_unnamed_requirements_compare_by_literal_token(tmp_path):
    from uv_stack.operations.diff import diff_environments, parse_lock

    # Same path on both sides should be identical
    a = _source("a", parse_lock(_lock(tmp_path, "/path/to/pkg.tar.gz\n", "a.lock")))
    b = _source("b", parse_lock(_lock(tmp_path, "/path/to/pkg.tar.gz\n", "b.lock")))
    assert diff_environments(a, b).pins.is_empty()
    
    # Different paths should differ
    c = _source("c", parse_lock(_lock(tmp_path, "/path/one.tar.gz\n", "c.lock")))
    d = _source("d", parse_lock(_lock(tmp_path, "/path/two.tar.gz\n", "d.lock")))
    diff = diff_environments(c, d)
    assert [e.name for e in diff.pins.only_in_a] == ["/path/one.tar.gz"]
    assert [e.name for e in diff.pins.only_in_b] == ["/path/two.tar.gz"]


def test_diff_pin_only_drift_is_different():
    from uv_stack.operations.diff import VERDICT_DIFFERENT, diff_environments

    # Two sources differing only in one pin's version should be DIFFERENT
    result = diff_environments(
        _env_source("a", {"numpy": "2.1.0"}),
        _env_source("b", {"numpy": "2.2.0"}),
    )

    assert result.verdict == VERDICT_DIFFERENT
    assert len(result.pins.version_differs) == 1
    assert result.pins.version_differs[0].name == "numpy"


def _run(config, *args):
    return CliRunner().invoke(cli, ["--root", str(config.root), *args])


def test_diff_json_payload_for_two_environments(config_tree):
    _declare_env(
        config_tree,
        "left",
        micromamba=("gdal",),
        channels=("bioconda",),
        pins="numpy==2.1.0\nrich==13.7.0\n",
    )
    _declare_env(config_tree, "right", pins="rich==14.0.0\n-e /src/tool\n")

    result = _run(config_tree, "diff", "left", "right", "--json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "a": "left",
        "b": "right",
        "verdict": "different",
        "python": {"a": "3.12", "b": "3.12"},
        "micromamba": {"only_in_a": ["gdal"], "only_in_b": []},
        "channels": {"a": ["conda-forge", "bioconda"], "b": ["conda-forge"]},
        "pins": {
            "only_in_a": [{"name": "numpy", "version": "2.1.0"}],
            "only_in_b": [{"name": "-e /src/tool", "version": None}],
            "version_differs": [{"name": "rich", "a": "13.7.0", "b": "14.0.0"}],
        },
    }


def test_diff_json_payload_for_an_identical_pair(config_tree):
    _declare_env(config_tree, "left")
    _declare_env(config_tree, "right")

    result = _run(config_tree, "diff", "left", "right", "--json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "a": "left",
        "b": "right",
        "verdict": "identical",
        "python": {"a": "3.12", "b": "3.12"},
        "micromamba": {"only_in_a": [], "only_in_b": []},
        "channels": {"a": ["conda-forge"], "b": ["conda-forge"]},
        "pins": {"only_in_a": [], "only_in_b": [], "version_differs": []},
    }


def test_diff_json_sorts_every_difference_collection(config_tree):
    # Each side lists its entries out of order, so a collection that merely
    # preserved input order would fail here.
    _declare_env(
        config_tree,
        "left",
        micromamba=("zlib", "gdal"),
        pins="zope==1.0\nattrs==1.0\ntyper==1.0\nclick==8.0\n",
    )
    _declare_env(config_tree, "right", pins="yarl==1.0\nanyio==1.0\ntyper==2.0\nclick==8.1\n")

    payload = json.loads(_run(config_tree, "diff", "left", "right", "--json").output)

    assert payload["micromamba"]["only_in_a"] == ["gdal", "zlib"]
    assert [entry["name"] for entry in payload["pins"]["only_in_a"]] == ["attrs", "zope"]
    assert [entry["name"] for entry in payload["pins"]["only_in_b"]] == ["anyio", "yarl"]
    differs = payload["pins"]["version_differs"]
    assert [change["name"] for change in differs] == ["click", "typer"]


def test_diff_json_does_not_sort_channels(config_tree):
    # Sorted, these would read bioconda first; the effective order is the
    # resolution priority the comparison reports on, so it must survive.
    _declare_env(config_tree, "left", channels=("bioconda",))
    _declare_env(config_tree, "right", channels=("bioconda",))

    payload = json.loads(_run(config_tree, "diff", "left", "right", "--json").output)

    assert payload["channels"]["a"] == ["conda-forge", "bioconda"]
    assert payload["channels"]["b"] == ["conda-forge", "bioconda"]
    assert payload["verdict"] == "identical"


def test_diff_json_nulls_every_layer_a_bare_lock_cannot_see(config_tree):
    _declare_env(config_tree, "left")
    lock = config_tree.env_requirements_lock("left")

    payload = json.loads(_run(config_tree, "diff", "left", str(lock), "--json").output)

    assert payload["python"] is None
    assert payload["micromamba"] is None
    assert payload["channels"] is None
    assert payload["verdict"] == "identical-where-comparable"


def test_diff_exits_zero_by_default_when_sources_differ(config_tree):
    _declare_env(config_tree, "left", pins="numpy==2.1.0\n")
    _declare_env(config_tree, "right", pins="numpy==2.2.0\n")

    result = _run(config_tree, "diff", "left", "right")

    assert result.exit_code == 0, result.output
    assert "Verdict: different" in result.output


def test_diff_exit_code_flag_exits_one_when_sources_differ(config_tree):
    _declare_env(config_tree, "left", pins="numpy==2.1.0\n")
    _declare_env(config_tree, "right", pins="numpy==2.2.0\n")

    result = _run(config_tree, "diff", "left", "right", "--exit-code")

    # CliRunner also reports 1 for an uncaught exception, so pin that the
    # comparison rendered in full before the deliberate exit.
    assert result.exit_code == 1
    assert "Verdict: different" in result.output


def test_diff_exit_code_flag_exits_zero_when_only_a_layer_was_unseen(config_tree):
    _declare_env(config_tree, "left")
    lock = config_tree.env_requirements_lock("left")

    result = _run(config_tree, "diff", "left", str(lock), "--exit-code")

    assert result.exit_code == 0, result.output
    assert "Verdict: identical-where-comparable" in result.output


@pytest.mark.parametrize("extra", [[], ["--exit-code"]])
def test_diff_exits_one_on_a_missing_source(config_tree, tmp_path, monkeypatch, extra):
    # 'nope' is tried as a path relative to the working directory, so pin it
    # to one known to hold nothing of that name. The wide console keeps the
    # error panel from folding the message.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "1000")
    _declare_env(config_tree, "left")

    result = _run(config_tree, "diff", "left", "nope", *extra)

    assert result.exit_code == 1
    assert "No source named nope." in result.output


def test_diff_says_which_layers_a_bare_lock_hid(config_tree):
    _declare_env(config_tree, "left")
    lock = config_tree.env_requirements_lock("left")

    output = _run(config_tree, "diff", "left", str(lock)).output

    assert "not compared" in output


def test_diff_renders_a_bracketed_identity_intact(config_tree, monkeypatch):
    # Wide enough that rich never wraps the cell, so a missing '[extra]' can
    # only mean it was parsed as markup.
    monkeypatch.setenv("COLUMNS", "1000")
    _declare_env(config_tree, "left", pins="-e /src/tool[extra]\n")
    _declare_env(config_tree, "right", pins="numpy==2.1.0\n")

    output = _run(config_tree, "diff", "left", "right").output

    assert "/src/tool[extra]" in output


def test_diff_renders_a_bracketed_label_intact(config_tree, tmp_path):
    import shutil

    _declare_env(config_tree, "left")
    copied = tmp_path / "from[laptop]" / "left"
    copied.parent.mkdir()
    shutil.copytree(config_tree.env_dir("left"), copied)

    output = _run(config_tree, "diff", "left", str(copied)).output

    assert "from[laptop]" in output


def test_diff_labels_an_unnamed_requirement_as_versionless(config_tree, monkeypatch):
    # An archive path has no version but is not editable, so the label must
    # not claim it is.
    monkeypatch.setenv("COLUMNS", "1000")
    _declare_env(config_tree, "left", pins="./wheels/tool-1.0.tar.gz\n")
    _declare_env(config_tree, "right", pins="numpy==2.1.0\n")

    output = _run(config_tree, "diff", "left", "right").output

    assert "./wheels/tool-1.0.tar.gz" in output
    assert "(no version)" in output
    assert "(editable)" not in output
