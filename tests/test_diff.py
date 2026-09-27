"""Tests for stack diff: source classification, lock grammar, comparison."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import _deadline
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
