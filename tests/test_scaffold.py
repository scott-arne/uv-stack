from __future__ import annotations

from unittest import mock

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.operations.scaffold import (
    write_bundle,
    write_env_sources,
    write_profile,
)


def test_write_profile_round_trips(config_tree: ConfigRoot):
    path = write_profile(
        config_tree, "viz", ["matplotlib", "seaborn"],
        description="Plotting", tags=["viz"],
    )
    assert path == config_tree.profile_path("viz")
    prof = config_tree.load_profile("viz")
    assert prof.includes == ["matplotlib", "seaborn"]
    assert prof.description == "Plotting"
    assert prof.tags == ["viz"]


def test_write_profile_minimal_omits_optional_keys(config_tree: ConfigRoot):
    path = write_profile(config_tree, "tiny", ["rich"])
    text = path.read_text()
    assert "description" not in text
    assert "tags" not in text
    assert config_tree.load_profile("tiny").includes == ["rich"]


def test_write_profile_refuses_overwrite(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_profile(config_tree, "ds", ["numpy"])
    assert "Profile 'ds' already exists" in str(excinfo.value)


def test_write_bundle_round_trips(config_tree: ConfigRoot):
    write_bundle(config_tree, "daily", ["ds", "pkg:httpx"], tags=["core"])
    bundle = config_tree.load_bundle("daily")
    assert bundle.includes == ["ds", "pkg:httpx"]
    assert bundle.tags == ["core"]


def test_write_bundle_refuses_overwrite(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "standard", ["ds"])
    assert "Bundle 'standard' already exists" in str(excinfo.value)


def test_write_env_sources_creates_stack_and_python(config_tree: ConfigRoot):
    written = write_env_sources(
        config_tree, "fresh", ["@standard", "httpx"], python="3.13"
    )
    assert config_tree.env_stack_path("fresh").read_text() == "@standard\nhttpx\n"
    assert config_tree.env_python_path("fresh").read_text() == "3.13\n"
    assert written == [
        config_tree.env_stack_path("fresh"),
        config_tree.env_python_path("fresh"),
    ]


def test_write_env_sources_without_python(config_tree: ConfigRoot):
    written = write_env_sources(config_tree, "nopy", ["ds"])
    assert written == [config_tree.env_stack_path("nopy")]
    assert not config_tree.env_python_path("nopy").exists()


def test_write_env_sources_refuses_existing_stack(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "main", ["ds"])
    assert "already has a stack.txt" in str(excinfo.value)


def test_no_temp_litter(config_tree: ConfigRoot):
    write_profile(config_tree, "clean", ["rich"])
    leftovers = list(config_tree.profiles_dir.glob("*.tmp"))
    assert leftovers == []


def test_write_starter_profile(config_tree: ConfigRoot):
    from uv_stack.operations.scaffold import write_starter_profile

    path = write_starter_profile(config_tree)
    assert path == config_tree.profile_path("starter")
    assert path.read_text().startswith("# A profile is a reusable")
    assert config_tree.load_profile("starter").includes == ["rich"]
    with pytest.raises(ConfigError):
        write_starter_profile(config_tree)


def test_write_profile_rejects_path_traversal(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_profile(config_tree, "a/b", ["pkg"])
    assert "Invalid profile name: 'a/b'" in str(excinfo.value)


def test_write_profile_rejects_dot_segments(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_profile(config_tree, "..", ["pkg"])
    assert "Invalid profile name: '..'" in str(excinfo.value)


def test_write_profile_rejects_empty_name(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_profile(config_tree, "", ["pkg"])
    assert "Invalid profile name: ''" in str(excinfo.value)


def test_write_bundle_rejects_path_traversal(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "a/b", ["ds"])
    assert "Invalid bundle name: 'a/b'" in str(excinfo.value)


def test_write_bundle_rejects_dot_segments(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "..", ["ds"])
    assert "Invalid bundle name: '..'" in str(excinfo.value)


def test_write_bundle_rejects_empty_name(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "", ["ds"])
    assert "Invalid bundle name: ''" in str(excinfo.value)


def test_write_env_sources_rejects_path_traversal(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "a/b", ["ds"])
    assert "Invalid environment name: 'a/b'" in str(excinfo.value)


def test_write_env_sources_rejects_dot_segments(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "..", ["ds"])
    assert "Invalid environment name: '..'" in str(excinfo.value)


def test_write_env_sources_rejects_empty_name(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "", ["ds"])
    assert "Invalid environment name: ''" in str(excinfo.value)


def test_write_env_sources_refuses_existing_python_txt(config_tree: ConfigRoot):
    config_tree.env_python_path("orphan").parent.mkdir(parents=True, exist_ok=True)
    config_tree.env_python_path("orphan").write_text("3.12\n")
    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "orphan", ["ds"], python="3.13")
    assert "already has a python.txt" in str(excinfo.value)
    assert not config_tree.env_stack_path("orphan").exists()


def test_write_env_sources_rollback_on_python_failure(config_tree: ConfigRoot):
    from uv_stack.fsutil import atomic_write_new

    call_count = 0

    def failing_write_new(path, text):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call (stack.txt) succeeds.
            return atomic_write_new(path, text)
        else:
            # Second call (python.txt) fails.
            raise RuntimeError("Simulated failure")

    with mock.patch("uv_stack.operations.scaffold.atomic_write_new", side_effect=failing_write_new):
        with pytest.raises(RuntimeError):
            write_env_sources(config_tree, "rollback-test", ["ds"], python="3.13")

    assert not config_tree.env_stack_path("rollback-test").exists()
    assert not config_tree.env_python_path("rollback-test").exists()


def test_write_env_sources_rollback_preserves_concurrent_replacement(config_tree: ConfigRoot):
    """Rollback should not unlink stack.txt if a concurrent process replaced it."""
    from uv_stack import operations

    original_publish = operations.scaffold._publish
    call_count = 0
    stack_path = config_tree.env_stack_path("concurrent-test")

    def publish_with_concurrent_replacement(path, text, message, hint):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call (stack.txt) succeeds normally.
            return original_publish(path, text, message, hint)
        else:
            # Second call (python.txt): first replace stack.txt, then fail.
            # Replacement has different content and a new inode.
            stack_path.unlink()
            stack_path.write_text("REPLACED\n")
            raise RuntimeError("Simulated failure after replacement")

    with mock.patch(
        "uv_stack.operations.scaffold._publish",
        side_effect=publish_with_concurrent_replacement,
    ):
        with pytest.raises(RuntimeError):
            write_env_sources(config_tree, "concurrent-test", ["ds"], python="3.13")

    # The replacement file should survive the rollback.
    assert stack_path.exists()
    assert stack_path.read_text() == "REPLACED\n"
    assert not config_tree.env_python_path("concurrent-test").exists()
