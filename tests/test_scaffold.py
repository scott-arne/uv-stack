from __future__ import annotations

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
