import pytest
from pydantic import ValidationError

from uv_stack.models import Bundle, EnvConfig, Profile, ResolvedStack


def test_profile_fields_and_defaults():
    p = Profile(name="ds", includes=["numpy", "pandas"])
    assert p.name == "ds"
    assert p.includes == ["numpy", "pandas"]
    assert p.description is None
    assert p.tags == []


def test_profile_with_description_and_tags():
    p = Profile(name="ds", description="Core DS", tags=["data"], includes=["numpy"])
    assert p.description == "Core DS"
    assert p.tags == ["data"]


def test_bundle_fields_and_defaults():
    b = Bundle(name="qsar", includes=["ds", "chem", "umap-learn"])
    assert b.name == "qsar"
    assert b.includes == ["ds", "chem", "umap-learn"]
    assert b.description is None
    assert b.tags == []


def test_profile_rejects_unknown_key():
    with pytest.raises(ValidationError):
        Profile(name="ds", includes=["numpy"], bogus="x")


def test_env_config_defaults():
    env = EnvConfig(name="main", stack=["@full"])
    assert env.python == "3.12"
    assert env.micromamba == []


def test_resolved_stack_defaults_empty():
    rs = ResolvedStack()
    assert rs.profiles == []
    assert rs.inline == []
