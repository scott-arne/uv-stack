import pytest
from pydantic import ValidationError

from uv_stack.models import Bundle, EnvConfig, Profile, ProjectTracking, ResolvedStack


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


def test_project_tracking_accepts_an_absent_or_plain_python():
    assert ProjectTracking(stack=["ds"]).python is None
    assert ProjectTracking(stack=["ds"], python="3.12").python == "3.12"
    assert ProjectTracking(stack=["ds"], python="main").python == "main"


def test_project_tracking_rejects_an_empty_python():
    """An emptied value must not read as "no preference".

    The selector falls through to the machine default for '', so without this
    the user who cleared the key silently gets a different interpreter than the
    one they were trying to change.
    """
    with pytest.raises(ValidationError) as excinfo:
        ProjectTracking(stack=["ds"], python="")
    assert "must not be empty" in str(excinfo.value)
    with pytest.raises(ValidationError):
        ProjectTracking(stack=["ds"], python="   ")


def test_project_tracking_rejects_a_padded_python():
    """Padding defeats the version test, so ' 3.12 ' is taken for an env name.

    Refused rather than stripped: the file stays the user's, and the failure
    names the real problem instead of quietly resolving it one way.
    """
    with pytest.raises(ValidationError) as excinfo:
        ProjectTracking(stack=["ds"], python=" 3.12 ")
    assert "must not be padded" in str(excinfo.value)
    with pytest.raises(ValidationError):
        ProjectTracking(stack=["ds"], python="3.12\n")
