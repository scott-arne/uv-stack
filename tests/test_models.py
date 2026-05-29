from uvstack.models import Bundle, EnvConfig, Profile, ResolvedStack


def test_profile_from_lines_cleans_input():
    p = Profile.from_lines("ds", ["numpy  # core", "", "# comment", "pandas"])
    assert p.name == "ds"
    assert p.requirements == ["numpy", "pandas"]


def test_bundle_from_lines_cleans_input():
    b = Bundle.from_lines("qsar", ["ds", "chem  # chemistry", "", "umap-learn"])
    assert b.name == "qsar"
    assert b.tokens == ["ds", "chem", "umap-learn"]


def test_env_config_defaults():
    env = EnvConfig(name="main", stack=["@full"])
    assert env.python == "3.12"
    assert env.micromamba == []


def test_resolved_stack_defaults_empty():
    rs = ResolvedStack()
    assert rs.profiles == []
    assert rs.inline == []
