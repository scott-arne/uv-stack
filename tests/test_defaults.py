from __future__ import annotations

from uv_stack.defaults import DEFAULT_BUNDLES, DEFAULT_PROFILES


def test_expected_profiles_present():
    expected = {
        "ds", "chem", "openeye", "marimo", "jupyter", "viz", "utils",
        "db-cloud", "web", "build", "docs", "dev", "livedesign", "local-editable",
    }
    assert expected <= set(DEFAULT_PROFILES)


def test_expected_bundles_present():
    expected = {"minimal", "standard", "notebook", "openeye", "full", "chemprop", "qsar"}
    assert expected <= set(DEFAULT_BUNDLES)


def test_ds_profile_content():
    assert "numpy" in DEFAULT_PROFILES["ds"]
    assert "pandas" in DEFAULT_PROFILES["ds"]


def test_standard_bundle_content():
    assert "ds" in DEFAULT_BUNDLES["standard"]
    assert "chem" in DEFAULT_BUNDLES["standard"]
