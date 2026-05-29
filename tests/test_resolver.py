from __future__ import annotations

import pytest

from uvstack.config import ConfigRoot
from uvstack.errors import ResolutionError
from uvstack.resolver import Resolver


def test_unqualified_profile_then_bundle_then_literal(config_tree: ConfigRoot):
    rs = Resolver(config_tree).resolve(["ds", "standard", "umap-learn"])
    # ds is a profile; standard is a bundle expanding to ds, chem, utils;
    # umap-learn is a literal package.
    assert rs.profiles == ["ds", "chem", "utils"]
    assert rs.inline == ["umap-learn"]


def test_explicit_prefixes(config_tree: ConfigRoot):
    rs = Resolver(config_tree).resolve(
        ["profile:chem", "bundle:standard", "pkg:ds", "@qsar"]
    )
    # profile:chem -> chem; bundle:standard -> ds,chem,utils; pkg:ds -> literal "ds";
    # @qsar -> standard(ds,chem,utils) + umap-learn
    assert rs.profiles == ["chem", "ds", "utils"]
    assert rs.inline == ["ds", "umap-learn"]


def test_editable_and_archive_are_inline(config_tree: ConfigRoot):
    rs = Resolver(config_tree).resolve(["-e /path/pkg", "/path/pkg.tar.gz"])
    assert rs.profiles == []
    assert rs.inline == ["-e /path/pkg", "/path/pkg.tar.gz"]


def test_dedup_preserves_first_order(config_tree: ConfigRoot):
    rs = Resolver(config_tree).resolve(["ds", "ds", "standard"])
    assert rs.profiles == ["ds", "chem", "utils"]


def test_missing_explicit_profile_raises(config_tree: ConfigRoot):
    with pytest.raises(ResolutionError):
        Resolver(config_tree).resolve(["profile:ghost"])


def test_missing_explicit_bundle_raises(config_tree: ConfigRoot):
    with pytest.raises(ResolutionError):
        Resolver(config_tree).resolve(["@ghost"])


def test_bundle_cycle_is_safe(config_tree: ConfigRoot):
    # a -> b -> a
    (config_tree.bundles_dir / "a.bundle").write_text("b\n")
    (config_tree.bundles_dir / "b.bundle").write_text("a\nnumpy-extra\n")
    rs = Resolver(config_tree).resolve(["@a"])
    assert rs.inline == ["numpy-extra"]
