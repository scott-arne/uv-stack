from __future__ import annotations

import pytest

from uv_stack.config import ConfigRoot
from uv_stack.errors import ResolutionError
from uv_stack.resolver import Resolver


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


def test_package_prefix_is_inline(config_tree: ConfigRoot):
    # 'package:' is the canonical inline prefix (what `stack resolve` prints);
    # 'pkg:' is an accepted alias. Both strip to a literal requirement.
    rs = Resolver(config_tree).resolve(["package:rdkit", "pkg:chemprop"])
    assert rs.profiles == []
    assert rs.inline == ["rdkit", "chemprop"]


def test_classify_output_round_trips_as_input(config_tree: ConfigRoot):
    # The specifiers printed by `stack resolve` must be re-readable as stack
    # tokens: resolving the classified output reproduces the same packages.
    classified = Resolver(config_tree).classify(["package:rdkit", "chem", "@standard"])
    assert classified == ["package:rdkit", "profile:chem", "bundle:standard"]
    rs = Resolver(config_tree).resolve(classified)
    assert rs.profiles == ["chem", "ds", "utils"]
    assert rs.inline == ["rdkit"]


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
    config_tree.bundle_path("a").write_text("includes:\n  - b\n")
    config_tree.bundle_path("b").write_text("includes:\n  - a\n  - numpy-extra\n")
    rs = Resolver(config_tree).resolve(["@a"])
    assert rs.inline == ["numpy-extra"]


def test_classify_labels_without_expanding(config_tree: ConfigRoot):
    out = Resolver(config_tree).classify(["standard", "ds", "numpy"])
    # standard is a bundle (NOT expanded), ds a profile, numpy a literal package.
    assert out == ["bundle:standard", "profile:ds", "package:numpy"]


def test_classify_honors_explicit_prefixes(config_tree: ConfigRoot):
    out = Resolver(config_tree).classify(
        ["@standard", "bundle:qsar", "profile:chem", "pkg:ds"]
    )
    # @ and bundle: -> bundle:; profile: passthrough; pkg: -> package: even though
    # a profile named "ds" exists.
    assert out == ["bundle:standard", "bundle:qsar", "profile:chem", "package:ds"]


def test_classify_dedups_preserving_order(config_tree: ConfigRoot):
    out = Resolver(config_tree).classify(["ds", "ds", "@standard", "standard"])
    assert out == ["profile:ds", "bundle:standard"]


def test_resolve_packages_expands_to_flat_list(config_tree: ConfigRoot):
    out = Resolver(config_tree).resolve_packages(["standard", "umap-learn"])
    # ds -> numpy, pandas; chem -> rdkit; utils -> rich; then inline umap-learn.
    assert out == ["numpy", "pandas", "rdkit", "rich", "umap-learn"]


def test_resolve_packages_dedups_across_profiles_and_inline(config_tree: ConfigRoot):
    # Add numpy as an explicit inline package; it already comes from ds.
    out = Resolver(config_tree).resolve_packages(["ds", "numpy"])
    assert out == ["numpy", "pandas"]


def test_resolve_packages_missing_profile_raises(config_tree: ConfigRoot):
    with pytest.raises(ResolutionError):
        Resolver(config_tree).resolve_packages(["profile:ghost"])
