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
    classified = Resolver(config_tree).classify(["package:rdkit", "chem", "@standard"]).entries
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
    out = Resolver(config_tree).classify(["standard", "ds", "numpy"]).entries
    # standard is a bundle (NOT expanded), ds a profile, numpy a literal package.
    assert out == ["bundle:standard", "profile:ds", "package:numpy"]


def test_classify_honors_explicit_prefixes(config_tree: ConfigRoot):
    out = Resolver(config_tree).classify(
        ["@standard", "bundle:qsar", "profile:chem", "pkg:ds"]
    ).entries
    # @ and bundle: -> bundle:; profile: passthrough; pkg: -> package: even though
    # a profile named "ds" exists.
    assert out == ["bundle:standard", "bundle:qsar", "profile:chem", "package:ds"]


def test_classify_dedups_preserving_order(config_tree: ConfigRoot):
    out = Resolver(config_tree).classify(["ds", "ds", "@standard", "standard"]).entries
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


def test_resolve_warns_on_near_miss(config_tree: ConfigRoot):
    # "standrd" is one edit from the "standard" bundle.
    rs = Resolver(config_tree).resolve(["standrd"])
    assert rs.inline == ["standrd"]
    assert rs.warnings == [
        "'standrd' resolved to a literal package; did you mean 'standard'? "
        "(use pkg:standrd to silence)"
    ]


def test_resolve_no_warning_for_ordinary_package(config_tree: ConfigRoot):
    rs = Resolver(config_tree).resolve(["numpy"])
    assert rs.inline == ["numpy"]
    assert rs.warnings == []


def test_resolve_warns_on_profile_shadowing_bundle(config_tree: ConfigRoot):
    # Create a bundle named like the existing "ds" profile.
    config_tree.bundle_path("ds").write_text("includes:\n  - rich\n")
    rs = Resolver(config_tree).resolve(["ds"])
    assert rs.profiles == ["ds"]
    assert rs.warnings == [
        "'ds' matches both a profile and a bundle; using the profile "
        "(use @ds for the bundle)"
    ]


def test_strict_rejects_bare_literal(config_tree: ConfigRoot):
    with pytest.raises(ResolutionError) as excinfo:
        Resolver(config_tree, strict=True).resolve(["numpy"])
    assert str(excinfo.value) == "Unqualified token 'numpy' resolved to a literal package."
    hint = excinfo.value.hint
    assert hint == "Use pkg:numpy for a literal package, or fix the profile/bundle name."


def test_strict_exempts_qualified_and_non_name_tokens(config_tree: ConfigRoot):
    rs = Resolver(config_tree, strict=True).resolve(
        ["pkg:numpy", "package:pandas", "numpy>=2", "-e /path/pkg",
         "/path/pkg.tar.gz", "ds"]
    )
    assert rs.profiles == ["ds"]
    assert rs.inline == ["numpy", "pandas", "numpy>=2", "-e /path/pkg",
                         "/path/pkg.tar.gz"]
    assert rs.warnings == []


def test_strict_exempts_flags_and_dot_paths(config_tree: ConfigRoot):
    rs = Resolver(config_tree, strict=True).resolve([".", "..", "--pre"])
    assert rs.profiles == []
    assert rs.inline == [".", "..", "--pre"]
    assert rs.warnings == []


def test_strict_applies_inside_bundles(config_tree: ConfigRoot):
    config_tree.bundle_path("typo").write_text("includes:\n  - numpyy\n")
    with pytest.raises(ResolutionError) as excinfo:
        Resolver(config_tree, strict=True).resolve(["@typo"])
    assert "Unqualified token 'numpyy'" in str(excinfo.value)


def test_classify_returns_entries_and_warnings(config_tree: ConfigRoot):
    result = Resolver(config_tree).classify(["standrd", "ds"])
    assert result.entries == ["package:standrd", "profile:ds"]
    assert result.warnings == [
        "'standrd' resolved to a literal package; did you mean 'standard'? "
        "(use pkg:standrd to silence)"
    ]


def test_classify_strict_rejects_bare_literal(config_tree: ConfigRoot):
    with pytest.raises(ResolutionError):
        Resolver(config_tree, strict=True).classify(["numpyy"])


def test_warnings_are_deduplicated(config_tree: ConfigRoot):
    rs = Resolver(config_tree).resolve(["standrd", "standrd"])
    assert len(rs.warnings) == 1


def test_flatten_matches_resolve_packages(config_tree: ConfigRoot):
    resolver = Resolver(config_tree)
    stack = resolver.resolve(["standard", "umap-learn"])
    assert resolver.flatten(stack) == resolver.resolve_packages(
        ["standard", "umap-learn"]
    )


def test_resolve_warns_on_bundle_self_reference(config_tree: ConfigRoot):
    """A bundle already on disk that names itself warns instead of expanding silently."""
    config_tree.bundle_path("loop").write_text("includes:\n  - '@loop'\n  - pkg:rich\n")
    stack = Resolver(config_tree).resolve(["@loop"])
    assert stack.inline == ["rich"]
    assert any("Bundle cycle skipped: loop -> loop" in w for w in stack.warnings)


def test_resolve_warns_on_mutual_bundle_cycle(config_tree: ConfigRoot):
    config_tree.bundle_path("left").write_text("includes:\n  - '@right'\n")
    config_tree.bundle_path("right").write_text("includes:\n  - '@left'\n  - pkg:rich\n")
    stack = Resolver(config_tree).resolve(["@left"])
    assert stack.inline == ["rich"]
    assert any("left -> right -> left" in w for w in stack.warnings)


def test_resolve_diamond_bundle_reference_is_silent(config_tree: ConfigRoot):
    """Two paths to the same bundle are a diamond, not a cycle — no warning."""
    config_tree.bundle_path("shared").write_text("includes:\n  - pkg:rich\n")
    config_tree.bundle_path("mid").write_text("includes:\n  - '@shared'\n")
    config_tree.bundle_path("top").write_text("includes:\n  - '@shared'\n  - '@mid'\n")
    stack = Resolver(config_tree).resolve(["@top"])
    assert stack.inline == ["rich"]
    assert stack.warnings == []
