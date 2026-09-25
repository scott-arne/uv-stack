from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.conftest import _deadline, _lock_held_by_another_process
from uv_stack.config import ConfigRoot
from uv_stack.fsutil import _LOCK_AVAILABLE
from uv_stack.operations.doctor import Finding, diagnose, repair
from uv_stack.operations.portable import (
    BEGIN_MARKER,
    END_MARKER,
    render_block,
    write_portable_ignore,
)

_IS_ROOT = getattr(os, "geteuid", lambda: -1)() == 0


def test_clean_tree_has_no_errors(config_tree: ConfigRoot):
    findings = diagnose(config_tree)
    assert [f for f in findings if f.level == "error"] == []


def test_missing_root_reports_error(tmp_path: Path):
    cfg = ConfigRoot(tmp_path / "does-not-exist")
    finding = next(f for f in diagnose(cfg) if f.kind == "missing-root")
    assert finding.level == "error"
    # The wording, not just the path: a finding that names the root and
    # diagnoses nothing about it would satisfy the path on its own.
    assert "Config root does not exist:" in finding.message
    assert str(cfg.root) in finding.message


def test_legacy_profile_in_file_flagged(config_tree: ConfigRoot):
    (config_tree.profiles_dir / "old.in").write_text("numpy\n")
    messages = [f.message for f in diagnose(config_tree)]
    assert any("Legacy profile file:" in m and "old.in" in m for m in messages)


def test_legacy_bundle_file_flagged(config_tree: ConfigRoot):
    (config_tree.bundles_dir / "old.bundle").write_text("ds\n")
    messages = [f.message for f in diagnose(config_tree)]
    assert any("Legacy bundle file:" in m and "old.bundle" in m for m in messages)


def test_legacy_profiles_txt_flagged(config_tree: ConfigRoot):
    (config_tree.env_dir("main") / "profiles.txt").write_text("ds\n")
    messages = [f.message for f in diagnose(config_tree)]
    # "profiles.txt" alone is also in the finding's own path, so it says
    # nothing about the message having diagnosed anything.
    assert any("Legacy profiles.txt in env 'main'" in m for m in messages)


def test_legacy_env_at_root_flagged(config_tree: ConfigRoot):
    # An env-like dir directly under root (not under envs/) with generated files.
    legacy = config_tree.root / "legacyenv"
    legacy.mkdir()
    (legacy / "requirements.in").write_text("# x\n")
    messages = [f.message for f in diagnose(config_tree)]
    assert any(
        "Env-like directory not under envs/:" in m and "legacyenv" in m for m in messages
    )


@pytest.mark.parametrize(
    "shape",
    [
        "directory",
        "dangling-symlink",
        pytest.param(
            "fifo",
            marks=pytest.mark.skipif(
                not hasattr(os, "mkfifo"), reason="platform lacks mkfifo"
            ),
        ),
    ],
)
def test_an_env_with_an_unusable_stack_txt_is_reported(config_tree: ConfigRoot, shape: str):
    # list_envs keeps a child only when its stack.txt is a regular file, so
    # each of these shapes drops the env out of every listing in the program
    # without a word. Doctor used the same test one line down and skipped it
    # too, reporting a clean root for a tree that normal env loading refuses.
    env = config_tree.env_dir("broken")
    env.mkdir()
    stack = env / "stack.txt"
    if shape == "directory":
        stack.mkdir()
    elif shape == "dangling-symlink":
        stack.symlink_to(env / "nowhere")
    else:
        os.mkfifo(stack)
    findings = diagnose(config_tree)
    # Exactly one finding: the missing-python-txt check probes the same
    # stack.txt with os.path.isfile, so it must stay silent rather than stack
    # a second, wrong diagnosis on top of this one.
    assert _kinds(findings) == ["unusable-env"]
    assert findings[0].level == "error"
    assert findings[0].path == stack
    assert "stack.txt that is not a regular file" in findings[0].message


def test_an_env_directory_without_a_stack_txt_is_not_a_finding(
    config_tree: ConfigRoot,
):
    # The probe above is deliberately 'present but unusable', not 'absent'.
    # Dropping its lexists conjunct would leave 'not isfile(stack.txt)', which
    # reads identically on a directory holding nothing but notes -- and an
    # absent stack.txt is not a state load_env rejects, it is simply not an
    # env. Without this the simplification passes the whole suite.
    env = config_tree.env_dir("notes")
    env.mkdir()
    (env / "README").write_text("scratch\n")

    assert diagnose(config_tree) == []


def test_env_missing_python_txt_flagged(config_tree: ConfigRoot):
    config_tree.env_python_path("main").unlink()
    messages = [f.message for f in diagnose(config_tree)]
    # As above: the finding's path ends in python.txt, so the bare filename
    # cannot tell the diagnosis apart from the path it points at. The env name
    # is interleaved with the wording for the same reason -- either half alone
    # is data the finding already carries.
    assert any("Env 'main' missing python.txt (will default to 3.12)" in m for m in messages)


_ENV_SOURCE_ACCESSORS = ["env_python_path", "env_micromamba_path", "env_channels_path"]

_UNUSABLE_SHAPES = [
    "directory",
    "dangling-symlink",
    pytest.param(
        "fifo",
        marks=pytest.mark.skipif(
            not hasattr(os, "mkfifo"), reason="platform lacks mkfifo"
        ),
    ),
]


def _replace_with_unusable(path: Path, shape: str) -> None:
    """Put something present but unreadable where a regular file was."""
    path.unlink()
    if shape == "directory":
        path.mkdir()
    elif shape == "dangling-symlink":
        path.symlink_to(path.parent / "nowhere")
    else:
        os.mkfifo(path)


@pytest.mark.parametrize("accessor", _ENV_SOURCE_ACCESSORS)
@pytest.mark.parametrize("shape", _UNUSABLE_SHAPES)
def test_an_unusable_env_source_is_reported_not_defaulted(
    config_tree: ConfigRoot, accessor: str, shape: str
):
    # read_clean_lines and first_clean_line answer for an unreadable path
    # exactly what they answer for an absent one, so none of these shapes was
    # a failure to load: the value was replaced by a default. Doctor printed
    # "No problems detected." over a root that converge would build against
    # 3.12 instead of the configured interpreter, or generate an
    # environment.yml for with no channels at all.
    path = getattr(config_tree, accessor)("main")
    _replace_with_unusable(path, shape)

    findings = diagnose(config_tree)

    assert [f.kind for f in findings] == ["unparseable-source"]
    assert str(path) in findings[0].message


@pytest.mark.parametrize("shape", _UNUSABLE_SHAPES)
def test_an_unusable_python_txt_is_not_reported_as_missing(
    config_tree: ConfigRoot, shape: str
):
    # 'missing python.txt (will default to 3.12)' about a file that is
    # emphatically present, offering a repair -- create it -- that cannot run,
    # because atomic_write_new opens O_EXCL and the entry is already there.
    # The presence test is what tells the two states apart.
    _replace_with_unusable(config_tree.env_python_path("main"), shape)

    assert "missing-python-txt" not in _kinds(diagnose(config_tree))


@pytest.mark.parametrize("accessor", ["env_micromamba_path", "env_channels_path"])
def test_a_genuinely_absent_optional_env_source_stays_silent(
    config_tree: ConfigRoot, accessor: str
):
    # The guard must not promote an optional file to a required one: a root
    # that declares no conda packages and no channels is the common case.
    getattr(config_tree, accessor)("main").unlink()

    assert diagnose(config_tree) == []


def test_diagnose_survives_every_env_source_being_unusable(config_tree: ConfigRoot):
    # All three at once, which is what a checkout that put directories where
    # files belong leaves behind. load_env is reached from inside _scan_sources
    # only, and its guard is what keeps the governing rule's first clause --
    # doctor must never fail because the thing it is diagnosing is broken --
    # true while the second one is closed.
    for accessor in _ENV_SOURCE_ACCESSORS:
        _replace_with_unusable(getattr(config_tree, accessor)("main"), "directory")

    findings = diagnose(config_tree)

    assert [f.kind for f in findings] == ["unparseable-source"]


def test_a_missing_directory_names_which_one(tmp_path):
    # diagnose reports the three required directories from one loop, so the
    # name is the only thing distinguishing the three messages. Asserting the
    # path alone would pass for whichever of them happened to be reported.
    root = tmp_path / "python-envs"
    (root / "profiles").mkdir(parents=True)
    (root / "envs").mkdir()
    config = ConfigRoot(root)

    missing = [f for f in diagnose(config) if f.kind == "missing-dir"]

    assert len(missing) == 1
    assert missing[0].level == "error"
    assert missing[0].path == config.bundles_dir
    assert missing[0].message == f"Missing bundles directory: {config.bundles_dir}"
    assert missing[0].fix is not None
    assert "stack config init" in missing[0].fix


def test_findings_carry_kinds(tmp_path):
    config = ConfigRoot(tmp_path / "python-envs")
    findings = diagnose(config)
    assert findings[0].kind == "missing-root"
    assert findings[0].path == config.root


def test_repair_creates_missing_dirs(tmp_path):
    config = ConfigRoot(tmp_path / "python-envs")
    actions = repair(config, diagnose(config))
    assert all(a.applied for a in actions)
    assert config.root.is_dir()
    # Re-diagnosing after creating the root reveals the missing subdirs;
    # a second repair pass clears those too.
    repair(config, diagnose(config))
    assert config.profiles_dir.is_dir()
    assert config.bundles_dir.is_dir()
    assert config.envs_dir.is_dir()
    assert diagnose(config) == []


def test_repair_renames_profiles_txt(config_tree: ConfigRoot):
    env_dir = config_tree.env_dir("legacy")
    env_dir.mkdir(parents=True)
    (env_dir / "profiles.txt").write_text("@standard\n")
    actions = repair(config_tree, diagnose(config_tree))
    renamed = [a for a in actions if a.finding.kind == "legacy-profiles-txt"]
    assert renamed and renamed[0].applied
    assert (env_dir / "stack.txt").read_text() == "@standard\n"
    assert not (env_dir / "profiles.txt").exists()


def test_repair_skips_profiles_txt_when_stack_exists(config_tree: ConfigRoot):
    env_dir = config_tree.env_dir("main")
    (env_dir / "profiles.txt").write_text("@standard\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-profiles-txt"]
    assert skipped and not skipped[0].applied
    assert skipped[0].reason is not None
    assert (env_dir / "profiles.txt").exists()


def test_repair_converts_legacy_profile_to_yaml(config_tree: ConfigRoot):
    legacy = config_tree.profiles_dir / "old.in"
    legacy.write_text("# comment\nnumpy\npandas>=2\n")
    actions = repair(config_tree, diagnose(config_tree))
    converted = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert converted and converted[0].applied
    assert config_tree.load_profile("old").includes == ["numpy", "pandas>=2"]
    assert not legacy.exists()
    assert (config_tree.profiles_dir / "old.in.bak").is_file()


def test_repair_skips_conversion_when_yaml_exists(config_tree: ConfigRoot):
    legacy = config_tree.profiles_dir / "ds.in"
    legacy.write_text("numpy\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert legacy.exists()


def test_repair_skips_conversion_when_backup_exists(config_tree: ConfigRoot):
    legacy = config_tree.profiles_dir / "old.in"
    legacy.write_text("numpy\n")
    (config_tree.profiles_dir / "old.in.bak").write_text("precious\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "old.in.bak already exists" == skipped[0].reason
    assert (config_tree.profiles_dir / "old.in.bak").read_text() == "precious\n"
    assert legacy.exists()


def test_repair_converts_legacy_bundle(config_tree: ConfigRoot):
    legacy = config_tree.bundles_dir / "oldb.bundle"
    legacy.write_text("ds\nchem\n")
    repair(config_tree, diagnose(config_tree))
    assert config_tree.load_bundle("oldb").includes == ["ds", "chem"]
    assert (config_tree.bundles_dir / "oldb.bundle.bak").is_file()


@pytest.mark.parametrize(
    ("directory", "filename", "kind"),
    [
        ("profiles_dir", "old.in", "legacy-profile"),
        ("bundles_dir", "oldb.bundle", "legacy-bundle"),
    ],
)
def test_repair_skips_converting_an_entry_uv_stack_would_not_write(
    config_tree: ConfigRoot, directory: str, filename: str, kind: str
):
    """The conversion is a durable writer of human-typed entries like any other.

    ``init``, ``create``, and ``edit`` all refuse this entry, so publishing it
    as YAML would leave behind a generated file every later command rejects --
    and the refusal would name that file rather than the legacy one the user
    actually wrote, which the conversion has by then hidden as a ``.bak``.
    """
    legacy = getattr(config_tree, directory) / filename
    legacy.write_text("numpy\n-r ${DEV}/extra.txt\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == kind]
    assert skipped and not skipped[0].applied
    assert skipped[0].reason is not None
    assert "uv-stack will not write" in skipped[0].reason
    assert "-r ${DEV}/extra.txt" in skipped[0].reason
    # Nothing published and nothing moved, so the file to fix is where it was.
    assert legacy.read_text() == "numpy\n-r ${DEV}/extra.txt\n"
    assert not legacy.with_suffix(".yaml").exists()
    assert not legacy.with_name(legacy.name + ".bak").exists()


def test_repair_skips_legacy_profile_when_bundle_exists_for_stem(config_tree: ConfigRoot):
    """Legacy profiles/old.in skipped when bundles/old.yaml exists (shadow guard)."""
    # Create existing bundle.
    (config_tree.bundles_dir / "old.yaml").write_text("includes:\n- pandas\n")
    # Create legacy profile with same stem.
    legacy = config_tree.profiles_dir / "old.in"
    legacy.write_text("numpy\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "would shadow the existing bundle" in skipped[0].reason
    # Legacy file intact, no profiles/old.yaml created.
    assert legacy.exists()
    assert not (config_tree.profiles_dir / "old.yaml").exists()


def test_repair_skips_legacy_bundle_when_profile_exists_for_stem(config_tree: ConfigRoot):
    """Legacy bundles/old.bundle skipped when profiles/old.yaml exists (shadow guard)."""
    # Create existing profile.
    (config_tree.profiles_dir / "old.yaml").write_text("includes:\n- numpy\n")
    # Create legacy bundle with same stem.
    legacy = config_tree.bundles_dir / "old.bundle"
    legacy.write_text("ds\n")
    actions = repair(config_tree, diagnose(config_tree))
    skipped = [a for a in actions if a.finding.kind == "legacy-bundle"]
    assert skipped and not skipped[0].applied
    assert "would be shadowed by the existing profile" in skipped[0].reason
    # Legacy file intact, no bundles/old.yaml created.
    assert legacy.exists()
    assert not (config_tree.bundles_dir / "old.yaml").exists()


def test_repair_moves_misplaced_env(config_tree: ConfigRoot):
    stray = config_tree.root / "straggler"
    stray.mkdir()
    (stray / "requirements.in").write_text("# x\n")
    repair(config_tree, diagnose(config_tree))
    assert not stray.exists()
    assert (config_tree.envs_dir / "straggler" / "requirements.in").is_file()


def test_repair_writes_default_python_txt(config_tree: ConfigRoot):
    config_tree.env_python_path("main").unlink()
    repair(config_tree, diagnose(config_tree))
    assert config_tree.env_python_path("main").read_text() == "3.12\n"


def test_repair_skips_stale_legacy_conversion(config_tree: ConfigRoot):
    """Stale conversion: legacy .in deleted after diagnose() → skip, no .yaml created."""
    legacy = config_tree.profiles_dir / "ephemeral.in"
    legacy.write_text("numpy\n")
    findings = diagnose(config_tree)
    # Delete the source after diagnosis.
    legacy.unlink()
    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "no longer exists" in skipped[0].reason
    assert not (config_tree.profiles_dir / "ephemeral.yaml").exists()


def test_repair_skips_python_txt_created_between_diagnose_and_repair(config_tree: ConfigRoot):
    """python.txt created between diagnose and repair → skipped, user content preserved."""
    config_tree.env_python_path("main").unlink()
    findings = diagnose(config_tree)
    # User creates python.txt with their own content after diagnosis.
    config_tree.env_python_path("main").write_text("3.11\n")
    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "missing-python-txt"]
    assert skipped and not skipped[0].applied
    assert "python.txt already exists" in skipped[0].reason
    assert config_tree.env_python_path("main").read_text() == "3.11\n"


def test_repair_skips_profiles_txt_rename_when_stack_txt_appears_after_diagnose(
    config_tree: ConfigRoot,
):
    """profiles.txt rename when stack.txt appears after diagnose → skipped, both intact."""
    env_dir = config_tree.env_dir("racy")
    env_dir.mkdir(parents=True)
    (env_dir / "profiles.txt").write_text("@standard\n")
    findings = diagnose(config_tree)
    # User creates stack.txt after diagnosis.
    (env_dir / "stack.txt").write_text("@stable\n")
    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profiles-txt"]
    assert skipped and not skipped[0].applied
    assert "stack.txt already exists" in skipped[0].reason
    assert (env_dir / "profiles.txt").exists()
    assert (env_dir / "stack.txt").read_text() == "@stable\n"


def test_repair_conversion_rollback_on_source_vanish(config_tree: ConfigRoot, monkeypatch):
    """Source vanishes after YAML publish → YAML removed (identity match), action skipped."""
    from uv_stack.operations import doctor

    legacy = config_tree.profiles_dir / "vanish.in"
    legacy.write_text("numpy\n")
    findings = diagnose(config_tree)

    # Monkeypatch _move_no_replace to unlink the source then raise FileNotFoundError.
    def fake_move(src, dst):
        src.unlink()
        raise FileNotFoundError(f"{src} disappeared")

    monkeypatch.setattr(doctor, "_move_no_replace", fake_move)

    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "disappeared" in skipped[0].reason
    # YAML was removed in rollback.
    assert not (config_tree.profiles_dir / "vanish.yaml").exists()
    # Source is gone (simulated vanish).
    assert not legacy.exists()


def test_repair_conversion_preserves_replaced_yaml(config_tree: ConfigRoot, monkeypatch):
    """Dest YAML replaced after publish → replacement survives, action skipped."""
    from uv_stack.operations import doctor

    legacy = config_tree.profiles_dir / "race.in"
    legacy.write_text("numpy\n")
    findings = diagnose(config_tree)

    def fake_move(src, dst):
        # Simulate concurrent replacement: unlink the YAML and rewrite it.
        yaml_path = config_tree.profiles_dir / "race.yaml"
        yaml_path.unlink()
        yaml_path.write_text("includes:\n- pandas\n")
        raise FileExistsError(f"{dst.with_suffix('.bak')} already exists")

    monkeypatch.setattr(doctor, "_move_no_replace", fake_move)

    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "race.in.bak already exists" in skipped[0].reason
    # Replacement YAML survives (identity check prevented deletion).
    yaml_path = config_tree.profiles_dir / "race.yaml"
    assert yaml_path.exists()
    assert "pandas" in yaml_path.read_text()
    # Source file untouched.
    assert legacy.exists()


def test_repair_misplaced_env_skips_when_dest_created_after_diagnose(config_tree: ConfigRoot):
    """Dest dir created after diagnose → skipped, both directories intact."""
    stray = config_tree.root / "concurrent"
    stray.mkdir()
    (stray / "requirements.in").write_text("# x\n")
    findings = diagnose(config_tree)
    # Destination appears after diagnosis.
    dest = config_tree.envs_dir / "concurrent"
    dest.mkdir()
    (dest / "environment.yml").write_text("name: x\n")
    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "misplaced-env"]
    assert skipped and not skipped[0].applied
    assert "concurrent already exists" in skipped[0].reason
    # Both directories intact.
    assert stray.exists()
    assert (stray / "requirements.in").exists()
    assert dest.exists()
    assert (dest / "environment.yml").read_text() == "name: x\n"


def test_finish_move_normal_case(tmp_path: Path):
    """Normal move: source identity matches linked inode → source removed."""
    import os

    from uv_stack.fsutil import Published
    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    _finish_move(src, dst, moved_stat, Published("link", moved_stat))
    assert not src.exists()
    assert dst.read_text() == "content\n"




def test_finish_move_src_replaced_after_link(tmp_path: Path):
    """Source replaced after link → link withdrawn, OSError raised, replacement survives."""
    import os

    from uv_stack.fsutil import Published
    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    # Replace source with new inode.
    src.unlink()
    src.write_text("replacement\n")
    try:
        _finish_move(src, dst, moved_stat, Published("link", moved_stat))
        raise AssertionError("Expected OSError")
    except OSError as e:
        assert "changed during move" in str(e)
    # Replacement survives.
    assert src.exists()
    assert src.read_text() == "replacement\n"
    # Link withdrawn.
    assert not dst.exists()


def test_finish_move_src_vanished_after_link(tmp_path: Path):
    """Source vanished after link → no error, dest preserves the inode."""
    import os

    from uv_stack.fsutil import Published
    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    src.unlink()
    # Should not raise.
    _finish_move(src, dst, moved_stat, Published("link", moved_stat))
    assert not src.exists()
    assert dst.read_text() == "content\n"


def test_finish_move_dst_replaced_after_link(tmp_path: Path):
    """Both src and dst replaced → the stranger's dst survives, OSError raised."""
    import os

    from uv_stack.fsutil import Published
    from uv_stack.operations.doctor import _finish_move

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    moved_stat = src.lstat()
    os.link(src, dst)
    # Replace both src and dst.
    src.unlink()
    src.write_text("new-src\n")
    dst.unlink()
    dst.write_text("new-dst\n")
    try:
        _finish_move(src, dst, moved_stat, Published("link", moved_stat))
        raise AssertionError("Expected OSError")
    except OSError as e:
        assert "changed during move" in str(e)
    # Both replacements survive.
    assert src.exists()
    assert src.read_text() == "new-src\n"
    assert dst.exists()
    assert dst.read_text() == "new-dst\n"


def test_move_no_replace_identity_check_in_rename(config_tree: ConfigRoot, monkeypatch):
    """File rename with source replaced after link → skipped, replacement survives."""

    env_dir = config_tree.env_dir("race")
    env_dir.mkdir(parents=True)
    profiles_txt = env_dir / "profiles.txt"
    stack_txt = env_dir / "stack.txt"
    profiles_txt.write_text("@standard\n")
    findings = diagnose(config_tree)

    # Simulate race: monkeypatch _finish_move to replace source before unlinking.
    from uv_stack.fsutil import Published
    from uv_stack.operations import doctor

    original_finish = doctor._finish_move

    def race_finish(
        src: Path, dst: Path, moved_stat: os.stat_result, published: Published
    ) -> None:
        # Replace source with new inode before calling original.
        if src == profiles_txt:
            src.unlink()
            src.write_text("@replaced\n")
        original_finish(src, dst, moved_stat, published)

    monkeypatch.setattr(doctor, "_finish_move", race_finish)

    actions = repair(config_tree, findings)

    skipped = [a for a in actions if a.finding.kind == "legacy-profiles-txt"]
    assert skipped and not skipped[0].applied
    assert "changed during move" in skipped[0].reason
    # Replacement survives.
    assert profiles_txt.exists()
    assert profiles_txt.read_text() == "@replaced\n"
    # Link withdrawn.
    assert not stack_txt.exists()


def _link_less(monkeypatch) -> None:
    """Make os.link raise EOPNOTSUPP, as FAT/exFAT and some network mounts do."""
    import errno
    import os

    def _no_link(src, dst, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links not supported")

    monkeypatch.setattr(os, "link", _no_link)


def test_move_no_replace_completes_on_a_link_less_filesystem(tmp_path: Path, monkeypatch):
    """The whole point of C10: a repair that is impossible today now completes."""
    from uv_stack.operations.doctor import _move_no_replace

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    _move_no_replace(src, dst)
    assert not src.exists()
    assert dst.read_text() == "content\n"


@pytest.mark.parametrize(
    ("mutation", "label"),
    [(b"mutated-and-much-longer\n", "different-length"), (b"MUTATED!\n", "same-length")],
)
def test_move_no_replace_copy_path_refuses_an_in_place_rewrite(
    tmp_path: Path, monkeypatch, mutation: bytes, label: str
):
    """A write through the source's own inode after the copy must not be lost.

    On the copy path ``dst`` is a snapshot, so an in-place write leaves
    st_dev/st_ino untouched while the bytes diverge. Unlinking the source on
    identity alone would destroy the post-write content. The same-length case
    is here so the size short-circuit cannot be what carries the test.
    """
    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_mutate(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        with open(src, "r+b") as handle:
            handle.write(mutation)
            handle.truncate()
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_mutate)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # The source survives, still the same inode, holding the post-copy bytes.
    assert src.read_bytes() == mutation
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    # The stale snapshot is withdrawn.
    assert not dst.exists()


def test_move_no_replace_link_path_survives_an_in_place_rewrite(tmp_path: Path, monkeypatch):
    """The paired link case still succeeds: both names reach the mutated inode.

    The content check belongs to the copy branch only. Applying it to the link
    branch would turn a move that loses nothing into a spurious failure.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_mutate(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        with open(src, "r+b") as handle:
            handle.write(b"MUTATED!\n")
            handle.truncate()
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_mutate)
    doctor._move_no_replace(src, dst)
    assert not src.exists()
    assert dst.read_bytes() == b"MUTATED!\n"


def test_move_no_replace_copy_path_withdraws_when_the_source_is_replaced(
    tmp_path: Path, monkeypatch
):
    """A replaced source takes the withdrawal path, leaving the replacement intact."""
    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_replace(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        src.unlink()
        src.write_text("replacement\n")
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_replace)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    assert src.read_text() == "replacement\n"
    assert not dst.exists()


def test_move_no_replace_copy_path_leaves_a_foreign_destination_alone(
    tmp_path: Path, monkeypatch
):
    """The rollback set is exactly the inode the exclusive create made.

    A destination replaced by a third party after publication is not ours to
    delete, however the move ends.
    """
    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_hijack(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        src.unlink()
        src.write_text("replacement\n")
        dst.unlink()
        dst.write_text("stranger\n")
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_hijack)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "nothing deleted" in str(excinfo.value)
    assert dst.read_text() == "stranger\n"


def test_same_bytes_compares_content_not_timestamps(tmp_path: Path):
    """Sizes first, then bytes — never mtime, which FAT resolves to two seconds."""
    from uv_stack.operations.doctor import _same_bytes

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    d = tmp_path / "d.txt"
    a.write_bytes(b"same\n")
    b.write_bytes(b"same\n")
    c.write_bytes(b"diff\n")
    d.write_bytes(b"longer content\n")
    assert _same_bytes(a, b)
    assert not _same_bytes(a, c)
    assert not _same_bytes(a, d)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform lacks mkfifo")
def test_move_no_replace_copy_path_refuses_a_fifo_source_during_same_bytes(
    tmp_path: Path, monkeypatch
):
    """A FIFO swapped for src during the byte comparison is refused, not hung on.

    The FIFO must be empty (st_size == 0) to reach the open — a non-empty source
    short-circuits the comparison to False. The guarded open refuses it rather
    than blocking.
    """
    import os

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("")  # Empty so size check passes.
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_fifo(left, right):
        # Replace src with a FIFO before calling the real comparison.
        src.unlink()
        os.mkfifo(src)
        return real_same_bytes(left, right)

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_fifo)
    with _deadline(5.0):
        with pytest.raises(OSError) as excinfo:
            doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # FIFO survives, dst is withdrawn.
    assert src.exists()
    assert not dst.exists()


def test_move_no_replace_copy_path_refuses_a_symlink_source_during_same_bytes(
    tmp_path: Path, monkeypatch
):
    """A symlink swapped for src during the byte comparison is refused via O_NOFOLLOW."""

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    target.write_text("content\n")
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_symlink(left, right):
        # Replace src with a symlink before calling the real comparison.
        src.unlink()
        src.symlink_to(target)
        return real_same_bytes(left, right)

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_symlink)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # Symlink survives, dst is withdrawn.
    assert src.is_symlink()
    assert not dst.exists()


def test_move_no_replace_copy_path_refuses_a_chmod_after_same_bytes(
    tmp_path: Path, monkeypatch
):
    """A chmod after the byte comparison but before the mode check withdraws dst.

    The mode check now re-stats src immediately before the unlink, so a chmod
    landing after the byte comparison is caught.
    """
    import os
    import stat

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    os.chmod(src, 0o640)
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_chmod(left, right):
        result = real_same_bytes(left, right)
        # chmod after the byte comparison returns.
        os.chmod(src, 0o600)
        return result

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_chmod)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # The source survives with the post-chmod mode.
    assert src.read_text() == "content\n"
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    assert stat.S_IMODE(src.stat().st_mode) == 0o600
    # The stale snapshot is withdrawn.
    assert not dst.exists()


def test_move_no_replace_copy_path_refuses_a_dst_replacement_after_same_bytes(
    tmp_path: Path, monkeypatch
):
    """Dst replaced with a same-byte file after the comparison withdraws nothing.

    The replacement is not ours to delete, and src survives.
    """
    import os
    import stat

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    replacement = tmp_path / "replacement.txt"
    src.write_text("content\n")
    replacement.write_text("content\n")  # Same bytes.
    os.chmod(replacement, 0o600)  # Different mode from default.
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_replace_dst(left, right):
        result = real_same_bytes(left, right)
        # Replace dst after the comparison returns.
        dst.unlink()
        os.rename(replacement, dst)
        return result

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_replace_dst)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    assert "nothing deleted" in str(excinfo.value)
    # The source survives unchanged.
    assert src.read_text() == "content\n"
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    # The replacement at dst survives (not ours to delete).
    assert dst.read_text() == "content\n"
    assert stat.S_IMODE(dst.stat().st_mode) == 0o600


def test_move_no_replace_copy_path_refuses_when_dst_vanishes_after_same_bytes(
    tmp_path: Path, monkeypatch
):
    """Dst vanishing after the comparison raises an error rather than silently succeeding.

    Src still exists and the destination is gone, so the move did not complete.
    """

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_same_bytes = doctor._same_bytes

    def same_bytes_then_remove_dst(left, right):
        result = real_same_bytes(left, right)
        # Remove dst after the comparison returns.
        dst.unlink()
        return result

    monkeypatch.setattr(doctor, "_same_bytes", same_bytes_then_remove_dst)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # The source survives.
    assert src.read_text() == "content\n"
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    # Dst is gone.
    assert not dst.exists()


def test_move_no_replace_copy_path_refuses_a_chmod_under_the_copy(tmp_path: Path, monkeypatch):
    """A chmod on src after publication leaves src in place and withdraws dst.

    The copy is a snapshot of the pre-chmod bits. Unlinking src on identity and
    bytes alone would discard the mode change.
    """
    import os
    import stat

    from uv_stack.operations import doctor

    _link_less(monkeypatch)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    os.chmod(src, 0o640)
    src_ident = (src.stat().st_dev, src.stat().st_ino)
    real_publish = doctor.link_or_copy_no_replace

    def publish_then_chmod(a, b, **kwargs):
        result = real_publish(a, b, **kwargs)
        os.chmod(src, 0o600)
        return result

    monkeypatch.setattr(doctor, "link_or_copy_no_replace", publish_then_chmod)
    with pytest.raises(OSError) as excinfo:
        doctor._move_no_replace(src, dst)
    assert "changed during move" in str(excinfo.value)
    # The source survives, still the same inode, holding the post-chmod mode.
    assert src.read_text() == "content\n"
    assert (src.stat().st_dev, src.stat().st_ino) == src_ident
    assert stat.S_IMODE(src.stat().st_mode) == 0o600
    # The stale snapshot is withdrawn.
    assert not dst.exists()


def test_move_no_replace_propagates_eperm_from_a_public_source(tmp_path: Path, monkeypatch):
    """EPERM from os.link on a user-visible file is a denial, not a fallback trigger.

    On Linux with fs.protected_hardlinks=1, EPERM means "you do not own this
    file." Treating it as link-less turns a denial into copy-then-unlink.
    """
    import errno
    import os

    from uv_stack.operations.doctor import _move_no_replace

    def _eperm(a, b, **kwargs):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", _eperm)
    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("content\n")
    with pytest.raises(OSError) as excinfo:
        _move_no_replace(src, dst)
    assert excinfo.value.errno == errno.EPERM
    assert src.read_text() == "content\n"
    assert not dst.exists()


def test_repair_conversion_source_replaced_between_read_and_move(
    config_tree: ConfigRoot, monkeypatch
):
    """Source replaced after read but before move → skipped, no YAML left, replacement intact."""
    from uv_stack.operations import doctor
    from uv_stack.parse import read_clean_lines as original_read

    legacy = config_tree.profiles_dir / "race.in"
    legacy.write_text("numpy\n")
    findings = diagnose(config_tree)

    # Monkeypatch read_clean_lines to replace the source after reading.
    def race_read(path):
        result = original_read(path)
        if path == legacy:
            # Replace source with new inode.
            path.unlink()
            path.write_text("pandas\n")
        return result

    monkeypatch.setattr(doctor, "read_clean_lines", race_read)

    actions = repair(config_tree, findings)
    skipped = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert skipped and not skipped[0].applied
    assert "changed during conversion" in skipped[0].reason
    # No YAML left.
    assert not (config_tree.profiles_dir / "race.yaml").exists()
    # Replacement intact.
    assert legacy.exists()
    assert legacy.read_text() == "pandas\n"


def test_move_no_replace_never_adopts_a_foreign_destination(tmp_path: Path, monkeypatch):
    """A dst holding a stranger's file is not ours to delete.

    The os.link stub models the reachable end state — our link published and
    then replaced by a foreign writer, with src replaced too — rather than a
    literal call sequence, since a real os.link would have raised
    FileExistsError against an existing dst. The inode relationships driving
    the code path are the same either way.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")

    def racing_link(_a, _b, **_kwargs):
        dst.write_text("foreign\n")
        src.unlink()
        src.write_text("replacement\n")

    monkeypatch.setattr(doctor.os, "link", racing_link)

    with pytest.raises(OSError, match="source.txt changed during move"):
        doctor._move_no_replace(src, dst)

    assert dst.read_text() == "foreign\n"
    assert src.read_text() == "replacement\n"


def test_move_no_replace_withdraws_a_link_to_a_replaced_source(
    tmp_path: Path, monkeypatch
):
    """A source replaced BEFORE the link leaves no stray link at dst.

    The link then publishes the replacement inode, which is neither the inode
    we measured nor a stranger's file — it is ours, and leaving it behind
    would block every later move to this destination.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("original\n")
    real_link = doctor.os.link

    def racing_link(a, b, **kwargs):
        # A concurrent writer replaces src between our lstat and the link.
        src.unlink()
        src.write_text("replacement\n")
        real_link(a, b, **kwargs)

    monkeypatch.setattr(doctor.os, "link", racing_link)

    with pytest.raises(OSError, match="source.txt changed during move"):
        doctor._move_no_replace(src, dst)

    assert not dst.exists()
    assert src.read_text() == "replacement\n"


def test_move_no_replace_refuses_a_symlinked_source(tmp_path: Path):
    """os.link would publish the TARGET, and the unlink would relocate it."""
    from uv_stack.operations.doctor import _move_no_replace

    target = tmp_path / "target.txt"
    target.write_text("target\n")
    src = tmp_path / "link.txt"
    src.symlink_to(target)
    dst = tmp_path / "dest.txt"

    with pytest.raises(OSError, match="not a regular file"):
        _move_no_replace(src, dst)

    assert src.is_symlink()
    assert target.read_text() == "target\n"
    assert not dst.exists()


def test_move_no_replace_refuses_when_the_destination_vanishes(
    tmp_path: Path, monkeypatch
):
    """A dst removed after we publish it must not cost src its last name.

    src still names the moved inode, so checking src alone succeeds; only
    re-checking dst keeps the unlink from destroying the file outright while
    reporting the move as done.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("precious\n")
    real_link = doctor.os.link

    def link_then_dst_vanishes(a, b, **kwargs):
        real_link(a, b, **kwargs)
        dst.unlink()

    monkeypatch.setattr(doctor.os, "link", link_then_dst_vanishes)

    with pytest.raises(OSError, match="vanished during move"):
        doctor._move_no_replace(src, dst)

    assert src.read_text() == "precious\n"


def test_move_no_replace_refuses_when_the_destination_is_replaced(
    tmp_path: Path, monkeypatch
):
    """A dst swapped for a stranger's file after we link leaves both intact."""
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("precious\n")
    real_link = doctor.os.link

    def link_then_dst_replaced(a, b, **kwargs):
        real_link(a, b, **kwargs)
        dst.unlink()
        dst.write_text("foreign\n")

    monkeypatch.setattr(doctor.os, "link", link_then_dst_replaced)

    with pytest.raises(OSError, match="dest.txt changed during move"):
        doctor._move_no_replace(src, dst)

    assert src.read_text() == "precious\n"
    assert dst.read_text() == "foreign\n"


def test_move_no_replace_completes_when_the_source_vanishes_before_the_unlink(
    tmp_path: Path, monkeypatch
):
    """A src removed after its check has already been moved; do not report failure.

    dst holds the moved inode by then, so the move is complete. Raising here
    would make _fix_convert_yaml withdraw a YAML file it had already published,
    leaving the tree with neither the source nor its conversion.
    """
    from uv_stack.operations import doctor

    src = tmp_path / "source.txt"
    dst = tmp_path / "dest.txt"
    src.write_text("precious\n")
    real_lstat = Path.lstat
    src_checks = 0
    removed = False

    def remove_src_right_after_its_identity_check(self, *args, **kwargs):
        # Anchored on src's own check, not on dst's — which merely happens to
        # sit between it and the unlink today. Reordering the two stats must
        # not quietly turn this into a test of the early-return path.
        nonlocal src_checks, removed
        result = real_lstat(self, *args, **kwargs)
        if self == src:
            src_checks += 1
            # The first check is _move_no_replace's; the second is the one
            # _finish_move's unlink relies on.
            if src_checks == 2:
                src.unlink()
                removed = True
        return result

    monkeypatch.setattr(Path, "lstat", remove_src_right_after_its_identity_check)

    doctor._move_no_replace(src, dst)

    assert removed, "the race never fired; this no longer tests the window"
    assert not src.exists()
    assert dst.read_text() == "precious\n"


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_repair_conversion_serializes_against_a_concurrent_create(
    config_tree: ConfigRoot, monkeypatch
):
    """A conversion is a create in the shared stem namespace, so it takes the lock.

    Publishing profiles/<stem>.yaml collides with a bundle create for the same
    stem, and the collision spans two paths, so no no-clobber write can reserve
    it. Without the lock the two commit on top of each other and diagnose then
    reports nothing wrong. Holding the stem lock from another process is what
    proves doctor asks for it: the conversion has no other reason to stop.

    The env's missing python.txt is repaired in the same pass on purpose. The
    lock raises ConfigError, which repair() does not catch, so an unavailable
    stem must cost its own finding and nothing else.
    """
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    legacy = config_tree.profiles_dir / "old.in"
    legacy.write_text("numpy\n")
    (config_tree.envs_dir / "main" / "python.txt").unlink()

    with _lock_held_by_another_process(config_tree.stem_lock_path("old")):
        actions = repair(config_tree, diagnose(config_tree))

    converted = [a for a in actions if a.finding.kind == "legacy-profile"]
    assert converted and not converted[0].applied
    assert "another stack process" in converted[0].reason
    assert legacy.exists()
    assert not (config_tree.profiles_dir / "old.yaml").exists()

    unrelated = [a for a in actions if a.finding.kind == "missing-python-txt"]
    assert unrelated and unrelated[0].applied, (
        "one unavailable stem lock aborted the rest of the repair pass"
    )


def test_diagnose_reports_degraded_locks(config_tree: ConfigRoot, monkeypatch):
    """A root that cannot lock is reported once, as a warning, with no repair."""
    from uv_stack.operations import doctor

    monkeypatch.setattr(doctor, "probe_locking", lambda path: False)
    findings = diagnose(config_tree)
    degraded = [f for f in findings if f.kind == "degraded-locks"]
    assert len(degraded) == 1
    assert degraded[0].level == "warn"
    assert str(config_tree.root) in degraded[0].message
    assert "not serialized" in degraded[0].message
    assert degraded[0].path == config_tree.locks_dir
    assert degraded[0].kind not in doctor._REPAIRS


def test_diagnose_is_silent_when_locking_works(config_tree: ConfigRoot):
    findings = diagnose(config_tree)
    assert [f for f in findings if f.kind == "degraded-locks"] == []


def test_diagnose_never_raises_when_locks_is_a_plain_file(config_tree: ConfigRoot):
    """The probe's non-raising contract, exercised through the caller that needs it."""
    config_tree.locks_dir.write_text("not a directory\n")
    findings = diagnose(config_tree)
    assert any(f.kind == "degraded-locks" for f in findings)


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_diagnose_never_raises_on_an_unsearchable_locks_dir(config_tree: ConfigRoot):
    """A .locks this user cannot traverse into: the degrade path with no exception."""
    config_tree.locks_dir.mkdir()
    os.chmod(config_tree.locks_dir, 0o600)
    try:
        findings = diagnose(config_tree)
    finally:
        # Restore, or tmp_path teardown cannot remove the directory.
        os.chmod(config_tree.locks_dir, 0o700)
    assert any(f.kind == "degraded-locks" for f in findings)


def test_diagnose_never_raises_on_a_non_regular_probe_lock(config_tree: ConfigRoot):
    """A planted FIFO at probe.lock is the shape name_lock refuses outright."""
    config_tree.locks_dir.mkdir()
    os.mkfifo(config_tree.probe_lock_path())
    findings = diagnose(config_tree)
    assert any(f.kind == "degraded-locks" for f in findings)


def test_repair_leaves_degraded_locks_alone(config_tree: ConfigRoot, monkeypatch):
    """Nothing here is machine-fixable, so --fix must report no action for it."""
    from uv_stack.operations import doctor

    monkeypatch.setattr(doctor, "probe_locking", lambda path: False)
    findings = diagnose(config_tree)
    actions = repair(config_tree, findings)
    assert [a for a in actions if a.finding.kind == "degraded-locks"] == []


def _kinds(findings: list[Finding]) -> list[str]:
    return [f.kind for f in findings]


def test_a_declared_variable_with_no_value_is_an_error(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("DEV\n")
    findings = diagnose(config_tree)
    finding = next(f for f in findings if f.kind == "undefined-variable")
    assert finding.level == "error"
    assert "Variable 'DEV' is declared in" in finding.message
    assert "has no value on this machine" in finding.message
    assert finding.fix is not None
    assert f"Add 'DEV=<value>' to {config_tree.variables_local_path()}" in finding.fix


def test_an_undeclared_reference_is_an_error(config_tree: ConfigRoot):
    config_tree.profile_path("dev").write_text("includes:\n  - -e ${NOPE}/pkg\n")
    findings = diagnose(config_tree)
    finding = next(f for f in findings if f.kind == "undeclared-variable")
    assert finding.level == "error"
    assert "references 'NOPE', which" in finding.message
    assert finding.path == config_tree.profile_path("dev")


def test_an_undeclared_reference_in_a_bundle_is_found(config_tree: ConfigRoot):
    # The scan set is three kinds of file, not one. Each gets its own test so
    # it cannot silently narrow back to profiles.
    config_tree.bundle_path("qsar").write_text("includes:\n  - -e ${NOPE}/pkg\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "undeclared-variable")
    assert finding.path == config_tree.bundle_path("qsar")


def test_an_undeclared_reference_in_a_stack_txt_is_found(config_tree: ConfigRoot):
    config_tree.env_stack_path("main").write_text("@standard\n-e ${NOPE}/pkg\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "undeclared-variable")
    assert finding.path == config_tree.env_stack_path("main")


def test_a_reference_only_in_a_comment_is_not_undeclared(config_tree: ConfigRoot):
    # Doctor must agree with the renderer about what an entry references. uv
    # discards the comment, so a name there is prose, and reporting it would
    # send the user to declare a variable nothing reads.
    config_tree.profile_path("dev").write_text("includes:\n  - numpy  # not ${NOPE}\n")
    kinds = {f.kind for f in diagnose(config_tree)}
    assert "undeclared-variable" not in kinds


@pytest.mark.parametrize("entry", ["-e ${UV-ROOT}/pkg", "-e ${1ROOT}/pkg", "-e ${ROOT"])
def test_a_malformed_reference_is_an_error(config_tree: ConfigRoot, entry):
    # A hyphen, a leading digit, and an unterminated opener: three ways to
    # write something that looks like a reference and is not one.
    config_tree.profile_path("dev").write_text(f"includes:\n  - {entry}\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "malformed-reference")
    assert finding.level == "error"
    assert "is not a well-formed reference" in finding.message


def test_a_misplaced_reference_is_an_error(config_tree: ConfigRoot):
    config_tree.variables_path().write_text("PKG\n")
    config_tree.variables_local_path().write_text("PKG=widget\n")
    config_tree.profile_path("dev").write_text("includes:\n  - ${PKG}>=2\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "misplaced-reference")
    assert finding.level == "error"
    assert "condition" in finding.message


def test_a_multiline_entry_is_an_error(config_tree: ConfigRoot):
    config_tree.profile_path("dev").write_text('includes:\n  - "numpy\\nrich"\n')
    finding = next(f for f in diagnose(config_tree) if f.kind == "multiline-entry")
    assert finding.level == "error"


def test_a_continuation_entry_is_an_error(config_tree: ConfigRoot):
    # placement_problem returns a fourth refusal kind that the other placement
    # tests never produce. The fix table is indexed, not queried, so a missing
    # row would take 'stack doctor' down with a KeyError on the one root it
    # exists to diagnose.
    config_tree.profile_path("dev").write_text("includes:\n  - -e lib/widget\\\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "continuation-entry")
    assert finding.level == "error"
    assert finding.fix is not None


def test_references_in_bundles_and_stack_txt_are_scanned(config_tree: ConfigRoot):
    config_tree.bundle_path("b").write_text("includes:\n  - -e ${B}/pkg\n")
    config_tree.env_stack_path("main").write_text("@standard\n-e ${E}/pkg\n")
    kinds = diagnose(config_tree)
    sources = {f.path for f in kinds if f.kind == "undeclared-variable"}
    assert config_tree.bundle_path("b") in sources
    assert config_tree.env_stack_path("main") in sources


def test_an_unparseable_source_is_reported_not_raised(config_tree: ConfigRoot):
    config_tree.profile_path("broken").write_text("includes: [unclosed\n")
    findings = diagnose(config_tree)
    finding = next(f for f in findings if f.kind == "unparseable-source")
    assert finding.level == "warn"
    assert finding.path == config_tree.profile_path("broken")
    # The detail is the other half of this message's job: a "Cannot read"
    # carrying no reason names a file without saying what is wrong with it.
    assert finding.message.startswith("Cannot read ")
    assert "Invalid YAML" in finding.message


@pytest.mark.parametrize(
    "document",
    [
        pytest.param("description: 2020-99-99\nincludes: []\n", id="value-error"),
        pytest.param('description: !!bool "nope"\nincludes: []\n', id="key-error"),
        pytest.param(
            'description: !!timestamp "nope"\nincludes: []\n', id="attribute-error"
        ),
    ],
)
def test_a_yaml_constructor_failure_is_reported_not_raised(
    config_tree: ConfigRoot, document: str
):
    # A YAML constructor that rejects its own scalar does not raise YAMLError:
    # it raises whatever the conversion raised. Three families reach here, and
    # none of them is one _scan_sources catches, so each took the run down.
    config_tree.profile_path("broken").write_text(document)
    finding = next(
        f
        for f in diagnose(config_tree)
        if f.kind == "unparseable-source"
        and f.path == config_tree.profile_path("broken")
    )
    assert finding.message.startswith("Cannot read ")
    assert str(config_tree.profile_path("broken")) in finding.message


def test_an_unparseable_variables_file_is_reported_not_raised(
    config_tree: ConfigRoot,
):
    config_tree.variables_path().write_bytes(b"\xff\xfe not utf-8\n")
    findings = diagnose(config_tree)
    assert any(
        f.kind == "unparseable-source" and f.path == config_tree.variables_path()
        for f in findings
    )


def test_a_broken_local_variables_file_is_blamed_on_itself(config_tree: ConfigRoot):
    """The finding names the file at fault, not the call doctor happened to make.

    load_variables reads two files, and the finding was built from the
    declaration path unconditionally -- so a malformed variables.local.txt was
    reported as "Cannot read variables.txt", with the message then naming the
    real file in its tail. A reader is told to fix a file that is fine, and the
    path a JSON consumer acts on is the wrong one.
    """
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV /srv/src\n")

    finding = next(f for f in diagnose(config_tree) if f.kind == "unparseable-source")

    assert finding.path == config_tree.variables_local_path()
    assert finding.message.startswith(f"Cannot read {config_tree.variables_local_path()}:")


def test_a_broken_declaration_file_is_still_blamed_on_itself(config_tree: ConfigRoot):
    # The control: blame follows the failing read, and the first read is the
    # declarations, so this one must keep naming variables.txt.
    config_tree.variables_path().write_text("not a name\n")

    finding = next(f for f in diagnose(config_tree) if f.kind == "unparseable-source")

    assert finding.path == config_tree.variables_path()


def test_a_bad_environment_value_is_blamed_on_the_declaration_file(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
):
    # An environment override belongs to no file, and the message says so. The
    # declaration file is what named it, so it is the closest thing to an
    # offender there is; the alternative is a finding with no path at all.
    config_tree.variables_path().write_text("DEV\n")
    monkeypatch.setenv("DEV", "/srv/with a space")

    finding = next(f for f in diagnose(config_tree) if f.kind == "unparseable-source")

    assert finding.path == config_tree.variables_path()
    assert "environment variable DEV" in finding.message


@pytest.mark.parametrize(
    "shape",
    [
        "directory",
        "dangling-symlink",
        pytest.param(
            "fifo",
            marks=pytest.mark.skipif(
                not hasattr(os, "mkfifo"), reason="platform lacks mkfifo"
            ),
        ),
    ],
)
def test_a_non_regular_variables_file_is_reported_not_raised(
    config_tree: ConfigRoot, shape: str
):
    # The loader used to read every one of these shapes as an empty file, so
    # doctor reported a clean root for a tree where no variable resolves.
    path = config_tree.variables_path()
    if shape == "directory":
        path.mkdir()
    elif shape == "dangling-symlink":
        path.symlink_to(path.parent / "nowhere")
    else:
        os.mkfifo(path)
    findings = diagnose(config_tree)
    assert [f.kind for f in findings] == ["unparseable-source"]
    assert findings[0].path == path
    assert str(path) in findings[0].message


@pytest.mark.parametrize(
    "shape",
    [
        "directory",
        "dangling-symlink",
        pytest.param(
            "fifo",
            marks=pytest.mark.skipif(
                not hasattr(os, "mkfifo"), reason="platform lacks mkfifo"
            ),
        ),
    ],
)
def test_a_non_regular_project_python_file_is_reported_not_raised(
    config_tree: ConfigRoot, shape: str
):
    # Every one of these shapes read as an absent file, so doctor reported a
    # clean root and none of them could be told apart from the root that
    # genuinely configures no default. The conversion is already here --
    # _project_python_findings catches ConfigError -- so this proves the whole
    # route, loader guard included, rather than the guard alone.
    path = config_tree.project_python_path()
    if shape == "directory":
        path.mkdir()
    elif shape == "dangling-symlink":
        path.symlink_to(path.parent / "nowhere")
    else:
        os.mkfifo(path)
    findings = diagnose(config_tree)
    assert [f.kind for f in findings] == ["unparseable-source"]
    assert findings[0].path == path
    assert str(path) in findings[0].message


def test_a_missing_editable_checkout_is_a_warning(
    config_tree: ConfigRoot, tmp_path: Path
):
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text(f"DEV={tmp_path / 'gone'}\n")
    config_tree.profile_path("dev").write_text("includes:\n  - -e ${DEV}/widget\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "missing-checkout")
    assert finding.level == "warn"
    assert "editable checkout does not exist:" in finding.message
    assert str(tmp_path / "gone" / "widget") in finding.message


def test_a_present_checkout_and_a_remote_editable_are_not_flagged(
    config_tree: ConfigRoot, tmp_path: Path
):
    (tmp_path / "co" / "widget").mkdir(parents=True)
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text(f"DEV={tmp_path / 'co'}\n")
    config_tree.profile_path("dev").write_text(
        "includes:\n  - -e ${DEV}/widget\n  - -e git+https://example.invalid/x#egg=x\n"
    )
    assert not [f for f in diagnose(config_tree) if f.kind == "missing-checkout"]


def test_the_long_editable_spelling_is_recognized(
    config_tree: ConfigRoot, tmp_path: Path
):
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text(f"DEV={tmp_path / 'gone'}\n")
    config_tree.profile_path("dev").write_text(
        "includes:\n  - --editable ${DEV}/widget\n"
    )
    assert "missing-checkout" in _kinds(diagnose(config_tree))


@pytest.mark.parametrize("flag", ["-e", "--editable", "-e=", "--editable="])
def test_every_editable_spelling_uv_accepts_names_the_same_target(
    config_tree: ConfigRoot, tmp_path: Path, flag: str
):
    # uv takes the operand attached with '=' as readily as separated, and
    # these entries go straight to 'uv pip compile'. A spelling doctor does
    # not recognise is a checkout it quietly stops checking.
    separator = "" if flag.endswith("=") else " "
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text(f"DEV={tmp_path / 'gone'}\n")
    config_tree.profile_path("dev").write_text(
        f"includes:\n  - {flag}{separator}${{DEV}}/widget\n"
    )
    finding = next(f for f in diagnose(config_tree) if f.kind == "missing-checkout")
    assert str(tmp_path / "gone" / "widget") in finding.message


@pytest.mark.parametrize("flag", ["-e=", "--editable="])
def test_an_attached_separator_with_a_detached_operand_names_the_target(
    config_tree: ConfigRoot, tmp_path: Path, flag: str
):
    # '-e= PATH' is a third spelling uv accepts: the '=' separates, and the
    # operand is the next token. Reading the empty text after the '=' as the
    # value silently stops checking the checkout.
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text(f"DEV={tmp_path / 'gone'}\n")
    config_tree.profile_path("dev").write_text(
        f"includes:\n  - {flag} ${{DEV}}/widget\n"
    )

    finding = next(f for f in diagnose(config_tree) if f.kind == "missing-checkout")

    assert str(tmp_path / "gone" / "widget") in finding.message


@pytest.mark.parametrize("flag", ["-e=", "--editable="])
def test_the_attached_spellings_keep_the_url_and_extras_handling(
    config_tree: ConfigRoot, tmp_path: Path, flag: str
):
    checkout = tmp_path / "widget"
    checkout.mkdir()
    config_tree.profile_path("dev").write_text(
        f"includes:\n  - {flag}git+https://example.invalid/x#egg=x\n"
        f"  - {flag}{checkout}[dev,test]\n"
    )
    assert "missing-checkout" not in _kinds(diagnose(config_tree))


def test_an_editable_operand_glued_to_the_flag_is_not_a_checkout(
    config_tree: ConfigRoot,
):
    # uv refuses '-e./widget' at the parser ("Expected '=' or whitespace,
    # found Some('.')"), so reading a path out of it would report a missing
    # checkout for an entry that cannot install for an unrelated reason.
    config_tree.profile_path("dev").write_text("includes:\n  - -e./widget\n")
    assert "missing-checkout" not in _kinds(diagnose(config_tree))


def test_a_plain_url_editable_is_not_a_checkout(config_tree: ConfigRoot):
    config_tree.profile_path("dev").write_text(
        "includes:\n  - -e https://example.invalid/x.tar.gz\n"
    )
    assert "missing-checkout" not in _kinds(diagnose(config_tree))


def test_an_editable_with_extras_is_read_as_a_path(
    config_tree: ConfigRoot, tmp_path: Path
):
    # '-e ./pkg[dev]' is valid pip syntax: the bracket suffix names extras and
    # is not part of the path. Probing the whole token reports a checkout that
    # is sitting right there as missing.
    checkout = tmp_path / "widget"
    checkout.mkdir()
    config_tree.profile_path("dev").write_text(f"includes:\n  - -e {checkout}[dev,test]\n")
    assert "missing-checkout" not in _kinds(diagnose(config_tree))

    # And the suffix does not buy an absent checkout a pass.
    config_tree.profile_path("gone").write_text(
        f"includes:\n  - -e {tmp_path / 'absent'}[dev]\n"
    )
    assert "missing-checkout" in _kinds(diagnose(config_tree))


def test_a_relative_editable_resolves_against_the_config_root(
    config_tree: ConfigRoot,
):
    # uv reads the generated requirements.in from the config root, so doctor
    # must resolve a relative path the same way rather than against the cwd.
    config_tree.profile_path("dev").write_text("includes:\n  - -e lib/widget\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "missing-checkout")
    assert "editable checkout does not exist:" in finding.message
    assert str(config_tree.root / "lib" / "widget") in finding.message
    (config_tree.root / "lib" / "widget").mkdir(parents=True)
    assert "missing-checkout" not in _kinds(diagnose(config_tree))


def test_an_editable_path_holding_a_space_is_read_whole(config_tree: ConfigRoot):
    """uv reads everything after the flag as one path; so must the diagnosis.

    Reading only the first token reports a missing checkout for a directory
    that is present under its real name -- a false positive from the command
    whose job is to say what is actually wrong.
    """
    checkout = config_tree.root / "my pkg"
    checkout.mkdir()
    config_tree.profile_path("ds").write_text("includes:\n  - -e my pkg\n")
    findings = diagnose(config_tree)
    assert not [f for f in findings if f.kind == "missing-checkout"]


def test_an_editable_path_holding_a_space_is_read_whole_when_absent(
    config_tree: ConfigRoot,
):
    """The complete path must appear in the message for an absent checkout.

    Strengthens the positive test: a parser that truncated the path would fail
    here, while both would pass if only the presence of the finding mattered.
    """
    config_tree.profile_path("ds").write_text("includes:\n  - -e my pkg\n")
    findings = diagnose(config_tree)
    missing = [f for f in findings if f.kind == "missing-checkout"]
    assert len(missing) == 1
    assert str(config_tree.root / "my pkg") in missing[0].message


def test_an_editable_entry_with_a_trailing_option_keeps_its_path_token(
    config_tree: ConfigRoot,
):
    """The rejoin must not swallow an option into the path.

    Pins one side of the boundary the fix draws: rejoining stops before a
    token that could be an option of its own.
    """
    config_tree.profile_path("ds").write_text(
        "includes:\n  - -e ./absent --config-settings editable_mode=compat\n"
    )
    findings = diagnose(config_tree)
    missing = [f for f in findings if f.kind == "missing-checkout"]
    assert len(missing) == 1
    assert "--config-settings" not in missing[0].message
    assert str(config_tree.root / "absent") in missing[0].message


def test_an_editable_path_holding_a_space_survives_a_trailing_option(
    config_tree: ConfigRoot,
):
    """Both halves of the rule at once: rejoin the path, then stop at the option.

    The other side of the boundary. An all-or-nothing rejoin -- give up
    entirely as soon as any later token looks like an option -- passes both
    tests above and still reports './my' here.
    """
    checkout = config_tree.root / "my pkg"
    checkout.mkdir()
    config_tree.profile_path("ds").write_text(
        "includes:\n  - -e my pkg --config-settings editable_mode=compat\n"
    )
    findings = diagnose(config_tree)
    assert not [f for f in findings if f.kind == "missing-checkout"]


def test_an_editable_path_with_attached_equals_reads_the_whole_spaced_path(
    config_tree: ConfigRoot,
):
    """The attached form must read the operand verbatim, not from split().

    '-e=my pkg' partitions parts[0] into ('-e', '=', 'my'), leaving 'pkg' in
    parts[1:]. Stopping at that attached operand truncates the path exactly as
    taking the first token alone did before the separated form was fixed.
    """
    checkout = config_tree.root / "my pkg"
    checkout.mkdir()
    config_tree.profile_path("ds").write_text("includes:\n  - -e=my pkg\n")
    findings = diagnose(config_tree)
    assert not [f for f in findings if f.kind == "missing-checkout"]


def test_an_editable_path_with_long_flag_attached_names_the_whole_path_when_absent(
    config_tree: ConfigRoot,
):
    """The --editable= spelling must report the complete path when absent.

    Covers the long-flag attached form and asserts on message content so a
    truncating parser would fail.
    """
    config_tree.profile_path("ds").write_text("includes:\n  - --editable=my pkg\n")
    findings = diagnose(config_tree)
    missing = [f for f in findings if f.kind == "missing-checkout"]
    assert len(missing) == 1
    assert str(config_tree.root / "my pkg") in missing[0].message


def test_an_editable_path_with_repeated_whitespace_preserves_the_run_length(
    config_tree: ConfigRoot,
):
    """Interior whitespace must be preserved exactly as written.

    split() discards run length and rejoin cannot restore it, so a directory
    whose real name holds two spaces is reported missing while it sits there.
    The parser must extract the operand from the original entry, not from
    the split tokens.
    """
    config_tree.profile_path("ds").write_text("includes:\n  - -e ./my  pkg\n")
    findings = diagnose(config_tree)
    missing = [f for f in findings if f.kind == "missing-checkout"]
    assert len(missing) == 1
    assert str(config_tree.root / "my  pkg") in missing[0].message


def test_an_editable_path_with_a_trailing_comment_is_read_without_the_comment(
    config_tree: ConfigRoot,
):
    """A comment marker after whitespace terminates the operand.

    uv treats '#' as a comment marker at the start of a line or after
    whitespace. The entry must be quoted in YAML — an unquoted
    '- -e ./pkg # note' has its comment eaten by the YAML parser before
    doctor sees it.
    """
    checkout = config_tree.root / "pkg"
    checkout.mkdir()
    # The entry must be quoted so the YAML parser preserves the comment.
    config_tree.profile_path("ds").write_text('includes:\n  - "-e ./pkg # local checkout"\n')
    findings = diagnose(config_tree)
    assert not [f for f in findings if f.kind == "missing-checkout"]


def test_an_editable_path_with_a_trailing_comment_names_the_path_when_absent(
    config_tree: ConfigRoot,
):
    """The message must name the path without the comment text.

    Proves the parser cut before the comment rather than merely accepting
    it as part of the path.
    """
    # The entry must be quoted so the YAML parser preserves the comment.
    config_tree.profile_path("ds").write_text('includes:\n  - "-e ./pkg # local"\n')
    findings = diagnose(config_tree)
    missing = [f for f in findings if f.kind == "missing-checkout"]
    assert len(missing) == 1
    assert str(config_tree.root / "pkg") in missing[0].message
    assert "# local" not in missing[0].message


def test_an_editable_spaced_path_with_a_trailing_comment_reads_both_rules(
    config_tree: ConfigRoot,
):
    """Whitespace preservation and comment termination in one entry.

    A spaced path plus a trailing comment exercises both rules: rejoin the
    path tokens, then stop before the comment.
    """
    checkout = config_tree.root / "my pkg"
    checkout.mkdir()
    # The entry must be quoted so the YAML parser preserves the comment.
    config_tree.profile_path("ds").write_text('includes:\n  - "-e my pkg # note"\n')
    findings = diagnose(config_tree)
    assert not [f for f in findings if f.kind == "missing-checkout"]


def test_an_editable_path_containing_a_hash_preserves_the_character(
    config_tree: ConfigRoot,
):
    """A '#' with no whitespace before it is part of the operand.

    uv treats '#' as a comment marker only at the start of a line or after
    whitespace. A '#' inside a token is ordinary text.
    """
    checkout = config_tree.root / "pkg#1"
    checkout.mkdir()
    config_tree.profile_path("ds").write_text("includes:\n  - -e ./pkg#1\n")
    findings = diagnose(config_tree)
    assert not [f for f in findings if f.kind == "missing-checkout"]


def test_an_editable_path_with_attached_equals_and_comment_cuts_at_the_comment(
    config_tree: ConfigRoot,
):
    """The attached spelling must also terminate at a comment marker.

    The two spellings must read alike, so '-e=./pkg # note' must cut at the
    comment just as '-e ./pkg # note' does.
    """
    checkout = config_tree.root / "pkg"
    checkout.mkdir()
    # The entry must be quoted so the YAML parser preserves the comment.
    config_tree.profile_path("ds").write_text('includes:\n  - "-e=./pkg # note"\n')
    findings = diagnose(config_tree)
    assert not [f for f in findings if f.kind == "missing-checkout"]


def test_an_editable_entry_that_is_only_a_comment_is_silent(
    config_tree: ConfigRoot,
):
    """When the operand is nothing but a comment, doctor stays silent.

    uv sees a bare '-e' here, which is a malformed entry, not a checkout.
    The entry must be quoted so the YAML parser preserves the comment.
    """
    # The entry must be quoted so the YAML parser preserves the comment.
    config_tree.profile_path("ds").write_text('includes:\n  - "-e # local checkout"\n')
    findings = diagnose(config_tree)
    # No missing-checkout and no other findings about this entry.
    assert not findings


def test_an_editable_entry_with_empty_attached_operand_and_comment_is_silent(
    config_tree: ConfigRoot,
):
    """The attached form with empty operand falls through to separated reading.

    '-e= # note' has an empty attached operand, so it falls through to the
    separated branch where the lstrip()ed remainder begins with '#' and the
    guard fires. The entry must be quoted so the YAML parser preserves the
    comment.
    """
    # The entry must be quoted so the YAML parser preserves the comment.
    config_tree.profile_path("ds").write_text('includes:\n  - "-e= # note"\n')
    findings = diagnose(config_tree)
    # No missing-checkout and no other findings about this entry.
    assert not findings


def test_an_editable_entry_with_glued_hash_names_a_path_beginning_with_hash(
    config_tree: ConfigRoot,
):
    """In the attached form, a '#' immediately after '=' is inside the token.

    '-e=#note' has no whitespace before the '#', so the '#' is ordinary text
    and uv installs from a directory named '#note'. The entry is quoted for
    consistency with the other hash tests, though the '#' has no space before
    it and would survive unquoted.
    """
    # Quoted for consistency with other hash tests, though not strictly required.
    config_tree.profile_path("ds").write_text('includes:\n  - "-e=#note"\n')
    findings = diagnose(config_tree)
    missing = [f for f in findings if f.kind == "missing-checkout"]
    assert len(missing) == 1
    assert str(config_tree.root / "#note") in missing[0].message


def test_the_checkout_check_is_skipped_when_a_variable_is_undefined(
    config_tree: ConfigRoot,
):
    config_tree.variables_path().write_text("DEV\n")
    config_tree.profile_path("dev").write_text("includes:\n  - -e ${DEV}/widget\n")
    kinds = _kinds(diagnose(config_tree))
    assert "undefined-variable" in kinds
    assert "missing-checkout" not in kinds


def test_the_checkout_check_is_skipped_when_a_reference_is_undeclared(
    config_tree: ConfigRoot,
):
    config_tree.profile_path("dev").write_text("includes:\n  - -e ${NOPE}/widget\n")
    kinds = _kinds(diagnose(config_tree))
    assert "undeclared-variable" in kinds
    assert "missing-checkout" not in kinds


def test_a_misplaced_reference_also_suppresses_the_checkout_check(
    config_tree: ConfigRoot, tmp_path: Path
):
    # Placement is judged per entry, but the suppression is per root: with any
    # placement problem outstanding, expansion is not trustworthy enough to
    # report a *second* diagnosis derived from it.
    config_tree.variables_path().write_text("DEV\nPKG\n")
    config_tree.variables_local_path().write_text(
        f"DEV={tmp_path / 'gone'}\nPKG=widget\n"
    )
    config_tree.profile_path("dev").write_text(
        "includes:\n  - ${PKG}>=2\n  - -e ${DEV}/widget\n"
    )
    kinds = _kinds(diagnose(config_tree))
    assert "misplaced-reference" in kinds
    assert "missing-checkout" not in kinds


def test_an_unsafe_expansion_is_reported_as_an_error(config_tree: ConfigRoot):
    # The one expansion failure the suppression guard does not cover: DEV is
    # declared and has a value here, and the entry's placement is admitted,
    # because the defect only exists after substitution. The expansion check is
    # therefore the first thing to meet it -- and the last. It must neither
    # carry the error out of 'stack doctor' nor drop it: no other check can see
    # that this root is one 'stack converge' will refuse.
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text("DEV=-r\n")
    config_tree.profile_path("dev").write_text("includes:\n  - -e ${DEV}/widget\n")
    findings = diagnose(config_tree)
    assert "missing-checkout" not in _kinds(findings)
    finding = next(f for f in findings if f.kind == "unsafe-expansion")
    assert finding.level == "error"
    assert finding.path == config_tree.profile_path("dev")
    # The source, the entry, and the explanation -- converge's own words.
    assert finding.message.startswith(f"{config_tree.profile_path('dev')}: ")
    assert "-e ${DEV}/widget" in finding.message
    assert "-r/widget" in finding.message
    # Converge's wording is deliberately not pinned here -- doctor quotes it
    # verbatim so the two cannot drift, and test_variables.py owns it. What is
    # doctor's own is the flattening: converge raises a multi-line ConfigError,
    # and a finding message is one line.
    assert "\n" not in finding.message
    assert finding.fix is not None


def test_an_unresolvable_tilde_user_is_reported_not_raised(config_tree: ConfigRoot):
    # Path.expanduser raises RuntimeError on a '~user' with no home directory,
    # and RuntimeError is neither UvStackError nor OSError -- the two families
    # the CLI turns into messages -- so the pathlib spelling tracebacks here.
    # No variable is involved: this entry never reaches expand_all's guard, so
    # only the checkout check's own spelling stands between it and a crash.
    config_tree.profile_path("dev").write_text(
        "includes:\n  - -e ~__no_such_user__/widget\n"
    )
    finding = next(f for f in diagnose(config_tree) if f.kind == "missing-checkout")
    assert "~__no_such_user__/widget" in finding.message


def test_a_nul_in_a_tilde_user_entry_is_reported_not_raised(config_tree: ConfigRoot):
    # posixpath resolves the '~user' form through pwd.getpwnam and catches only
    # KeyError, so an embedded NUL comes back as ValueError -- neither
    # UvStackError nor OSError, which means nothing between here and the CLI
    # catches it. Unlike the permission cases around it, this one raises on
    # every interpreter this project supports, so it is the one test in this
    # area that cannot quietly stop discriminating.
    config_tree.env_stack_path("main").write_text("-e ~ab\0cd/widget\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "missing-checkout")
    # The text is kept as written and resolved against the root, which is the
    # same answer the unresolvable-'~user' case gets one test above.
    assert finding.path == config_tree.root / "~ab\0cd/widget"


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_an_unsearchable_checkout_parent_is_reported_not_raised(
    config_tree: ConfigRoot, tmp_path: Path
):
    # The probed path is built from a variable value, so an ancestor this user
    # cannot search is ordinary input. Path.exists re-raises EACCES on every
    # interpreter this project supports, which would abort the whole run over
    # one entry -- on the happy path, with nothing else wrong with the root.
    closed = tmp_path / "closed"
    (closed / "widget").mkdir(parents=True)
    config_tree.variables_path().write_text("DEV\n")
    config_tree.variables_local_path().write_text(f"DEV={closed}\n")
    config_tree.profile_path("dev").write_text("includes:\n  - -e ${DEV}/widget\n")
    os.chmod(closed, 0o000)
    try:
        kinds = _kinds(diagnose(config_tree))
    finally:
        # Restore, or tmp_path teardown cannot remove the directory.
        os.chmod(closed, 0o700)
    assert "missing-checkout" in kinds


def test_a_project_python_path_is_a_warning(config_tree: ConfigRoot):
    config_tree.project_python_path().write_text("/opt/envs/x/bin/python\n")
    finding = next(f for f in diagnose(config_tree) if f.kind == "project-python-path")
    assert finding.level == "warn"
    assert "holds an interpreter path:" in finding.message
    assert "/opt/envs/x/bin/python" in finding.message


def test_a_project_python_naming_an_undeclared_env_is_a_warning(
    config_tree: ConfigRoot,
):
    config_tree.project_python_path().write_text("scratch\n")
    finding = next(
        f for f in diagnose(config_tree) if f.kind == "project-python-undeclared-env"
    )
    assert finding.level == "warn"
    assert "names environment 'scratch', which this root does not declare" in finding.message
    assert finding.fix is not None
    assert "Declared environments: main." in finding.fix


def test_portable_project_python_values_are_not_flagged(config_tree: ConfigRoot):
    for value in ("3.12", "cpython@3.12", "pypy-3.10", "main"):
        config_tree.project_python_path().write_text(value + "\n")
        assert not [
            f for f in diagnose(config_tree) if f.kind.startswith("project-python-")
        ]


def test_a_missing_ignore_block_in_a_repository_is_a_warning(
    config_tree: ConfigRoot,
):
    (config_tree.root / ".git").mkdir()
    finding = next(f for f in diagnose(config_tree) if f.kind == "stale-ignore-block")
    assert finding.level == "warn"
    # Anchored to the end of the message rather than searched for: the message
    # quotes the config root, and pytest builds that path out of this test's
    # own name, so a substring test for 'missing' passes whatever the wording.
    assert finding.message.endswith("is missing.")
    assert finding.fix is not None
    assert "stack config portable" in finding.fix


def test_a_stale_ignore_block_says_out_of_date(config_tree: ConfigRoot):
    # A block that exists but no longer matches what the writer would produce
    # is a different state from one that is absent, and it is worded
    # differently. Writing the markers by hand keeps the test independent of
    # which patterns the block currently holds.
    (config_tree.root / ".git").mkdir()
    (config_tree.root / ".gitignore").write_text(
        f"{BEGIN_MARKER}\nstale-entry\n{END_MARKER}\n"
    )
    finding = next(f for f in diagnose(config_tree) if f.kind == "stale-ignore-block")
    assert "out of date" in finding.message


def test_a_stale_ignore_block_in_a_nested_root_is_reported(config_tree: ConfigRoot):
    # The root is a subdirectory of the working tree rather than the working
    # tree itself, which is the 'git clone <url> ~/.config/python-envs' layout
    # inside a dotfiles repo. Its files are tracked all the same, so a stale
    # block is the same defect here as one directory up — and gating on
    # <root>/.git alone reported this root clean, which is the one answer a
    # check that could not tell is never allowed to give.
    (config_tree.root.parent / ".git").mkdir()
    (config_tree.root / ".gitignore").write_text(
        f"{BEGIN_MARKER}\nstale-entry\n{END_MARKER}\n"
    )
    finding = next(f for f in diagnose(config_tree) if f.kind == "stale-ignore-block")
    assert "out of date" in finding.message


def test_a_missing_ignore_block_in_a_nested_root_is_reported(config_tree: ConfigRoot):
    (config_tree.root.parent / ".git").mkdir()
    finding = next(f for f in diagnose(config_tree) if f.kind == "stale-ignore-block")
    assert finding.message.endswith("is missing.")


def test_a_current_ignore_block_is_not_flagged(config_tree: ConfigRoot):
    (config_tree.root / ".git").mkdir()
    write_portable_ignore(config_tree)
    assert not [f for f in diagnose(config_tree) if f.kind == "stale-ignore-block"]


def test_a_symlinked_ignore_file_is_reported_not_reported_clean(
    config_tree: ConfigRoot,
):
    # A link whose target already holds the current block reads as up to date
    # through the link, so the staleness check has nothing to say about it.
    # What travels is the link, not the rules, and reporting clean on a root
    # whose ignore rules will not survive a clone is the one answer this scan
    # may not give. The writer's refusal is what doctor reports.
    (config_tree.root / ".git").mkdir()
    destination = config_tree.root / "ignore-target"
    destination.write_text(render_block(config_tree) + "\n")
    (config_tree.root / ".gitignore").symlink_to(destination)

    findings = diagnose(config_tree)

    assert not [f for f in findings if f.kind == "stale-ignore-block"]
    finding = next(
        f
        for f in findings
        if f.kind == "unparseable-source"
        and f.path == config_tree.root / ".gitignore"
    )
    assert "Symlinked ignore file" in finding.message


def _case_insensitive(root: Path) -> bool:
    """Whether this filesystem resolves two spellings to one directory."""
    probe = root / "CaseProbe"
    probe.mkdir()
    try:
        return (root / "caseprobe").is_dir()
    finally:
        probe.rmdir()


def test_a_case_variant_of_a_root_directory_is_not_called_misplaced(
    config_tree: ConfigRoot,
):
    """On a case-insensitive filesystem 'Lib' IS the root's lib directory.

    Falling through to the marker check prints 'mv <root>/Lib
    <root>/envs/Lib' -- a proposal to move one of the root's own directories
    into envs/, from the command whose job is to say what is actually wrong.

    'Lib' rather than 'Envs' because scandir reports the name a directory was
    created with: the fixture already makes 'envs', so mkdir('Envs') there is
    a no-op and the entry stays lowercase. The fixture makes no 'lib'.
    """
    if not _case_insensitive(config_tree.root):
        pytest.skip("two spellings only name one directory on a case-insensitive filesystem")
    variant = config_tree.root / "Lib"
    variant.mkdir()
    (variant / "requirements.in").write_text("numpy\n")

    findings = diagnose(config_tree)

    assert not [f for f in findings if f.kind == "misplaced-env"]


def test_a_genuinely_separate_case_variant_directory_is_still_reported(
    config_tree: ConfigRoot,
):
    """The case-sensitive platform must keep its real finding.

    Here 'Lib' is a different directory from any 'lib', so an env sitting in
    it is genuinely misplaced. Casefolding the name would suppress this, which
    is why the fix asks the filesystem instead -- the same reasoning
    doctor.py:280-288 already applies to the marker names.
    """
    if _case_insensitive(config_tree.root):
        pytest.skip("the two spellings are one directory on a case-insensitive filesystem")
    stray = config_tree.root / "Lib"
    stray.mkdir()
    (stray / "requirements.in").write_text("numpy\n")

    findings = diagnose(config_tree)

    assert [f for f in findings if f.kind == "misplaced-env" and f.path == stray]


def test_a_non_repository_root_is_never_flagged(config_tree: ConfigRoot):
    assert not [f for f in diagnose(config_tree) if f.kind == "stale-ignore-block"]


# --- OSError containment -----------------------------------------------------
# A file that exists but cannot be opened raises OSError, not UvStackError.
# Each read the portability scan performs gets its own proof that the failure
# becomes a finding rather than a traceback out of 'stack doctor'.


def _denied(*args, **kwargs):
    """Stand in for any read that fails at the filesystem layer.

    :raises PermissionError: Always.
    """
    raise PermissionError(13, "Permission denied")


def test_an_unopenable_profile_is_reported_not_raised(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.setattr(ConfigRoot, "load_profile", _denied)
    findings = diagnose(config_tree)
    assert any(
        f.kind == "unparseable-source" and f.path == config_tree.profile_path("ds")
        for f in findings
    )


@pytest.mark.parametrize(
    ("listing", "directory"),
    [
        ("list_profiles", "profiles_dir"),
        ("list_bundles", "bundles_dir"),
        ("list_envs", "envs_dir"),
    ],
)
def test_an_unreadable_source_directory_is_reported_not_raised(
    config_tree: ConfigRoot, monkeypatch, listing: str, directory: str
):
    # list_envs walks the directory with iterdir, which raises outright; the
    # other two use glob, which swallows a permission error today but is not
    # contracted to. All three are guarded, so all three are proved.
    monkeypatch.setattr(ConfigRoot, listing, _denied)
    findings = diagnose(config_tree)
    assert any(
        f.kind == "unparseable-source" and f.path == getattr(config_tree, directory)
        for f in findings
    )


def test_an_unopenable_variables_file_is_reported_not_raised(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.setattr(ConfigRoot, "load_variables", _denied)
    findings = diagnose(config_tree)
    assert any(
        f.kind == "unparseable-source" and f.path == config_tree.variables_path()
        for f in findings
    )


def test_an_unopenable_project_python_file_is_reported_not_raised(
    config_tree: ConfigRoot, monkeypatch
):
    monkeypatch.setattr(ConfigRoot, "default_project_python", _denied)
    findings = diagnose(config_tree)
    assert any(
        f.kind == "unparseable-source" and f.path == config_tree.project_python_path()
        for f in findings
    )


def test_an_unopenable_gitignore_is_reported_not_raised(
    config_tree: ConfigRoot, monkeypatch
):
    (config_tree.root / ".git").mkdir()
    monkeypatch.setattr("uv_stack.operations.doctor.write_portable_ignore", _denied)
    findings = diagnose(config_tree)
    assert any(
        f.kind == "unparseable-source" and f.path == config_tree.root / ".gitignore"
        for f in findings
    )


@pytest.mark.parametrize("attribute", ["root", "envs_dir"])
def test_an_unwalkable_directory_is_reported_not_raised(
    config_tree: ConfigRoot, monkeypatch, attribute: str
):
    # diagnose walks the config root and envs/ with iterdir directly, and both
    # walks run before the portability scan. Patching ConfigRoot.list_envs
    # cannot see either one -- neither goes through it -- which is why the two
    # need their own proof. Path.iterdir is patched selectively so only the
    # target directory fails; a blanket failure would prove nothing about
    # which walk is guarded.
    target = getattr(config_tree, attribute)
    real_iterdir = Path.iterdir

    def _selective(self: Path):
        # A generator function, because Path.iterdir is one on Python 3.12:
        # the failure surfaces on the first iteration, not at the call. 3.13
        # made it eager, so an ordinary function here would quietly stop
        # testing that _children materializes the walk inside its guard.
        if self == target:
            raise PermissionError(13, "Permission denied")
        yield from real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _selective)
    findings = diagnose(config_tree)
    # The envs_dir case reports twice -- once from diagnose's own walk and once
    # from list_envs inside the source scan. Both are correct, so this asserts
    # that one exists rather than counting them.
    assert any(f.kind == "unparseable-source" and f.path == target for f in findings)


def test_an_unreadable_envs_directory_does_not_escape_the_project_check(
    config_tree: ConfigRoot, monkeypatch
):
    # 'scratch' is not a declared environment, so python_travel_problem has to
    # consult list_envs to say so, and the fix line joins the same listing.
    # Both sit past the file read, so a guard around default_project_python()
    # alone lets a PermissionError out of 'stack doctor'.
    config_tree.project_python_path().write_text("scratch\n")
    monkeypatch.setattr(ConfigRoot, "list_envs", _denied)
    findings = diagnose(config_tree)
    assert any(
        f.kind == "unparseable-source"
        and f.path == config_tree.project_python_path()
        for f in findings
    )


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
@pytest.mark.parametrize("unsearchable", ["envs-dir", "one-env", "root-child"])
def test_an_unsearchable_directory_does_not_abort_diagnose(
    config_tree: ConfigRoot, unsearchable: str
):
    # Mode 400 is the state that bites: the directory still lists, so the walk
    # guard hands back names, and the very next statement probes one of them.
    # The pathlib probes re-raise EACCES there, taking the run down one line
    # after the guard did its job.
    #
    # Each case leaves a different probe holding the bag: the envs directory
    # reaches the per-env isdir, one env reaches the source-file probes inside
    # it, and a directory under the root reaches the misplaced-env probes.
    # All three assert on project-python-path, which is produced by the last
    # stage of the run: if it survives, every probe above it did.
    config_tree.project_python_path().write_text("/opt/envs/x/bin/python\n")
    legacy = config_tree.root / "legacyenv"
    legacy.mkdir()
    target = {
        "envs-dir": config_tree.envs_dir,
        "one-env": config_tree.env_dir("main"),
        "root-child": legacy,
    }[unsearchable]
    os.chmod(target, 0o400)
    try:
        kinds = _kinds(diagnose(config_tree))
    finally:
        # Restore, or tmp_path teardown cannot remove the directory.
        os.chmod(target, 0o700)
    assert "project-python-path" in kinds


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_an_unreadable_python_txt_does_not_abort_diagnose(
    config_tree: ConfigRoot, tmp_path: Path
):
    # The python.txt probe is the second operand of an 'and', so every state
    # that makes the first operand fail short-circuits past it: the mode-400
    # cases elsewhere in this file reach it never. Pointing python.txt at a
    # directory this user cannot search, with the real stack.txt left beside
    # it, is the shape that makes both operands run, and the one that makes
    # the presence test answer for a path whose target it cannot reach.
    closed = tmp_path / "closed"
    closed.mkdir()
    (closed / "python.txt").write_text("3.12\n")
    config_tree.env_python_path("main").unlink()
    config_tree.env_python_path("main").symlink_to(closed / "python.txt")
    os.chmod(closed, 0o000)
    try:
        kinds = _kinds(diagnose(config_tree))
    finally:
        # Restore, or tmp_path teardown cannot remove the directory.
        os.chmod(closed, 0o700)
    # Reported by name, and not a traceback. The symlink is present, so the
    # presence test declines to call it missing -- which is what it is for:
    # the old 'missing python.txt (will default to 3.12)' arrived alongside
    # this finding and contradicted it, offering a repair that O_EXCL refuses
    # because the entry is already there.
    assert kinds == ["unparseable-source"]
    assert "missing-python-txt" not in kinds


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_an_unsearchable_config_root_reports_instead_of_raising(config_tree: ConfigRoot):
    # The same state one level up. The root still lists, so every fixed-path
    # probe beneath it -- and the .git probe in the ignore-block check -- gets
    # an EACCES the pathlib spelling would re-raise. The finding asserted on
    # comes from the source scan at the end of the run, so it stands for the
    # run having finished rather than for any one probe.
    os.chmod(config_tree.root, 0o400)
    try:
        findings = diagnose(config_tree)
    finally:
        os.chmod(config_tree.root, 0o700)
    assert any(
        f.kind == "unparseable-source" and f.path == config_tree.profiles_dir for f in findings
    )


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_an_unsearchable_root_does_not_call_its_directories_missing(config_tree: ConfigRoot):
    """The same tree as above, asserted on what it must not say.

    Stat'ing profiles/ needs search permission on the root, not on profiles/,
    so an unsearchable root turns all three fixed-path probes False for
    directories that are plainly there. The run then contradicted itself: three
    errors saying they are missing, each offering a mkdir that cannot run, and
    below them three warns correctly naming the same paths as unreadable.
    """
    os.chmod(config_tree.root, 0o400)
    try:
        findings = diagnose(config_tree)
    finally:
        os.chmod(config_tree.root, 0o700)

    assert [f.message for f in findings if f.kind == "missing-dir"] == []
    # Suppressing the three errors may not cost the paths their mention: the
    # warns are what is left saying anything about them at all.
    unreadable = {f.path for f in findings if f.kind == "unparseable-source"}
    assert {
        config_tree.profiles_dir,
        config_tree.bundles_dir,
        config_tree.envs_dir,
    } <= unreadable


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_an_unsearchable_root_names_each_directory_once(config_tree: ConfigRoot):
    """The same tree again, asserted on how many times it says it.

    Two independent probes reach the same conclusion here: the scaffold-directory
    stat fails EACCES, and so does the listing _scan_sources does for the same
    path. Both render the errno through _unparseable, so the findings come out
    byte-identical. Printed twice they read as two distinct problems.
    """
    os.chmod(config_tree.root, 0o400)
    try:
        findings = diagnose(config_tree)
    finally:
        os.chmod(config_tree.root, 0o700)

    for directory in (
        config_tree.profiles_dir,
        config_tree.bundles_dir,
        config_tree.envs_dir,
    ):
        named = [
            f for f in findings if f.kind == "unparseable-source" and f.path == directory
        ]
        assert len(named) == 1, named


def test_a_genuinely_absent_directory_under_a_searchable_root_is_still_missing(
    config_tree: ConfigRoot,
):
    # The control for the test above: the suppression is conditioned on the
    # root being unsearchable, not on the finding being inconvenient.
    shutil.rmtree(config_tree.bundles_dir)
    findings = diagnose(config_tree)
    missing = [f for f in findings if f.kind == "missing-dir"]
    assert [f.path for f in missing] == [config_tree.bundles_dir]
    assert missing[0].level == "error"


def test_an_unstattable_scaffold_directory_is_reported_as_unreadable(
    config_tree: ConfigRoot,
):
    """A directory the kernel will not describe must not read as clean.

    Suppressing missing-dir for a residual errno was right -- the mkdir it
    offers fails for the same reason the stat did -- but nothing else picked
    the path up. The legacy scan is guarded by os.path.isdir, which answers
    False for that same errno, so _children never ran and no walk named the
    directory either. The whole run came back empty for a root whose profiles/
    cannot be used at all.

    A self-referential symlink is the portable way to force the errno (ELOOP)
    without root privileges and without a chmod, which would need the geteuid
    guard the permission-based tests above carry.
    """
    shutil.rmtree(config_tree.profiles_dir)
    os.symlink("profiles", config_tree.profiles_dir)

    findings = diagnose(config_tree)

    about_profiles = [f for f in findings if f.path == config_tree.profiles_dir]
    assert [(f.level, f.kind) for f in about_profiles] == [("warn", "unparseable-source")]
    assert str(config_tree.profiles_dir) in about_profiles[0].message
    # The suppression this sits next to still holds: no error offering a mkdir
    # that would fail exactly as the stat did.
    assert [f for f in findings if f.kind == "missing-dir"] == []
    # The two directories that are fine stay silent, so the warn is a report
    # about profiles/ and not about the run having given up.
    assert [
        f
        for f in findings
        if f.path in (config_tree.bundles_dir, config_tree.envs_dir)
    ] == []


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_an_unsearchable_parent_of_the_root_reports_a_missing_root(tmp_path: Path):
    # The root's own probe needs search permission on the root's parent, which
    # is the only place it is reachable. "Does not exist" is not the whole
    # truth about a root that cannot be stat'd, but it is an error with a fix
    # attached, which a traceback is not.
    outer = tmp_path / "outer"
    (outer / "python-envs").mkdir(parents=True)
    os.chmod(outer, 0o400)
    try:
        kinds = _kinds(diagnose(ConfigRoot(outer / "python-envs")))
    finally:
        os.chmod(outer, 0o700)
    assert kinds == ["missing-root"]


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
@pytest.mark.parametrize("directory", ["profiles", "bundles"])
def test_an_unreadable_legacy_scan_directory_is_not_reported_clean(
    config_tree: ConfigRoot, directory: str
):
    # glob answers a directory it cannot read with an empty match, so a legacy
    # scan written with it lets 'stack doctor' print a clean bill of health for
    # a directory it never read. The listings inside list_profiles and
    # list_bundles are globs too, so nothing else here would notice.
    target = config_tree.profiles_dir if directory == "profiles" else config_tree.bundles_dir
    os.chmod(target, 0o000)
    try:
        findings = diagnose(config_tree)
    finally:
        os.chmod(target, 0o700)
    assert any(f.kind == "unparseable-source" and f.path == target for f in findings)


@pytest.mark.skipif(_IS_ROOT, reason="root ignores the directory mode this relies on")
def test_an_unsearchable_directory_under_root_is_not_reported_clean(
    config_tree: ConfigRoot,
):
    # Stat'ing a directory needs search permission on its parent, not on
    # itself, so the root walk hands this child back and the marker test runs
    # against it. Written with exists() that test collapsed EACCES to False
    # and doctor returned nothing at all -- for a directory that, one chmod
    # later, produces a real finding. The control below is the whole point:
    # this is a diagnosis lost, not a diagnosis never attempted.
    legacy = config_tree.root / "legacyenv"
    legacy.mkdir()
    (legacy / "requirements.in").write_text("# x\n")
    os.chmod(legacy, 0o000)
    try:
        findings = diagnose(config_tree)
    finally:
        # Restore, or tmp_path teardown cannot remove the directory.
        os.chmod(legacy, 0o755)
    assert _kinds(findings) == ["unparseable-source"]
    assert findings[0].path == legacy

    assert _kinds(diagnose(config_tree)) == ["misplaced-env"]


def test_a_readable_directory_under_root_without_markers_is_not_a_finding(
    config_tree: ConfigRoot,
):
    # Every unknown top-level directory is enumerated now, not just the
    # env-like ones, so the marker test is the only thing keeping an ordinary
    # directory out of the findings. Without this, reporting each one that was
    # merely walked would pass the rest of the suite.
    notes = config_tree.root / "notes"
    notes.mkdir()
    (notes / "README").write_text("scratch\n")

    assert diagnose(config_tree) == []


def test_a_case_variant_marker_is_diagnosed_where_the_filesystem_resolves_it(
    config_tree: ConfigRoot,
):
    # Membership is decided from the enumerated names, which are exact-case.
    # On a case-insensitive filesystem -- APFS, the macOS default -- that name
    # is the same file config.py, render.py, status.py and upgrade.py all
    # open, so a name test on its own would report clean on a misplaced env
    # the rest of the tool would use. The assertion branches because the right
    # answer genuinely differs by platform: where the filesystem does not
    # resolve it, this is a different file uv-stack never reads, and flagging
    # it would offer an mv repair for a directory that is not an env.
    legacy = config_tree.root / "legacyenv"
    legacy.mkdir()
    (legacy / "Requirements.in").write_text("# x\n")

    findings = diagnose(config_tree)
    if os.path.exists(legacy / "requirements.in"):
        assert _kinds(findings) == ["misplaced-env"]
    else:
        assert findings == []


def test_a_marker_the_filesystem_will_not_resolve_is_still_a_misplaced_env(
    config_tree: ConfigRoot,
):
    # The other half of the membership test. A dangling symlink is listed by
    # the enumeration and answered False by exists(), so deciding membership
    # from the probes alone drops it -- and placement is what this finding is
    # about, not whether the marker resolves. The mv repair moves the broken
    # link along with everything else in the directory.
    legacy = config_tree.root / "legacyenv"
    legacy.mkdir()
    (legacy / "requirements.in").symlink_to(legacy / "nowhere")

    assert _kinds(diagnose(config_tree)) == ["misplaced-env"]
