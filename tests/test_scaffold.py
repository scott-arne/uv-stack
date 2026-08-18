from __future__ import annotations

from unittest import mock

import pytest

from tests.conftest import _lock_held_by_another_process
from uv_stack.config import ConfigRoot
from uv_stack.errors import ConfigError
from uv_stack.fsutil import _LOCK_AVAILABLE
from uv_stack.operations.scaffold import (
    _SHADOW_HINT,
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


def test_write_profile_refuses_collision_with_bundle(config_tree: ConfigRoot):
    """Creating a profile with a name matching an existing bundle raises ConfigError."""
    with pytest.raises(ConfigError) as excinfo:
        write_profile(config_tree, "standard", ["numpy"])
    assert "would shadow the existing bundle" in str(excinfo.value)
    # Verify the file was not created.
    assert not config_tree.profile_path("standard").exists()


def test_write_bundle_round_trips(config_tree: ConfigRoot):
    write_bundle(config_tree, "daily", ["ds", "pkg:httpx"], tags=["core"])
    bundle = config_tree.load_bundle("daily")
    assert bundle.includes == ["ds", "pkg:httpx"]
    assert bundle.tags == ["core"]


def test_write_bundle_refuses_overwrite(config_tree: ConfigRoot):
    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "standard", ["ds"])
    assert "Bundle 'standard' already exists" in str(excinfo.value)


def test_write_bundle_refuses_collision_with_profile(config_tree: ConfigRoot):
    """Creating a bundle with a name matching an existing profile raises ConfigError."""
    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "ds", ["numpy"])
    assert "would be shadowed by the existing profile" in str(excinfo.value)
    # Verify the file was not created.
    assert not config_tree.bundle_path("ds").exists()


@pytest.mark.parametrize("token", ["daily", "@daily", "bundle:daily"])
def test_write_bundle_refuses_self_reference(config_tree: ConfigRoot, token: str):
    """A bundle naming itself resolves to nothing; refuse it where intent is recoverable."""
    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "daily", ["ds", token])
    assert "cannot include itself" in str(excinfo.value)
    assert not config_tree.bundle_path("daily").exists()


def test_write_bundle_allows_qualified_package_of_same_name(config_tree: ConfigRoot):
    """pkg: is the documented escape hatch — it is a package, not a self-reference."""
    write_bundle(config_tree, "httpx", ["pkg:httpx"])
    assert config_tree.load_bundle("httpx").includes == ["pkg:httpx"]


def test_write_bundle_refuses_self_reference_non_plain_name(config_tree: ConfigRoot):
    """A name the resolver accepts but ``_PLAIN_NAME_RE`` rejects is still a self-reference."""
    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "daily+cpu", ["ds", "daily+cpu"])
    assert "cannot include itself" in str(excinfo.value)
    assert not config_tree.bundle_path("daily+cpu").exists()


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


def test_write_env_sources_withdraws_python_when_stack_publish_fails(
    config_tree: ConfigRoot,
):
    """An ordinary stack.txt failure withdraws the python.txt this call published.

    The fake is keyed on the FILE, not the call number: publishing python.txt
    first is half of what makes the crash case retryable, so the test has to
    fail when the order is wrong, not only when the withdrawal is missing.
    """
    from uv_stack.fsutil import atomic_write_new

    published: list[str] = []

    def failing_write_new(path, text):
        published.append(path.name)
        if path.name == "python.txt":
            return atomic_write_new(path, text)
        raise RuntimeError("Simulated stack.txt publish failure")

    with mock.patch(
        "uv_stack.operations.scaffold.atomic_write_new", side_effect=failing_write_new
    ):
        with pytest.raises(RuntimeError):
            write_env_sources(config_tree, "racy", ["ds"], python="3.13")

    assert published == ["python.txt", "stack.txt"]
    # Nothing of ours is left attached to whatever env now owns this directory.
    assert not config_tree.env_python_path("racy").exists()
    assert not config_tree.env_stack_path("racy").exists()


def test_write_env_sources_withdrawal_spares_a_concurrent_replacement(
    config_tree: ConfigRoot,
):
    """The withdrawal is inode-matched, so a third party's python.txt survives."""
    import os

    from uv_stack.fsutil import atomic_write_new

    python_path = config_tree.env_python_path("racy2")

    def failing_write_new(path, text):
        if path.name == "python.txt":
            return atomic_write_new(path, text)
        # A third writer replaces python.txt before our stack.txt publish fails.
        replacement = python_path.parent / "replacement.txt"
        replacement.write_text("3.99\n", encoding="utf-8")
        os.replace(replacement, python_path)
        raise RuntimeError("Simulated stack.txt publish failure")

    with mock.patch(
        "uv_stack.operations.scaffold.atomic_write_new", side_effect=failing_write_new
    ):
        with pytest.raises(RuntimeError):
            write_env_sources(config_tree, "racy2", ["ds"], python="3.13")

    assert python_path.read_text(encoding="utf-8") == "3.99\n"


def test_write_env_sources_does_not_withdraw_python_when_stack_lands_before_interrupt(
    config_tree: ConfigRoot,
):
    """A stack.txt that lands before the publish raises must NOT have its python.txt withdrawn.

    The unfixed code treated any BaseException from the stack.txt publish as
    proof that stack.txt did not land, and withdrew python.txt. That assumption
    is false: atomic_write_new publishes with os.link and then still has work
    to do (returning identity, unlinking the temp). An interrupt arriving after
    the link but before the return propagates with stack.txt already on disk.
    Withdrawing python.txt then creates an environment with no interpreter pin,
    and the retry hits the "already has a stack.txt" refusal — exactly the
    unretryable state the publish-order inversion was meant to prevent.
    """
    from uv_stack.fsutil import atomic_write_new

    def interrupt_after_stack_link(path, text):
        result = atomic_write_new(path, text)
        if path.name == "stack.txt":
            raise KeyboardInterrupt("Simulated interrupt after stack.txt link")
        return result

    with mock.patch(
        "uv_stack.operations.scaffold.atomic_write_new", side_effect=interrupt_after_stack_link
    ):
        with pytest.raises(KeyboardInterrupt):
            write_env_sources(config_tree, "interrupted", ["ds"], python="3.13")

    # stack.txt landed before the interrupt.
    assert config_tree.env_stack_path("interrupted").exists()
    assert config_tree.env_stack_path("interrupted").read_text() == "ds\n"
    # python.txt must NOT be withdrawn — it sits beside the stack.txt that did land.
    assert config_tree.env_python_path("interrupted").exists()
    assert config_tree.env_python_path("interrupted").read_text() == "3.13\n"


def test_write_env_sources_withdraws_python_when_a_third_party_wins_the_stack_race(
    config_tree: ConfigRoot,
):
    """The motivating race, end to end: their stack.txt stands, our pin is withdrawn.

    ``os.link`` lost, so the ``stack.txt`` on disk is somebody else's and the
    ``python.txt`` this call just published would otherwise hang off their
    environment. This is the only test that travels the
    ``isinstance(exc, ConfigError)`` disjunct with the withdrawal allowed to
    succeed: the failed-withdrawal test forces ``Path.unlink`` to raise, so
    dropping that disjunct from the condition moves nothing there but a
    substring of an error message. Here it changes what is on disk — without
    it the withdrawal never runs and ``python.txt`` survives beside a
    ``stack.txt`` that is not ours.
    """
    from uv_stack.fsutil import atomic_write_new

    python_path = config_tree.env_python_path("lost-race")
    stack_path = config_tree.env_stack_path("lost-race")

    def losing_write_new(path, text):
        if path.name == "python.txt":
            return atomic_write_new(path, text)
        # A concurrent writer wins the race; os.link would raise EEXIST here.
        stack_path.parent.mkdir(parents=True, exist_ok=True)
        stack_path.write_text("their-tokens\n", encoding="utf-8")
        raise FileExistsError("Stack file exists")

    with mock.patch(
        "uv_stack.operations.scaffold.atomic_write_new", side_effect=losing_write_new
    ):
        with pytest.raises(ConfigError) as excinfo:
            write_env_sources(config_tree, "lost-race", ["ds"], python="3.13")

    # Our pin is gone; their environment keeps exactly what they wrote.
    assert not python_path.exists()
    assert stack_path.read_text(encoding="utf-8") == "their-tokens\n"
    # The plain refusal, not the residual variant — the withdrawal succeeded.
    assert "already has a stack.txt" in str(excinfo.value)
    assert "could not be removed" not in str(excinfo.value)


def test_write_env_sources_adopts_matching_orphan_python(config_tree: ConfigRoot):
    """A hard crash leaves this orphan; the retry adopts it and completes.

    The orphan is placed by hand because that is what a killed process leaves:
    no handler ran, so nothing was withdrawn.
    """
    config_tree.env_python_path("crashy").parent.mkdir(parents=True, exist_ok=True)
    config_tree.env_python_path("crashy").write_text("3.13\n")

    written = write_env_sources(config_tree, "crashy", ["ds"], python="3.13")

    assert written == [config_tree.env_stack_path("crashy")]
    assert config_tree.env_stack_path("crashy").read_text() == "ds\n"
    assert config_tree.env_python_path("crashy").read_text() == "3.13\n"


def test_write_env_sources_adopts_multiply_linked_orphan_python(
    config_tree: ConfigRoot,
):
    """A crash between link and unlink leaves python.txt with st_nlink == 2.

    The genuine crash artifact has two links: one at python.txt and one at the
    .tmp name that was never unlinked. The adoption probe deliberately does
    NOT check st_nlink == 1 for exactly this reason — copying that condition
    from atomic_write would break the retry this task enables.
    """
    import os

    python_path = config_tree.env_python_path("multi-link")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("3.13\n")

    # Simulate the .tmp leftover by creating a second hard link.
    tmp_link = python_path.parent / "python.txt.tmp"
    os.link(python_path, tmp_link)
    assert python_path.stat().st_nlink == 2

    written = write_env_sources(config_tree, "multi-link", ["ds"], python="3.13")

    assert written == [config_tree.env_stack_path("multi-link")]
    assert config_tree.env_stack_path("multi-link").read_text() == "ds\n"
    assert python_path.read_text() == "3.13\n"


def test_write_env_sources_refuses_python_txt_swapped_after_descriptor_opened(
    config_tree: ConfigRoot,
):
    """A python.txt replaced after opening yields matching bytes but a wrong inode.

    The descriptor-bound read proves only that the *opened* inode held matching
    bytes, while adoption is a claim about the *pathname*: skipping the publish
    asserts that the file now at python.txt is correct. Swapping the file after
    the descriptor is open makes those two diverge, so the identity recheck must
    fail and the call must refuse rather than publish an environment around an
    interpreter pin it never wrote or verified.
    """
    import os

    python_path = config_tree.env_python_path("swapped")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("3.13\n")

    real_fdopen = os.fdopen
    swap_done = False

    def swapping_fdopen(fd, mode="r", *args, **kwargs):
        nonlocal swap_done
        handle = real_fdopen(fd, mode, *args, **kwargs)
        if not swap_done and python_path.exists():
            swap_done = True
            replacement = python_path.parent / "replacement.txt"
            replacement.write_text("3.99\n")
            os.replace(replacement, python_path)
        return handle

    with mock.patch("os.fdopen", side_effect=swapping_fdopen):
        with pytest.raises(ConfigError) as excinfo:
            write_env_sources(config_tree, "swapped", ["ds"], python="3.13")

    assert "already has a python.txt" in str(excinfo.value)
    assert not config_tree.env_stack_path("swapped").exists()
    assert python_path.read_text() == "3.99\n"


def test_write_env_sources_refuses_orphan_python_without_the_open_flags(
    config_tree: ConfigRoot, monkeypatch: pytest.MonkeyPatch
):
    """Without both open flags the adoption preflight does not run, so nothing is adopted.

    The preflight opens a file whose type is not known in advance, which needs
    O_NOFOLLOW and O_NONBLOCK; where either constant is absent it is skipped
    entirely rather than run unguarded. The orphan here is byte-for-byte the one
    test_write_env_sources_adopts_matching_orphan_python adopts, so the gate is
    what decides between the two outcomes: with it off, the file must meet the
    "already has a python.txt" refusal — the behavior that predates adoption —
    and be left untouched for the user to clear.
    """
    from uv_stack.operations import scaffold

    monkeypatch.setattr(scaffold, "_FASTPATH_AVAILABLE", False)
    python_path = config_tree.env_python_path("noflags")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("3.13\n")

    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "noflags", ["ds"], python="3.13")
    assert "already has a python.txt" in str(excinfo.value)
    assert not config_tree.env_stack_path("noflags").exists()
    assert python_path.read_text() == "3.13\n"


def test_write_env_sources_refuses_orphan_python_with_other_content(
    config_tree: ConfigRoot,
):
    """A python.txt that does not match is a user edit, not our debris."""
    config_tree.env_python_path("edited").parent.mkdir(parents=True, exist_ok=True)
    config_tree.env_python_path("edited").write_text("3.11\n")

    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "edited", ["ds"], python="3.13")
    assert "already has a python.txt" in str(excinfo.value)
    assert not config_tree.env_stack_path("edited").exists()


def test_write_env_sources_refuses_symlinked_python_txt(config_tree: ConfigRoot):
    """A symlinked python.txt is refused even when its target holds the requested version."""
    target = config_tree.root / "elsewhere.txt"
    target.write_text("3.13\n")
    python_path = config_tree.env_python_path("symlinked")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.symlink_to(target)

    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "symlinked", ["ds"], python="3.13")
    assert "already has a python.txt" in str(excinfo.value)
    assert not config_tree.env_stack_path("symlinked").exists()
    # The symlink and its target survived untouched.
    assert python_path.is_symlink()
    assert target.read_text() == "3.13\n"


def test_write_env_sources_refuses_unreadable_python_txt(config_tree: ConfigRoot):
    """A python.txt that cannot be read as UTF-8 is refused."""
    python_path = config_tree.env_python_path("unreadable")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_bytes(b"\xff\xfe")

    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "unreadable", ["ds"], python="3.13")
    assert "already has a python.txt" in str(excinfo.value)
    assert not config_tree.env_stack_path("unreadable").exists()


def test_write_env_sources_refuses_fifo_python_txt(config_tree: ConfigRoot):
    """A FIFO at python.txt is refused without blocking on a read.

    The descriptor-bound adoption guard refuses a FIFO before any blocking
    read attempt. The SIGALRM timeout ensures a regression to path-based
    reading fails loudly rather than hanging the suite.

    The handler raises a subclass of BaseException rather than of Exception so
    that the adoption probe cannot swallow it. With an ordinary TimeoutError —
    an OSError subclass — a regression to path-based reading would block inside
    the probe's try, the alarm exception would be caught by its
    `except (OSError, UnicodeDecodeError)`, and the call would refuse with
    exactly the ConfigError this test asserts, so the test would pass against
    the bug. A BaseException that is not an Exception cannot be caught by that
    handler, nor by any broader `except Exception` a future edit introduces.
    """
    import os
    import signal

    python_path = config_tree.env_python_path("fifo")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(python_path)

    class _Blocked(BaseException):
        pass

    def timeout_handler(signum, frame):
        raise _Blocked("write_env_sources blocked on FIFO read")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, 2)
        try:
            with pytest.raises(ConfigError) as excinfo:
                write_env_sources(config_tree, "fifo", ["ds"], python="3.13")
            assert "already has a python.txt" in str(excinfo.value)
            assert not config_tree.env_stack_path("fifo").exists()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def test_write_env_sources_refuses_directory_python_txt(config_tree: ConfigRoot):
    """A directory at python.txt is refused.

    This pins the outcome, not the mechanism: with the ``S_ISREG`` guard removed
    the refusal still arrives, because reading a directory descriptor raises
    ``IsADirectoryError`` into the probe's own ``except OSError``. Do not read
    this as a regression test for that guard.
    """
    python_path = config_tree.env_python_path("directory")
    python_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "directory", ["ds"], python="3.13")
    assert "already has a python.txt" in str(excinfo.value)
    assert not config_tree.env_stack_path("directory").exists()


def test_write_env_sources_refuses_python_txt_appearing_after_preflight(
    config_tree: ConfigRoot, monkeypatch
):
    """The publish-site refusal matches the preflight message and hint.

    A python.txt that appears between the preflight and the publish is refused
    by the atomic_write_new call with the same message and hint the preflight
    uses. The two sites must stay synchronized — this test pins both.
    """
    from uv_stack.fsutil import atomic_write_new

    # Capture the preflight refusal hint first, without monkeypatching.
    python_path = config_tree.env_python_path("preflight")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("3.12\n")
    with pytest.raises(ConfigError) as preflight_exc:
        write_env_sources(config_tree, "preflight", ["ds"], python="3.13")
    preflight_hint = preflight_exc.value.hint

    # Now test the publish-site refusal — the preflight passes (no python.txt
    # yet), but the publish raises FileExistsError.
    def racing_write_new(path, text):
        if path.name == "python.txt":
            # A concurrent writer creates python.txt before we can.
            raise FileExistsError("Python file exists")
        return atomic_write_new(path, text)

    monkeypatch.setattr("uv_stack.operations.scaffold.atomic_write_new", racing_write_new)

    with pytest.raises(ConfigError) as publish_exc:
        write_env_sources(config_tree, "race-case", ["ds"], python="3.13")

    # The publish-site refusal must match the preflight refusal in both the
    # message pattern and the hint — the hint is the part that drifted.
    assert "already has a python.txt" in publish_exc.value.message
    assert publish_exc.value.hint == preflight_hint


def test_write_env_sources_reports_failed_python_withdrawal(
    config_tree: ConfigRoot, monkeypatch
):
    """When stack.txt publish fails and python.txt withdrawal fails, state the residual."""
    from pathlib import Path

    from uv_stack.fsutil import atomic_write_new

    python_path = config_tree.env_python_path("withdrawal-failure")
    stack_path = config_tree.env_stack_path("withdrawal-failure")
    original_unlink = Path.unlink

    def racing_write_new(path, text):
        # python.txt write succeeds.
        if path.name == "python.txt":
            return atomic_write_new(path, text)
        # A concurrent writer creates stack.txt before we can.
        stack_path.parent.mkdir(parents=True, exist_ok=True)
        stack_path.write_text("concurrent\n", encoding="utf-8")
        raise FileExistsError("Stack file exists")

    def failing_unlink(self, missing_ok=False):
        if self == python_path:
            raise PermissionError("Simulated permission error")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr("uv_stack.operations.scaffold.atomic_write_new", racing_write_new)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(ConfigError) as excinfo:
        write_env_sources(config_tree, "withdrawal-failure", ["ds"], python="3.13")

    # The message should NOT contain the ".;" defect.
    assert ".;" not in str(excinfo.value)
    assert "was just written and could not be removed" in str(excinfo.value)
    assert str(python_path) in str(excinfo.value)
    assert str(python_path) in str(excinfo.value.hint)
    # The python.txt is left on disk.
    assert python_path.exists()
    # The concurrent writer's stack.txt survived untouched.
    assert stack_path.read_text(encoding="utf-8") == "concurrent\n"


def test_write_env_sources_spares_adopted_python_from_withdrawal(
    config_tree: ConfigRoot, monkeypatch
):
    """An adopted python.txt survives when stack.txt publish fails.

    write_env_sources short-circuits withdrawal when published_python is
    None, which is the adopted case. A regression would delete the
    python.txt this call never wrote.
    """
    from pathlib import Path

    from uv_stack.fsutil import atomic_write_new

    python_path = config_tree.env_python_path("adopted-orphan")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("3.13\n")

    unlinked_paths: list[Path] = []
    original_unlink = Path.unlink

    def failing_write_new(path, text):
        if path.name == "stack.txt":
            raise RuntimeError("Simulated stack.txt publish failure")
        return atomic_write_new(path, text)

    def tracking_unlink(self, missing_ok=False):
        unlinked_paths.append(self)
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr("uv_stack.operations.scaffold.atomic_write_new", failing_write_new)
    monkeypatch.setattr(Path, "unlink", tracking_unlink)

    with pytest.raises(RuntimeError):
        write_env_sources(config_tree, "adopted-orphan", ["ds"], python="3.13")

    # The adopted python.txt was never unlinked and still exists.
    assert python_path not in unlinked_paths
    assert python_path.exists()
    assert python_path.read_text() == "3.13\n"


def test_write_env_sources_inherits_orphan_python_with_no_python_arg(
    config_tree: ConfigRoot,
):
    """A retry with no --python inherits the orphan python.txt with no comparison.

    The orphan's content deliberately would NOT match any request, proving
    the call skips the adoption preflight entirely when python= is omitted.
    """
    python_path = config_tree.env_python_path("no-python-arg")
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("3.13-orphaned-pin\n")

    written = write_env_sources(config_tree, "no-python-arg", ["ds"])

    assert written == [config_tree.env_stack_path("no-python-arg")]
    assert config_tree.env_stack_path("no-python-arg").exists()
    assert python_path.read_text() == "3.13-orphaned-pin\n"


@pytest.mark.parametrize("bad", ["pkg:x", "@x", "a b", "-x", "a\tb"])
def test_validate_name_rejects_token_shaped_names(config_tree: ConfigRoot, bad):
    with pytest.raises(ConfigError):
        write_profile(config_tree, bad, ["numpy"])
    with pytest.raises(ConfigError):
        write_bundle(config_tree, bad, ["ds"])
    with pytest.raises(ConfigError):
        write_env_sources(config_tree, bad, ["ds"])


def test_write_profile_withdraws_when_bundle_appears_during_publish(
    config_tree: ConfigRoot, monkeypatch
):
    """A bundle created in the pre-check/publish window is caught after the fact.

    `bundle_exists` is made to answer False before our file lands and True
    afterwards, which is exactly what a concurrent `stack create bundle`
    produces. The published profile must be withdrawn, not left shadowing.
    """
    path = config_tree.profile_path("racy")
    bundle_path = config_tree.bundle_path("racy")

    def racing_bundle_exists(name: str) -> bool:
        if name == "racy" and path.exists() and not bundle_path.exists():
            bundle_path.write_text("includes: [racing-bundle]\n")
        return name == "racy" and bundle_path.exists()

    monkeypatch.setattr(config_tree, "bundle_exists", racing_bundle_exists)

    with pytest.raises(ConfigError) as excinfo:
        write_profile(config_tree, "racy", ["numpy"])
    assert "would shadow the existing bundle" in str(excinfo.value)
    assert excinfo.value.hint == _SHADOW_HINT
    assert not path.exists()
    assert bundle_path.exists()
    assert bundle_path.read_text() == "includes: [racing-bundle]\n"


def test_write_profile_withdrawal_spares_a_concurrent_replacement(
    config_tree: ConfigRoot, monkeypatch
):
    """Withdrawal is inode-matched: a third writer's file is never unlinked."""
    import os
    path = config_tree.profile_path("racy")

    def racing_bundle_exists(name: str) -> bool:
        if name != "racy" or not path.exists():
            return False
        # Simulate the third writer replacing our file before we withdraw.
        temp = path.with_suffix(".replacement")
        temp.write_text("includes: [someone-else]\n")
        os.replace(temp, path)
        return True

    monkeypatch.setattr(config_tree, "bundle_exists", racing_bundle_exists)

    with pytest.raises(ConfigError) as excinfo:
        write_profile(config_tree, "racy", ["numpy"])
    assert excinfo.value.hint == _SHADOW_HINT
    assert path.read_text() == "includes: [someone-else]\n"


def test_write_bundle_withdraws_when_profile_appears_during_publish(
    config_tree: ConfigRoot, monkeypatch
):
    """A profile created in the pre-check/publish window is caught after the fact."""
    path = config_tree.bundle_path("racy")
    profile_path = config_tree.profile_path("racy")

    def racing_profile_exists(name: str) -> bool:
        if name == "racy" and path.exists() and not profile_path.exists():
            profile_path.write_text("includes: [racing-profile]\n")
        return name == "racy" and profile_path.exists()

    monkeypatch.setattr(config_tree, "profile_exists", racing_profile_exists)

    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "racy", ["ds"])
    assert "would be shadowed by the existing profile" in str(excinfo.value)
    assert excinfo.value.hint == _SHADOW_HINT
    assert not path.exists()
    assert profile_path.exists()
    assert profile_path.read_text() == "includes: [racing-profile]\n"


def test_write_starter_profile_withdraws_when_bundle_appears_during_publish(
    config_tree: ConfigRoot, monkeypatch
):
    """A bundle named 'starter' created mid-publish is caught after the fact."""
    from uv_stack.operations.scaffold import write_starter_profile

    path = config_tree.profile_path("starter")
    bundle_path = config_tree.bundle_path("starter")

    def racing_bundle_exists(name: str) -> bool:
        if name == "starter" and path.exists() and not bundle_path.exists():
            bundle_path.write_text("includes: [racing-bundle]\n")
        return name == "starter" and bundle_path.exists()

    monkeypatch.setattr(config_tree, "bundle_exists", racing_bundle_exists)

    with pytest.raises(ConfigError) as excinfo:
        write_starter_profile(config_tree)
    assert "would shadow the existing bundle" in str(excinfo.value)
    assert excinfo.value.hint == _SHADOW_HINT
    assert not path.exists()
    assert bundle_path.exists()
    assert bundle_path.read_text() == "includes: [racing-bundle]\n"


def test_write_profile_reports_failed_withdrawal(
    config_tree: ConfigRoot, monkeypatch
):
    """When withdrawal fails, the error message states so and names the path."""
    from pathlib import Path

    path = config_tree.profile_path("racy")
    bundle_path = config_tree.bundle_path("racy")
    original_unlink = Path.unlink

    def racing_bundle_exists(name: str) -> bool:
        if name == "racy" and path.exists() and not bundle_path.exists():
            bundle_path.write_text("includes: [racing-bundle]\n")
        return name == "racy" and bundle_path.exists()

    def failing_unlink(self, missing_ok=False):
        if self == path:
            raise PermissionError("Simulated permission error")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(config_tree, "bundle_exists", racing_bundle_exists)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(ConfigError) as excinfo:
        write_profile(config_tree, "racy", ["numpy"])
    assert "would shadow the existing bundle" in str(excinfo.value)
    assert "was just written and could not be removed" in str(excinfo.value)
    assert str(path) in str(excinfo.value)
    assert str(path) in str(excinfo.value.hint)
    assert "Delete" in excinfo.value.hint
    assert path.exists()
    assert bundle_path.exists()


def test_write_bundle_probe_failure_after_publish(
    config_tree: ConfigRoot, monkeypatch
):
    """Post-publish probe raising is treated as a collision and triggers withdrawal.

    The probe is the only thing between a successful write and an undetected
    cross-kind collision. If it cannot answer, the safe behavior is to withdraw
    the file and raise, not to trust the write.
    """
    path = config_tree.bundle_path("probe-failure")

    def probe_raises(name: str) -> bool:
        if name == "probe-failure":
            if path.exists():
                raise PermissionError("Simulated config dir probe failure")
        return False

    monkeypatch.setattr(config_tree, "profile_exists", probe_raises)

    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "probe-failure", ["ds"])
    assert "Could not check for a conflicting file after writing" in str(excinfo.value)
    assert str(path) in str(excinfo.value)
    assert "config directory is readable" in excinfo.value.hint
    assert not path.exists()


def test_write_bundle_probe_failure_with_failed_withdrawal(
    config_tree: ConfigRoot, monkeypatch
):
    """When both probe and withdrawal fail, the error names the residual."""
    from pathlib import Path

    path = config_tree.bundle_path("probe-failure-residual")
    original_unlink = Path.unlink

    def probe_raises(name: str) -> bool:
        if name == "probe-failure-residual":
            if path.exists():
                raise PermissionError("Simulated config dir probe failure")
        return False

    def failing_unlink(self, missing_ok=False):
        if self == path:
            raise PermissionError("Simulated permission error")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(config_tree, "profile_exists", probe_raises)
    monkeypatch.setattr(Path, "unlink", failing_unlink)

    with pytest.raises(ConfigError) as excinfo:
        write_bundle(config_tree, "probe-failure-residual", ["ds"])
    assert "Could not check for a conflicting file after writing" in str(excinfo.value)
    assert "was just written and could not be removed" in str(excinfo.value)
    assert str(path) in str(excinfo.value)
    assert str(path) in str(excinfo.value.hint)
    assert "Delete" in excinfo.value.hint
    assert path.exists()


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_write_profile_waits_for_another_process_holding_the_stem_lock(
    tmp_path, monkeypatch
):
    """write_profile is serialized against another process on the same stem.

    Shortening the module default rather than passing a timeout is deliberate:
    it proves write_profile reaches the lock at all, which a parameter the test
    supplies itself could not.
    """
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    config = ConfigRoot(tmp_path)

    with _lock_held_by_another_process(config.stem_lock_path("x")):
        with pytest.raises(ConfigError) as excinfo:
            write_profile(config, "x", ["rich"])

    assert "another stack process" in str(excinfo.value)
    # Nothing was published while the competitor held the name.
    assert not config.profile_path("x").exists()


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_write_bundle_contends_on_the_same_stem_lock_as_write_profile(
    tmp_path, monkeypatch
):
    """The two kinds share one lock file — that is what makes them serialize.

    A bundle taking a bundle-specific lock would pass a same-kind test and
    still leave the cross-kind collision this task exists to close wide open.
    """
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    config = ConfigRoot(tmp_path)

    with _lock_held_by_another_process(config.stem_lock_path("x")):
        with pytest.raises(ConfigError):
            write_bundle(config, "x", ["rich"])

    assert not config.bundle_path("x").exists()


@pytest.mark.skipif(not _LOCK_AVAILABLE, reason="requires fcntl")
def test_write_env_sources_waits_for_another_process_on_the_same_env(
    tmp_path, monkeypatch
):
    """The adopter surface is serialized too, not just the profile/bundle one."""
    monkeypatch.setattr("uv_stack.fsutil._LOCK_TIMEOUT", 0.2)
    config = ConfigRoot(tmp_path)

    with _lock_held_by_another_process(config.env_lock_path("myenv")):
        with pytest.raises(ConfigError) as excinfo:
            write_env_sources(config, "myenv", ["rich"], python="3.12")

    assert "another stack process" in str(excinfo.value)
    assert not config.env_stack_path("myenv").exists()
    assert not config.env_python_path("myenv").exists()


def test_writers_still_work_when_locking_is_unavailable(tmp_path, monkeypatch):
    """Degrading to a no-op must not change the ordinary success path."""
    monkeypatch.setattr("uv_stack.fsutil._LOCK_AVAILABLE", False)
    config = ConfigRoot(tmp_path)

    assert write_profile(config, "x", ["rich"]).is_file()
    assert write_bundle(config, "y", ["x"]).is_file()
    assert all(p.is_file() for p in write_env_sources(config, "e", ["x"], python="3.12"))
    assert not config.locks_dir.exists()
