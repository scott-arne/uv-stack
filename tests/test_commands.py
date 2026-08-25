from __future__ import annotations

from pathlib import Path

from uv_stack import commands
from uv_stack.commands import (
    micromamba_create,
    micromamba_python_info,
    micromamba_python_path,
    micromamba_remove,
    uv_add,
    uv_init,
    uv_pip_check,
    uv_pip_compile,
    uv_pip_compile_for_version,
    uv_pip_sync,
    uv_sync,
)


def test_uv_pip_compile_basic():
    cmd = uv_pip_compile("/py", Path("requirements.in"), Path("out.lock"))
    assert cmd.args == [
        "uv", "pip", "compile", "--python", "/py",
        "requirements.in", "-o", "out.lock",
    ]


def test_uv_pip_compile_upgrade_all():
    cmd = uv_pip_compile("/py", Path("requirements.in"), Path("out.lock"), upgrade=True)
    assert "--upgrade" in cmd.args


def test_uv_pip_compile_upgrade_packages():
    cmd = uv_pip_compile(
        "/py", Path("r.in"), Path("o"), upgrade_packages=["pandas", "numpy"]
    )
    assert cmd.args.count("--upgrade-package") == 2
    assert "pandas" in cmd.args
    assert "numpy" in cmd.args
    assert "--upgrade" not in cmd.args


def test_uv_pip_compile_for_version_basic():
    cmd = uv_pip_compile_for_version("3.13", Path("requirements.in"), Path("out.lock"))
    assert cmd.args == [
        "uv", "pip", "compile", "--python-version", "3.13",
        "requirements.in", "-o", "out.lock",
    ]


def test_uv_pip_compile_for_version_upgrade_flags():
    cmd = uv_pip_compile_for_version(
        "3.13", Path("r.in"), Path("o"), upgrade=True, upgrade_packages=["pandas"]
    )
    assert "--upgrade" in cmd.args
    assert cmd.args.count("--upgrade-package") == 1
    assert "pandas" in cmd.args


def test_compile_builders_differ_only_in_the_interpreter_selector():
    # The path form's argv is pinned by existing callers and tests; the version
    # form must be the same command with a different selector, nothing else.
    by_path = uv_pip_compile("/py", Path("r.in"), Path("o"), upgrade=True)
    by_version = uv_pip_compile_for_version("3.13", Path("r.in"), Path("o"), upgrade=True)
    assert by_path.args[:3] == by_version.args[:3] == ["uv", "pip", "compile"]
    assert by_path.args[3:5] == ["--python", "/py"]
    assert by_version.args[3:5] == ["--python-version", "3.13"]
    assert "--python-version" not in by_path.args
    assert "--python" not in by_version.args
    assert by_path.args[5:] == by_version.args[5:]


def test_uv_pip_sync():
    cmd = uv_pip_sync("/py", Path("lock.txt"))
    assert cmd.args == [
        "uv", "pip", "sync", "--python", "/py",
        "-C", "editable_mode=compat", "lock.txt",
    ]


def test_uv_pip_sync_editable_mode_override():
    cmd = uv_pip_sync("/py", Path("lock.txt"), editable_mode="strict")
    assert "editable_mode=strict" in cmd.args


def test_uv_pip_check():
    assert uv_pip_check("/py").args == ["uv", "pip", "check", "--python", "/py"]


def test_micromamba_create_uses_mamba_exe(monkeypatch):
    # micromamba shell init installs micromamba as a shell function and exports
    # the real binary path as MAMBA_EXE; the builder must exec that path, not the
    # bare name (which subprocess cannot resolve to a shell function).
    monkeypatch.setenv("MAMBA_EXE", "/opt/mm/bin/micromamba")
    assert micromamba_create(Path("env.yml")).args[0] == "/opt/mm/bin/micromamba"


def test_micromamba_create(monkeypatch):
    monkeypatch.setenv("MAMBA_EXE", "micromamba")
    assert micromamba_create(Path("env.yml")).args == [
        "micromamba", "create", "-f", "env.yml", "-y",
    ]


def test_micromamba_remove(monkeypatch):
    monkeypatch.setenv("MAMBA_EXE", "micromamba")
    assert micromamba_remove("main").args == [
        "micromamba", "remove", "-n", "main", "--all", "-y",
    ]


def test_micromamba_python_path(monkeypatch):
    monkeypatch.setenv("MAMBA_EXE", "micromamba")
    cmd = micromamba_python_path("main")
    assert cmd.args[:4] == ["micromamba", "run", "-n", "main"]
    assert "python" in cmd.args


def test_micromamba_python_info(monkeypatch):
    import subprocess
    import sys

    monkeypatch.setenv("MAMBA_EXE", "micromamba")
    cmd = micromamba_python_info("main")
    assert cmd.args[:4] == ["micromamba", "run", "-n", "main"]
    assert "python" in cmd.args
    # Verify the snippet prints executable then version.
    snippet = [arg for arg in cmd.args if "import sys" in arg][0]
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2
    assert lines[0] == sys.executable
    # Verify the version line is three dot-separated integers.
    version_parts = lines[1].split(".")
    assert len(version_parts) == 3
    for part in version_parts:
        assert part.isdigit()


def test_micromamba_exe_prefers_mamba_exe_over_path(monkeypatch):
    monkeypatch.setenv("MAMBA_EXE", "/opt/mm/bin/micromamba")
    monkeypatch.setattr(commands.shutil, "which", lambda _: "/usr/bin/micromamba")
    assert commands._micromamba_exe() == "/opt/mm/bin/micromamba"


def test_micromamba_exe_falls_back_to_path(monkeypatch):
    monkeypatch.delenv("MAMBA_EXE", raising=False)
    monkeypatch.setattr(commands.shutil, "which", lambda _: "/usr/bin/micromamba")
    assert commands._micromamba_exe() == "/usr/bin/micromamba"


def test_micromamba_exe_falls_back_to_bare_name(monkeypatch):
    monkeypatch.delenv("MAMBA_EXE", raising=False)
    monkeypatch.setattr(commands.shutil, "which", lambda _: None)
    assert commands._micromamba_exe() == "micromamba"


def test_uv_init_with_name_and_python():
    cmd = uv_init("3.11", name="proj")
    assert cmd.args == ["uv", "init", "--bare", "--name", "proj", "--python", "3.11"]


def test_uv_init_without_name():
    cmd = uv_init("3.12")
    assert cmd.args == ["uv", "init", "--bare", "--python", "3.12"]


def test_uv_add_and_sync():
    assert uv_add(Path("r.txt")).args == ["uv", "add", "--no-sync", "-r", "r.txt"]
    assert uv_sync().args == ["uv", "sync"]


def test_uv_sync_with_python():
    assert uv_sync("/envs/main/bin/python").args == [
        "uv", "sync", "--python", "/envs/main/bin/python",
    ]


def test_uv_remove_always_no_sync():
    from uv_stack.commands import uv_remove

    cmd = uv_remove(["numpy", "pandas"])
    assert cmd.args == ["uv", "remove", "--no-sync", "numpy", "pandas"]
