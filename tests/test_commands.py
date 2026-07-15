from __future__ import annotations

from pathlib import Path

from uv_stack import commands
from uv_stack.commands import (
    micromamba_create,
    micromamba_python_path,
    micromamba_remove,
    uv_add,
    uv_init,
    uv_pip_check,
    uv_pip_compile,
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
