from __future__ import annotations

from pathlib import Path

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


def test_micromamba_create():
    assert micromamba_create(Path("env.yml")).args == [
        "micromamba", "create", "-f", "env.yml", "-y",
    ]


def test_micromamba_remove():
    assert micromamba_remove("main").args == [
        "micromamba", "remove", "-n", "main", "--all", "-y",
    ]


def test_micromamba_python_path():
    cmd = micromamba_python_path("main")
    assert cmd.args[:4] == ["micromamba", "run", "-n", "main"]
    assert "python" in cmd.args


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
