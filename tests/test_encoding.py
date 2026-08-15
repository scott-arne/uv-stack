"""uv-stack's own files are UTF-8 regardless of the process locale.

Every reader and writer in the package pins ``encoding="utf-8"``. Without
that pin the interpreter falls back to the locale's preferred encoding, so a
profile description, a ``stack.txt`` comment, or a requirement URL containing
a single non-ASCII byte crashes with an unhandled ``UnicodeDecodeError`` on a
machine running under ``LC_ALL=C`` — on a file uv-stack itself wrote as UTF-8.

The locale is fixed at interpreter start-up and cannot be changed from inside
a running process, so this has to run in a subprocess. It launches the same
interpreter pytest is running under (nothing external, no network), points it
at a tree the test builds, and refuses to pass unless the ASCII locale
actually took effect — otherwise the whole file would be silently vacuous.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# A non-ASCII character that is one byte in latin-1 and two in UTF-8, so an
# ASCII decoder rejects it outright rather than mojibake-ing it.
ACCENT = "é"  # é

_DRIVER = '''
"""Exercise every uv-stack text reader and writer under the ambient locale."""
import locale
import sys
from pathlib import Path

root = Path(sys.argv[1])
project = Path(sys.argv[2])
fresh = Path(sys.argv[3])

# Guard against a vacuous run: if the locale override did not take, the
# readers below would pass for the wrong reason.
preferred = locale.getencoding()
if "utf" in preferred.lower().replace("-", "") or sys.flags.utf8_mode:
    print("VACUOUS: locale override did not take (%s, utf8_mode=%d)"
          % (preferred, sys.flags.utf8_mode))
    raise SystemExit(2)

from uv_stack.config import ConfigRoot
from uv_stack.operations.project import (
    ProjectOptions,
    RefreshOptions,
    init_project,
    refresh_project,
)
from uv_stack.operations.pyproject import (
    read_project_dependency_names,
    read_tracking,
)
from uv_stack.operations.status import _read_text_or_none
from uv_stack.parse import read_clean_lines
from uv_stack.runner import Command, CommandResult, RecordingRunner

ACCENT = "\\u00e9"
config = ConfigRoot(root)

# config.py — YAML profile body.
assert config.load_profile("uni").description == "caf" + ACCENT + " chemistry"

# parse.py — env stack file (the comment is stripped, but must decode first).
assert read_clean_lines(config.env_dir("main") / "stack.txt") == ["uni"]

# status.py — the tolerant reader used for staleness comparison.
local = _read_text_or_none(config.env_dir("main") / "requirements.local.in")
assert local is not None and ACCENT in local, repr(local)

# pyproject.py — the tracking table and the dependency-name scan.
tracking = read_tracking(project / "pyproject.toml")
assert tracking is not None and tracking.stack == ["uni"]
assert read_project_dependency_names(project / "pyproject.toml") == {"numpy"}

# project.py — the temp requirements files handed to `uv add`, one per entry
# point. Both carry the profile's non-ASCII direct reference.
written = []


def capture(cmd: Command) -> CommandResult:
    for arg in cmd.args:
        if arg.endswith(".txt") and Path(arg).is_file():
            written.append(Path(arg).read_bytes())
    return CommandResult(returncode=0, stdout="")


refresh_project(
    config, RecordingRunner(responder=capture),
    RefreshOptions(python="3.12", no_sync=True), cwd=project,
)
init_project(
    config, RecordingRunner(responder=capture), ["uni"],
    ProjectOptions(python="3.12", no_sync=True, track=False), cwd=fresh,
)
assert len(written) == 2, len(written)
for blob in written:
    assert ACCENT in blob.decode("utf-8"), repr(blob)

print("OK")
'''


def _ascii_locale_env() -> dict[str, str]:
    """The current environment forced to an ASCII locale."""
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    for name in ("UV_STACK_ROOT", "UV_ENV_ROOT", "UV_STACK_PROJECT_PYTHON"):
        env.pop(name, None)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["PYTHONUTF8"] = "0"
    return env


def test_utf8_files_readable_and_writable_under_ascii_locale(tmp_path: Path):
    root = tmp_path / "python-envs"
    (root / "profiles").mkdir(parents=True)
    (root / "envs" / "main").mkdir(parents=True)
    # A direct reference whose URL carries the non-ASCII byte: it survives the
    # ownership filter by name and lands in the temp requirements file.
    (root / "profiles" / "uni.yaml").write_text(
        f"description: caf{ACCENT} chemistry\n"
        f"includes:\n  - numpy\n  - pkg @ https://h/caf{ACCENT}.whl\n",
        encoding="utf-8",
    )
    env_dir = root / "envs" / "main"
    env_dir.joinpath("python.txt").write_text("3.12\n", encoding="utf-8")
    env_dir.joinpath("stack.txt").write_text(
        f"# le caf{ACCENT} stack\nuni\n", encoding="utf-8"
    )
    env_dir.joinpath("requirements.local.in").write_text(
        f"# caf{ACCENT}\nsome-local-pkg\n", encoding="utf-8"
    )

    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "caf{ACCENT}"\nversion = "0.1.0"\n'
        'dependencies = ["numpy"]\n'
        '\n[tool.uv-stack]\nversion = 1\nstack = ["uni"]\napplied = ["numpy"]\n',
        encoding="utf-8",
    )
    fresh = tmp_path / "fresh"
    fresh.mkdir()

    result = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(root), str(project), str(fresh)],
        env=_ascii_locale_env(),
        capture_output=True,
        text=True,
        errors="backslashreplace",
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert result.stdout.strip().endswith("OK")
