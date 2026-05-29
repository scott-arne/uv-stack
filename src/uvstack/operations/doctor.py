"""Detect-only diagnostics for a uvstack config tree.

``diagnose`` never mutates the filesystem; it returns a list of findings the CLI
prints with suggested fixes. It flags missing directories, legacy names
(``*.profiles``, ``profiles.txt``), env-like directories left at the root, and
envs missing their source files.
"""

from __future__ import annotations

from dataclasses import dataclass

from uvstack.config import ConfigRoot

_KNOWN_TOP_LEVEL = {"profiles", "bundles", "envs", "lib"}


@dataclass
class Finding:
    """A single diagnostic result.

    :param level: ``"error"`` or ``"warn"``.
    :param message: What was detected.
    :param fix: Optional suggested remediation.
    """

    level: str
    message: str
    fix: str | None = None


def diagnose(config: ConfigRoot) -> list[Finding]:
    """Inspect the config tree and return findings.

    :param config: The configuration root to inspect.
    :returns: A list of :class:`Finding` (empty if everything looks correct).
    """
    findings: list[Finding] = []

    if not config.root.is_dir():
        findings.append(
            Finding(
                "error",
                f"Config root does not exist: {config.root}",
                fix=f"Run 'uvstack config init' or create {config.root}.",
            )
        )
        return findings

    for name, directory in (
        ("profiles", config.profiles_dir),
        ("bundles", config.bundles_dir),
        ("envs", config.envs_dir),
    ):
        if not directory.is_dir():
            findings.append(
                Finding(
                    "error",
                    f"Missing {name} directory: {directory}",
                    fix=f"Create {directory} (or run 'uvstack config init').",
                )
            )

    # Legacy bundle extension.
    if config.bundles_dir.is_dir():
        for legacy in config.bundles_dir.glob("*.profiles"):
            findings.append(
                Finding(
                    "warn",
                    f"Legacy bundle file: {legacy}",
                    fix=f"Rename to {legacy.with_suffix('.bundle')}.",
                )
            )

    # Env-like directories left directly under root.
    for child in config.root.iterdir():
        if not child.is_dir() or child.name in _KNOWN_TOP_LEVEL:
            continue
        if (child / "requirements.in").exists() or (child / "environment.yml").exists():
            findings.append(
                Finding(
                    "warn",
                    f"Env-like directory not under envs/: {child.name}",
                    fix=f"Move it: mv {child} {config.envs_dir / child.name}",
                )
            )

    # Per-env source-file checks.
    if config.envs_dir.is_dir():
        for env_dir in config.envs_dir.iterdir():
            if not env_dir.is_dir():
                continue
            if (env_dir / "profiles.txt").exists():
                findings.append(
                    Finding(
                        "warn",
                        f"Legacy profiles.txt in env '{env_dir.name}'",
                        fix=f"Rename {env_dir / 'profiles.txt'} to stack.txt.",
                    )
                )
            if (env_dir / "stack.txt").is_file() and not (
                env_dir / "python.txt"
            ).is_file():
                findings.append(
                    Finding(
                        "warn",
                        f"Env '{env_dir.name}' missing python.txt (will default to 3.12)",
                        fix=f"Create {env_dir / 'python.txt'}.",
                    )
                )

    return findings
