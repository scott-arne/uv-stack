"""``stack create``: create a new environment or project."""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from uv_stack.cli._render import console, echo, print_activation_hint, render_warnings
from uv_stack.cli.upgrade import _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.operations.project import ProjectOptions, init_project
from uv_stack.operations.scaffold import write_bundle, write_env_sources, write_profile
from uv_stack.operations.upgrade import UpgradeOptions
from uv_stack.resolver import Resolver
from uv_stack.runner import SubprocessRunner


@click.group("create")
def create() -> None:
    """Create a new environment or project."""


@create.command("env")
@click.argument("name")
@click.argument("tokens", nargs=-1)
@click.option(
    "--python",
    "python",
    default=None,
    help="Write python.txt with this version (requires TOKENS).",
)
@click.option("--recreate", is_flag=True, help="Remove and recreate the env first.")
@click.option(
    "--strict",
    is_flag=True,
    help="Fail if an unqualified token falls through to a literal package.",
)
@click.pass_obj
def create_env(
    config: ConfigRoot,
    name: str,
    tokens: tuple[str, ...],
    python: str | None,
    recreate: bool,
    strict: bool,
) -> None:
    """Create environment NAME, then upgrade it ('--recreate' wipes it first).

    With TOKENS, scaffold envs/NAME/stack.txt first (and python.txt when
    '--python' is given).
    """
    if python is not None and not tokens:
        raise click.UsageError(
            "--python requires TOKENS (it only applies when scaffolding a new env)."
        )
    if python is not None and not python.strip():
        raise click.UsageError("--python requires a non-empty version.")
    if tokens:
        # Validate before anything durable is written: strict failures and
        # missing explicit references must not leave a half-created env.
        Resolver(config, strict=strict).resolve(list(tokens))
        for path in write_env_sources(config, name, list(tokens), python=python):
            echo(f"Wrote {path}")
    options = (
        UpgradeOptions(recreate=True, strict=strict)
        if recreate
        else UpgradeOptions(create=True, strict=strict)
    )
    _run_upgrade(config, [name], options)
    echo("")
    print_activation_hint(name)


@create.command("project")
@click.argument("tokens", nargs=-1, required=True)
@click.option(
    "--python",
    "python",
    default=None,
    help=(
        "Python version or micromamba env name for the project interpreter. "
        "Defaults to $UV_STACK_PROJECT_PYTHON, the config-root default, then 3.12."
    ),
)
@click.option("--name", "name", default=None, help="Project name for uv init.")
@click.option("--no-sync", is_flag=True, help="Add dependencies but do not sync.")
@click.option("--force", is_flag=True, help="Add to an existing pyproject.toml.")
@click.option(
    "--strict",
    is_flag=True,
    help="Fail if an unqualified token falls through to a literal package.",
)
@click.pass_obj
def create_project(
    config: ConfigRoot,
    tokens: tuple[str, ...],
    python: str | None,
    name: str | None,
    no_sync: bool,
    force: bool,
    strict: bool,
) -> None:
    """Create a uv project from the resolved stack TOKENS."""
    options = ProjectOptions(python=python, name=name, no_sync=no_sync, force=force, strict=strict)
    warnings = init_project(
        config, SubprocessRunner(), list(tokens), options, cwd=Path.cwd()
    )
    render_warnings(warnings)
    console.print("[green]Project initialized.[/green]")


@create.command("profile")
@click.argument("name")
@click.argument("packages", nargs=-1, required=True)
@click.option("--description", default=None, help="One-line description stored in the YAML.")
@click.option("--tag", "tags", multiple=True, help="Tag stored in the YAML (repeatable).")
@click.pass_obj
def create_profile(
    config: ConfigRoot,
    name: str,
    packages: tuple[str, ...],
    description: str | None,
    tags: tuple[str, ...],
) -> None:
    """Create profiles/NAME.yaml from PACKAGES."""
    path = write_profile(
        config, name, list(packages), description=description, tags=list(tags)
    )
    echo(f"Wrote {path}")


@create.command("bundle")
@click.argument("name")
@click.argument("tokens", nargs=-1, required=True)
@click.option("--description", default=None, help="One-line description stored in the YAML.")
@click.option("--tag", "tags", multiple=True, help="Tag stored in the YAML (repeatable).")
@click.option(
    "--strict",
    is_flag=True,
    help="Fail if an unqualified token falls through to a literal package.",
)
@click.pass_obj
def create_bundle(
    config: ConfigRoot,
    name: str,
    tokens: tuple[str, ...],
    description: str | None,
    tags: tuple[str, ...],
    strict: bool,
) -> None:
    """Create bundles/NAME.yaml from stack TOKENS."""
    # Full resolution (not classify) so explicit profile:/bundle: references
    # must exist before anything durable is written.
    stack = Resolver(config, strict=strict).resolve(list(tokens))
    render_warnings(stack.warnings)
    path = write_bundle(
        config, name, list(tokens), description=description, tags=list(tags)
    )
    echo(f"Wrote {path}")
