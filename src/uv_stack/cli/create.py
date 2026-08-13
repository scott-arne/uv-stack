"""``stack create``: create a new environment or project."""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from uv_stack.cli._render import console, echo, print_activation_hint, render_warnings
from uv_stack.cli.upgrade import _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.errors import UvStackError
from uv_stack.operations.project import ProjectOptions, init_project
from uv_stack.operations.scaffold import write_bundle, write_env_sources, write_profile
from uv_stack.operations.upgrade import UpgradeOptions
from uv_stack.parse import read_clean_lines
from uv_stack.resolver import Resolver
from uv_stack.runner import SubprocessRunner


def _bare_usage_warnings(
    config: ConfigRoot, name: str, kind: str, *, exclude_bundle: str | None = None
) -> list[str]:
    """Warn where ``name`` is already used as a bare token.

    A new profile/bundle takes precedence over the literal package a bare
    token used to resolve to; these warnings point at each usage without
    blocking creation (refusal was deliberately rejected in review).
    """
    warnings: list[str] = []
    template = (
        "'{name}' is used as a bare token in {location}; it now resolves to "
        "this {kind} (use pkg:{name} there for the literal package)"
    )
    for env in config.list_envs():
        if name in read_clean_lines(config.env_stack_path(env)):
            warnings.append(
                template.format(name=name, location=f"envs/{env}/stack.txt", kind=kind)
            )
    for bundle_name in config.list_bundles():
        if bundle_name == exclude_bundle:
            continue
        try:
            includes = config.load_bundle(bundle_name).includes
        except UvStackError:
            continue  # malformed bundles are doctor's job, not create's
        if name in includes:
            warnings.append(
                template.format(
                    name=name, location=f"bundles/{bundle_name}.yaml", kind=kind
                )
            )
    return warnings


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
        # Validate before anything durable is written: strict failures,
        # missing explicit references, and malformed profile YAML must not
        # leave a half-created env. flatten() loads every referenced profile.
        resolver = Resolver(config, strict=strict)
        resolver.flatten(resolver.resolve(list(tokens)))
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
    "--no-track",
    "no_track",
    is_flag=True,
    help="Do not record [tool.uv-stack] tracking metadata.",
)
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
    no_track: bool,
    strict: bool,
) -> None:
    """Create a uv project from the resolved stack TOKENS."""
    options = ProjectOptions(
        python=python,
        name=name,
        no_sync=no_sync,
        force=force,
        track=not no_track,
        strict=strict,
    )
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
    render_warnings(_bare_usage_warnings(config, name, "profile"))


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
    # Strict and near-miss rules apply to the DIRECT tokens only...
    direct = Resolver(config, strict=strict).classify(list(tokens))
    render_warnings(direct.warnings)
    # ...while existence of explicit references (recursively) is validated
    # without re-judging existing bundles' own contents.
    Resolver(config).resolve(list(tokens))
    path = write_bundle(
        config, name, list(tokens), description=description, tags=list(tags)
    )
    echo(f"Wrote {path}")
    render_warnings(_bare_usage_warnings(config, name, "bundle", exclude_bundle=name))
