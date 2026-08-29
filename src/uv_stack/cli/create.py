"""``stack create``: create a shared environment, project, profile, or bundle."""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from uv_stack.cli._render import console, echo, print_activation_hint, render_warnings
from uv_stack.cli.upgrade import _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.errors import UvStackError
from uv_stack.operations.project import ProjectOptions, init_project
from uv_stack.operations.scaffold import (
    write_bundle,
    write_env_python,
    write_env_sources,
    write_profile,
)
from uv_stack.operations.upgrade import UpgradeOptions
from uv_stack.parse import read_clean_lines
from uv_stack.pyversion import is_comparable
from uv_stack.resolver import Resolver
from uv_stack.runner import SubprocessRunner


def _require_plain_python(python: str) -> None:
    """Refuse a ``--python`` value a recreate cannot resolve the lock against.

    A recreate resolves the lock against this value before rebuilding the
    environment, and the operations layer refuses one uv cannot resolve
    against. Refusing here too keeps a doomed run from leaving stack.txt and
    python.txt written.

    :param python: The stripped ``--python`` value.
    :raises click.UsageError: If the value is not a plain dotted version.
    """
    if not is_comparable(python):
        raise click.UsageError(
            f"--python must be a plain version such as 3.14, not '{python}'."
        )


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

    def _matches(token: str) -> bool:
        """Check if token matches the created name, accounting for case-insensitive filesystems."""
        if token == name:
            return True
        if token.casefold() == name.casefold():
            try:
                if kind == "profile":
                    return config.profile_path(token).is_file()
                else:  # kind == "bundle"
                    return config.bundle_path(token).is_file()
            except OSError:
                return False
        return False

    try:
        envs = config.list_envs()
    except OSError:
        envs = []
    for env in envs:
        try:
            tokens = read_clean_lines(config.env_stack_path(env))
            if any(_matches(token) for token in tokens):
                warnings.append(
                    template.format(name=name, location=f"envs/{env}/stack.txt", kind=kind)
                )
        except (UvStackError, OSError):
            continue  # unreadable envs are doctor's job, not create's
    try:
        bundles = config.list_bundles()
    except OSError:
        bundles = []
    for bundle_name in bundles:
        if bundle_name == exclude_bundle:
            continue
        try:
            includes = config.load_bundle(bundle_name).includes
        except (UvStackError, OSError, UnicodeDecodeError):
            continue  # malformed/unreadable bundles are doctor's job, not create's
        if any(_matches(token.strip()) for token in includes):
            warnings.append(
                template.format(
                    name=name, location=f"bundles/{bundle_name}.yaml", kind=kind
                )
            )
    return warnings


@click.group("create")
def create() -> None:
    """Create a shared environment, project, profile, or bundle."""


@create.command("env")
@click.argument("name")
@click.argument("tokens", nargs=-1)
@click.option(
    "--python",
    "python",
    default=None,
    help=(
        "Write python.txt with this version (with TOKENS: a new env; "
        "without: an existing env, requires --recreate)."
    ),
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
    '--python' is given). Without TOKENS, '--python VERSION --recreate' changes
    an existing environment's interpreter: the lock is compiled against VERSION
    before the environment is rebuilt.
    """
    if python is not None and not python.strip():
        raise click.UsageError("--python requires a non-empty version.")
    if python is not None:
        # Normalize before the is_comparable check below, so the CLI judges the
        # same string first_clean_line will hand back to the operations layer.
        # Judging the raw value would refuse ' 3.14 ', which reads back as the
        # plain version the recreate accepts.
        python = python.strip()
    if python is not None and not tokens:
        # Refusing up front is deliberate: the alternative writes python.txt
        # and then fails in upgrade_env, leaving the source edited and the env
        # untouched.
        stack_path = config.env_stack_path(name)
        if not stack_path.exists():
            raise click.UsageError(
                "--python requires TOKENS when creating a new environment."
            )
        if not recreate:
            raise click.UsageError(
                "Changing an existing environment's Python version requires "
                "--recreate; the interpreter is only rebuilt then."
            )
        _require_plain_python(python)
        path = write_env_python(config, name, python)
        echo(f"Wrote {path}")
        options = UpgradeOptions(recreate=True, strict=strict)
        _run_upgrade(config, [name], options)
        echo("")
        print_activation_hint(name)
        return
    if tokens:
        if python is not None and recreate:
            # Only a recreate resolves the lock against this value. Creating
            # without --recreate compiles against the built interpreter's
            # path instead, so a conda match spec is still legal there.
            _require_plain_python(python)
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
    help=r"Do not record \[tool.uv-stack] tracking metadata.",
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
    """Create a uv project in the current directory from the resolved stack TOKENS."""
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
    recursive = Resolver(config).resolve(list(tokens))
    # De-duplicate: direct.warnings are already rendered above.
    new_warnings = [w for w in recursive.warnings if w not in direct.warnings]
    render_warnings(new_warnings)
    path = write_bundle(
        config, name, list(tokens), description=description, tags=list(tags)
    )
    echo(f"Wrote {path}")
    render_warnings(_bare_usage_warnings(config, name, "bundle", exclude_bundle=name))
