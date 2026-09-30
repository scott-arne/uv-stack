"""``stack export`` and ``stack import``: move definitions between machines."""

from __future__ import annotations

import sys
from pathlib import Path

import rich_click as click

from uv_stack.cli._render import echo, render_error, render_warnings
from uv_stack.cli.upgrade import _run_upgrade
from uv_stack.config import ConfigRoot
from uv_stack.fsutil import atomic_write
from uv_stack.hints import render_positional_arg
from uv_stack.models import ExportDocument
from uv_stack.operations.export import (
    AmbiguousItemError,
    ExportResult,
    build_document,
    serialize_document,
    source_platform,
)
from uv_stack.operations.importing import (
    STATUS_LABELS,
    BuildRequest,
    ConflictError,
    ImportOptions,
    ImportPlan,
    change_diff,
    load_document,
    pin_report,
    prepared_import,
    read_document,
)
from uv_stack.operations.upgrade import UpgradeOptions
from uv_stack.runner import SubprocessRunner


def export_items(config: ConfigRoot, items: tuple[str, ...] | list[str]) -> ExportResult:
    """Build the export document, turning an ambiguous item into exit 2.

    The panel goes through :func:`render_error`, which renders user text
    literally; a Click usage error would pass a bracketed name through markup.
    """
    try:
        return build_document(config, items)
    except AmbiguousItemError as error:
        render_error(error)
        sys.exit(2)


def print_header(document: ExportDocument) -> None:
    """Say what is being imported, and whether pins may re-resolve here."""
    echo(f"Importing {len(document.items)} item(s) exported by {document.created_by} "
         f"on {document.source_platform}.")
    here = source_platform()
    if here != document.source_platform:
        echo(f"This machine is {here}: pins re-resolved where this platform differs.")


def print_plan(plan: ImportPlan) -> None:
    """List each file's outcome and any variables the import declares."""
    for change in plan.changes:
        echo(f"  {STATUS_LABELS[change.status]}: {change.key}")
    if plan.missing_variables:
        echo(f"  declare in variables.txt: {', '.join(plan.missing_variables)}")


def print_conflicts(error: ConflictError) -> None:
    """Print each conflicting file's diff and the environments that use it."""
    for change in error.conflicts:
        click.echo(change_diff(change), nl=False)
        users = error.used_by.get(change.key)
        if users:
            echo(f"{change.key} is used by: {', '.join(users)}")


def dependents_line(names: list[str]) -> str:
    """Name the environments that use changed definitions but were not rebuilt."""
    args = " ".join(render_positional_arg(n) for n in names)
    return (f"Not rebuilt, but using changed definitions: {', '.join(names)}. "
            f"Rebuild them with 'stack sync env {args}'.")


@click.command("export")
@click.argument("items", nargs=-1, metavar="ITEMS...")
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
    help="Write the document to this file instead of standard output.",
)
@click.pass_obj
def export_cmd(config: ConfigRoot, items: tuple[str, ...], output: Path | None) -> None:
    """Export environments, profiles, and bundles as one portable document.

    ITEMS are env:NAME, profile:NAME, bundle:NAME, @NAME, or a bare NAME
    that names exactly one item. With no ITEMS the whole root is exported.
    Everything the items reach is included, with each environment's lock as
    a seed. Import it elsewhere with 'stack import FILE', or pull straight
    from another machine with 'ssh HOST stack export ITEMS | stack import -'.
    """
    result = export_items(config, items)
    render_warnings(result.warnings, styled=False)
    text = serialize_document(result.document)
    if output is None:
        click.echo(text, nl=False)
    else:
        atomic_write(output, text)


@click.command("import")
@click.argument("source", metavar="FILE|-")
@click.option("--overwrite", is_flag=True,
              help="Replace files that differ and remove target-only environment files.")
@click.option("--dry-run", is_flag=True, help="Report what would change; write nothing.")
@click.option("--strict", is_flag=True,
              help="Refuse unqualified names that fall through to package literals.")
@click.option("--no-build", is_flag=True, help="Install the definitions without building.")
@click.option("--recreate", is_flag=True,
              help="Wipe and rebuild each imported environment from the shipped pins.")
@click.pass_obj
def import_cmd(config: ConfigRoot, source: str, overwrite: bool, dry_run: bool,
               strict: bool, no_build: bool, recreate: bool) -> None:
    """Import a document written by 'stack export'.

    FILE is the document, or - to read standard input. New files are
    installed and identical ones left alone. A file that differs from this
    machine's copy is refused with a diff unless --overwrite is given. Each
    imported environment is then built, with the shipped pins as
    preferences, unless --no-build is given.
    """
    if no_build and recreate:
        raise click.UsageError("--recreate rebuilds environments, so it cannot be combined "
                               "with --no-build.")
    document = load_document(read_document(source, click.get_binary_stream("stdin")))
    print_header(document)
    options = ImportOptions(overwrite=overwrite, strict=strict)
    build = None if no_build else BuildRequest(SubprocessRunner(), recreate=recreate)
    try:
        with prepared_import(config, document, options, dry_run=dry_run, build=build) as plan:
            try:
                render_warnings(plan.warnings)
                print_plan(plan)
                if dry_run:
                    for step in plan.builds:
                        echo(f"  build: {step.name} ({step.action})")
                elif plan.builds:
                    _build(config, plan, recreate=recreate, strict=strict)
            finally:
                if plan.dependents:
                    echo(dependents_line(plan.dependents))
    except ConflictError as error:
        # First, so a warning that an environment could not be read qualifies
        # the "used by" lists below.
        render_warnings(error.resolution_warnings)
        print_conflicts(error)
        raise
    if dry_run:
        echo("Dry run: nothing written.")


def _build(config: ConfigRoot, plan: ImportPlan, *, recreate: bool, strict: bool) -> None:
    names = [step.name for step in plan.builds]
    built: list[str] = []

    def report(name: str) -> None:
        built.append(name)
        seed = plan.document.seeds.get(name)
        if seed is not None:
            for line in pin_report(name, seed, config.env_requirements_lock(name)):
                echo(line)

    options = UpgradeOptions(create=True, recreate=recreate, no_upgrade=True, strict=strict)
    try:
        _run_upgrade(config, names, options, seeds=plan.document.seeds, on_success=report,
                     rule_verb="Building", all_succeeded="All imported environments built.")
    finally:
        failed = [n for n in names if n not in built]
        if failed:
            args = " ".join(render_positional_arg(n) for n in failed)
            echo(f"Re-run the failed build(s) once the cause is fixed: stack sync env {args}")
