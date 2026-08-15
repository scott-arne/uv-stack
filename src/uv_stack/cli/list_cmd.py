"""``stack list KIND``: list environments, profiles, or bundles."""

from __future__ import annotations

import json

import rich_click as click
from rich.markup import escape

from uv_stack.cli._render import console, echo, render_table
from uv_stack.config import ConfigRoot

_KINDS = ("env", "profile", "bundle")

#: Max rendered width of the Description and Tags cells before truncation.
_DESCRIPTION_WIDTH = 40
_TAGS_WIDTH = 20

_EMPTY_HINTS = {
    "env": "No environments yet — run 'stack init' or 'stack create env NAME TOKENS...'.",
    "profile": "No profiles yet — run 'stack create profile NAME PACKAGE...'.",
    "bundle": "No bundles yet — run 'stack create bundle NAME TOKEN...'.",
}


def _print_empty_hint(kind: str, wanted: set[str]) -> None:
    """Print the empty-result hint for ``kind`` (tag-aware) instead of a table."""
    if wanted:
        console.print(
            f"[dim]No {kind}s match tags: {escape(', '.join(sorted(wanted)))}.[/dim]"
        )
    else:
        console.print(f"[dim]{_EMPTY_HINTS[kind]}[/dim]")


def _truncate(text: str, width: int) -> str:
    """Return ``text`` shortened to ``width`` characters with an ellipsis.

    :param text: The full cell text (may be empty).
    :param width: Maximum number of characters in the result.
    :returns: ``text`` unchanged when within ``width``, else its first
        ``width - 1`` characters followed by ``…``.
    """
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


@click.command("list")
@click.argument("kind", type=click.Choice(_KINDS))
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Filter profiles/bundles to those carrying any of these tags (repeatable).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_obj
def list_resources(
    config: ConfigRoot, kind: str, tags: tuple[str, ...], as_json: bool
) -> None:
    """List resources of KIND (env, profile, or bundle)."""
    if kind == "env" and tags:
        raise click.UsageError(
            "--tag is not valid for 'env' (environments have no tags)."
        )

    wanted = set(tags)
    if kind == "env":
        env_data: list[dict[str, object]] = []
        env_rows: list[tuple[str, ...]] = []
        for env in config.list_envs():
            cfg = config.load_env(env)
            env_data.append({"name": env, "python": cfg.python, "stack": cfg.stack})
            env_rows.append((env, cfg.python, str(len(cfg.stack))))
        if as_json:
            echo(json.dumps(env_data, indent=2))
            return
        if not env_data:
            _print_empty_hint("env", wanted)
            return
        render_table(
            "envs",
            [("Env", "left"), ("Python", "left"), ("Stack", "right")],
            env_rows,
            config.envs_dir,
        )
    elif kind == "profile":
        profile_data: list[dict[str, object]] = []
        profile_rows: list[tuple[str, ...]] = []
        for name in config.list_profiles():
            prof = config.load_profile(name)
            if wanted and wanted.isdisjoint(prof.tags):
                continue
            profile_data.append(
                {
                    "name": name,
                    "packages": prof.includes,
                    "tags": prof.tags,
                    "description": prof.description,
                }
            )
            profile_rows.append(
                (
                    name,
                    str(len(prof.includes)),
                    _truncate(", ".join(prof.tags), _TAGS_WIDTH),
                    _truncate(prof.description or "", _DESCRIPTION_WIDTH),
                )
            )
        if as_json:
            echo(json.dumps(profile_data, indent=2))
            return
        if not profile_data:
            _print_empty_hint("profile", wanted)
            return
        render_table(
            "profiles",
            [
                ("Profile", "left"),
                ("Packages", "right"),
                ("Tags", "left"),
                ("Description", "left"),
            ],
            profile_rows,
            config.profiles_dir,
        )
    else:
        bundle_data: list[dict[str, object]] = []
        bundle_rows: list[tuple[str, ...]] = []
        for name in config.list_bundles():
            bundle = config.load_bundle(name)
            if wanted and wanted.isdisjoint(bundle.tags):
                continue
            bundle_data.append(
                {
                    "name": name,
                    "entries": bundle.includes,
                    "tags": bundle.tags,
                    "description": bundle.description,
                }
            )
            bundle_rows.append(
                (
                    name,
                    str(len(bundle.includes)),
                    _truncate(", ".join(bundle.tags), _TAGS_WIDTH),
                    _truncate(bundle.description or "", _DESCRIPTION_WIDTH),
                )
            )
        if as_json:
            echo(json.dumps(bundle_data, indent=2))
            return
        if not bundle_data:
            _print_empty_hint("bundle", wanted)
            return
        render_table(
            "bundles",
            [
                ("Bundle", "left"),
                ("Entries", "right"),
                ("Tags", "left"),
                ("Description", "left"),
            ],
            bundle_rows,
            config.bundles_dir,
        )
