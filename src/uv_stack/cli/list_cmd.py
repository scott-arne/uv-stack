"""``stack list KIND``: list environments, profiles, or bundles."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import render_table
from uv_stack.config import ConfigRoot

_KINDS = ("env", "profile", "bundle")

#: Max rendered width of the Description and Tags cells before truncation.
_DESCRIPTION_WIDTH = 40
_TAGS_WIDTH = 20


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
@click.pass_obj
def list_resources(config: ConfigRoot, kind: str, tags: tuple[str, ...]) -> None:
    """List resources of KIND (env, profile, or bundle)."""
    if kind == "env" and tags:
        raise click.UsageError(
            "--tag is not valid for 'env' (environments have no tags)."
        )

    wanted = set(tags)
    rows: list[tuple[str, ...]]
    if kind == "env":
        rows = []
        for env in config.list_envs():
            cfg = config.load_env(env)
            rows.append((env, cfg.python, str(len(cfg.stack))))
        render_table(
            "envs",
            [("Env", "left"), ("Python", "left"), ("Stack", "right")],
            rows,
            config.envs_dir,
        )
    elif kind == "profile":
        rows = []
        for name in config.list_profiles():
            prof = config.load_profile(name)
            if wanted and wanted.isdisjoint(prof.tags):
                continue
            rows.append(
                (
                    name,
                    str(len(prof.includes)),
                    _truncate(", ".join(prof.tags), _TAGS_WIDTH),
                    _truncate(prof.description or "", _DESCRIPTION_WIDTH),
                )
            )
        render_table(
            "profiles",
            [
                ("Profile", "left"),
                ("Packages", "right"),
                ("Tags", "left"),
                ("Description", "left"),
            ],
            rows,
            config.profiles_dir,
        )
    else:
        rows = []
        for name in config.list_bundles():
            bundle = config.load_bundle(name)
            if wanted and wanted.isdisjoint(bundle.tags):
                continue
            rows.append(
                (
                    name,
                    str(len(bundle.includes)),
                    _truncate(", ".join(bundle.tags), _TAGS_WIDTH),
                    _truncate(bundle.description or "", _DESCRIPTION_WIDTH),
                )
            )
        render_table(
            "bundles",
            [
                ("Bundle", "left"),
                ("Entries", "right"),
                ("Tags", "left"),
                ("Description", "left"),
            ],
            rows,
            config.bundles_dir,
        )
