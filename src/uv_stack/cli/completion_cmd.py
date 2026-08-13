"""``stack completion``: print the shell-completion script for a shell."""

from __future__ import annotations

import rich_click as click

from uv_stack.cli._render import echo

_INSTALL_HINTS = {
    "bash": '# Requires bash >= 4.4. Add to ~/.bashrc: eval "$(stack completion bash)"',
    "zsh": '# Add to ~/.zshrc: eval "$(stack completion zsh)"',
    "fish": "# Add to ~/.config/fish/config.fish: stack completion fish | source",
}


@click.command("completion")
@click.argument("shell", type=click.Choice(("bash", "zsh", "fish")))
def completion(shell: str) -> None:
    """Print the shell-completion script for SHELL (bash, zsh, or fish)."""
    from click.shell_completion import get_completion_class

    from uv_stack.cli import cli as root_cli

    comp_cls = get_completion_class(shell)
    if comp_cls is None:  # pragma: no cover - click always ships these three
        raise click.UsageError(f"No completion support for '{shell}'.")
    comp = comp_cls(root_cli, {}, "stack", "_STACK_COMPLETE")
    echo(_INSTALL_HINTS[shell])
    echo(comp.source())
