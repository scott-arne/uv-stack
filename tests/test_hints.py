from __future__ import annotations

from uv_stack.hints import render_positional_arg


def test_plain_name_unchanged():
    assert render_positional_arg("main") == "main"


def test_plain_name_with_dash_unchanged():
    assert render_positional_arg("my-env") == "my-env"


def test_shell_metacharacter_quoted():
    assert render_positional_arg("bad;touch") == "'bad;touch'"


def test_single_quote_nested():
    """The name contains a single quote, so shlex.quote uses nested quoting."""
    assert render_positional_arg("a'b") == """'a'"'"'b'"""


def test_leading_dash_prefixed():
    assert render_positional_arg("--recreate") == "-- --recreate"


def test_leading_dash_and_metacharacter_both():
    """A name that needs both the separator and quoting gets both."""
    assert render_positional_arg("--bad;touch") == "-- '--bad;touch'"
