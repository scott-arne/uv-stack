from __future__ import annotations

import pytest

from uv_stack.hints import escape_controls, has_control, render_positional_arg


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


@pytest.mark.parametrize(
    "char", ["\x00", "\x07", "\t", "\n", "\x1b", "\x1f", "\x7f", "\x80", "\x9b", "\x9f"]
)
def test_has_control_finds_each_control_range(char):
    assert has_control(f"a{char}b")


@pytest.mark.parametrize("text", ["", "main", "my env", "caf\u00e9", "a\u200bb", "a\u00a0b"])
def test_has_control_passes_text_without_one(text):
    """Only category Cc counts: a space, a format character, and NBSP do not."""
    assert not has_control(text)


@pytest.mark.parametrize(
    ("text", "escaped"),
    [
        ("a\x1b[2Kb", "a\\x1b[2Kb"),
        ("a\x9bb", "a\\x9bb"),
        ("a\x00b", "a\\x00b"),
        ("line\nnext\tcell", "line\\nnext\\tcell"),
        ("caf\u00e9\x07", "caf\u00e9\\x07"),
    ],
)
def test_escape_controls_spells_each_control_character(text, escaped):
    assert escape_controls(text) == escaped
    assert not has_control(escape_controls(text))


@pytest.mark.parametrize("text", ["profiles/ds.yaml", "caf\u00e9", "a\u200bb", "a\\x1bb"])
def test_escape_controls_returns_text_without_one_unchanged(text):
    assert escape_controls(text) == text


def test_escape_controls_keeps_layout_but_nothing_else():
    """Newlines and tabs lay a message out; a CR can overwrite the line it is on."""
    assert escape_controls("a\nb\tc", keep_layout=True) == "a\nb\tc"
    assert escape_controls("a\x1b[2J\rb\x7fc\x9bd", keep_layout=True) == (
        "a\\x1b[2J\\rb\\x7fc\\x9bd"
    )
    assert escape_controls("a\nb") == "a\\nb"


@pytest.mark.parametrize("keep_layout", [False, True])
def test_escape_controls_spells_a_lone_surrogate(keep_layout):
    """A lone surrogate cannot be encoded for a UTF-8 terminal, so it is escaped."""
    assert escape_controls("a\udcffb", keep_layout=keep_layout) == "a\\udcffb"
