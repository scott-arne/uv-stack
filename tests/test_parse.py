from pathlib import Path

from uv_stack.parse import clean_line, first_clean_line, read_clean_lines


def test_clean_line_strips_comment_and_whitespace():
    assert clean_line("  numpy  # pin later ") == "numpy"
    assert clean_line("# whole line comment") == ""
    assert clean_line("   ") == ""
    assert clean_line("-e /path/to/pkg") == "-e /path/to/pkg"


def test_read_clean_lines_skips_blank_and_comment(tmp_path: Path):
    f = tmp_path / "ds.in"
    f.write_text("numpy\n# comment\n\n  pandas  \n")
    assert read_clean_lines(f) == ["numpy", "pandas"]


def test_read_clean_lines_missing_file_returns_empty(tmp_path: Path):
    assert read_clean_lines(tmp_path / "nope.in") == []


def test_first_clean_line_returns_default_when_empty(tmp_path: Path):
    f = tmp_path / "python.txt"
    f.write_text("# only a comment\n")
    assert first_clean_line(f, default="3.12") == "3.12"


def test_first_clean_line_returns_first_value(tmp_path: Path):
    f = tmp_path / "python.txt"
    f.write_text("\n3.11\n3.10\n")
    assert first_clean_line(f, default="3.12") == "3.11"
