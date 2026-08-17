from pathlib import Path

import pytest

from uv_stack.parse import clean_line, first_clean_line, read_clean_lines, requirement_name


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


@pytest.mark.parametrize(
    "requirement, expected",
    [
        ("numpy", "numpy"),
        ("numpy>=2", "numpy"),
        ("pkg[extra]==1.0", "pkg"),
        ("pkg ; python_version<'3.13'", "pkg"),
        ("pkg @ https://host/x.whl", None),   # slash → not a plain name
        ("./dist/x.tar.gz", None),
        ("-e ./tool", None),
        ("--pre", None),
        ("", None),
        ("   ", None),
    ],
)
def test_requirement_name(requirement, expected):
    assert requirement_name(requirement) == expected


@pytest.mark.parametrize(
    "raw, canonical",
    [("My_Pkg", "my-pkg"), ("my.pkg", "my-pkg"), ("a--b__c..d", "a-b-c-d"), ("plain", "plain")],
)
def test_canonical_name(raw, canonical):
    from uv_stack.parse import canonical_name

    assert canonical_name(raw) == canonical


@pytest.mark.parametrize(
    "entry, expected",
    [
        ("numpy>=2", "numpy"),
        ("pkg[extra]==1.0", "pkg"),
        ("pkg @ https://host/x.whl", "pkg"),  # PEP 508 direct reference
        ("pkg[extra] @ https://host/x.whl", "pkg"),  # with extras
        ("git+https://h/r.git@v1", None),  # VCS → contains /
        ("-e ./tool", None),  # editable
        ("./dist/x.tar.gz", None),  # path
        ("", None),
    ],
)
def test_ownership_name(entry, expected):
    from uv_stack.parse import ownership_name

    assert ownership_name(entry) == expected


def test_direct_reference_is_owned_but_not_removable():
    """Ownership and removability are deliberately asymmetric for `name @ url`.

    ownership_name resolves the head so the ledger can claim the entry;
    requirement_name refuses the whole string so it can never be handed to
    `uv remove` as a bare name, which would remove whatever else claims it.
    """
    from uv_stack.parse import ownership_name, requirement_name

    entry = "torch @ https://example.invalid/torch-2.0-py3-none-any.whl"
    assert ownership_name(entry) == "torch"
    assert requirement_name(entry) is None
