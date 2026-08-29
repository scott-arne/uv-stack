from __future__ import annotations

from uv_stack.pyversion import (
    is_comparable,
    is_near_miss_version,
    near_miss_version_notice,
    parse_python_info,
    satisfies,
)


def test_parse_python_info_both_lines():
    stdout = "/envs/main/bin/python\n3.12.7\n"
    executable, version = parse_python_info(stdout)
    assert executable == "/envs/main/bin/python"
    assert version == "3.12.7"


def test_parse_python_info_missing_second_line():
    stdout = "/envs/main/bin/python\n"
    executable, version = parse_python_info(stdout)
    assert executable == "/envs/main/bin/python"
    assert version is None


def test_parse_python_info_blank_input():
    stdout = ""
    executable, version = parse_python_info(stdout)
    assert executable is None
    assert version is None


def test_parse_python_info_trailing_whitespace():
    stdout = "  /envs/main/bin/python  \n  3.12.7  \n"
    executable, version = parse_python_info(stdout)
    assert executable == "/envs/main/bin/python"
    assert version == "3.12.7"


def test_parse_python_info_blank_lines():
    stdout = "\n\n/envs/main/bin/python\n\n3.12.7\n\n"
    executable, version = parse_python_info(stdout)
    assert executable == "/envs/main/bin/python"
    assert version == "3.12.7"


def test_is_comparable_plain_version():
    assert is_comparable("3.14") is True
    assert is_comparable("3.12.7") is True
    assert is_comparable("3") is True


def test_is_comparable_with_operators():
    assert is_comparable(">=3.12") is False
    assert is_comparable("<3.14") is False
    assert is_comparable("=3.12") is False
    assert is_comparable("!=3.11") is False


def test_is_comparable_with_wildcards():
    assert is_comparable("3.12.*") is False
    assert is_comparable("3.*") is False


def test_is_comparable_with_separators():
    assert is_comparable("3.11|3.12") is False
    assert is_comparable("3.12,3.13") is False


def test_is_comparable_empty():
    assert is_comparable("") is False
    assert is_comparable("   ") is False


def test_is_comparable_with_whitespace():
    assert is_comparable("3.12 ") is False
    assert is_comparable(" 3.12") is False


def test_is_comparable_non_integer_component():
    assert is_comparable("3.x") is False
    assert is_comparable("3.12.alpha") is False
    assert is_comparable("3.-1") is False


def test_is_comparable_rejects_what_int_would_accept():
    """int() is not the predicate: uv's --python-version is stricter than it.

    A leading '+', PEP 515 underscores, and non-ASCII decimal digits all round
    trip through int() and all reach uv as something it rejects.
    """
    assert is_comparable("+3.14") is False
    assert is_comparable("3.1_2") is False
    assert is_comparable("٣.١٢") is False


def test_is_near_miss_version_catches_botched_versions():
    assert is_near_miss_version("3.12.x") is True
    assert is_near_miss_version("3.12.*") is True
    assert is_near_miss_version("3.l2") is True
    assert is_near_miss_version("3.12rc1") is True
    assert is_near_miss_version("3.") is True


def test_is_near_miss_version_passes_plain_versions():
    assert is_near_miss_version("3.12") is False
    assert is_near_miss_version("3.12.7") is False
    assert is_near_miss_version("3") is False


def test_is_near_miss_version_leaves_env_names_alone():
    """A leading digit is not enough; the predicate must also see a dot.

    The whole value of this test is that it fires only where the alternative
    reading is wrong. '3d-modeling' is a perfectly good environment name, and
    telling its owner it looks like a botched version is the same misdirection
    running the other way.
    """
    assert is_near_miss_version("nope") is False
    assert is_near_miss_version("main") is False
    assert is_near_miss_version("3d-modeling") is False
    assert is_near_miss_version("2024-baseline") is False
    assert is_near_miss_version("") is False


def test_is_near_miss_version_ignores_conda_operator_specs():
    """Operator forms stay unflagged; only digit-dot openings are judged.

    A deliberate limit rather than an oversight: the message names the value as
    an attempted version, and this predicate earns that claim from the shape of
    the value alone. Widening it to leading operators is defensible, but it is
    a separate decision from the one this function makes.
    """
    assert is_near_miss_version(">=3.11,<3.13") is False
    assert is_near_miss_version("<3.13") is False


def test_near_miss_version_notice_quotes_the_spec():
    notice = near_miss_version_notice("3.12.x")
    assert "'3.12.x'" in notice
    assert "micromamba environment" in notice


def test_satisfies_matching_versions():
    assert satisfies("3.14.7", "3.14.7") is True
    assert satisfies("3.12", "3.12") is True


def test_satisfies_prefix_match():
    assert satisfies("3.14", "3.14.7") is True
    assert satisfies("3", "3.14.7") is True


def test_satisfies_prefix_mismatch():
    assert satisfies("3.13", "3.14.7") is False
    assert satisfies("3.14.2", "3.14.7") is False


def test_satisfies_actual_shorter_than_configured():
    assert satisfies("3.14.7", "3.14") is False


def test_satisfies_unparseable_returns_true():
    # Permissive fallback: don't manufacture false drift reports.
    assert satisfies(">=3.12", "3.14.7") is True
    assert satisfies("3.12", "not-a-version") is True
    assert satisfies("invalid", "3.14.7") is True
