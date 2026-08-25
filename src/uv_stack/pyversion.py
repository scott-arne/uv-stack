"""Pure helpers for comparing Python versions.

These are pure helpers for comparing the interpreter version an env *config*
requests against the one an env *actually* runs.
"""

from __future__ import annotations


def parse_python_info(stdout: str) -> tuple[str | None, str | None]:
    """Parse the two-line output of :func:`micromamba_python_info`.

    :param stdout: Raw stdout: the executable path, then the version.
    :returns: ``(executable, version)``; either element is ``None`` when its
        line is missing or blank.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    executable = lines[0] if len(lines) > 0 else None
    version = lines[1] if len(lines) > 1 else None
    return (executable, version)


def is_comparable(configured: str) -> bool:
    """Whether ``configured`` is a plain dotted version this module can compare.

    ``python.txt`` normally holds a plain version such as ``3.14``, but conda
    accepts full match specs. Anything carrying a comparison operator, wildcard,
    build string, or list separator is not something a component-wise compare
    can judge, so it is reported as incomparable rather than guessed at.
    """
    if not configured or not configured.strip():
        return False

    # Reject if contains any operator, wildcard, separator, or whitespace.
    invalid_chars = {"<", ">", "=", "!", "*", ",", "|", " ", "\t", "\n"}
    if any(char in configured for char in invalid_chars):
        return False

    # Verify each dot-separated component is a non-negative integer.
    components = configured.split(".")
    for component in components:
        if not component:
            return False
        try:
            value = int(component)
            if value < 0:
                return False
        except ValueError:
            return False

    return True


def satisfies(configured: str, actual: str) -> bool:
    """Whether the running version ``actual`` satisfies ``configured``.

    ``python=3.14`` in an ``environment.yml`` pins the 3.14 series, so a
    configured ``3.14`` is satisfied by an actual ``3.14.7``: the configured
    value is a prefix constraint, compared component-wise only as far as it is
    specified.
    """
    # Parse configured version.
    try:
        configured_parts = [int(part) for part in configured.split(".")]
    except (ValueError, AttributeError):
        # Unparseable input: fail-open to avoid false drift reports.
        return True

    # Parse actual version.
    try:
        actual_parts = [int(part) for part in actual.split(".")]
    except (ValueError, AttributeError):
        # Unparseable input: fail-open to avoid false drift reports.
        return True

    # Actual must have at least as many components as configured.
    if len(actual_parts) < len(configured_parts):
        return False

    # Compare component-wise for the length of configured.
    for configured_part, actual_part in zip(configured_parts, actual_parts, strict=False):
        if configured_part != actual_part:
            return False

    return True
