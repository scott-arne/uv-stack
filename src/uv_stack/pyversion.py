"""Pure helpers for reading and comparing Python version specs.

These are pure helpers for comparing the interpreter version an env *config*
requests against the one an env *actually* runs, and for judging whether a
*project* interpreter spec is a version at all.

The two jobs answer to different grammars and must not be mixed. An env's
``python.txt`` is handed to conda, which takes full match specs — ``3.12.*``
and ``>=3.11,<3.13`` are legal there. A project's ``[tool.uv-stack].python``
is handed to uv, which takes none of those, so a value conda would accept is
often a mistake in a project.
"""

from __future__ import annotations

import re

#: A project interpreter spec uv understands as a version. Anything else that
#: is not a path or an implementation form is read as a micromamba env name.
#: Deliberately looser than :func:`is_comparable`, which additionally demands
#: ASCII digits because it feeds ``uv pip compile --python-version``.
PLAIN_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")

#: The shape of a value the user was plainly *trying* to write as a version.
#: Anchored on a digit followed by a dot so that plausible environment names
#: beginning with a digit — ``3d-modeling``, ``2024-baseline`` — do not match.
_VERSION_ATTEMPT_RE = re.compile(r"^\d+\.")


def is_near_miss_version(spec: str) -> bool:
    """Whether ``spec`` reads as an attempted version that is not one.

    A project spec that is not a version, a path, or an implementation form is
    silently taken to name a micromamba environment, and every message from
    there on talks about environments. For ``3.12.x`` or ``3.12.*`` that is a
    misdirection: the user was writing a version, and being told to create an
    environment called ``3.12.x`` sends them the wrong way entirely.

    Only for *project* specs. An env's ``python.txt`` is read by conda, where
    ``3.12.*`` is a correct match spec rather than a botched version.

    :param spec: The raw interpreter spec.
    :returns: ``True`` when ``spec`` starts with digits and a dot but is not a
        plain dotted version.
    """
    return bool(_VERSION_ATTEMPT_RE.match(spec)) and not PLAIN_VERSION_RE.match(spec)


def near_miss_version_notice(spec: str) -> str:
    """The one sentence every surface uses for a near-miss project version.

    Shared so the warning ``stack edit`` prints before anything runs and the
    hint ``stack refresh`` attaches after the probe fails cannot drift apart:
    they describe one condition seen at two moments.

    :param spec: The offending spec.
    :returns: The message, without a trailing newline.
    """
    return (
        f"'{spec}' looks like a Python version but is not a plain one "
        "(digits and dots only), so it is read as a micromamba environment "
        "name. Use a plain version such as 3.12, or name an environment that "
        "exists."
    )


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

    :param configured: The value read from ``python.txt``.
    :returns: ``True`` when every dot-separated component is a plain ASCII
        run of digits.
    """
    if not configured or not configured.strip():
        return False

    # Reject if contains any operator, wildcard, separator, or whitespace.
    invalid_chars = {"<", ">", "=", "!", "*", ",", "|", " ", "\t", "\n"}
    if any(char in configured for char in invalid_chars):
        return False

    # Verify each dot-separated component is a run of plain ASCII digits.
    # An int() round-trip is too permissive for this job: it accepts a leading
    # '+', PEP 515 underscores, and non-ASCII decimal digits, none of which
    # uv takes for --python-version.
    components = configured.split(".")
    for component in components:
        if not (component.isascii() and component.isdigit()):
            return False

    return True


def satisfies(configured: str, actual: str) -> bool:
    """Whether the running version ``actual`` satisfies ``configured``.

    ``python=3.14`` in an ``environment.yml`` pins the 3.14 series, so a
    configured ``3.14`` is satisfied by an actual ``3.14.7``: the configured
    value is a prefix constraint, compared component-wise only as far as it is
    specified.

    :param configured: The version ``python.txt`` requests.
    :param actual: The version the environment's interpreter reports.
    :returns: ``True`` when ``actual`` matches ``configured`` component-wise,
        and also when either side cannot be parsed.
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
