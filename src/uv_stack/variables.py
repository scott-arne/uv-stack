"""Variable references inside requirement strings.

A config root that travels between machines cannot hard-code one machine's
filesystem, so a requirement entry may carry ``${NAME}`` references that expand
against values the local machine supplies. This module owns the grammar, the
rules about where a reference may appear, and the substitution itself.

It is pure: it reads no files and runs no commands. :class:`Variables` is loaded
by :meth:`uv_stack.config.ConfigRoot.load_variables`; expansion happens at the
single boundary that materializes ``requirements.in``, so every durable
uv-stack record keeps the unexpanded text and stays portable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from uv_stack.errors import ConfigError
from uv_stack.parse import ownership_name

#: A well-formed reference. Braces are mandatory, so a bare ``$`` in a direct
#: reference URL is never mistaken for one.
REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Any opening of a reference, well-formed or not. Every occurrence in an entry
#: must begin a :data:`REFERENCE_RE` match; anything else is malformed. Without
#: this rule a typo like ``${DEV`` would install a package literally named
#: ``${DEV`` instead of failing.
OPENER = "${"

#: A legal variable name: the conventional shell identifier.
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Options whose operand is another requirements file. A reference inside one
#: would pull in named dependencies that differ per machine, which is exactly
#: what the ledger cannot see. Matched in every spelling pip accepts: bare,
#: attached-short (``-rFILE``), and attached-long (``--requirement=FILE``).
RECURSIVE_OPTIONS = frozenset({"-r", "--requirement", "-c", "--constraint"})

#: How much of an entry to quote when reporting a malformed reference.
_FRAGMENT_WINDOW = 24

#: A comment marker: a ``#`` that *begins* a whitespace-separated token, so it
#: sits at the start of the entry or directly after whitespace. The preceding
#: whitespace is part of the match so that slicing at :attr:`re.Match.start`
#: drops the separator along with the comment.
_COMMENT_RE = re.compile(r"(?:^|\s)#")

#: Hint shown when a reference names an undeclared variable.
UNDECLARED_HINT = (
    "Add each name on its own line in variables.txt. Declaring it "
    "is what makes it part of the root's portable contract, so an "
    "environment variable alone is not enough."
)


@dataclass(frozen=True)
class Variables:
    """The names a config root declares and the values this machine supplies.

    :ivar declared: Names from ``<root>/variables.txt``, in file order. A
        reference to a name outside this tuple is an error even when the
        environment happens to define it — the declaration is what makes a
        reference part of the root's portable contract.
    :ivar values: Resolved values for the subset of names this machine defines.
    """

    declared: tuple[str, ...]
    values: Mapping[str, str]

    def undefined(self) -> list[str]:
        """Declared names this machine supplies no value for.

        :returns: The missing names, in declaration order.
        """
        return [name for name in self.declared if name not in self.values]


def referenced_names(text: str) -> list[str]:
    """Names referenced by ``text``, de-duplicated, in first-appearance order.

    Only the code portion is read, per :func:`_code_portion`. uv discards a
    comment, so a ``${...}`` there is prose: substituting it would rewrite text
    uv never sees, and requiring it declared would refuse an entry that
    :func:`placement_problem` — which has judged the code portion since it
    learned about comments — admits. Every caller wants that reading: the two
    declaration checks and doctor's undeclared-variable report all ask what the
    entry references *as an entry*.

    :param text: Any string that may hold references.
    :returns: The referenced names.
    """
    names: list[str] = []
    for match in REFERENCE_RE.finditer(_code_portion(text)):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


_CONDITION_TEXT = {
    1: (
        "condition 1: a reference may not sit in the distribution-name "
        "position, because the ledger records what a stack depends on by name"
    ),
    2: (
        "condition 2: an entry holding a reference must begin with '-' or "
        "with a token containing a path separator"
    ),
    3: (
        "condition 3: a reference may not appear in a '-r'/'--requirement' or "
        "'-c'/'--constraint' include"
    ),
}


def _malformed_fragment(entry: str) -> str | None:
    """The first ``${`` in ``entry`` that does not open a reference.

    :param entry: One requirement entry.
    :returns: A bounded quote of the offending fragment, or ``None``.
    """
    starts = {match.start() for match in REFERENCE_RE.finditer(entry)}
    index = entry.find(OPENER)
    while index != -1:
        if index not in starts:
            return entry[index : index + _FRAGMENT_WINDOW]
        index = entry.find(OPENER, index + 1)
    return None


def _first_token(entry: str) -> str:
    """The first whitespace-separated token of ``entry``, or ``""``."""
    tokens = entry.split()
    return tokens[0] if tokens else ""


def _code_portion(entry: str) -> str:
    """The part of ``entry`` uv actually reads, with any comment cut away.

    A requirements file treats ``#`` as a comment marker at the start of a line
    or after whitespace — exactly a token that begins with one — and discards it
    along with the rest of the line. A ``#`` inside a token is ordinary text, so
    ``pkg#egg=thing`` stays one requirement rather than becoming a bare ``pkg``.
    This is the same definition of a comment :func:`expansion_problem` uses on
    the other side of substitution.

    :param entry: One requirement entry, a single physical line.
    :returns: The text before the comment marker, or the whole entry when there
        is none. The empty string when the entry is nothing but a comment.
    """
    match = _COMMENT_RE.search(entry)
    return entry if match is None else entry[: match.start()]


def _expand_code(entry: str, values: Mapping[str, str]) -> str:
    """Substitute every reference in ``entry``'s code portion.

    The comment is carried over verbatim rather than rebuilt, so its spacing
    survives. Substituting inside it would be wrong twice over: uv discards it,
    and :func:`referenced_names` does not report the names in it, so the
    declaration and definition checks above have not vouched for them — a
    ``sub`` over the whole line would raise :class:`KeyError` on the first one.

    :param entry: One requirement entry, a single physical line.
    :param values: This machine's values, keyed by name.
    :returns: The entry with its code portion expanded.
    """
    code = _code_portion(entry)
    return REFERENCE_RE.sub(lambda match: values[match.group(1)], code) + entry[len(code) :]


def _is_recursive_include(token: str) -> bool:
    """Whether ``token`` is a requirements-file or constraints-file option.

    Covers the bare (``-r``), attached-short (``-rFILE``), and attached-long
    (``--requirement=FILE``) spellings; pip accepts all three.

    :param token: One whitespace-separated token of an entry.
    :returns: ``True`` when the token opens a recursive include.
    """
    if token in RECURSIVE_OPTIONS:
        return True
    for option in RECURSIVE_OPTIONS:
        if option.startswith("--"):
            if token.startswith(option + "="):
                return True
        elif token.startswith(option) and len(token) > len(option):
            return True
    return False


def placement_problem(entry: str) -> tuple[str, str] | None:
    """Classify one entry, or return ``None`` when it is admitted.

    Four refusals are distinguished so a caller that reports rather than
    raises (``stack doctor``) can give each its own finding:

    - ``multiline-entry`` — the entry spans more than one physical line. This
      applies whether or not it holds a reference, because the admission
      conditions below are meaningless on a multi-line string: an admitted
      first line can smuggle an arbitrary second one.
    - ``continuation-entry`` — the entry ends in a backslash. This applies
      whether or not it holds a reference, for the same reason: a requirements
      file joins such a line to the one after it, so the entry consumes
      whichever requirement the render writes next. The entry is still a single
      physical line, which is why it is not a ``multiline-entry``.
    - ``malformed-reference`` — some ``${`` does not open a well-formed
      reference. Checked before placement: it is the more basic defect, and a
      malformed opener makes the rest of the analysis unreliable.
    - ``misplaced-reference`` — the entry holds a reference somewhere the
      expansion would not be safe. Every violated condition is named, so the
      message distinguishes an entry that fails one condition from one that
      fails several.

    Only the last of the four is judged on the entry's code portion — the text
    before any comment marker, per :func:`_code_portion`. uv discards a comment
    entirely, so a reference inside one occupies no position and opens no
    include; judging the raw entry refuses ``-e ${DEV}/pkg # do not use -r
    here``, which is a correct line.

    The first three deliberately stay on the raw entry. The asymmetry is not an
    oversight:

    - a comment cannot contain a newline, so ``multiline-entry`` reads the same
      either way;
    - a trailing backslash inside a comment still continues the *physical*
      line, so ``-e ${DEV}/pkg # note \\`` swallows the requirement written
      after it. Judging continuation on the code portion would admit exactly
      that;
    - a malformed ``${`` in a comment cannot hurt uv, which never reads it.
      Naming it anyway is a judgment call and not a consequence: a typo in a
      reference is worth hearing about wherever it was written, and a user who
      meant the line to be inert loses nothing by fixing it.

    :param entry: One requirement entry, unexpanded.
    :returns: ``(kind, explanation)`` or ``None``.
    """
    if "\n" in entry or "\r" in entry:
        return (
            "multiline-entry",
            "condition 0: the entry spans more than one line; write one "
            "requirement per line",
        )
    if entry.rstrip().endswith("\\"):
        return (
            "continuation-entry",
            "a trailing backslash continues onto the next line and swallows "
            "the requirement after it; write the path without a trailing "
            "separator",
        )
    fragment = _malformed_fragment(entry)
    if fragment is not None:
        return (
            "malformed-reference",
            f"'{fragment}' is not a well-formed reference; write ${{NAME}}",
        )
    # Everything from here down reads the code portion, for the reason given in
    # the docstring: a reference uv never reads cannot be misplaced. Computed
    # once so the four reads below cannot disagree about where the comment
    # starts.
    code = _code_portion(entry)
    if not referenced_names(code):
        return None

    token = _first_token(code)
    failed = []
    if ownership_name(code) is not None:
        failed.append(1)
    if not (token.startswith("-") or "/" in token or "\\" in token):
        failed.append(2)
    # Judged over every token, not just the first. uv reads an option line as
    # one option, so a value-taking option swallows a trailing '-r' -- but a
    # BOOLEAN one does not: uv reads '--no-index -r ${DEV}/reqs.txt' as
    # --no-index followed by a real include, and does read that file. pip is
    # looser still and accepts a recursive option after any option at all. A
    # first-token-only test therefore admits exactly the machine-local second
    # requirements file condition 3 exists to refuse. This is the same
    # whole-sequence reading expansion_problem already applies. It stops at the
    # comment, which is what keeps the scan from reading prose as an option.
    if any(_is_recursive_include(other) for other in code.split()):
        failed.append(3)
    if not failed:
        return None
    return ("misplaced-reference", "; ".join(_CONDITION_TEXT[number] for number in failed))


def check_placement(
    entries: Sequence[str],
    *,
    source: str | None = None,
    sources: Sequence[str | None] | None = None,
) -> None:
    """Refuse a multiline, continuation, malformed, or misplaced entry.

    Reports every offender at once, naming the condition each one failed, so a
    caller fixing a profile sees the whole list rather than one item per run.

    :param entries: The entries as written, unexpanded.
    :param source: The one file the entries came from, for the message. For a
        caller whose sequence spans several files, use ``sources`` instead.
    :param sources: The file each entry came from, positionally aligned with
        ``entries``, or ``None`` for an entry with no single file. Lets a
        flattened sequence keep per-entry blame while still reporting every
        offender in one error.
    :raises ConfigError: When any entry is refused.
    :raises ValueError: When both ``source`` and ``sources`` are given — the
        header would name one file and the offender lines another. Also when
        ``sources`` is given and is not the same length as ``entries`` — a
        misalignment would attach the wrong file to a refusal, which is worse
        than attaching none.
    """
    if source is not None and sources is not None:
        raise ValueError("cannot pass both source and sources")
    if sources is not None and len(sources) != len(entries):
        raise ValueError(
            f"sources has {len(sources)} items for {len(entries)} entries"
        )
    problems: list[str] = []
    for index, entry in enumerate(entries):
        problem = placement_problem(entry)
        if problem is None:
            continue
        origin = sources[index] if sources is not None else None
        held_by = f" (in {origin})" if origin else ""
        problems.append(f"  {entry!r}{held_by}: {problem[1]}")
    if not problems:
        return
    where = f" in {source}" if source else ""
    noun = "entry" if len(problems) == 1 else "entries"
    raise ConfigError(
        f"Refused {len(problems)} requirement {noun}{where}:\n" + "\n".join(problems),
        hint=(
            "A variable reference may only stand in a path or an option value: "
            "'-e ${DEV}/pkg', '--index-url ${HOST}/simple', or '${DEV}/pkg'. "
            "It may not name a distribution, and it may not sit inside a "
            "'-r'/'-c' include. A trailing backslash is a line continuation "
            "and must be removed."
        ),
    )


def _option_class(token: str) -> str | None:
    """The option a token introduces, or ``None`` when it is an operand.

    An attached value counts: pip and uv both accept ``-rFILE`` and
    ``--requirement=FILE``, so the option has to be read off the front of the
    token rather than assumed to be the whole of it.

    :param token: One whitespace-separated token.
    :returns: The option name, or ``None``.
    """
    if not token.startswith("-"):
        return None
    if token.startswith("--"):
        return token.split("=", 1)[0]
    return token[:2]


def expansion_problem(entry: str, expanded: str) -> str | None:
    """Report an expansion that changed what an entry means to uv.

    Placement is judged before substitution, on the entry as written, so it
    cannot see what a value turns the entry into. Values are arbitrary
    whitespace-free text, so the admitted path operand ``${DEV}/deps.txt``
    becomes the attached recursive include ``-r/deps.txt`` under ``DEV=-r`` —
    admitted going in, a second requirements file coming out.

    Refusing whitespace leaves six ways a value can still rewrite the entry,
    and each gets its own comparison across the substitution:

    - it carries an option of its own, caught by comparing each token's option
      class;
    - it opens a comment. A requirements file treats ``#`` as a comment marker
      at the start of a line or after whitespace — exactly a token that begins
      with one — so ``DEV=#`` turns ``${DEV}/pkg`` into a line uv never reads,
      and turns ``--index-url ${DEV}/simple`` into a bare, valueless option. A
      ``#`` inside a token is ordinary text to uv and stays admitted;
    - it opens a line continuation. An entry ending in a backslash joins the
      next physical line, so a value that is a lone backslash turns
      ``-e ${DEV}`` into an entry that swallows whichever requirement the
      render writes after it. A backslash anywhere else is a path separator
      and stays admitted. An entry written with a trailing backslash is
      refused at placement. Both sides are compared after stripping trailing
      whitespace, matching placement: the rendered line is stripped, so a
      trailing space cannot stop the backslash from landing at its end;
    - it introduces a newline. A value that is pure whitespace preserves the
      token count but splits the entry across multiple lines;
    - it introduces or changes the ownership name. A value like
      ``victim@https:/`` turns ``${ROOT}/files.example/pkg.whl`` into a direct
      reference whose distribution name came from a machine-local value,
      breaking the invariant that expansion never changes what a requirement
      is named;
    - it leaves a ``${...}`` behind in the code portion. Substitution is
      single-pass and values are opaque, so ``DEV=/a/${OTHER}/b`` leaves
      ``${OTHER}`` in the result. uv expands environment variables in
      requirements files, so the residual reference resolves against the
      environment and bypasses this module's declared-name and undefined-value
      checks. A ``${...}`` in the comment is not a residue: nothing there was
      ever substituted, because uv discards the comment before it expands
      anything.

    The first five checks ask whether the substitution *introduced* the syntax,
    not whether the result already held it: an entry written that way was
    placement's to judge, and re-judging it here would refuse text no value
    produced. The last one cannot take that shape, for the reason given where
    it is implemented.

    :param entry: The entry as written, unexpanded.
    :param expanded: The same entry after substitution.
    :returns: An explanation, or ``None`` when the expansion is safe.
    """
    before = entry.split()
    after = expanded.split()
    if len(before) != len(after):
        return (
            f"expansion changed the entry from {len(before)} token(s) to "
            f"{len(after)}"
        )
    for original, result in zip(before, after, strict=True):
        if _option_class(original) != _option_class(result):
            return (
                f"expansion turned the token {original!r} into {result!r}, "
                "which carries a different option"
            )
        if result.startswith("#") and not original.startswith("#"):
            return (
                f"expansion turned the token {original!r} into {result!r}, "
                "which starts a comment"
            )
    if expanded.rstrip().endswith("\\") and not entry.rstrip().endswith("\\"):
        return (
            "expansion left the entry ending in a backslash, which continues "
            "onto the next line"
        )
    if ("\n" in expanded or "\r" in expanded) and not (
        "\n" in entry or "\r" in entry
    ):
        return (
            "expansion introduced a newline, turning a single entry into "
            "multiple lines"
        )
    if ownership_name(expanded) != ownership_name(entry):
        return (
            f"expansion changed the ownership name from {ownership_name(entry)!r} "
            f"to {ownership_name(expanded)!r}"
        )
    # Every ${...} in an admitted entry's code portion is either a well-formed
    # reference (and therefore substituted) or already refused by
    # _malformed_fragment, so none may survive expansion there. Asking this of
    # the *result's* code portion is safe because the comment check above has
    # already refused any value that introduced a marker, so the portion still
    # ends where it did going in.
    #
    # This is NOT an introduced-vs-already-there check: every entry with a
    # reference contains OPENER before substitution, so "OPENER in expanded and
    # OPENER not in entry" would be dead code. The invariant is that expansion
    # removes every opener uv would go on to act on.
    if OPENER in _code_portion(expanded):
        return f"expansion left an unsubstituted reference in {expanded!r}"
    return None


def expand_all(
    entries: Sequence[str],
    variables: Variables,
    *,
    sources: Sequence[str | None] | None = None,
) -> list[str]:
    """Substitute every reference in ``entries``.

    Three checks run in order over the whole sequence, so one error names every
    offender: placement first (a misplaced reference is a defect in the entry,
    not in the machine), then undeclared names, then declared names with no
    local value. The order matters for the message a user sees on a freshly
    cloned root: a typo in a profile should not read as a missing local value.

    All three checks and the substitution itself read the entry's code
    portion, the same view :func:`placement_problem` judges. An entry may
    therefore mention a name in its comment without declaring it: uv discards
    the comment, so the name is prose, and the alternative was an entry legal
    at placement and refused one step later.

    Substitution is a single pass; a value is opaque text, so a ``${...}``
    inside one is not itself a reference. The replacement is supplied as a
    function rather than a template so a backslash in a Windows path is not
    interpreted as a regex escape.

    :param entries: The entries as written.
    :param variables: The declared names and this machine's values.
    :param sources: The file each entry came from, positionally aligned with
        ``entries``, for the placement message. A caller flattening several
        files into one sequence passes it so the refusal still names them.
    :returns: The expanded entries, positionally aligned with ``entries``.
    :raises ConfigError: On a refused entry, an undeclared name, a declared
        name with no value on this machine, or a value whose substitution
        would change which options an entry carries.
    """
    check_placement(entries, sources=sources)

    undeclared: list[str] = []
    undefined: list[str] = []
    for entry in entries:
        for name in referenced_names(entry):
            if name not in variables.declared:
                if name not in undeclared:
                    undeclared.append(name)
            elif name not in variables.values and name not in undefined:
                undefined.append(name)

    if undeclared:
        names = ", ".join(undeclared)
        raise ConfigError(
            f"Requirement sources reference {len(undeclared)} name(s) that are "
            f"not declared by the config root: {names}.",
            hint=UNDECLARED_HINT,
        )
    if undefined:
        names = ", ".join(undefined)
        raise ConfigError(
            f"{len(undefined)} declared variable(s) have no value on this "
            f"machine: {names}.",
            hint=(
                "Set each one in variables.local.txt as 'NAME=value', or export "
                "it in the environment. Run 'stack doctor' to see every "
                "unsatisfied name at once."
            ),
        )

    expanded = [_expand_code(entry, variables.values) for entry in entries]

    # Placement was judged on the entries as written. This is the same
    # judgement applied to what the values actually produced, and it is the
    # only check that can see a value carrying requirements-file syntax of
    # its own.
    unsafe: list[str] = []
    for entry, result in zip(entries, expanded, strict=True):
        problem = expansion_problem(entry, result)
        if problem is not None:
            unsafe.append(f"  {entry!r} -> {result!r}: {problem}")
    if unsafe:
        noun = "entry" if len(unsafe) == 1 else "entries"
        raise ConfigError(
            f"Refused {len(unsafe)} expanded requirement {noun}:\n"
            + "\n".join(unsafe),
            hint=(
                "A variable's value fills in a path or an option value; it may "
                "not introduce an option, a comment, or a line continuation of "
                "its own. Check the values in variables.local.txt and in the "
                "environment."
            ),
        )
    return expanded
