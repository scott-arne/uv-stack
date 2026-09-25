from __future__ import annotations

import pytest

from uv_stack.errors import ConfigError
from uv_stack.parse import ownership_name
from uv_stack.variables import (
    Variables,
    check_placement,
    expand_all,
    expansion_problem,
    placement_problem,
    referenced_names,
)


def _vars(**values: str) -> Variables:
    return Variables(declared=tuple(sorted(values)), values=dict(values))


def test_referenced_names_finds_each_name_once_in_order():
    assert referenced_names("-e ${DEV}/a ${OTHER} ${DEV}/b") == ["DEV", "OTHER"]


def test_referenced_names_ignores_a_bare_dollar():
    assert referenced_names("pkg @ https://host/$path") == []


def test_referenced_names_skips_a_comment():
    # uv discards the comment, so a name there is prose and not a reference.
    assert referenced_names("${DEV}/pkg.whl  # rebuilt from ${SRC}") == ["DEV"]


def test_reference_requires_braces():
    # Only '${' opens a reference; a bare '$DEV' is ordinary text.
    assert referenced_names("-e $DEV/pkg") == []


def test_undefined_lists_declared_names_without_values():
    variables = Variables(declared=("DEV", "WORK"), values={"DEV": "/d"})
    assert variables.undefined() == ["WORK"]


def test_undefined_is_empty_when_every_name_has_a_value():
    assert _vars(DEV="/d").undefined() == []

# (entry, the kind of refusal it earns). Condition 0 is single-line,
# 1 is ownership-free, 2 is head-qualified, 3 is not-a-recursive-include.
#: Admitted entries that actually carry a reference. Separated from the
#: literal-only rows below because the invariant tests only apply to these.
_ADMITTED_WITH_REFERENCES = [
    "-e ${DEV}/pkg",
    "--editable ${DEV}/pkg",
    "${DEV}/pkg",
    "--index-url ${HOST}/simple",
]

#: Entries whose only reference sits in a comment, plus one that carries a real
#: reference alongside a comment. Kept out of _ADMITTED_WITH_REFERENCES because
#: the invariant tests there describe a reference uv actually reads: the rows
#: below expand to text uv discards, and the second one legitimately names a
#: distribution ('pkg') that no value can change.
_ADMITTED_WITH_COMMENTS = [
    "-e ${DEV}/pkg # do not use -r here",
    "pkg # see ${DEV} for the path",
    "# install ${DEV}/pkg manually",
    "${DEV}/pkg # plain note",
]

_ADMITTED = [
    *_ADMITTED_WITH_REFERENCES,
    *_ADMITTED_WITH_COMMENTS,
    "numpy",
    "-e /absolute/path",
    "-r /abs/reqs.txt",
    "--index-url https://example.invalid/simple",
]

_REFUSED = [
    ("${PACKAGE}", "misplaced-reference"),
    ("${PACKAGE}>=2", "misplaced-reference"),
    ("${PACKAGE}[extra]", "misplaced-reference"),
    ('${PACKAGE}>=2 ; os_name != "n/a"', "misplaced-reference"),
    ("pkg@${DEV}/x.whl", "misplaced-reference"),
    ("pkg @ file://${DEV}/x.whl", "misplaced-reference"),
    ("-r ${DEV}/reqs.txt", "misplaced-reference"),
    ("-c ${DEV}/constraints.txt", "misplaced-reference"),
    ("-r${DEV}/reqs.txt", "misplaced-reference"),
    ("--requirement=${DEV}/reqs.txt", "misplaced-reference"),
    ("-e ${DEV}/safe\n-r ${DEV}/reqs.txt", "multiline-entry"),
    ("-e ${DEV/pkg", "malformed-reference"),
    # A '#' inside a token is ordinary text to uv, so this entry's reference is
    # code and condition 2 still judges it. Pinned here so the comment rule
    # above cannot be widened into a substring search.
    ("pkg#egg=thing ${DEV}", "misplaced-reference"),
]


@pytest.mark.parametrize("entry", _ADMITTED)
def test_admitted_entries_have_no_placement_problem(entry):
    assert placement_problem(entry) is None


@pytest.mark.parametrize("entry,kind", _REFUSED)
def test_refused_entries_report_their_kind(entry, kind):
    problem = placement_problem(entry)
    assert problem is not None
    assert problem[0] == kind


def test_ownership_position_is_the_only_failure_for_an_at_entry():
    # 'pkg@${DEV}/x.whl' names a distribution (condition 1) but its first token
    # does hold a path separator, so condition 2 is satisfied.
    _, explanation = placement_problem("pkg@${DEV}/x.whl")
    assert "condition 1" in explanation
    assert "condition 2" not in explanation


def test_a_marker_with_a_slash_does_not_hide_the_ownership_name():
    # The '/' inside the marker must not make ownership_name() return None,
    # which would let a plain requirement sneak past condition 1.
    problem = placement_problem('${PACKAGE};os_name!="a/b"')
    assert problem is not None
    _, explanation = problem
    assert "condition 1" in explanation


def test_marker_entry_violates_conditions_1_and_2():
    # The marker no longer hides the name, so this entry violates both
    # condition 1 (ownership position) and condition 2 (not head-qualified),
    # but still not condition 3.
    _, explanation = placement_problem('${PACKAGE}>=2 ; os_name != "n/a"')
    assert "condition 1" in explanation
    assert "condition 2" in explanation
    assert "condition 3" not in explanation


@pytest.mark.parametrize(
    "entry",
    [
        "-r ${DEV}/reqs.txt",
        "-c ${DEV}/constraints.txt",
        "-r${DEV}/reqs.txt",
        "--requirement=${DEV}/reqs.txt",
    ],
)
def test_recursive_includes_fail_only_the_include_condition(entry):
    _, explanation = placement_problem(entry)
    assert "condition 3" in explanation
    assert "condition 1" not in explanation
    assert "condition 2" not in explanation


@pytest.mark.parametrize(
    "entry",
    [
        "--no-index -r ${DEV}/reqs.txt",
        "--no-index -c ${DEV}/constraints.txt",
        "--no-index -r${DEV}/reqs.txt",
        "--no-index --requirement=${DEV}/reqs.txt",
        "--index-url https://example.invalid/simple -r ${DEV}/reqs.txt",
        "--pre --no-index -r ${DEV}/reqs.txt",
    ],
)
def test_a_recursive_include_is_caught_behind_a_preceding_option(entry):
    # Condition 3 is judged over every token, not just the first: uv reads
    # '--no-index -r <file>' as a boolean option followed by a real include
    # and does read that file, so a first-token-only test admits exactly the
    # machine-local second requirements file the condition exists to refuse.
    problem = placement_problem(entry)
    assert problem is not None, entry
    assert "condition 3" in problem[1]


def test_a_recursive_option_inside_a_comment_is_not_an_include():
    # The whole-sequence scan above must not reach past the comment marker: uv
    # discards everything from '#' to the end of the line, so this entry opens
    # no second requirements file at all.
    assert placement_problem("-e ${DEV}/pkg # do not use -r here") is None


def test_a_hash_inside_a_token_does_not_start_a_comment():
    # The counterexample that fixes the comment rule at a token boundary. A
    # '#' in mid-token is ordinary text to uv -- 'pkg#egg=thing' is one
    # requirement -- so the reference after it is code and stays refused.
    problem = placement_problem("pkg#egg=thing ${DEV}")
    assert problem is not None
    assert problem[0] == "misplaced-reference"


def test_a_comment_does_not_rescue_a_trailing_backslash():
    # Continuation is judged on the raw entry on purpose: the backslash ends
    # the physical line whether or not a comment precedes it, so a code-portion
    # reading would admit an entry that swallows the next requirement.
    kind, _ = placement_problem("-e ${DEV}/pkg # note \\")
    assert kind == "continuation-entry"


def test_a_malformed_reference_inside_a_comment_is_still_named():
    # The other deliberate asymmetry. uv would never read this text, so the
    # typo cannot hurt it; naming it anyway is a judgment call about which
    # mistake a user would rather hear about.
    kind, _ = placement_problem("pkg # see ${DEV for the path")
    assert kind == "malformed-reference"


def test_multiline_is_refused_even_without_a_reference():
    problem = placement_problem("numpy\npandas")
    assert problem is not None
    assert problem[0] == "multiline-entry"


def test_carriage_return_counts_as_multiline():
    assert placement_problem("numpy\rpandas")[0] == "multiline-entry"


def test_trailing_backslash_is_refused_at_placement():
    # An entry ending in a backslash continues onto the next line and swallows
    # the requirement after it. The bare shape is the one uv actually joins, so
    # it is asserted directly; a raw string cannot end in a backslash, which is
    # why the other cases here are written with trailing whitespace instead.
    kind, explanation = placement_problem("-e ${DEV}\\")
    assert kind == "continuation-entry"
    assert "backslash" in explanation
    # Trailing whitespace does not rescue it: the readers strip lines, so the
    # backslash still lands at the end of the rendered line.
    assert placement_problem(r"-e ${DEV}\ ")[0] == "continuation-entry"


def test_reference_free_entry_ending_in_backslash_is_refused():
    # The continuation check applies to all entries, not just those with
    # references.
    kind, _ = placement_problem(r"numpy\ ")
    assert kind == "continuation-entry"


def test_check_placement_refuses_trailing_backslash():
    with pytest.raises(ConfigError) as excinfo:
        check_placement([r"-e /some/path\ "])
    assert "backslash" in excinfo.value.message


def test_malformed_reference_is_reported_before_placement():
    # This entry is BOTH malformed and in the ownership position; malformed wins.
    kind, explanation = placement_problem("${DEV")
    assert kind == "malformed-reference"
    assert "${DEV" in explanation


def test_check_placement_reports_every_offender_at_once():
    with pytest.raises(ConfigError) as excinfo:
        check_placement(["${A}", "-r ${B}/x.txt", "numpy"], source="profiles/ds.yaml")
    message = excinfo.value.message
    assert "profiles/ds.yaml" in message
    assert "${A}" in message
    assert "-r ${B}/x.txt" in message
    assert "numpy" not in message


def test_check_placement_accepts_admitted_entries():
    check_placement(_ADMITTED)


@pytest.mark.parametrize("entry", _ADMITTED_WITH_REFERENCES)
def test_no_admitted_entry_carries_an_ownership_name(entry):
    # The property the whole placement rule exists to guarantee: expansion can
    # never change what a requirement is *named*, because nothing admitted has
    # a name for it to change.
    expanded = entry.replace("${DEV}", "/home/me/dev").replace(
        "${HOST}", "https://pypi.invalid"
    )
    assert ownership_name(entry) is None
    assert ownership_name(expanded) is None


def test_an_entry_is_judged_as_a_whole_not_per_reference():
    # Two references in an admitted entry keep it admitted; the same two in an
    # ownership position make the whole entry illegal. Legality is a property
    # of the entry, never of an individual reference.
    assert placement_problem("-e ${DEV}/${NAME}") is None
    assert placement_problem("${NAME}@${DEV}/x.whl") is not None


def test_expand_all_substitutes_every_reference():
    variables = _vars(DEV="/home/me/dev", HOST="https://pypi.invalid")
    assert expand_all(["-e ${DEV}/pkg", "--index-url ${HOST}/simple"], variables) == [
        "-e /home/me/dev/pkg",
        "--index-url https://pypi.invalid/simple",
    ]


def test_expand_all_leaves_entries_without_references_alone():
    assert expand_all(["numpy", "pandas>=2"], _vars(DEV="/d")) == ["numpy", "pandas>=2"]


def test_a_value_is_substituted_once_and_is_not_rescanned():
    # OTHER is declared and has a value, so a second pass would resolve the
    # '${OTHER}' that DEV's value contains. The refusal quotes the reference
    # verbatim and never mentions '/zzz', which is what proves the single pass.
    variables = Variables(
        declared=("DEV", "OTHER"), values={"DEV": "/a/${OTHER}/b", "OTHER": "/zzz"}
    )
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${DEV}/pkg"], variables)
    assert "${OTHER}" in excinfo.value.message
    assert "/zzz" not in excinfo.value.message


def test_a_backslash_in_a_value_survives_substitution():
    variables = _vars(DEV=r"C:\dev")
    assert expand_all([r"-e ${DEV}\pkg"], variables) == [r"-e C:\dev\pkg"]


def test_a_reference_only_in_a_comment_needs_no_declaration():
    """Placement admits the entry, so expansion must not then refuse it.

    The two judged different views of the same line: placement read the code
    portion and expansion read the whole of it, so an entry could be legal at
    one stage and illegal at the next over a name that is only ever prose.
    """
    entry = "numpy  # see ${DEV} for the path"
    assert placement_problem(entry) is None
    assert expand_all([entry], _vars()) == [entry]


def test_expansion_leaves_a_comment_byte_for_byte():
    # uv discards the comment, so there is nothing in it to substitute -- and
    # the run of spaces proves the tail is carried over rather than rebuilt.
    entry = "--find-links ${DEV}/wheels  #  built   in ${DEV}  "
    assert expand_all([entry], _vars(DEV="/opt/d")) == [
        "--find-links /opt/d/wheels  #  built   in ${DEV}  "
    ]


def test_a_reference_in_the_code_portion_still_needs_declaration():
    # The negative control for the two above: comment-awareness must not have
    # turned the declaration requirement off for the part uv actually reads.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["${DEV}/pkg.whl"], _vars())
    assert "DEV" in excinfo.value.message


def test_expand_all_refuses_a_misplaced_reference_before_checking_names():
    # Placement runs first: the message is about placement, not about 'NOPE'
    # being undeclared.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["${NOPE}"], _vars(DEV="/d"))
    assert "condition" in excinfo.value.message


def test_expand_all_refuses_an_undeclared_name():
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${NOPE}/pkg"], _vars(DEV="/d"))
    assert "NOPE" in excinfo.value.message
    assert "not declared" in excinfo.value.message


def test_a_declared_name_with_no_value_is_a_different_error():
    variables = Variables(declared=("DEV", "WORK"), values={"DEV": "/d"})
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${WORK}/pkg"], variables)
    assert "WORK" in excinfo.value.message
    assert "no value on this machine" in excinfo.value.message


def test_undeclared_is_reported_before_undefined():
    variables = Variables(declared=("WORK",), values={})
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${NOPE}/a", "-e ${WORK}/b"], variables)
    assert "not declared" in excinfo.value.message


def test_every_undeclared_name_is_reported_at_once():
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${A}/x", "-e ${B}/y"], _vars(DEV="/d"))
    assert "A" in excinfo.value.message
    assert "B" in excinfo.value.message


@pytest.mark.parametrize("entry", _ADMITTED_WITH_REFERENCES)
def test_expansion_preserves_the_token_count(entry):
    # A value holding embedded whitespace would split one token into two and
    # turn an editable path into a second, unintended requirement. Task 2's
    # refusal of whitespace in a value keeps this invariant true for the
    # embedded case; expansion_problem's newline refusal covers the
    # whitespace-only case that preserves token count but spans multiple lines.
    expanded = expand_all([entry], _vars(DEV="/home/me/dev", HOST="https://h"))[0]
    assert len(expanded.split()) == len(entry.split())


@pytest.mark.parametrize(
    "entry", ["-e ${DEV}/pkg", "--editable ${DEV}/pkg", "--index-url ${HOST}/simple"]
)
def test_expansion_never_rewrites_a_leading_option(entry):
    expanded = expand_all([entry], _vars(DEV="/home/me/dev", HOST="https://h"))[0]
    assert expanded.split()[0] == entry.split()[0]


def test_a_value_may_not_smuggle_in_a_recursive_include():
    # The counterexample the placement rule alone cannot see: '${DEV}/deps.txt'
    # is an admitted path operand, and 'DEV=-r' is a legal whitespace-free
    # value, but together they are the attached include '-r/deps.txt'.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["${DEV}/deps.txt"], _vars(DEV="-r"))
    assert "-r/deps.txt" in excinfo.value.message
    assert "different option" in excinfo.value.message


@pytest.mark.parametrize("value", ["-r", "-c", "--requirement=", "-e", "--index-url="])
def test_no_value_may_introduce_an_option(value):
    with pytest.raises(ConfigError):
        expand_all(["${DEV}/pkg"], _vars(DEV=value))


def test_a_value_may_complete_an_option_it_did_not_write():
    # The only shape that reaches the option-class branch through a token that
    # already looks like an option. Placement admits '-${OPT}' on condition 2
    # because the token starts with '-', and a reference past the first two
    # characters can never change the '-x' prefix _option_class reads -- so
    # completing an option is reachable where replacing one is not. Writing
    # this as '${OPT} /some/pkg' with OPT='-e' does not test expansion at all:
    # check_placement refuses the entry on condition 2 before a value is ever
    # substituted.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-${OPT} /some/pkg"], _vars(OPT="e"))
    assert "different option" in excinfo.value.message


def test_an_empty_value_that_drops_a_token_is_refused():
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${DEV}"], _vars(DEV=""))
    assert "token(s) to 1" in excinfo.value.message


@pytest.mark.parametrize("entry", _ADMITTED_WITH_REFERENCES)
def test_an_ordinary_path_value_is_not_refused(entry):
    # The validator must not fire on the legitimate case: a value holding a
    # hyphen mid-path is a path, not an option.
    expand_all([entry], _vars(DEV="/home/me/my-dev", HOST="https://my-host"))


def test_a_value_may_not_comment_out_the_entry():
    # uv reads a requirements file with pip's grammar, where '#' opens a
    # comment at the start of a line or after whitespace -- exactly a token
    # that begins with one. '#' is a legal value: the file grammar strips it
    # as a comment, but the environment has no grammar, which
    # test_an_environment_value_may_contain_a_hash pins. Expanded here it is
    # the difference between a requirement and a line uv never reads.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["${DEV}/pkg"], _vars(DEV="#"))
    assert "starts a comment" in excinfo.value.message


def test_a_value_may_not_comment_out_an_options_value():
    # The same defect one token later: the option survives the truncation and
    # its value does not, so uv sees a bare '--index-url'.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["--index-url ${HOST}/simple"], _vars(HOST="#"))
    assert "starts a comment" in excinfo.value.message


def test_a_hash_inside_a_token_is_not_a_comment():
    # The admission case the refusal must not swallow: a comment opens only at
    # a token boundary, so a '#' mid-path reaches uv intact.
    assert expand_all(["-e ${DEV}/pkg"], _vars(DEV="/home/me/c#sharp")) == [
        "-e /home/me/c#sharp/pkg"
    ]


def test_a_value_may_not_continue_the_entry_onto_the_next_line():
    # A trailing backslash joins the next physical line, so this entry would
    # swallow whichever requirement the render writes after it -- silently,
    # and differently depending on how the entries happen to be ordered.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${DEV}"], _vars(DEV="\\"))
    assert "backslash" in excinfo.value.message


def test_a_value_may_not_introduce_a_newline():
    # A pure-whitespace value preserves the token count but splits the entry
    # across multiple lines, which is the multiline-entry shape placement
    # refuses at condition 0.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${DEV}/pkg"], _vars(DEV="\n"))
    assert "newline" in excinfo.value.message


def test_a_value_may_not_introduce_or_change_the_ownership_name():
    # A value can introduce an ownership name where there was none, turning
    # an admitted path into a direct reference whose distribution name came
    # from a machine-local value.
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["${ROOT}/files.example/pkg.whl"], _vars(ROOT="victim@https:/"))
    assert "ownership" in excinfo.value.message


def test_a_value_may_not_leave_a_reference_behind():
    # Substitution is single-pass and values are opaque, so a nested reference
    # survives into the expanded result where uv would expand it from the
    # environment, bypassing this module's checks.
    variables = Variables(
        declared=("DEV", "OTHER"), values={"DEV": "/a/${OTHER}/b", "OTHER": "/zzz"}
    )
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${DEV}/pkg"], variables)
    assert "unsubstituted" in excinfo.value.message


def test_a_comment_does_not_excuse_a_reference_left_in_the_code():
    # The guard reads the code portion so a ${...} in the comment is not
    # mistaken for residue; this is the other half of that, proving the
    # exemption did not extend to the part uv actually reads.
    variables = Variables(
        declared=("DEV", "OTHER"), values={"DEV": "/a/${OTHER}/b", "OTHER": "/zzz"}
    )
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${DEV}/pkg  # nested"], variables)
    assert "unsubstituted" in excinfo.value.message


def test_trailing_whitespace_does_not_hide_an_introduced_continuation():
    # The entry survives placement because its own tail is not a backslash, so
    # the expansion guard is the only thing standing between this value and a
    # line that swallows the next requirement. It compares stripped tails for
    # the same reason placement does: the rendered line is stripped, so the
    # trailing space disappears and the backslash ends the line.
    variables = Variables(declared=("DEV",), values={"DEV": "\\"})
    assert placement_problem("-e ${DEV} ") is None
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["-e ${DEV} "], variables)
    assert "backslash" in excinfo.value.message


def test_a_value_may_append_a_semicolon_to_a_path_entry():
    # Measured against uv, not assumed: uv honours an environment marker only on
    # a NAME-shaped requirement ('requests;os_name=="never"' resolves to nothing,
    # silently). On a path-shaped one it folds the ';' into the path instead --
    # './pkg;os_name=="never"' and '-e /abs/pkg;os_name=="never"' both fail with
    # 'Distribution not found at: file://...;os_name==%22never%22'. So a value
    # that appends a marker to a path cannot suppress a dependency; it only picks
    # a different path, which is what values are for.
    variables = Variables(declared=("TAIL",), values={"TAIL": ';os_name=="never"'})
    assert expand_all(["./${TAIL}"], variables) == ['./;os_name=="never"']
    # The name-shaped case, where uv would honour the marker, is unreachable: it
    # is exactly the case ownership_name() claims, so the ownership comparison
    # refuses it. The two notions of "is this a name" coincide.
    variables = Variables(declared=("T",), values={"T": 'victim;os_name=="never"'})
    with pytest.raises(ConfigError) as excinfo:
        expand_all(["${T}/../pkg"], variables)
    assert "ownership name" in excinfo.value.message


def test_expansion_problem_is_none_for_a_safe_substitution():
    assert expansion_problem("-e ${DEV}/pkg", "-e /home/me/dev/pkg") is None
    assert expansion_problem("${DEV}/pkg", "/home/me/dev/pkg") is None
    # An entry written as a comment stays one: expansion introduced nothing,
    # and placement already had its chance to refuse the entry.
    assert expansion_problem("#${DEV}/pkg", "#/home/me/dev/pkg") is None
    # Placement refuses entries ending in a backslash, so expansion_problem
    # deliberately does not re-judge them.
    assert expansion_problem("${DEV}\\", "/home/me/dev\\") is None


def test_check_placement_names_each_offenders_own_file():
    """One error still lists every offender, and now says where each one lives.

    The flattened call in render_requirements_in exists so a user sees the
    whole list rather than one item per run; without a per-entry source that
    list names entries and no files.
    """
    with pytest.raises(ConfigError) as excinfo:
        check_placement(
            ["-r ${DEV}/a.txt", "good-pkg", "${DEV}"],
            sources=["/cfg/profiles/ds.yaml", None, "/cfg/profiles/chem.yaml"],
        )
    message = excinfo.value.message
    assert "Refused 2 requirement entries" in message
    assert "'-r ${DEV}/a.txt' (in /cfg/profiles/ds.yaml):" in message
    assert "'${DEV}' (in /cfg/profiles/chem.yaml):" in message


def test_check_placement_refuses_misaligned_sources():
    """A short sources list would report the last entries with no file.

    The ValueError is deliberate: zip would truncate silently, and a caller
    would claim the feature worked while showing no file for entries that do
    have one. Better to fail loudly than to lie.
    """
    with pytest.raises(ValueError) as excinfo:
        check_placement(["pkg-a", "pkg-b", "pkg-c"], sources=["a.yaml", "b.yaml"])
    assert "sources has 2 items for 3 entries" in str(excinfo.value)


def test_check_placement_refuses_both_source_and_sources():
    """Passing both source and sources would give contradictory blame.

    The header composes from source and each offender line from sources, so
    accepting both would name two different files for one refusal. The source
    docstring already tells multi-file callers to use sources instead.
    """
    with pytest.raises(ValueError) as excinfo:
        check_placement(["pkg"], source="whole.yaml", sources=["entry.yaml"])
    assert "both source and sources" in str(excinfo.value)
