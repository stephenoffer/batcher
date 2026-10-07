"""The plan-time regex preflight and the capture-group scan behind `extract_groups`.

The engine's regex is the Rust `regex` crate, which has no lookaround, no backreferences
and no atomic groups. A pattern using one used to build a plan that failed only when the
scan started. These pin that it is refused when the expression is *built* -- before any
data source is touched -- and that the scan does not refuse what the engine accepts.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.namespaces._dialect import check_regex, regex_group_names

pytestmark = pytest.mark.unit

_REFUSED = [
    (r"a(?=b)", "lookahead"),
    (r"a(?!b)", "negative lookahead"),
    (r"(?<=a)b", "lookbehind"),
    (r"(?<!a)b", "negative lookbehind"),
    (r"(?>ab)", "atomic group"),
    (r"(a)\1", "backreference"),
    (r"(?P<n>a)\k<n>", "backreference"),
]

_ACCEPTED = [
    r"\(?=",  # an escaped parenthesis is a literal
    r"[(?=]",  # so is one inside a character class
    r"[]()?=]",  # a leading `]` does not close the class
    r"\\1",  # an escaped backslash followed by a digit
    r"(?<name>a)",  # a named group, not a lookbehind
    r"(?P<name>a)(?i:b)(?:c)",
    r"\d{3}-\p{L}+",
]

#: Every `.str` method that lowers a user pattern to a regex.
_METHODS = [
    lambda s, p: s.regexp_matches(p),
    lambda s, p: s.contains(p, literal=False),
    lambda s, p: s.match(p),
    lambda s, p: s.extract(p),
    lambda s, p: s.extract_all(p),
    lambda s, p: s.extract_groups(p),
    lambda s, p: s.regexp_split(p),
    lambda s, p: s.count_matches(p),
    lambda s, p: s.regexp_replace(p, "x"),
    lambda s, p: s.replace_all(p, "x"),
]


@pytest.mark.parametrize(("pattern", "construct"), _REFUSED)
def test_an_unsupported_construct_is_refused_with_its_name(pattern, construct):
    with pytest.raises(PlanError, match=construct):
        check_regex(pattern, "regexp_matches")


@pytest.mark.parametrize("pattern", _ACCEPTED)
def test_a_supported_pattern_passes_unchanged(pattern):
    assert check_regex(pattern, "regexp_matches") == pattern


@pytest.mark.parametrize("build", _METHODS)
def test_every_regex_method_refuses_at_build_time(build):
    with pytest.raises(PlanError, match="lookahead"):
        build(bt.col("s").str, r"(x)(?=y)")


@pytest.mark.parametrize("build", _METHODS)
def test_every_regex_method_accepts_a_supported_pattern(build):
    assert build(bt.col("s").str, r"(x)y").to_ir()["e"] == "str"


def test_a_literal_pattern_is_not_preflighted():
    """`count_matches(literal=True)` escapes its text, so `(?=` is just characters."""
    expr = bt.col("s").str.count_matches("(?=", literal=True)
    assert expr.pattern == r"\(\?="


def test_the_message_names_an_alternative():
    with pytest.raises(PlanError, match=r"Instead, .*extract_groups"):
        bt.col("s").str.regexp_matches(r"(a)\1")


@pytest.mark.parametrize(
    ("pattern", "names"),
    [
        (r"abc", []),
        (r"(a)(b)", ["1", "2"]),
        (r"(?P<k>a)(b)(?<v>c)", ["k", "2", "v"]),
        (r"(?:a)(?i)(?i:b)(c)", ["1"]),
        (r"\((a)\)", ["1"]),
        (r"[(](a)", ["1"]),
        (r"[[:alpha:](](a)", ["1"]),
    ],
)
def test_group_names_follow_the_engines_numbering(pattern, names):
    assert regex_group_names(pattern) == names
