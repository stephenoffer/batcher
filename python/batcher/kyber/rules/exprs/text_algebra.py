"""String structure: de-specializing the remaining regex calls, and composing substrings.

`exprs/text` turns a metacharacter-free `regexp_matches` into a substring search. This
module finishes that job for the two other regex entry points -- `regexp_replace_all`
and `regexp_split` -- and composes stacked `substr` calls into one.

The replace rewrite carries a trap worth naming, because the obvious version of it is
wrong. `regexp_replace` replaces the **first** match only, while the plain `replace`
replaces **every** occurrence; confirmed against the engine, where
`regexp_replace('abcabc', 'b', 'X')` is `'aXcabc'` and `replace` of the same is
`'aXcaXc'`. Only `regexp_replace_all` corresponds to `replace`, and only that one is
rewritten here. The single-match form has no plain equivalent and is left alone.

Substring composition needs the indexing convention pinned rather than assumed.
Batcher's `substr` is **1-based**, matching DuckDB exactly, including the quirk that
`substr(s, 0, 2)` yields one character because position zero consumes a slot. Two
stacked calls therefore compose to `start = s1 + s2 - 1` with the inner length
clipped, which was verified end to end: `substr(2, 3)` then `substr(2, 2)` equals
`substr(3, 2)` on the same input.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.text import plain_pattern
from batcher.kyber.rules.leaf_rewrite import rewrite_node
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_ir.func_nodes import StrFunc
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "compose_nested_substr",
    "regexp_replace_all_plain_to_replace",
    "regexp_split_plain_to_split",
]


def _plain_replacement(replacement: str | None) -> bool:
    """Whether a replacement string is literal text with no capture-group syntax.

    A `$1` or `\\1` in the replacement refers back to the pattern's groups, which the
    plain `replace` has no notion of. A pattern with no metacharacters cannot have
    groups, so this is belt-and-braces -- but it costs nothing and keeps the rule
    correct if the pattern test is ever loosened.
    """
    return replacement is not None and "$" not in replacement and "\\" not in replacement


def _replace_all(expr: Expr) -> Expr:
    if isinstance(expr, StrFunc) and expr.fn == "regexp_replace_all":
        body = plain_pattern(expr.pattern)
        if body is not None and _plain_replacement(expr.replacement):
            return StrFunc("replace", expr.input, pattern=body, replacement=expr.replacement)
    return expr


@rule(
    name="regexp_replace_all_plain_to_replace",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_replace_all,
)
def regexp_replace_all_plain_to_replace(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`regexp_replace_all(x, 'ab', 'y') -> replace(x, 'ab', 'y')` for a plain pattern.

    Both replace every occurrence, so the two agree value for value -- verified against
    the engine including the null and empty-string rows. What changes is that a regex
    automaton stepped over every character of every row becomes a substring scan.

    Only the `_all` variant qualifies. `regexp_replace` replaces the *first* match
    only, which the plain `replace` cannot express, and rewriting it would silently
    change every row containing two or more matches."""
    return rewrite_node(node, _replace_all)


def _split(expr: Expr) -> Expr:
    if isinstance(expr, StrFunc) and expr.fn == "regexp_split":
        body = plain_pattern(expr.pattern)
        if body is not None:
            return StrFunc("split", expr.input, pattern=body)
    return expr


@rule(
    name="regexp_split_plain_to_split",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_split,
)
def regexp_split_plain_to_split(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`regexp_split(x, '-') -> split(x, '-')` for a metacharacter-free separator.

    Splitting on a literal separator is the same partition either way, verified on the
    multi-field, no-match, null, and empty-string cases -- all four agree, down to the
    single-element list a non-matching row produces. The plain split scans for bytes
    instead of running an automaton, and the saving lands per row."""
    return rewrite_node(node, _split)


def _compose_substr(expr: Expr) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn == "substr"):
        return expr
    inner = expr.input
    if not (isinstance(inner, StrFunc) and inner.fn == "substr"):
        return expr
    outer_start, inner_start = expr.start, inner.start
    if outer_start is None or inner_start is None or outer_start < 1 or inner_start < 1:
        return expr
    start = inner_start + outer_start - 1
    if inner.length is None:
        length = expr.length
    else:
        # What the inner window still offers once the outer offset is applied.
        remaining = max(inner.length - (outer_start - 1), 0)
        length = remaining if expr.length is None else min(expr.length, remaining)
    return StrFunc("substr", inner.input, start=start, length=length)


@rule(
    name="compose_nested_substr",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_compose_substr,
)
def compose_nested_substr(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold two stacked `substr` calls into one.

    Batcher's `substr` is 1-based, matching DuckDB, so taking `substr(s2, l2)` of
    `substr(s1, l1)` starts at `s1 + s2 - 1` in the original string. The composed
    length is the inner window minus what the outer offset skipped, then clipped by the
    outer length and clamped at zero, so an outer window starting past the inner one's
    end yields the empty string rather than a negative length.

    Verified end to end: `substr(2, 3)` then `substr(2, 2)` produces exactly what
    `substr(3, 2)` produces on the same input.

    Restricted to starts of 1 or more. Position zero is a special case in this
    convention -- it consumes a slot without emitting a character -- and composing
    through it is not worth the subtlety when these arise from generated SQL that
    almost always uses 1-based offsets."""
    return rewrite_node(node, _compose_substr)
