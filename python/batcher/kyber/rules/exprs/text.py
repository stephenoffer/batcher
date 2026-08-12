"""Regex de-specialization and the remaining string identities.

`extra/strings` de-specializes `LIKE` into anchored searches and folds a string
function over a literal. This module does the same job for `regexp_matches`, which is
where the larger win is: a `LIKE` pattern compiles to a regex the engine can often
recognize, but a hand-written regex goes straight to the regex engine, and a
backtracking match against every row of a string column is one of the most expensive
per-row operations the data plane performs. Recognizing that a pattern has no
metacharacters and turning it into `contains`, `starts_with`, `ends_with`, or `=`
replaces that with a memchr-class substring scan.

The metacharacter test is the whole soundness argument, so it is deliberately
paranoid: a pattern qualifies only when every character is alphanumeric, a space, or
one of a tiny punctuation whitelist. Anything outside that -- a backslash, a bracket,
a quantifier, a dot, a Unicode category escape -- declines. Being wrong in the
permissive direction silently changes which rows match, and no test that does not
happen to use that character would catch it.

The identities that close the module (`reverse(reverse(x))`, `repeat(x, 1)`,
`translate(x, '', ...)`, a full-range `substr`) were each confirmed against the engine
on the null row as well as the value rows. Every one of them preserves null, which is
what lets them apply inside a `Project` rather than only at the top of a `Filter`.
"""

from __future__ import annotations

import string as _string

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.guards import is_string, schema_rule
from batcher.kyber.rules.leaf_rewrite import collapse_involution, rewrite_node
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_ir.func_nodes import StrFunc
from batcher.plan.logical import Filter, LogicalPlan, Project
from batcher.plan.schema import SchemaRef

__all__ = [
    "drop_full_substr",
    "drop_translate_with_empty_source",
    "plain_pattern",
    "regexp_anchored_to_equality",
    "regexp_plain_to_contains",
    "regexp_prefix_to_starts_with",
    "regexp_suffix_to_ends_with",
    "repeat_once_to_input",
    "str_reverse_involution",
]

#: The characters a regex pattern may contain and still be a plain substring. Every
#: regex metacharacter is excluded, and so is the backslash, which introduces an
#: escape whose meaning this rule does not model. Deliberately a whitelist: a
#: blacklist silently admits whatever it forgot.
_PLAIN_REGEX_CHARS = frozenset(_string.ascii_letters + _string.digits + " _-/:;,@#%&=~'\"<>!`")


def plain_pattern(pattern: str | None) -> str | None:
    """The literal substring a regex matches, or ``None`` if it is not plain text.

    Shared with `exprs/text_algebra`, which de-specializes the replace and split entry
    points against the same whitelist. Public so that test is written once.
    """
    if pattern is None or not pattern:
        return None
    return pattern if all(c in _PLAIN_REGEX_CHARS for c in pattern) else None


def _regexp_body(pattern: str | None, *, prefix: bool, suffix: bool) -> str | None:
    """Strip the requested anchors off `pattern` and return the plain body inside.

    Returns ``None`` unless the pattern is anchored exactly as asked and everything
    between the anchors is metacharacter-free.
    """
    if pattern is None:
        return None
    body = pattern
    if prefix:
        if not body.startswith("^"):
            return None
        body = body[1:]
    elif body.startswith("^"):
        return None
    if suffix:
        if not body.endswith("$"):
            return None
        body = body[:-1]
    elif body.endswith("$"):
        return None
    return plain_pattern(body)


def _string_typed(node: LogicalPlan, leaf) -> LogicalPlan | None:
    """Lift a leaf rewrite that needs to know whether its operand is a UTF-8 string.

    A string function coerces a `Binary` column to `Utf8`, so a rule that removes the
    last one around a value would change the output type unless the value is already
    a string. Declines when the schema cannot answer.

    Delegates to `guards.schema_rule` with `StrFunc` as the carried type, so a node with
    no string call in it costs one `isinstance` sweep and never resolves a schema.
    """
    return schema_rule(node, leaf, carries=(StrFunc,))


# --- regex de-specialization ---------------------------------------------------


def _regexp_contains(expr: Expr) -> Expr:
    if isinstance(expr, StrFunc) and expr.fn == "regexp_matches":
        body = _regexp_body(expr.pattern, prefix=False, suffix=False)
        if body is not None:
            return StrFunc("contains", expr.input, pattern=body)
    return expr


@rule(
    name="regexp_plain_to_contains",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_regexp_contains,
    expr_matches=(StrFunc,),
)
def regexp_plain_to_contains(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`regexp_matches(x, 'abc') -> contains(x, 'abc')` when the pattern is plain text.

    An unanchored regex over a plain substring is exactly a substring search, and the
    two agree on null too -- both propagate it, as the engine confirms. What changes is
    the cost: a regex automaton stepped over every character of every row becomes a
    substring scan, the single largest per-row saving in this module.

    The pattern must contain no metacharacter at all, judged by whitelist. `_plain_pattern`
    also rejects the empty pattern, which matches every row and is a different rewrite."""
    return rewrite_node(node, _regexp_contains)


def _regexp_starts(expr: Expr) -> Expr:
    if isinstance(expr, StrFunc) and expr.fn == "regexp_matches":
        body = _regexp_body(expr.pattern, prefix=True, suffix=False)
        if body is not None:
            return StrFunc("starts_with", expr.input, pattern=body)
    return expr


@rule(
    name="regexp_prefix_to_starts_with",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_regexp_starts,
    expr_matches=(StrFunc,),
)
def regexp_prefix_to_starts_with(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`regexp_matches(x, '^abc') -> starts_with(x, 'abc')`.

    A start-anchored plain pattern is a prefix test, which compares at most as many
    bytes as the prefix is long instead of scanning the row. It also feeds the
    existing sargable normalizer, which turns a prefix test into a half-open range and
    hands zone-map pruning something it can act on -- so this can eliminate whole row
    groups, not just cycles."""
    return rewrite_node(node, _regexp_starts)


def _regexp_ends(expr: Expr) -> Expr:
    if isinstance(expr, StrFunc) and expr.fn == "regexp_matches":
        body = _regexp_body(expr.pattern, prefix=False, suffix=True)
        if body is not None:
            return StrFunc("ends_with", expr.input, pattern=body)
    return expr


@rule(
    name="regexp_suffix_to_ends_with",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_regexp_ends,
    expr_matches=(StrFunc,),
)
def regexp_suffix_to_ends_with(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`regexp_matches(x, 'abc$') -> ends_with(x, 'abc')`. An end-anchored plain
    pattern is a suffix test, decided by comparing the tail rather than by running an
    automaton forward over the whole string."""
    return rewrite_node(node, _regexp_ends)


def _regexp_equality(expr: Expr) -> Expr:
    if isinstance(expr, StrFunc) and expr.fn == "regexp_matches":
        body = _regexp_body(expr.pattern, prefix=True, suffix=True)
        if body is not None:
            return Binary("eq", expr.input, Lit(body))
    return expr


@rule(
    name="regexp_anchored_to_equality",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_regexp_equality,
    expr_matches=(StrFunc,),
)
def regexp_anchored_to_equality(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`regexp_matches(x, '^abc$') -> x = 'abc'`. A fully anchored plain pattern
    matches one string and nothing else.

    This is the most valuable member of the family by some distance. Equality against
    a literal is the shape every downstream consumer understands: zone maps prune row
    groups by it, the runtime-filter rules build a Bloom probe from it, join-key
    inference propagates it across an equi-join, and constant propagation substitutes
    it into the rest of the predicate. None of those can see anything through a regex
    call. Null still propagates: `NULL = 'abc'` is `NULL`, exactly as the match was."""
    return rewrite_node(node, _regexp_equality)


# --- string identities ---------------------------------------------------------


#: `reverse(reverse(s))` over a string, through the shared involution factory. `_reverse_leaf`
#: below adds the coercion guard this family needs and lists cannot: dropping both calls also
#: drops a Binary-to-Utf8 coercion the result may still depend on.
_reverse_involution = collapse_involution(StrFunc, "reverse")


def _reverse_leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
    out = _reverse_involution(expr)
    # Both calls go away, so the value must already be Utf8 -- otherwise the pair was
    # also performing a Binary-to-Utf8 coercion the result still needs.
    return out if out is expr or is_string(out, schema) else expr


@rule(
    name="str_reverse_involution",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_matches=(StrFunc,),
    expr_ops=("reverse",),
    expr_schema=_reverse_leaf,
)
def str_reverse_involution(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`reverse(reverse(x)) -> x`. String reversal is an involution over the code
    points, so a doubled call is the identity, null row included.

    Guarded on `x` already being a UTF-8 string. A string function coerces a `Binary`
    column to `Utf8`, and this rule removes *both* calls -- so on a binary column the
    pair is doing type work the bare column would not, and the rewrite declines."""
    return _string_typed(node, _reverse_leaf)


def _repeat_once_leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, StrFunc)
        and expr.fn == "repeat"
        and expr.length == 1
        and is_string(expr.input, schema)
    ):
        return expr.input
    return expr


@rule(
    name="repeat_once_to_input",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_schema=_repeat_once_leaf,
    expr_matches=(StrFunc,),
)
def repeat_once_to_input(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`repeat(x, 1) -> x`. One copy of a string is the string, and the engine agrees
    on the null row (both sides yield null) and the empty one. Removing the call
    avoids allocating a fresh string buffer per row.

    A repeat count of zero is deliberately *not* folded to the empty string: the
    engine returns null for a null input there, and this IR has no null literal to
    express the result with, so a constant fold would be wrong on those rows.

    Utf8-guarded because this removes the last string function around the value."""
    return _string_typed(node, _repeat_once_leaf)


def _full_substr(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, StrFunc)
        and expr.fn == "substr"
        and (expr.start is None or expr.start in (0, 1))
        and expr.length is None
        and is_string(expr.input, schema)
    ):
        return expr.input
    return expr


@rule(
    name="drop_full_substr",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_schema=_full_substr,
    expr_matches=(StrFunc,),
)
def drop_full_substr(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`substr(x, 0)` or `substr(x, 1)` with no length `-> x`.

    `substr` is 1-based here (matching DuckDB), so position 1 is the first character
    and an unbounded substring from there is the whole string. Position 0 behaves the
    same way when no length is given -- both were checked against the engine. Either
    way the call copies every row to reproduce its input.

    Utf8-guarded, since dropping it also drops a `Binary`-to-`Utf8` coercion."""
    return _string_typed(node, _full_substr)


def _empty_translate(expr: Expr, schema: SchemaRef | None) -> Expr:
    if (
        isinstance(expr, StrFunc)
        and expr.fn == "translate"
        and expr.pattern == ""
        and is_string(expr.input, schema)
    ):
        return expr.input
    return expr


@rule(
    name="drop_translate_with_empty_source",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr_schema=_empty_translate,
    expr_matches=(StrFunc,),
)
def drop_translate_with_empty_source(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`translate(x, '', y) -> x`. With no source characters there is nothing to map,
    so every character passes through unchanged -- confirmed against the engine, null
    row included. Utf8-guarded like the rest of this group."""
    return _string_typed(node, _empty_translate)
