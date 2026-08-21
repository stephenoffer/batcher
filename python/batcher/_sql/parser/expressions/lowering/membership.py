"""``IN``, ``BETWEEN`` and ``IS DISTINCT FROM`` — SQL's set and null-safe comparisons.

Each is defined in terms of comparisons the IR already has, and each has a null rule that
is the whole reason it needs a module rather than a table row: scalar ``IN`` is
three-valued, *row* ``IN`` is not, and ``IS DISTINCT FROM`` is total by construction.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher._sql.parser.expressions.lowering.nulls import null_boolean
from batcher.plan.expr_ir import Expr, coalesce

__all__ = ["between", "in_membership", "is_distinct_from"]


def in_membership(tr, node) -> Expr:
    items = node.expressions
    if node.args.get("query") is not None:
        # A membership test in *value* position — `SELECT x IN (SELECT …)`, or one buried in
        # a CASE — has no relation to join against, which is why this used to refuse. It does
        # not need one when the set is small: `subquery.core` already collects such a set and
        # hands it to `Expr.is_in`, which is three-valued, so the same machinery answers here
        # and returns an expression rather than a rewritten relation. A set past the inline cap
        # still refuses, because the mark join it would need lives at the WHERE level.
        from batcher._sql.parser.subquery.in_set import inline_in_subquery_values

        values = inline_in_subquery_values(tr, node)
        if values is None:
            raise NotImplementedError(
                "IN (subquery) must be handled at the WHERE level, not as a scalar"
            )
        return tr._scalar(node.this).is_in(values)
    if not items:
        raise NotImplementedError("IN requires an explicit value list")
    if isinstance(node.this, exp.Tuple):
        return _row_in(tr, node.this, items)
    target = tr._scalar(node.this)
    # A NULL in the list is not a comparable value: `x IN (a, NULL)` is TRUE when x = a
    # and NULL otherwise (never FALSE), which is exactly `(x IN (a)) OR NULL` under SQL's
    # three-valued OR. Comparing against it instead built `x = NULL`, and the untyped NULL
    # literal lowered as Int64 — so `g IN ('a', NULL)` on a text column died with
    # `Invalid comparison operation: Utf8 == Int64` rather than answering.
    values = [i for i in items if not isinstance(i, exp.Null)]
    if len(values) != len(items):
        if not values:
            # `x IN (NULL)` is NULL for every row, x included.
            return null_boolean()
        return _in_values(tr, target, values) | null_boolean()
    return _in_values(tr, target, values)


def _row_in(tr, row, items) -> Expr:
    """`(a, b) IN ((1, 2), (3, 4))` — row-wise membership.

    DuckDB compares the tuples element by element with **null-safe** equality, so the
    answer is never NULL: `(NULL, NULL) IN ((1, 10))` is FALSE and
    `(NULL, NULL) IN ((NULL, NULL))` is TRUE — unlike scalar `IN`, which is three-valued.
    This is therefore the disjunction over the candidates of the conjunction over the
    columns of `IS NOT DISTINCT FROM`. There is no row value in the IR, so building it
    from the comparisons it is defined as is the whole translation; without it the query
    died on "unsupported SQL expression: Tuple".

    Args:
        tr: The translator.
        row: The `Tuple` on the left of `IN`.
        items: The candidate tuples.

    Returns:
        The membership predicate.

    Raises:
        NotImplementedError: A candidate is not a tuple of the same width.
    """
    targets = [tr._scalar(e) for e in row.expressions]
    result: Expr | None = None
    for item in items:
        if not isinstance(item, exp.Tuple) or len(item.expressions) != len(targets):
            raise NotImplementedError(
                f"IN over a row of {len(targets)} columns needs candidates of the same width"
            )
        conj: Expr | None = None
        for target, value in zip(targets, item.expressions, strict=True):
            other = tr._scalar(value)
            eq = coalesce(target == other, target.is_null() & other.is_null())
            conj = eq if conj is None else (conj & eq)
        result = conj if result is None else (result | conj)
    return result


def _in_values(tr, target: Expr, items) -> Expr:
    # x IN (a, b, c)  →  (x == a) | (x == b) | (x == c)
    result: Expr | None = None
    for item in items:
        eq = target == tr._scalar(item)
        result = eq if result is None else (result | eq)
    return result


def between(tr, node) -> Expr:
    # x BETWEEN lo AND hi  →  (x >= lo) & (x <= hi)
    target = tr._scalar(node.this)
    low = tr._scalar(node.args["low"])
    high = tr._scalar(node.args["high"])
    return (target >= low) & (target <= high)


def is_distinct_from(tr, node) -> Expr:
    """`a IS DISTINCT FROM b` — null-safe inequality (NULL is a comparable
    value). Built as the negation of null-safe *equality* (both null, or both
    non-null and equal); that form is null-free (the `a == b` term is masked by
    `~an & ~bn`, so it never leaks a NULL into the boolean result).
    """
    a = tr._scalar(node.this)
    b = tr._scalar(node.expression)
    # `a == b` is NULL when either side is NULL; `coalesce` then falls back to
    # "are both NULL?" — giving a null-free null-safe-equality without relying
    # on Kleene `and`/`or` (which the engine does not implement).
    not_distinct = coalesce(a == b, a.is_null() & b.is_null())
    return ~not_distinct
