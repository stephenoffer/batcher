"""Quantified comparison predicates — ``x = ANY (SELECT ...)`` and ``x <> ALL (...)``.

SQL's quantified comparisons say "compare against *every* row the subquery returns, and
combine the results with OR (``ANY``/``SOME``) or AND (``ALL``)". Two of the forms are
exactly the set-membership predicates the translator already decorrelates:

    x =  ANY (S)   is   x IN (S)
    x <> ALL (S)   is   x NOT IN (S)

which is the definition, not an approximation — ``x = ANY (S)`` is true when some row of S
equals x, and that is what ``IN`` means. Rewriting them here means they arrive at
`core._apply_in_subquery` as ordinary `IN`/`NOT IN` and inherit everything it already gets
right: the semi/anti join, the multi-column row form, and — for ``NOT IN`` — the
three-valued logic that a NULL anywhere in S makes the whole predicate un-true
(`core._not_in_antijoin`).

The **inequality** forms (``> ANY``, ``>= ALL``, …) are deliberately not rewritten. The
tempting rewrite is ``x > ANY (S)`` → ``x > (SELECT min(c) FROM S)``, and for ``ANY`` it is
even correct. For ``ALL`` it is not: ``min``/``max`` skip NULLs, so ``x > ALL (S)`` over an
S containing a NULL would answer TRUE where SQL says UNKNOWN — a silently wrong row in the
result rather than an error. They raise with the rewrite spelled out instead.

Run as a pre-pass over the whole statement (`normalize_quantified`) rather than inside the
WHERE folder, so the same rewrite reaches HAVING, a CASE arm, and a nested subquery.
"""

from __future__ import annotations

from sqlglot import expressions as exp

__all__ = ["normalize_quantified"]

#: The comparison operators a quantified subquery may carry, spelled for an error message.
_INEQUALITIES = {
    exp.GT: (">", "max", "min"),
    exp.GTE: (">=", "max", "min"),
    exp.LT: ("<", "min", "max"),
    exp.LTE: ("<=", "min", "max"),
}


def normalize_quantified(ast):
    """Rewrite every ``ANY``/``SOME``/``ALL`` comparison under `ast` into `IN`/`NOT IN`.

    Mutates `ast` in place and returns it, which is how sqlglot's own transforms work and
    what lets the caller stay a single line.

    Args:
        ast: The parsed statement to normalize.

    Returns:
        The same statement, with the quantified comparisons replaced.

    Raises:
        NotImplementedError: For a quantified form with no faithful `IN` rewrite.
    """
    for node in list(ast.find_all(exp.Any, exp.All)):
        parent = node.parent
        if parent is None or node.arg_key != "expression":
            # An `ALL` that is not the right operand of a comparison is not a quantified
            # predicate at all — `SELECT ALL x` and `UNION ALL` reuse the token. Leaving
            # those alone is what keeps this pass from touching unrelated syntax.
            continue
        _rewrite(parent, node)
    return ast


def _rewrite(compare, quantifier) -> None:
    """Replace one ``<lhs> <op> ANY/ALL (S)`` comparison with its `IN` equivalent."""
    quantified_any = isinstance(quantifier, exp.Any)
    query = quantifier.this
    if not isinstance(query, (exp.Select, exp.Union, exp.Subquery)):
        # `x = ANY([1, 2])` over an array literal is a different feature (array membership),
        # and lowering it as a subquery would silently mis-parse it.
        return
    lhs = compare.this
    if isinstance(compare, exp.EQ) and quantified_any:
        compare.replace(exp.In(this=lhs.copy(), query=_subquery(query)))
        return
    if isinstance(compare, exp.NEQ) and not quantified_any:
        compare.replace(exp.Not(this=exp.In(this=lhs.copy(), query=_subquery(query))))
        return
    _reject(compare, quantified_any)


def _subquery(query):
    """`query` as the `Subquery` node an `exp.In` expects."""
    return query.copy() if isinstance(query, exp.Subquery) else exp.Subquery(this=query.copy())


def _reject(compare, quantified_any: bool) -> None:
    """Raise for a quantified comparison with no faithful `IN` rewrite, naming the fix."""
    word = "ANY" if quantified_any else "ALL"
    op, _, _ = _INEQUALITIES.get(type(compare), ("=", "", ""))
    if type(compare) in _INEQUALITIES:
        _, all_agg, any_agg = _INEQUALITIES[type(compare)]
        agg = any_agg if quantified_any else all_agg
        detail = f"rewrite it as `x {op} (SELECT {agg}(c) FROM ...)`" + (
            ""
            if quantified_any
            else f", and add `AND NOT EXISTS (SELECT 1 FROM ... WHERE c IS NULL)` — "
            f"`{op} {word}` is UNKNOWN, not TRUE, when the subquery yields a NULL"
        )
    else:
        detail = "only `= ANY` and `<> ALL` have an exact IN/NOT IN equivalent"
    raise NotImplementedError(f"`{op} {word} (subquery)` is not supported: {detail}")
