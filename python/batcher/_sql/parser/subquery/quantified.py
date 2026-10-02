"""Quantified comparison predicates — ``x <op> ANY (SELECT ...)`` and ``x <op> ALL (...)``.

SQL's quantified comparisons say "compare against *every* row the subquery returns, and
combine the results with OR (``ANY``/``SOME``) or AND (``ALL``)". Two of the forms are
exactly the set-membership predicates the translator already decorrelates:

    x =  ANY (S)   is   x IN (S)
    x <> ALL (S)   is   x NOT IN (S)

which is the definition, not an approximation. Rewriting them here means they arrive at
`core._apply_in_subquery` as ordinary `IN`/`NOT IN` and inherit everything it already gets
right: the semi/anti join, the multi-column row form, and — for ``NOT IN`` — the
three-valued logic that a NULL anywhere in S makes the whole predicate un-true.

Every other operator (``>``, ``>=``, ``<``, ``<=`` with either quantifier, and ``= ALL`` /
``<> ANY``) becomes a ``CASE`` over three scalar aggregates of S: its row count, its
non-null count, and the ``min``/``max`` that decides the comparison. That is the full
three-valued truth table, not the ``x > (SELECT max(c) ...)`` shortcut, which is wrong twice:
an empty S makes ``x > ALL (S)`` TRUE where ``x > max`` is UNKNOWN, and a NULL in S makes it
UNKNOWN where ``max`` (which skips NULLs) answers TRUE. For ``ALL``:

    CASE WHEN count(*) = 0              THEN TRUE      -- vacuous truth
         WHEN x IS NULL                 THEN NULL
         WHEN <some non-null c refutes> THEN FALSE     -- e.g. x <= max(c) for `>`
         WHEN count(*) > count(c)       THEN NULL      -- a NULL could still refute
         ELSE TRUE END

and ``ANY`` is its dual (empty → FALSE, a witness → TRUE, a NULL → UNKNOWN, else FALSE).
The value is exact in a select list and under ``NOT``, not only as a filter. Each aggregate
is an ordinary scalar subquery, so an uncorrelated S is collected once per aggregate and a
correlated one decorrelates as any correlated scalar aggregate does.

Run as a pre-pass over the whole statement (`normalize_quantified`) rather than inside the
WHERE folder, so the same rewrite reaches HAVING, a CASE arm, and a nested subquery.
"""

from __future__ import annotations

from sqlglot import expressions as exp

__all__ = ["normalize_quantified"]

#: For each comparison: the aggregate that decides ``ALL`` and the comparison that refutes it
#: (a non-null ``c`` with ``NOT (x op c)``), and the aggregate that decides ``ANY`` and the
#: comparison that witnesses it (a non-null ``c`` with ``x op c``). `= ALL` / `<> ANY` need
#: both extremes, so they are handled on their own.
_ORDERED = {
    exp.GT: (("max", exp.LTE), ("min", exp.GT)),
    exp.GTE: (("max", exp.LT), ("min", exp.GTE)),
    exp.LT: (("min", exp.GTE), ("max", exp.LT)),
    exp.LTE: (("min", exp.GT), ("max", exp.LTE)),
}

#: The derived-table alias and column a non-simple subquery is wrapped under.
_Q, _C = "__bc_quant", "__bc_quant_c"


def normalize_quantified(ast):
    """Rewrite every ``ANY``/``SOME``/``ALL`` comparison under `ast` into supported SQL.

    Mutates `ast` in place and returns it, which is how sqlglot's own transforms work and
    what lets the caller stay a single line.

    Args:
        ast: The parsed statement to normalize.

    Returns:
        The same statement, with the quantified comparisons replaced.

    Raises:
        NotImplementedError: For a row-valued quantified form other than ``= ANY`` /
            ``<> ALL``, which has no scalar extreme to compare against.
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
    """Replace one ``<lhs> <op> ANY/ALL (S)`` comparison with its exact equivalent."""
    quantified_any = isinstance(quantifier, exp.Any)
    query = quantifier.this
    if isinstance(query, exp.Subquery):
        query = query.this
    if not isinstance(query, (exp.Select, exp.Union)):
        # `x = ANY([1, 2])` over an array literal is a different feature (array membership),
        # and lowering it as a subquery would silently mis-parse it.
        return
    lhs = compare.this
    if isinstance(compare, exp.EQ) and quantified_any:
        compare.replace(exp.In(this=lhs.copy(), query=exp.Subquery(this=query.copy())))
        return
    if isinstance(compare, exp.NEQ) and not quantified_any:
        compare.replace(
            exp.Not(this=exp.In(this=lhs.copy(), query=exp.Subquery(this=query.copy())))
        )
        return
    if isinstance(lhs, exp.Tuple) or len(query.selects) != 1:
        word = "ANY" if quantified_any else "ALL"
        raise NotImplementedError(
            f"a row-valued `{compare.key} {word} (subquery)` is not supported: only "
            "`= ANY` and `<> ALL` compare rows. Compare one column at a time"
        )
    compare.replace(_truth_table(compare, lhs, query, quantified_any))


def _truth_table(compare, lhs, query, quantified_any: bool):
    """The ``CASE`` that evaluates one quantified comparison under three-valued logic."""
    x = lhs.copy()
    if type(compare) in _ORDERED:
        all_rule, any_rule = _ORDERED[type(compare)]
        agg, test = any_rule if quantified_any else all_rule
        decided = test(this=x.copy(), expression=_aggregate(query, agg))
    elif isinstance(compare, (exp.EQ, exp.NEQ)):
        # `= ALL` is refuted, and `<> ANY` witnessed, by any non-null c other than x — which
        # exists exactly when x differs from the smallest or the largest of them.
        decided = exp.or_(
            exp.NEQ(this=x.copy(), expression=_aggregate(query, "min")),
            exp.NEQ(this=x.copy(), expression=_aggregate(query, "max")),
        )
    else:
        raise NotImplementedError(f"unsupported quantified comparison {compare.sql()!r}")
    rows = _aggregate(query, "count", star=True)
    non_null = _aggregate(query, "count")
    unknown = exp.cast(exp.Null(), "BOOLEAN")
    on_empty, on_decided, on_rest = (
        (exp.false(), exp.true(), exp.false())
        if quantified_any
        else (exp.true(), exp.false(), exp.true())
    )
    return exp.Paren(
        this=exp.Case(
            ifs=[
                exp.If(this=exp.EQ(this=rows, expression=exp.Literal.number(0)), true=on_empty),
                exp.If(this=exp.Is(this=x.copy(), expression=exp.Null()), true=unknown.copy()),
                exp.If(this=decided, true=on_decided),
                exp.If(this=exp.GT(this=rows.copy(), expression=non_null), true=unknown.copy()),
            ],
            default=on_rest,
        )
    )


def _aggregate(query, func: str, *, star: bool = False):
    """A scalar subquery computing `func` over the single column `query` returns.

    A plain ``SELECT e FROM ... WHERE ...`` is aggregated in place (``SELECT max(e) FROM ...``),
    which keeps a correlated predicate in the subquery's own WHERE where the decorrelator
    looks for it. Anything else — a UNION, a GROUP BY, a LIMIT, DISTINCT — is wrapped as a
    derived table and aggregated from outside, which is exact for every shape.
    """
    if _aggregates_in_place(query):
        inner = query.copy()
        target = inner.selects[0]
        value = target.this if isinstance(target, exp.Alias) else target
        inner.set("expressions", [_call(func, None if star else value.copy())])
        return exp.Subquery(this=inner)
    derived = exp.Subquery(
        this=query.copy(),
        alias=exp.TableAlias(this=exp.to_identifier(_Q), columns=[exp.to_identifier(_C)]),
    )
    column = None if star else exp.column(_C, table=_Q)
    return exp.Subquery(this=exp.select(_call(func, column)).from_(derived))


def _call(func: str, arg):
    """``func(arg)``, or ``count(*)`` when `arg` is None."""
    if func == "count":
        return exp.Count(this=exp.Star() if arg is None else arg)
    return (exp.Max if func == "max" else exp.Min)(this=arg)


def _aggregates_in_place(query) -> bool:
    """Whether `query`'s one projection can be replaced by an aggregate of itself."""
    if not isinstance(query, exp.Select):
        return False
    blocking = ("group", "having", "distinct", "limit", "offset", "qualify", "windows", "with")
    if any(query.args.get(key) for key in blocking):
        return False
    target = query.selects[0]
    return not isinstance(target, exp.Star) and not any(
        True for _ in target.find_all(exp.AggFunc, exp.Window)
    )
