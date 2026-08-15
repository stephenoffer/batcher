"""Lowering `x IN (SELECT …)` to a predicate — as a literal set, or as a mark join.

An uncorrelated `IN (SELECT …)` under an `OR` cannot become a semi-join, so
`in_marker` below builds a *mark join*: probe column, left join against the distinct
set, collapse the null-extension back to a boolean. That is the right shape for a set of any
size and the wrong one for a set of nine — which is what TPC-DS q45's
`i_item_id IN (SELECT i_item_id FROM item WHERE i_item_sk IN (2,3,…,29))` is. Measured at
scale 1: **124.3 ms as a mark join, 8.8 ms as a literal set**, against DuckDB's 18.9.

The set is materialized either way — the mark-join path already collects it once, to ask
whether it holds a NULL — so what this trades is a join against a hash-set probe per row.
Nothing about the semantics is re-derived: `Expr.is_in` already implements SQL's three-valued
`IN` (a NULL member turns a would-be FALSE into NULL; a NULL probe is NULL), so the collected
values are handed to it verbatim, nulls included, and `NOT` over the result is `NOT IN`.
"""

from __future__ import annotations

from sqlglot import expressions as exp

from batcher.api.dataset import Dataset
from batcher.plan.expr_ir import col, lit

#: Distinct values below which an uncorrelated `IN (subquery)` is inlined as a literal set
#: rather than joined against.
#:
#: The set is materialized either way — the marker path below already collects it to ask
#: whether it holds a NULL — so what this trades is a join against a hash-set probe per row.
#: The cap exists because the probe is per row and the set rides in the plan: a few hundred
#: members is an `InList` the engine hashes once, a few hundred thousand is a relation, and
#: that is what the join is for.
_INLINE_SET_MAX = 256


def _collect_small_set(inner_ds: Dataset, column: str) -> list | None:
    """`inner_ds`'s distinct values, or `None` once there are more than the inline cap allows.

    `limit(cap + 1)` is what makes this affordable on a set that turns out to be large: it is
    the question "are there more than `cap`", not "how many are there". Measured against
    DuckDB on a subquery over 2.9 M rows, the whole query still reads 12.9 ms to its 21.1.
    """
    collected = inner_ds.distinct().limit(_INLINE_SET_MAX + 1).collect()
    if collected.num_rows > _INLINE_SET_MAX:
        return None
    return collected.column(column).to_pylist() if collected.num_rows else []


def inline_in_subquery_values(tr, node) -> list | None:
    """The distinct values of an uncorrelated `IN (subquery)`, or `None` if it is not small.

    The *value*-position entry point — `expressions.scalar._in`, which has no relation to
    attach a mark join to and therefore refused this shape outright until there was a way to
    answer it without one. `None` covers every reason a set cannot be inlined (correlated,
    multi-column, a row-value target, or simply too large), and the caller then reports its
    original refusal. The two AST helpers are imported inside the function because `core`
    imports this module.
    """
    from batcher._sql.parser.subquery.core import _in_subquery_select, _reject_correlated

    if isinstance(node.this, exp.Tuple):
        return None
    inner = _in_subquery_select(node).copy()
    try:
        _reject_correlated(inner)
    except NotImplementedError:
        return None
    inner_ds = tr.statement(inner)
    if len(inner_ds.columns) != 1:
        return None
    return _collect_small_set(inner_ds, inner_ds.columns[0])


def inline_small_set(
    tr, ds: Dataset, target, marker: str, inner_ds: Dataset, key: str, *, negate: bool
):
    """`x IN (SELECT …)` as a literal membership test, when the set is small enough to be one.

    The subquery under an `OR` cannot become a semi-join, so it became a *mark join*: probe
    column, left join against the distinct set, collapse the null-extension to a boolean. That
    is the right shape for a set of any size and the wrong one for a set of nine — which is
    what TPC-DS q45's `i_item_id IN (SELECT i_item_id FROM item WHERE i_item_sk IN (2,3,…,29))`
    is. Measured at scale 1: **124.3 ms as a mark join, 8.8 ms as a literal set**, against
    DuckDB's 18.9 — a query that lost 7.3x and now wins.

    Nothing about the *semantics* changes, and none of it is re-derived here:
    `Expr.is_in` already implements SQL's three-valued `IN` (a NULL member turns a would-be
    FALSE into NULL; a NULL probe is NULL), so the collected values are handed to it verbatim,
    nulls included, and `NOT` over the result is `NOT IN`. That is also why this path needs one
    collect where the join path needs one *and* a separate null probe.

    Returns `(ds, ast)` like its caller, or `None` when the set is too large — in which case
    nothing has been consumed and the mark join proceeds.
    """
    values = _collect_small_set(inner_ds, key)
    if values is None:
        return None
    ds = ds.with_columns(**{marker: tr._scalar(target).is_in(values)})
    ast = exp.column(marker)
    return ds, exp.Paren(this=exp.Not(this=ast) if negate else ast)


def in_marker(tr, ds: Dataset, node, *, negate: bool):
    """`x IN (SELECT c …)` as a boolean *column*, for a predicate that cannot become a join.

    The `IN` counterpart to `_exists_marker`, and it has to answer the two objections that
    kept it from existing. Both are recorded in `_apply_single_predicate`, and neither is
    waved away here:

    * **Three-valued logic.** ``x IN (…)`` is NULL — not FALSE — when `x` is NULL, or when
      the set holds a NULL and nothing matches. A bare existence bit cannot carry that, so
      this builds the full three-way answer instead: matched ⇒ TRUE, else NULL when either
      null source applies, else FALSE. Whether the set holds a NULL is one extra probe of
      the (uncorrelated) inner relation, decided before the marker is built.
    * **Name capture.** The earlier attempt rewrote to ``EXISTS (SELECT 1 FROM S WHERE
      c = x)`` and wrote `x` as the user spelled it; for the ordinary
      ``category IN (SELECT category FROM vip)`` that unqualified name rebound to the
      *inner* relation, the predicate became ``category = category``, and every row
      matched. Here the outer value is materialized into a synthesized ``__in<n>_v``
      column and the inner key aliased to ``__in<n>_k``, so neither side can capture the
      other regardless of what the two relations call their columns.

    Only the **uncorrelated** single-column form is handled; anything else returns None and
    the caller reports the original refusal. TPC-DS q45 is the uncorrelated form.

    Args:
        tr: The translator, used to plan the inner SELECT.
        ds: The outer relation the marker is attached to.
        node: The `In` AST node.
        negate: True for `NOT IN`.

    Returns:
        `(ds, ast)` — the relation carrying the marker and the AST to substitute for the
        `IN` node — or None when the shape is not markerizable.
    """
    from batcher._sql.parser.subquery.core import (
        IN_MARKER_PREFIX,
        _in_subquery_select,
        _reject_correlated,
    )

    target = node.this
    if isinstance(target, exp.Tuple):
        return None  # a row-value IN has no single probe column
    inner = _in_subquery_select(node).copy()  # detach from the outer AST scope
    try:
        _reject_correlated(inner)
    except NotImplementedError:
        return None

    n = getattr(tr, "_in_marker_n", 0)
    tr._in_marker_n = n + 1
    probe = f"{IN_MARKER_PREFIX}{n}_v"
    key = f"{IN_MARKER_PREFIX}{n}_k"
    marker = f"{IN_MARKER_PREFIX}{n}_m"

    inner_ds = tr.statement(inner)
    if len(inner_ds.columns) != 1:
        return None
    inner_ds = inner_ds.rename({inner_ds.columns[0]: key})
    inlined = inline_small_set(tr, ds, target, marker, inner_ds, key, negate=negate)
    if inlined is not None:
        return inlined
    # Past the inline cap, so the set is **not empty** — which this arm relies on without
    # being able to check it. An empty set makes `x IN (…)` FALSE for every row *including* a
    # NULL `x`, where the `probe IS NULL` branch below answers NULL; `NULL NOT IN (empty)` is
    # then dropped where SQL keeps it (DuckDB agrees, and
    # `test_in_subquery_under_or_matches_duckdb` covers it). `_collect_small_set` returns a
    # list — possibly the empty one — for every set of at most `_INLINE_SET_MAX` values and
    # `None` only above it, so an empty set is inlined by construction and never arrives here.
    #
    # Does the set hold a NULL? It decides whether an unmatched row is FALSE or NULL, and
    # for an uncorrelated set it is one answer for the whole query. `limit(1)` stops at the
    # first one rather than scanning the relation.
    set_has_null = inner_ds.filter(col(key).is_null()).limit(1).collect().num_rows > 0

    values = inner_ds.filter(col(key).is_not_null()).distinct().with_columns(**{marker: lit(True)})
    ds = ds.with_columns(**{probe: tr._scalar(target)})
    ds = ds.join(values, left_on=[probe], right_on=[key], how="left")
    if key in ds.columns:
        ds = ds.drop(key)
    # Matched ⇒ the tag survives; unmatched ⇒ the left join null-extends it. Collapsing that
    # to a real boolean here keeps the substituted AST a plain column reference.
    ds = ds.with_columns(**{marker: col(marker).is_not_null()})

    # `NULLIF(TRUE, TRUE)` is a *boolean* NULL. A bare `NULL` would lower to an Int64 one
    # and clash with the CASE's boolean branches.
    null_bool = exp.Nullif(this=exp.true(), expression=exp.true())
    case = exp.case().when(exp.column(marker), exp.true())
    if set_has_null:
        ast = case.else_(null_bool)
    else:
        probe_is_null = exp.Is(this=exp.column(probe), expression=exp.Null())
        ast = case.when(probe_is_null, null_bool).else_(exp.false())
    return ds, exp.Paren(this=exp.Not(this=ast) if negate else ast)
