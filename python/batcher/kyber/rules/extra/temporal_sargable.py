"""NORMALIZE-phase rewrites: temporal extraction predicates → sargable ranges.

These rules complement `date_trunc_to_range` (in `rules/normalize.py`). Where that
one handles ``date_trunc(unit, col) = lit``, these handle *field-extraction*
comparisons such as ``year(col) = 2021`` or ``year(col) >= 2020`` and turn them into
plain half-open range predicates on the raw column
(``col >= DATE '2021-01-01' AND col < DATE '2022-01-01'``). The opaque `DateFunc`
extraction is blind to zone-map pruning and source predicate pushdown; the raw-column
range is not — so partitioned/clustered lakehouse scans can skip whole row groups and
files, and the per-row extraction disappears.

Only **contiguous, monotonic** extractions are range-sargable, and only those are
rewritten:

* ``year`` — the set of instants with ``year(col) = Y`` is exactly the contiguous
  interval ``[Jan 1 Y, Jan 1 Y+1)``; ``year`` is monotonic, so the inequalities map to
  clean single bounds too.
* ``decade`` — DuckDB's ``decade`` is ``floor(year / 10)`` (2021 → 202), so
  ``decade(col) = D`` is exactly ``[Jan 1 10·D, Jan 1 10·D+10)`` — again contiguous and
  monotonic.

Deliberately **not** rewritten (documented so the omission is a decision, not a gap):

* ``month`` / ``quarter`` / ``week`` — these *recur* every year, so ``month(col) = 6``
  is a union of one interval per year in the data, never a single contiguous range;
  not range-sargable.
* ``day`` / ``dayofweek`` / ``dayofyear`` / ``dayname`` — likewise recurring, not
  contiguous.
* ``iso_year`` — its year boundary is a mid-week ISO date that depends on the weekday
  of Jan 1, not a clean calendar literal; excluded to stay provably exact.
* ``!=`` on ``year``/``decade`` — the complement of a range is a *disjunction* of two
  open half-lines (``col < lo OR col >= hi``); it prunes nothing on either end, so it
  is left to the engine (matching `date_trunc_to_range`, which is equality-only).

Every rewrite is conservative: it fires only for a bare ``Col`` input, an *integer*
literal, and a column whose type the plan schema proves to be a naive ``date`` or
naive ``timestamp`` (so the emitted literal has the exact type the engine compares
against — a ``date`` literal for a date column, a ``datetime`` literal for a timestamp
column). Timezone-aware timestamps and out-of-`datetime.date`-range years are left
untouched. Nulls are preserved: ``fn(NULL)`` is NULL and every emitted comparison on a
NULL column is NULL, so both forms drop the same rows.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Callable

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.plan.expr_ir import Binary, Col, Expr, Lit
from batcher.plan.expr_ir.func_nodes import DateFunc
from batcher.plan.expr_rewrite import transform_expr_up
from batcher.plan.ir_tags import COMPARISON_FLIP, COMPARISON_OPS, COMPARISON_ORDER
from batcher.plan.logical import Filter, LogicalPlan

__all__ = [
    "TEMPORAL_SARGABLE_RULES",
    "rewrite_temporal_filter",
]

# The comparison operators we rewrite (equality + the four inequalities). `ne` is
# excluded on purpose — see the module docstring.

# When the literal sits on the *left* (`Y < year(col)`), the effective operator on the
# extraction is the mirror of the written one.

# A contiguous, monotonic year-bucket extraction: `first_year(V)` is the first calendar
# year in bucket `V`, and `span` is how many calendar years the bucket covers. `year`
# is one year per bucket; DuckDB's `decade` is `floor(year/10)`, i.e. ten years starting
# at `10·V`.
#
# `century` and `millennium` are the same shape with a different origin, and the origin is
# the whole subtlety: they are **1-based**, so century 20 is 1901–2000 rather than
# 1900–1999 (verified against the engine — `century(1900-01-01)` is 19 and
# `century(1901-01-01)` is 20). Getting that off by one would shift every bound by a year.
#
# `iso_year` looks like it belongs here and does not: an ISO year boundary is a Monday, not
# 1 January, so its buckets are not calendar-year aligned and `_year_start` cannot name them
# (the engine puts `2000-01-01` in ISO year 1999).
_BUCKETS: dict[str, tuple[Callable[[int], int], int]] = {
    "year": (lambda v: v, 1),
    "decade": (lambda v: v * 10, 10),
    "century": (lambda v: v * 100 - 99, 100),
    "millennium": (lambda v: v * 1000 - 999, 1000),
}


def _year_start(year: int, *, is_timestamp: bool) -> _dt.date | None:
    """Jan 1 of `year` as the column's literal type, or None if unrepresentable.

    A `datetime.date`/`datetime` only spans years 1-9999; a bound outside that range
    (e.g. the upper bound of ``year(col) <= 9999``) cannot be built, so the caller
    leaves the predicate untouched rather than emit a wrong or partial range.
    """
    try:
        return _dt.datetime(year, 1, 1) if is_timestamp else _dt.date(year, 1, 1)
    except ValueError:
        return None


def _column_kind(node: Filter, name: str) -> bool | None:
    """Whether `name` is a naive timestamp (True) or date (False) column, else None.

    The node-taking form, kept for the standalone `rewrite_temporal_filter` entry point
    that the unit tests drive. The fused path calls `_kind_from_schema` directly with the
    schema the driver already resolved.
    """
    return _kind_from_schema(node.input.available_schema(), name)


def _kind_from_schema(schema, name: str) -> bool | None:
    """Whether `name` is a naive timestamp (True) or date (False) in `schema`, else None.

    Returns None when the schema is unknown, the column is absent, the column is
    neither date nor timestamp, or the timestamp carries a timezone (whose extraction
    semantics depend on the session zone — not provably exact here).
    """
    if schema is None or not schema.has(name):
        return None
    dtype = schema.field(name).type
    if pa.types.is_date(dtype):
        return False
    if pa.types.is_timestamp(dtype) and dtype.tz is None:
        return True
    return None


def _match_extraction(expr: Binary) -> tuple[str, str, int, str] | None:
    """`(fn, column, value, effective_op)` if `expr` is a bucketed extraction compared
    to an integer literal, else None.

    Handles the literal on either side (mirroring the operator when it is on the left),
    a bare-``Col`` extraction input, and rejects bool literals (a bool is an ``int``
    subclass but not a valid year).
    """
    if expr.op not in COMPARISON_OPS:
        return None
    left, right = expr.left, expr.right
    if isinstance(left, DateFunc) and isinstance(right, Lit):
        func, lit, op = left, right, expr.op
    elif isinstance(right, DateFunc) and isinstance(left, Lit):
        func, lit, op = right, left, COMPARISON_FLIP[expr.op]
    else:
        return None
    if func.fn not in _BUCKETS or not isinstance(func.input, Col):
        return None
    value = lit.value
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return func.fn, func.input.name, value, op


def _range_expr(column: str, op: str, lo: _dt.date, hi: _dt.date) -> Expr:
    """The sargable predicate on the raw column for `op` against bounds `[lo, hi)`."""
    col = Col(column)
    ge_lo = Binary("ge", col, Lit(lo))
    lt_lo = Binary("lt", col, Lit(lo))
    lt_hi = Binary("lt", col, Lit(hi))
    ge_hi = Binary("ge", col, Lit(hi))
    if op == "eq":
        return Binary("and", ge_lo, lt_hi)
    if op == "ne":
        # The De Morgan complement of the `eq` band: *outside* `[lo, hi)` on either side.
        # Both operators flip, which is what makes it a disjunction of the two strict
        # half-bounds rather than the same pair rejoined by an `OR`.
        return Binary("or", lt_lo, ge_hi)
    if op == "lt":
        return lt_lo
    if op == "le":
        return lt_hi
    if op == "gt":
        return ge_hi
    return ge_lo  # "ge"


def rewrite_temporal_filter(node: Filter, fn: str, op: str) -> Filter | None:
    """Rewrite ``fn(col) <op> Y`` conjuncts in `node`'s predicate to raw-column ranges.

    Fires only for the given extraction `fn` and effective operator `op` (so it is a
    distinct, independently-indexable rule per pair). Returns a new `Filter` when at
    least one comparison was rewritten, else None (nothing to do — the fixpoint stops
    and idempotence holds, since the rewritten form contains no `DateFunc`).

    Args:
        node: The `Filter` whose predicate is scanned.
        fn: The extraction family to match (``year`` or ``decade``).
        op: The effective comparison operator to match (one of `eq/lt/le/gt/ge`).
    """
    changed = False

    def rewrite(expr: Expr) -> Expr:
        nonlocal changed
        if not isinstance(expr, Binary):
            return expr
        matched = _match_extraction(expr)
        if matched is None:
            return expr
        matched_fn, column, value, eff_op = matched
        if matched_fn != fn or eff_op != op:
            return expr
        is_timestamp = _column_kind(node, column)
        if is_timestamp is None:
            return expr
        first_year, span = _BUCKETS[fn]
        lo = _year_start(first_year(value), is_timestamp=is_timestamp)
        hi = _year_start(first_year(value) + span, is_timestamp=is_timestamp)
        if lo is None or hi is None:
            return expr
        changed = True
        return _range_expr(column, op, lo, hi)

    new_pred = transform_expr_up(node.predicate, rewrite)
    if not changed:
        return None
    return Filter(node.input, new_pred)


def _make_rule(fn: str, op: str):
    """Build the node-local `f(node, ctx)` closure for one (fn, op) pair."""

    def _apply(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
        return rewrite_temporal_filter(node, fn, op)

    return _apply


def _make_leaf(fn: str, op: str):
    """The same rewrite as a *schema leaf*, so the driver can fuse it.

    Twenty-four rules each walking the filter predicate themselves was the single largest
    remaining cost in planning (46,992 traversal steps on the profiler's benchmark). As a
    leaf the driver offers each expression to all twenty-four in one shared walk, with the
    schema resolved once for the node instead of once per rule.
    """

    def leaf(expr: Expr, schema) -> Expr:
        if not isinstance(expr, Binary):
            return expr
        matched = _match_extraction(expr)
        if matched is None:
            return expr
        matched_fn, column, value, eff_op = matched
        if matched_fn != fn or eff_op != op:
            return expr
        is_timestamp = _kind_from_schema(schema, column)
        if is_timestamp is None:
            return expr
        first_year, span = _BUCKETS[fn]
        lo = _year_start(first_year(value), is_timestamp=is_timestamp)
        hi = _year_start(first_year(value) + span, is_timestamp=is_timestamp)
        if lo is None or hi is None:
            return expr
        return _range_expr(column, op, lo, hi)

    return leaf


# Register one distinct rule per (extraction, operator) pair: 4 × 6 = 24 rules. Each is
# indexed on `Filter` so the driver skips it when no filter is present, and each is
# individually idempotent (its rewritten output has no `DateFunc` left to match).
TEMPORAL_SARGABLE_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"{fn}_{op}_to_range",
            Phase.NORMALIZE,
            _make_rule(fn, op),
            matches=(Filter,),
            expr_schema_fn=_make_leaf(fn, op),
            expr_matches=(Binary,),
            expr_ops=(op, COMPARISON_FLIP[op]),
        )
    )
    for fn in _BUCKETS
    for op in COMPARISON_ORDER
]
