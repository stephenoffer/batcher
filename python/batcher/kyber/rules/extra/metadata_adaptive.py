"""Metadata-adaptive rewrites — skip or simplify work a proven-EXACT stat makes dead.

Where `adaptive_meta` folds *limits* and *empty* inputs a provably-EXACT cardinality
unlocks, this module attacks the other pipeline breakers — `Sort`, `Distinct`, and a
`Filter` comparing two columns — using EXACT per-column metadata (`ndv`, `min`/`max`,
`null_count`) the estimator propagates. Every rule fires *only* on `Provenance.EXACT`
proof: a learned/sketch ndv or a merely-estimated bound can never drive a rewrite here,
because dropping a `Distinct` or a `Sort` on a wrong guess is silent data corruption.

- `skip_sort_of_single_row` drops a `Sort` over a subtree proven to hold ≤ 1 row (a
  global aggregate, a `Limit … 1`, a one-row source): a relation of at most one row is
  already ordered, so the sort — a breaker Carbonite budgets — is pure overhead.
- `prune_constant_sort_keys` removes `ORDER BY` keys whose column is proven a single
  constant with no nulls (`min == max`, `null_count == 0`, EXACT): a constant key never
  breaks a tie, so it contributes nothing to the ordering. When *every* key is constant
  (and there is no top-N cap) the whole sort is dropped.
- `drop_distinct_when_unique` drops a `Distinct` when some column is proven unique — its
  EXACT distinct count reaches the EXACT row count — so every row is already distinct and
  the dedup breaker does nothing. This is the metadata-adaptive dedup skip.
- `prune_filter_col_comparison` decides a `col OP col` filter from the two columns' EXACT
  bounds: always-true drops the filter, always-false folds to the empty relation. It
  extends `zonemap_prune_filter` (which only handles `col OP literal`) to the column-vs-
  column case that rule leaves untouched.

Every rewrite is byte-identical in result (same rows, same multiset) and idempotent
(returns None once nothing is left to simplify).
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.zonemap_pruning import _float_order_is_ambiguous
from batcher.plan.expr_ir import Binary, Col
from batcher.plan.ir_tags import COMPARISON_OPS
from batcher.plan.logical import Distinct, Filter, Limit, LogicalPlan, Sort
from batcher.plan.stats import ColumnStat, Provenance

__all__ = [
    "drop_distinct_when_unique",
    "prune_constant_sort_keys",
    "prune_filter_col_comparison",
    "skip_sort_of_single_row",
]

# Comparison operators (the `Binary.op` tags) this module reasons about — the same set
# `zonemap_prune_filter` decides for `col OP literal`, reused here for `col OP col`.


def _is_constant_column(stat: ColumnStat) -> bool:
    """Whether `stat` proves the column is a single constant value with no nulls.

    Requires `Provenance.EXACT` (a bound is only a *guess* otherwise), a present
    ``min == max``, and a known-zero ``null_count`` — a surviving null would sort into
    its own position, so a column that is one value *plus* nulls is not constant for
    ordering purposes.
    """
    return (
        stat.provenance is Provenance.EXACT
        and stat.min is not None
        and stat.null_count == 0
        and stat.min == stat.max
    )


def _exact_bounds(stat: ColumnStat) -> bool:
    """Whether `stat` carries an EXACT, fully-populated ``[min, max]`` range."""
    return stat.provenance is Provenance.EXACT and stat.min is not None and stat.max is not None


@rule(name="skip_sort_of_single_row", phase=Phase.REWRITE, matches=(Sort,))
def skip_sort_of_single_row(node: Sort, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `Sort` whose input is provably no larger than one row.

    A relation of ≤ 1 row is trivially already ordered under any key set, so the sort
    (a pipeline breaker) produces its input unchanged. Gated on an EXACT row count — a
    global aggregate is exactly one row, a `Limit(_, 1)` over an exact input caps at one,
    a one-row source declares one — so an estimate can never drop a real sort. A top-N
    `limit` below the row count still selects rows, so the rule stands down there
    (``limit < rows``); returns None once no such sort remains (idempotent).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.optimizer import Optimizer
            >>> from batcher.plan.logical import Sort
            >>> ds = bt.from_pydict({"x": [5]})  # exactly one row
            >>> out = Optimizer(sources=ds._sources).logical_rewrite(ds.sort("x")._plan)
            >>> isinstance(out, Sort)
            False
    """
    if ctx is None:
        return None
    stats = ctx.estimator.estimate(node.input)
    if not stats.rows_exact or stats.rows > 1:
        return None
    # A top-N cap below the (already ≤ 1) row count would still drop rows — keep it.
    if node.limit is not None and node.limit < stats.rows:
        return None
    return node.input


@rule(name="prune_constant_sort_keys", phase=Phase.REWRITE, matches=(Sort,))
def prune_constant_sort_keys(node: Sort, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop `ORDER BY` keys whose column is proven a single non-null constant.

    A key that is the same value in every row never breaks a tie, so ordering by
    ``(const, x)`` equals ordering by ``(x)`` — for ascending, descending, or
    nulls-first alike (with no nulls, all directions coincide). Each pruned key must be
    proven constant from EXACT metadata (``min == max``, ``null_count == 0``), e.g. a
    literal column from `with_columns` or a single-valued partition column. When some
    but not all keys are constant, the surviving keys re-form the sort (preserving any
    top-N `limit`, which a constant key never affects); when *every* key is constant and
    there is no `limit`, the sort is a no-op and is dropped entirely. Returns None when
    no key is provably constant (idempotent — the rebuilt sort has only live keys).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.optimizer import Optimizer
            >>> from batcher.plan.logical import Sort
            >>> from batcher.plan.visitor import walk
            >>> ds = bt.from_pydict({"x": [3, 1, 2]})
            >>> plan = ds.with_columns(k=7).sort("k", "x")._plan  # k is a constant key
            >>> out = Optimizer(sources=ds._sources).logical_rewrite(plan)
            >>> [k.expr.name for s in walk(out) if isinstance(s, Sort) for k in s.keys]
            ['x']
    """
    if ctx is None:
        return None
    stats = ctx.estimator.estimate(node.input)
    kept = []
    pruned_any = False
    for key in node.keys:
        if isinstance(key.expr, Col) and _is_constant_column(stats.column(key.expr.name)):
            pruned_any = True
            continue
        kept.append(key)
    if not pruned_any:
        return None
    if kept:
        return Sort(node.input, tuple(kept), node.limit)
    # Every key is constant: the ordering is arbitrary. Without a top-N cap the sort is a
    # pure no-op (input order is a valid ordering); with a `limit` it still selects rows,
    # so leave that case to the top-N machinery.
    if node.limit is None:
        return node.input
    return None


@rule(name="drop_distinct_when_unique", phase=Phase.REWRITE, matches=(Distinct,))
def drop_distinct_when_unique(node: Distinct, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `Distinct` whose input is proven already duplicate-free.

    `Distinct` deduplicates whole rows; when some column's EXACT distinct count reaches
    the input's EXACT row count, that column holds a different value in every row, so
    every *row* is already distinct and the dedup breaker is pure overhead. Both the
    ndv and the row count MUST be `Provenance.EXACT` — a learned/sketch ndv (the usual
    HLL-derived kind) is an estimate and can never drop a real dedup. The single unique
    column suffices regardless of nulls: a duplicate null would pull the distinct count
    below the row count under either null convention, so ``ndv >= rows`` already excludes
    it. Returns None when no column proves uniqueness (idempotent — it returns the input,
    not another `Distinct`).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.kyber.optimizer import Optimizer
            >>> from batcher.plan.source_stats import SourceStatistics
            >>> from batcher.plan.stats import ColumnStat, Provenance
            >>> from batcher.plan.logical import Distinct
            >>> ds = bt.from_pydict({"id": [1, 2, 3]})
            >>> stats = [SourceStatistics(row_count=3, columns={
            ...     "id": ColumnStat(ndv=3, null_count=0, provenance=Provenance.EXACT)})]
            >>> plan = ds.distinct()._plan
            >>> out = Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(plan)
            >>> isinstance(out, Distinct)
            False
    """
    if ctx is None:
        return None
    stats = ctx.estimator.estimate(node.input)
    if not stats.rows_exact:
        return None
    rows = stats.rows
    for stat in stats.columns.values():
        if stat.provenance is Provenance.EXACT and stat.ndv is not None and stat.ndv >= rows:
            return node.input
    return None


@rule(name="prune_filter_col_comparison", phase=Phase.SELECTION, matches=(Filter,))
def prune_filter_col_comparison(node: Filter, ctx: OptimizerContext) -> LogicalPlan | None:
    """Decide a `col OP col` filter from the two columns' EXACT bounds.

    Extends `zonemap_prune_filter` (which decides `col OP literal`) to a comparison of
    two columns: when their EXACT ``[min, max]`` ranges prove every row passes, the
    filter is dropped; when they prove no row passes, it folds to the canonical empty
    relation ``Limit(x, 0)``. Both columns' bounds MUST be `Provenance.EXACT` (footer /
    manifest metadata); an always-*true* verdict additionally needs both null counts
    zero (a null comparison is null → the row is dropped, so it would not pass), while an
    always-*false* verdict needs only the bounds (a filter drops nulls regardless).
    Returns None when the comparison is undecidable from bounds (idempotent).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher import col
            >>> from batcher.kyber.optimizer import Optimizer
            >>> from batcher.plan.source_stats import SourceStatistics
            >>> from batcher.plan.stats import ColumnStat, Provenance
            >>> from batcher.plan.logical import Filter
            >>> ds = bt.from_pydict({"a": [0, 1], "b": [10, 20]})
            >>> exact = lambda lo, hi: ColumnStat(min=lo, max=hi, null_count=0,
            ...                                   provenance=Provenance.EXACT)
            >>> stats = [SourceStatistics(row_count=2,
            ...     columns={"a": exact(0, 1), "b": exact(10, 20)})]
            >>> plan = ds.filter(col("a") < col("b"))._plan  # max(a)=1 < min(b)=10
            >>> out = Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(plan)
            >>> isinstance(out, Filter)
            False
    """
    if ctx is None:
        return None
    pred = node.predicate
    if not (isinstance(pred, Binary) and pred.op in COMPARISON_OPS):
        return None
    if not (isinstance(pred.left, Col) and isinstance(pred.right, Col)):
        return None
    stats = ctx.estimator.estimate(node.input)
    left = stats.column(pred.left.name)
    right = stats.column(pred.right.name)
    if not (_exact_bounds(left) and _exact_bounds(right)):
        return None
    status = _decide_col_cmp(pred.op, left, right)
    if status is True:
        return node.input  # every row passes → the filter is dead
    if status is False:
        return Limit(node.input, 0)  # no row passes → empty, schema-preserving
    return None


def _decide_col_cmp(op: str, a: ColumnStat, b: ColumnStat) -> bool | None:
    """Tri-state verdict for ``a OP b`` from the columns' EXACT bounds.

    Returns True (always passes — needs no nulls), False (never passes — empty), or
    None (undecidable from bounds). Mirrors `zonemap_pruning._decide` but with a column
    range on *both* sides. An incomparable pair of bound types is undecidable.
    """
    amin, amax = a.min, a.max
    bmin, bmax = b.min, b.max
    # A NaN or zero float bound cannot decide a comparison: the engine compares floats on
    # their total order (NaN greatest, `-0.0 < 0.0`) while the comparisons below are Python's
    # (NaN unordered, `-0.0 == 0.0`), so folding on one of those bounds can delete a row the
    # engine would keep. Same rule, same reason, as `zonemap_pruning._float_order_is_ambiguous`.
    if any(_float_order_is_ambiguous(v) for v in (amin, amax, bmin, bmax)):
        return None
    no_nulls = a.null_count == 0 and b.null_count == 0
    try:
        if op == "lt":  # a < b
            if amin >= bmax:
                return False
            return True if (amax < bmin and no_nulls) else None
        if op == "le":  # a <= b
            if amin > bmax:
                return False
            return True if (amax <= bmin and no_nulls) else None
        if op == "gt":  # a > b
            if amax <= bmin:
                return False
            return True if (amin > bmax and no_nulls) else None
        if op == "ge":  # a >= b
            if amax < bmin:
                return False
            return True if (amin >= bmax and no_nulls) else None
        if op == "eq":  # a == b
            if amax < bmin or bmax < amin:  # disjoint ranges → never equal
                return False
            return True if (amin == amax == bmin == bmax and no_nulls) else None
        if op == "ne":  # a != b
            if amin == amax == bmin == bmax:  # both the same lone constant → always equal
                return False
            return True if ((amax < bmin or bmax < amin) and no_nulls) else None
    except TypeError:
        return None  # incomparable bound types → undecidable
    return None
