"""Aggregate rewrites driven by *proven* metadata — uniqueness, constancy, exact counts.

`agg_extra` holds the aggregate rewrites that fire on plan shape alone; this is the metadata-driven
half. Every rule reads the estimator's `RelStats` and fires **only** on `Provenance.EXACT` proof: a
sketched ndv or a learned row count is a guess, and a guess that drops a group key or folds an
aggregate is silent data corruption. The SQL semantics each rule is reasoned against:

* Grouping changes the **row count**. A key may only be dropped when a *proven-unique* key
  determines it — EXACT `ndv >= rows`. Under either NULL convention that bound implies every group
  holds one row: if ndv excludes NULLs it forces zero nulls and all-distinct values; if it counts
  the NULL group as a value it admits at most one null row.
* NULLs group together (one NULL group); `COUNT(x)` skips NULLs, `COUNT(*)` does not.
* Over **empty** input a *grouped* aggregate emits 0 rows but a *global* one emits 1 row (COUNT =
  0, else NULL). Folds that would change that are gated on a proven-non-empty input, and no rule
  leaves an aggregate with neither key nor function.
* Every rewrite goes through `_checked`: refused unless the rewritten subtree's schema is
  *identical* — folding an aggregate is where a type shifts (`SUM` widens, `Lit(5)` is not Int32).
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# Shared with `agg_extra` rather than re-spelled here: copy-paste is the one *wrong* way to share
# (python-quality.md), so the aggregate-rewrite helpers keep a single home.
from batcher.kyber.rules.extra.agg_extra import (
    _count_star_alias,
    _fresh_alias,
    _project_over,
    _used_aliases,
)
from batcher.plan.expr_ir import AggExpr, Col, Expr, Lit, referenced_columns, when
from batcher.plan.logical import Aggregate, AggregateSpec, Limit, LogicalPlan, Project, Projection
from batcher.plan.stats import Provenance, RelStats
from batcher.plan.types import infer_type

__all__ = [
    "count_distinct_of_unique_column",
    "count_of_non_null_column",
    "drop_aggregate_over_single_row_input",
    "drop_dead_aggregate_output",
    "drop_group_key_functionally_determined_by_another",
    "global_count_star_from_exact_cardinality",
    "global_min_max_from_exact_bounds",
    "mean_of_constant_column",
    "merge_adjacent_aggregates_when_second_is_over_group_keys",
    "min_max_of_constant_column",
    "sum_of_constant_column",
]

# Over a one-row group these are that row's input value — and their output type *is* the input type,
# so replacing them by the input expression is type-neutral (`_checked` proves it, and so rejects a
# mistyped `bool_and(<int>)`). The counts are 1 for a non-null input, 0 for a NULL one (Int64).
_ROW_VALUE_AGGS = frozenset({"min", "max", "mode", "bool_and", "bool_or"})
_ROW_COUNT_AGGS = frozenset({"count", "count_distinct"})
# Averaging aggregates: over a constant column their value is that constant. Float64 out.
_MEAN_AGGS = frozenset({"mean", "median"})
# The types the engine defines `MIN` on: an ordered scalar. Grouping by a list column is legal, but
# `min` over one is a runtime error — hence `_min_is_defined`.
_ORDERED = (pa.types.is_integer, pa.types.is_floating, pa.types.is_temporal, pa.types.is_string)


def _child_stats(node: LogicalPlan, ctx: OptimizerContext | None) -> RelStats | None:
    """The estimator's statistics for `node`'s input, or None when there is no context."""
    if ctx is None:
        return None
    return ctx.estimator.estimate(node.input)  # type: ignore[attr-defined]


def _unique_column(stats: RelStats, name: str) -> bool:
    """Whether `name` provably holds a different value in every row — EXACT ``ndv >= rows``.

    Both facets must be EXACT; an HLL-derived ndv is a `SKETCH`, never a proof.
    """
    if not stats.rows_exact:
        return False
    stat = stats.column(name)
    return stat.provenance is Provenance.EXACT and stat.ndv is not None and stat.ndv >= stats.rows


def _constant_value(stats: RelStats, name: str):
    """`name`'s single value when it is *proven* a non-null constant, else None.

    Requires EXACT ``min == max`` and a zero ``null_count`` (a NULL makes it "one value *plus*
    NULLs").

    **Floats are refused outright.** A columnar footer omits NaN from its min/max statistics
    (the Parquet spec drops it; so does the KLL sketch), so a float column holding a NaN *and*
    one other value records ``min == max == that value`` and looks constant when it is not.
    Folding ``max(f)`` of such a column to that value then returns it instead of the NaN that
    SQL's total order — the one our own ``ORDER BY`` uses — makes the true maximum: a wrong
    *result*, not merely a wrong estimate. Signed zero (``-0.0 == 0.0``) is the same hazard in
    miniature. The sibling `global_min_max_from_exact_bounds` refuses floats for exactly this
    reason; sharing the gate keeps a constant-column fold and a bound fold from disagreeing on
    when a float bound may be trusted. (A NaN-aware in-memory source records the NaN in its
    bounds, so ``min != max`` there and the fold already declines — the refusal only costs the
    rare genuinely-constant float column a fold, never a correct one.)
    """
    stat = stats.column(name)
    if stat.provenance is not Provenance.EXACT or stat.null_count != 0:
        return None
    if stat.min is None or stat.min != stat.max:
        return None
    if isinstance(stat.min, float):
        return None
    return stat.min


def _agg_column(agg: AggExpr) -> str | None:
    """The bare column a unary aggregate reads, else None (a second input, a param, an expr)."""
    if isinstance(agg.input, Col) and agg.input2 is None and agg.param is None:
        return agg.input.name
    return None


def _min_is_defined(expr: Expr, node: LogicalPlan) -> bool:
    """Whether ``MIN(expr)`` is defined over `node`'s schema — an ordered scalar type."""
    schema = node.available_schema()
    t = infer_type(expr, schema) if schema is not None else None
    return t is not None and any(pred(t) for pred in _ORDERED)


def _checked(node: LogicalPlan, rewritten: LogicalPlan | None) -> LogicalPlan | None:
    """`rewritten`, but only when its schema is *identical* to `node`'s (un-inferable = refuse)."""
    if rewritten is None:
        return None
    before, after = node.available_schema(), rewritten.available_schema()
    if before is None or after is None or not before.arrow.equals(after.arrow):
        return None
    return rewritten


def _fold_with(node: Aggregate, stats: RelStats, fold) -> LogicalPlan | None:
    """Fold every aggregate `fold` maps to an expression, re-deriving it in a `Project` above.

    A *global* aggregate whose every output folds is one row of literals: ``Project(Limit(x, 1),
    …)`` is that row — sound only because those folds are gated on a proven **non-empty** input.
    """
    kept: list[AggregateSpec] = []
    folded: dict[str, Expr] = {}
    for spec in node.aggregates:
        value = fold(spec.agg)
        if value is None:
            kept.append(spec)
        else:
            folded[spec.alias] = value
    if not folded:
        return None
    if kept or node.group_keys:
        inner = dataclasses.replace(node, aggregates=tuple(kept))
        return _checked(node, _project_over(node, inner, agg_map=folded))
    if not (stats.rows_exact and stats.rows >= 1):
        return None
    items = tuple(Projection(s.alias, folded[s.alias]) for s in node.aggregates)
    return _checked(node, Project(Limit(node.input, 1), items))


def _swap_aggregates(node: Aggregate, swap) -> LogicalPlan | None:
    """Replace in place each aggregate `swap` rewrites (same alias, same output type)."""
    new: list[AggregateSpec] = []
    changed = False
    for spec in node.aggregates:
        agg = swap(spec.agg)
        if agg is None:
            new.append(spec)
        else:
            new.append(AggregateSpec(spec.alias, agg))
            changed = True
    if not changed:
        return None
    return _checked(node, dataclasses.replace(node, aggregates=tuple(new)))


def _single_row_value(agg: AggExpr) -> Expr | None:
    """The value `agg` takes over a group of exactly one row, or None if not foldable.

    `sum`/`mean`/`var`/`arg_min`… are absent by design: `SUM` widens its type, `AVG` is Float64, a
    Bessel-corrected `VAR` over one row is NULL (not 0), and ``ARG_MIN(x, y)`` is NULL when `y` is.
    """
    if agg.input2 is not None or agg.param is not None:
        return None
    if agg.func == "count_star" and agg.input is None:
        return Lit(1)
    if agg.input is None:
        return None
    if agg.func in _ROW_VALUE_AGGS:
        return agg.input
    if agg.func in _ROW_COUNT_AGGS:
        return when(agg.input.is_null()).then(0).otherwise(1)
    return None


def _fold_to_project(node: Aggregate, source: LogicalPlan) -> LogicalPlan | None:
    """Replace `node` by a `Project` over `source` — sound only when every group is one row.

    One unfoldable aggregate abandons the rewrite (a partial fold keeps the breaker anyway).
    """
    items: list[Projection] = [Projection(k.alias, k.expr) for k in node.group_keys]
    for spec in node.aggregates:
        value = _single_row_value(spec.agg)
        if value is None:
            return None
        items.append(Projection(spec.alias, value))
    if not items:
        return None
    return _checked(node, Project(source, tuple(items)))


@rule(
    name="drop_group_key_functionally_determined_by_another",
    phase=Phase.REWRITE,
    matches=(Aggregate,),
)
def drop_group_key_functionally_determined_by_another(
    node: Aggregate, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Group by a *proven-unique* key alone, carrying the other keys as ``MIN`` of themselves.

    A key proven unique on the input (EXACT ``ndv >= rows``) determines every other column:
    grouping by ``(a, b)`` forms exactly the groups ``a`` does, each of **one** row — so ``b`` is
    carried as ``MIN(b)``, which over a one-row group *is* ``b`` (NULLs and NaNs included), and a
    projection restores the original names and order. Refuses an unproven key or a type without
    ``MIN``.
    """
    if len(node.group_keys) < 2:
        return None
    stats = _child_stats(node, ctx)
    if stats is None:
        return None
    unique = [
        k for k in node.group_keys if isinstance(k.expr, Col) and _unique_column(stats, k.expr.name)
    ]
    if not unique:
        return None
    determinant, used = unique[0], _used_aliases(node)
    carried: dict[str, Expr] = {}
    extra: list[AggregateSpec] = []
    for key in node.group_keys:
        if key.alias == determinant.alias:
            continue
        if not _min_is_defined(key.expr, node.input):
            return None
        alias = _fresh_alias(used) if f"__fd_{key.alias}" in used else f"__fd_{key.alias}"
        used.add(alias)
        extra.append(AggregateSpec(alias, AggExpr("min", key.expr)))
        carried[key.alias] = Col(alias)
    inner = dataclasses.replace(
        node, group_keys=(determinant,), aggregates=node.aggregates + tuple(extra)
    )
    return _checked(node, _project_over(node, inner, key_map=carried))


@rule(name="count_distinct_of_unique_column", phase=Phase.REWRITE, matches=(Aggregate,))
def count_distinct_of_unique_column(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """``COUNT(DISTINCT c)`` → ``COUNT(c)`` when `c` is proven unique on the input.

    A proven-unique column repeats no value, so within *any* group (the groups partition the rows)
    its non-null values are already distinct — and both counts skip NULLs and return Int64. Holds
    globally and over empty input (both 0).
    """
    stats = _child_stats(node, ctx)
    if stats is None:
        return None

    def swap(agg: AggExpr) -> AggExpr | None:
        name = _agg_column(agg)
        if agg.func == "count_distinct" and name is not None and _unique_column(stats, name):
            return AggExpr("count", Col(name))
        return None

    return _swap_aggregates(node, swap)


@rule(name="count_of_non_null_column", phase=Phase.REWRITE, matches=(Aggregate,))
def count_of_non_null_column(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """``COUNT(c)`` → ``COUNT(*)`` when `c` is proven to hold no NULL.

    With an EXACT ``null_count == 0`` the non-null count *is* the row count in every group, and
    ``COUNT(*)`` never reads the column. Both are Int64 and 0 over empty input, so this holds
    grouped and global alike. A filter downgrades the provenance; the rule then stands down.
    """
    stats = _child_stats(node, ctx)
    if stats is None:
        return None

    def swap(agg: AggExpr) -> AggExpr | None:
        name = _agg_column(agg)
        if agg.func != "count" or name is None:
            return None
        stat = stats.column(name)
        null_free = stat.provenance is Provenance.EXACT and stat.null_count == 0
        return AggExpr("count_star", None) if null_free else None

    return _swap_aggregates(node, swap)


@rule(name="min_max_of_constant_column", phase=Phase.REWRITE, matches=(Aggregate,))
def min_max_of_constant_column(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold ``min``/``max``/``mode``/``bool_and``/``bool_or`` of a proven-constant column.

    A column with an EXACT ``min == max`` and no NULLs holds one value ``v`` in every row, so these
    are ``v`` on any **non-empty** group. Every group of a *grouped* aggregate is non-empty; a
    *global* one needs a proven-non-empty input (over 0 rows it must return NULL) — see
    `_fold_with`.
    """
    stats = _child_stats(node, ctx)
    if stats is None:
        return None

    def fold(agg: AggExpr) -> Expr | None:
        name = _agg_column(agg)
        if agg.func not in _ROW_VALUE_AGGS or name is None:
            return None
        value = _constant_value(stats, name)
        return None if value is None else Lit(value)

    return _fold_with(node, stats, fold)


@rule(name="mean_of_constant_column", phase=Phase.REWRITE, matches=(Aggregate,))
def mean_of_constant_column(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold ``mean``/``median`` of a proven-constant **integer** column to that constant.

    The average (and median) of ``n`` copies of ``v`` is ``v``, and both return Float64 — the fold
    is ``Lit(float(v))``. Integer constants only: summing ``n`` copies of a *float* ``v``
    accumulates rounding (``0.1 + 0.1 + 0.1`` is not ``0.3``), so the engine's answer would differ.
    """
    stats = _child_stats(node, ctx)
    if stats is None:
        return None

    def fold(agg: AggExpr) -> Expr | None:
        name = _agg_column(agg)
        if agg.func not in _MEAN_AGGS or name is None:
            return None
        value = _constant_value(stats, name)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        return Lit(float(value))

    return _fold_with(node, stats, fold)


@rule(name="sum_of_constant_column", phase=Phase.REWRITE, matches=(Aggregate,))
def sum_of_constant_column(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """Grouped ``SUM(c)`` → ``v * COUNT(*)`` when `c` is a proven-constant **integer** column.

    Summing ``v`` over ``n`` rows is ``v * n`` and ``COUNT(*)`` is ``n``, so the column is never
    read. Integer constants only (repeated float addition is not bit-identical to one multiply).
    Grouped only: a *global* ``SUM`` over empty input is NULL while ``v * 0`` is 0.
    """
    if not node.group_keys:
        return None
    stats = _child_stats(node, ctx)
    if stats is None:
        return None
    targets: dict[str, int] = {}
    for spec in node.aggregates:
        name = _agg_column(spec.agg)
        value = _constant_value(stats, name) if spec.agg.func == "sum" and name else None
        if isinstance(value, int) and not isinstance(value, bool):
            targets[spec.alias] = value
    if not targets:
        return None
    cnt_alias = _count_star_alias(node)
    extra: tuple[AggregateSpec, ...] = ()
    if cnt_alias is None:
        cnt_alias = _fresh_alias(_used_aliases(node))
        extra = (AggregateSpec(cnt_alias, AggExpr("count_star", None)),)
    kept = [s for s in node.aggregates if s.alias not in targets]
    inner = dataclasses.replace(node, aggregates=tuple(kept) + extra)
    folded = {alias: Lit(value) * Col(cnt_alias) for alias, value in targets.items()}
    return _checked(node, _project_over(node, inner, agg_map=folded))


@rule(name="global_min_max_from_exact_bounds", phase=Phase.REWRITE, matches=(Aggregate,))
def global_min_max_from_exact_bounds(node: Aggregate, ctx: OptimizerContext) -> LogicalPlan | None:
    """A *global* ``MIN(c)``/``MAX(c)`` is the column's EXACT bound — read it, don't scan.

    An EXACT ``[min, max]`` (a footer, a manifest, an in-memory relation) *is* the answer to an
    unfiltered global extreme: SQL's extremes skip NULLs and so do those bounds. Gated on a proven
    non-empty input (over 0 rows both are NULL); floats refused (a bound may be NaN or a signed
    zero).
    """
    if node.group_keys:
        return None
    stats = _child_stats(node, ctx)
    if stats is None or not (stats.rows_exact and stats.rows >= 1):
        return None

    def fold(agg: AggExpr) -> Expr | None:
        name = _agg_column(agg)
        if agg.func not in ("min", "max") or name is None:
            return None
        stat = stats.column(name)
        if stat.provenance is not Provenance.EXACT:
            return None
        value = stat.min if agg.func == "min" else stat.max
        return None if value is None or isinstance(value, float) else Lit(value)

    return _fold_with(node, stats, fold)


@rule(name="global_count_star_from_exact_cardinality", phase=Phase.REWRITE, matches=(Aggregate,))
def global_count_star_from_exact_cardinality(
    node: Aggregate, ctx: OptimizerContext
) -> LogicalPlan | None:
    """A *global* ``COUNT(*)`` over a provably-sized input is that size, as a literal.

    An EXACT row count answers ``COUNT(*)`` without counting — empty input included, where the
    global aggregate emits one row holding 0. Grouped aggregates are untouched (a per-group count
    is not the relation's row count).
    """
    if node.group_keys:
        return None
    stats = _child_stats(node, ctx)
    if stats is None or not stats.rows_exact:
        return None
    rows = int(stats.rows)

    def fold(agg: AggExpr) -> Expr | None:
        return Lit(rows) if agg.func == "count_star" and agg.input is None else None

    return _fold_with(node, stats, fold)


@rule(name="drop_aggregate_over_single_row_input", phase=Phase.REWRITE, matches=(Aggregate,))
def drop_aggregate_over_single_row_input(
    node: Aggregate, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Fold an aggregate whose input is *proven* to hold at most one row into a `Project`.

    One input row means one group of one row: each key is that row's key expression and each
    aggregate that row's value. The empty case splits — a *grouped* aggregate over 0 rows emits 0
    rows, as the projection does; a *global* one emits 1 row (COUNT 0, rest NULL) it cannot, so it
    needs 1 row.
    """
    stats = _child_stats(node, ctx)
    if stats is None or not stats.rows_exact:
        return None
    if not (stats.rows == 1 or (node.group_keys and stats.rows == 0)):
        return None
    return _fold_to_project(node, node.input)


@rule(
    name="merge_adjacent_aggregates_when_second_is_over_group_keys",
    phase=Phase.REWRITE,
    matches=(Aggregate,),
)
def merge_adjacent_aggregates_when_second_is_over_group_keys(
    node: Aggregate, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Fold an aggregate that re-groups an aggregate by *all* of its group keys.

    The inner aggregate emits one row per distinct key tuple, so re-grouping by those same keys
    forms groups of exactly one row: the grouping is the identity and the outer aggregates are
    single-row values. The keys must be plain columns covering *every* inner key — a subset would
    merge groups.
    """
    inner = node.input
    if not isinstance(inner, Aggregate) or not inner.group_keys:
        return None
    outer_keys: set[str] = set()
    for key in node.group_keys:
        if not isinstance(key.expr, Col):
            return None
        outer_keys.add(key.expr.name)
    if outer_keys != {k.alias for k in inner.group_keys}:
        return None
    return _fold_to_project(node, inner)


@rule(name="drop_dead_aggregate_output", phase=Phase.PUSHDOWN, matches=(Project,))
def drop_dead_aggregate_output(node: Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Project(Aggregate(x, keys, aggs), items)` → drop the `aggs` no `items` expression reads.

    The column pruner narrows an aggregate's *input* columns but keeps every function it finds, so
    an unread output still maintains a whole accumulator (a hash set, a sketch) for a discarded
    column. Groups come from the keys alone, so dropping one changes neither rows nor surviving
    values.
    """
    agg = node.input
    if not isinstance(agg, Aggregate):
        return None
    used: set[str] = set()
    for item in node.items:
        used |= referenced_columns(item.expr)
    kept = tuple(spec for spec in agg.aggregates if spec.alias in used)
    if len(kept) == len(agg.aggregates) or (not kept and not agg.group_keys):
        return None
    return dataclasses.replace(node, input=dataclasses.replace(agg, aggregates=kept))
