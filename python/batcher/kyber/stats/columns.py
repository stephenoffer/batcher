"""Per-operator column-statistics propagation.

Alongside row counts, the estimator carries per-column `ColumnStat`
(min/max/null_count/ndv) through the plan so aggregate, pruning, and existence
shortcuts can be answered from metadata. The cardinal rule is provenance
discipline: an operator may carry a column's *values* forward only as strongly
as it can still vouch for them. A `Sort` preserves the exact value set
(`EXACT` survives); a `Filter` or `Limit` keeps min/max as valid *bounds* but
must downgrade away from `EXACT` because it may have dropped the extremes. The
`weakest`/`downgrade` combiners in `plan.stats` are the only way provenance ever
changes, so nothing can silently over-claim.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from batcher.kyber.stats.aggregate_columns import (
    global_aggregate_columns,
    grouped_aggregate_columns,
)
from batcher.kyber.stats.constants import constant_projection_stat
from batcher.kyber.stats.derived import derived_projection_stat
from batcher.plan.expr_ir import Cast, Col, Lit
from batcher.plan.logical import AsofJoin, Join, Projection
from batcher.plan.schema import SchemaRef
from batcher.plan.stats import ColumnStat, Provenance, RelStats, weakest
from batcher.plan.types import DTYPE_REGISTRY

__all__ = [
    "asof_join_columns",
    "distinct_columns",
    "filter_columns",
    "global_aggregate_columns",
    "grouped_aggregate_columns",
    "join_columns",
    "limit_columns",
    "project_columns",
    "sample_columns",
    "scan_columns",
    "union_columns",
    "unnest_columns",
    "unpivot_columns",
    "window_columns",
]


def scan_columns(
    source_columns: Mapping[str, ColumnStat],
    learned: Mapping[str, ColumnStat],
) -> dict[str, ColumnStat]:
    """Seed a `Scan`'s column stats from the connector's declared statistics,
    supplemented by the statistics measured for **this source** in past runs.

    Source-declared stats are authoritative and carry their own provenance (footer min/max
    is `EXACT`, byte-truncated string bounds weaker). The learned bundle — an HLL distinct
    count, a KLL quantile grid, Misra-Gries top values, a measured byte width, none of them
    exact — fills in around them.

    The `ndv` merge is the one that matters. A Parquet footer gives EXACT min/max and null
    counts but **never a distinct count**, so the only ndv such a column can have is a
    measured (HLL) one. This used to refuse it — because a `ColumnStat` carried one provenance
    for the whole bundle, so attaching an approximate ndv to an EXACT column would have tagged
    it EXACT and let it answer `count_distinct`. The cost of that refusal was severe: every
    Parquet column reached the optimizer with no ndv, join cardinality fell back to
    `max(|L|, |R|)`, and join ordering went blind (TPC-H q9 applied its most selective filter
    last). `ColumnStat.ndv_provenance` now carries the ndv's *own* tag, so a measured count
    rides alongside exact bounds while `ndv_is_exact` still refuses to answer from it.

    The descriptive statistics (quantiles, mcv, avg_bytes) never carried that risk — nothing
    answers a query from them — so they attach to any column, exact or not.
    """
    cols: dict[str, ColumnStat] = dict(source_columns)
    for name, measured in learned.items():
        existing = cols.get(name)
        if existing is None:
            cols[name] = measured
            continue
        take_ndv = measured.ndv is not None and existing.ndv is None
        cols[name] = dataclasses.replace(
            existing,
            ndv=measured.ndv if take_ndv else existing.ndv,
            # A measured ndv is a sketch, whatever the bundle's bounds are worth.
            ndv_provenance=Provenance.SKETCH if take_ndv else existing.ndv_provenance,
            quantiles=existing.quantiles or measured.quantiles,
            mcv=existing.mcv or measured.mcv,
            avg_bytes=existing.avg_bytes or measured.avg_bytes,
        )
    return cols


def project_columns(
    items: tuple[Projection, ...],
    child: RelStats,
    input_schema: SchemaRef | None = None,
) -> dict[str, ColumnStat]:
    """Project/select output column stats.

    A `col(x)` output carries `x`'s stats through under its alias (exact stays
    exact — projection touches no values); a literal becomes a constant column
    (`min == max == value`, ndv 1, no nulls, `EXACT`); an *identity* `cast` (the
    target type equals the column's own type — the redundant cast the FFI boundary
    already performed on a narrow numeric) is a no-op that carries the source stats
    through unchanged (`input_schema` supplies the source type); any other
    expression is dropped (its output distribution is unknown).
    """
    out: dict[str, ColumnStat] = {}
    for item in items:
        if isinstance(item.expr, Col):
            src = child.columns.get(item.expr.name)
            if src is not None:
                out[item.alias] = src
        elif isinstance(item.expr, Lit):
            value = item.expr.value
            out[item.alias] = ColumnStat(
                min=value, max=value, null_count=0, ndv=1, provenance=Provenance.EXACT
            )
        elif isinstance(item.expr, Cast):
            carried = _identity_cast_column(item.expr, input_schema, child)
            if carried is not None:
                out[item.alias] = carried
        else:
            # A fully-constant expression folds to an EXACT single value; otherwise a
            # monotonic transform of one column (`x + k`, `x * pos`, `-x`) carries its bounds
            # through so a downstream range predicate on the derived column stays sharp.
            derived = constant_projection_stat(item.expr, child) or derived_projection_stat(
                item.expr, child
            )
            if derived is not None:
                out[item.alias] = derived
    return out


def _identity_cast_column(
    cast: Cast, input_schema: SchemaRef | None, child: RelStats
) -> ColumnStat | None:
    """The carried stat for a `Cast(Col(x), T)` that is a provable no-op, else None.

    Only an *identity* cast — `T` equal to `x`'s own (post-widening) type — is
    treated as value-preserving: it changes nothing, so `x`'s full `ColumnStat`
    (including `EXACT` min/max/ndv) carries through under the output alias. Any
    value-changing cast (a widening/narrowing/opaque conversion, or a `try_cast`
    that can turn a failed conversion into a null) is dropped — its output
    distribution can no longer be vouched for from the input's stats.
    """
    if cast.try_cast or not isinstance(cast.input, Col):
        return None
    if input_schema is None or not input_schema.has(cast.input.name):
        return None
    target = DTYPE_REGISTRY.get(cast.dtype)
    if target is None or not input_schema.field(cast.input.name).type.equals(target):
        return None
    return child.columns.get(cast.input.name)


def filter_columns(child: RelStats, max_ndv: float | None = None) -> dict[str, ColumnStat]:
    """Filter output column stats: min/max survive as *bounds* (a filter can only
    shrink the value range), but provenance drops to `DEFAULT` because the
    extremes may have been removed. null_count becomes unknown; ndv is an upper bound.

    The measured distributional stats (quantiles, mcv, avg_bytes) and the totals
    (total_sum, mean) carry through by `downgrade`, which weakens provenance but keeps the
    values. They must: they describe the column's *shape*, they are read only to estimate,
    and a relation above a filter is exactly where an estimate is still needed — dropping
    them here would mean any predicate at all blinded every operator above it to the very
    statistics the metadata loop measured (a `WHERE` clause would cost a wide string column
    at the flat 64-byte default and forfeit its broadcast join).

    `max_ndv` (the operator's surviving row count) caps each column's distinct count: a
    learned ndv reflects the *unfiltered* source and can exceed the rows a selective filter
    leaves, and a stale-large ndv deflates a downstream join estimate (`|L||R|/max(ndv)`) —
    an under-budget risk. Capping only *lowers* ndv, which *raises* that estimate (the safe
    direction), exactly as `join_columns` already caps a preserved column at the output rows.
    """
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        ndv = stat.ndv
        if max_ndv is not None and ndv is not None and ndv > max_ndv:
            ndv = max(1.0, max_ndv)
        out[name] = dataclasses.replace(
            stat.downgrade(Provenance.DEFAULT),
            null_count=None,  # filter may drop nulls; count no longer known
            ndv=ndv,
        )
    return out


def limit_columns(child: RelStats, max_ndv: float | None = None) -> dict[str, ColumnStat]:
    """Limit output column stats: like a filter, min/max are retained as bounds
    but downgraded (a prefix of rows may exclude the extremes). `max_ndv` caps the
    distinct count at the surviving rows (see `filter_columns`)."""
    return filter_columns(child, max_ndv)


def sample_columns(child: RelStats, max_ndv: float | None = None) -> dict[str, ColumnStat]:
    """Sample output column stats: a sample is a row-shrinking operator, so min/max
    survive only as *bounds* and are downgraded from `EXACT` (the sampled subset may
    drop the extremes), exactly like a filter/limit. `max_ndv` caps the distinct count
    at the sampled rows — a 1% sample cannot hold the source's full distinct count."""
    return filter_columns(child, max_ndv)


def unnest_columns(node, child: RelStats) -> dict[str, ColumnStat]:
    """Explode output column stats: every column except the exploded one repeats once per
    element, so its *values* are preserved — carried as downgraded bounds (the fan-out
    changes row counts, so null_count and EXACT no longer hold). The exploded list column is
    replaced by its element values, whose distribution is unknown, so it is dropped. Without
    this an explode blinded every operator above it (a downstream join over a passthrough key
    lost its bounds and byte width, forfeiting a broadcast)."""
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        if name == node.column:
            continue
        out[name] = dataclasses.replace(stat.downgrade(Provenance.DEFAULT), null_count=None)
    return out


def unpivot_columns(node, child: RelStats) -> dict[str, ColumnStat]:
    """Unpivot output column stats: the `index` columns repeat once per melted column, so
    their values are preserved (downgraded bounds, null_count dropped). The `variable`/`value`
    columns hold a new melted distribution and are left unknown."""
    keep = set(node.index)
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        if name in keep:
            out[name] = dataclasses.replace(stat.downgrade(Provenance.DEFAULT), null_count=None)
    return out


# Window functions whose output range is data-independent, so a downstream QUALIFY on them
# (`WHERE percent_rank < 0.1`) gets a sharp selectivity no other bound could supply. The
# *ranking* functions (`row_number`/`rank`) are deliberately excluded: their `[1, rows]` range
# would make a partitioned `rank <= k` estimate `k/rows`, which under-counts (each partition
# contributes k rows) — the unsafe direction. `percent_rank`/`cume_dist` are `[0, 1]` within
# *every* partition, so the bound is exact regardless of partitioning.
_UNIT_RANGE_WINDOW_FUNCS = frozenset({"percent_rank", "cume_dist"})


def window_columns(node, child: RelStats) -> dict[str, ColumnStat]:
    """Window output column stats: a `Window` is strictly row-count preserving and
    only *appends* function columns, so every input column's stats carry through
    unchanged — `EXACT` survives (no value is added, removed, or reordered). Most appended
    function columns are left unknown, but a `percent_rank`/`cume_dist` output is bounded to
    `[0, 1]` for free — a data-independent range that sharpens a downstream percentile filter.
    """
    out = dict(child.columns)
    for spec in node.functions:
        if spec.func in _UNIT_RANGE_WINDOW_FUNCS:
            out[spec.alias] = ColumnStat(min=0.0, max=1.0, provenance=Provenance.DEFAULT)
    return out


def distinct_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Distinct output column stats: dedup preserves the exact *value set*, so
    min/max/ndv pass through at their original provenance; null_count is no
    longer known (dedup collapses duplicate nulls to one).

    The measured width carries through (dedup drops rows, it does not change how wide a
    value is), and so does the quantile grid as a description of the value *set*. The
    most-common-values do not: dedup collapses every duplicate, which is precisely the
    frequency information an MCV records — a value held by 90% of rows is one row after a
    `DISTINCT`, so carrying its frequency forward would claim a skew that no longer exists.
    """
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        out[name] = dataclasses.replace(
            stat,
            null_count=None,
            total_sum=None,  # sum changes under dedup
            mean=None,  # so does the mean
            mcv=None,  # dedup destroys frequency; see above
        )
    return out


def union_columns(children: list[RelStats], output_names: list[str]) -> dict[str, ColumnStat]:
    """UNION ALL output column stats, by output position.

    For each output column, min = min over branches, max = max over branches;
    null_count = sum over branches. Each is `EXACT` only when every branch's
    corresponding stat is `EXACT` and present. ndv is left unknown (distinct
    values may overlap across branches).
    """
    out: dict[str, ColumnStat] = {}
    # Resolve each branch's columns in its own output order, aligned by position.
    branch_cols = [list(c.columns.values()) for c in children]
    for pos, name in enumerate(output_names):
        stats: list[ColumnStat] = []
        for bi, branch in enumerate(children):
            # Prefer name match; fall back to positional alignment across branches.
            if name in branch.columns:
                stats.append(branch.columns[name])
            elif pos < len(branch_cols[bi]):
                stats.append(branch_cols[bi][pos])
        if len(stats) != len(children):
            continue  # a branch lacks this column → can't combine safely
        out[name] = _merge_union_column(stats)
    return out


def _merge_union_column(stats: list[ColumnStat]) -> ColumnStat:
    prov = weakest(*(s.provenance for s in stats))
    mins = [s.min for s in stats if s.min is not None]
    maxs = [s.max for s in stats if s.max is not None]
    nulls = [s.null_count for s in stats]
    widths = [s.avg_bytes for s in stats if s.avg_bytes is not None]
    all_min = len(mins) == len(stats)
    all_max = len(maxs) == len(stats)
    all_null = all(n is not None for n in nulls)
    return ColumnStat(
        min=_safe_min(mins) if all_min else None,
        max=_safe_max(maxs) if all_max else None,
        null_count=sum(n for n in nulls if n is not None) if all_null else None,
        ndv=None,
        provenance=prov,
        # A width is a per-row property, so the union's is the widest branch's — the safe
        # direction for the memory/broadcast sizing it feeds. Quantiles and MCVs describe a
        # *distribution* and do not merge by any sound rule here, so they are dropped.
        avg_bytes=max(widths) if widths else None,
    )


def join_columns(
    node: Join, left: RelStats, right: RelStats, out_rows: float | None = None
) -> dict[str, ColumnStat]:
    """Column stats for a join's output: each side's values as downgraded *bounds*.

    A join only removes rows from a side or repeats them (an FK match), never invents
    a new value, so a preserved column's `min`/`max` stay valid *bounds* — but the
    extremes may be dropped and provenance must fall from `EXACT` (a join is a
    row-shrinking/duplicating operator). `null_count` is dropped (a match may duplicate
    rows or an outer join add nulls). The membership bloom survives — a value absent
    from a side stays absent in any join output of it.

    `ndv` is carried forward as `min(ndv_in, out_rows)`. Because a join invents no
    values, the output's distinct count for a preserved column can only *fall* (rows
    are dropped) — never rise — and it is trivially bounded by the output row count;
    the minimum of the two is therefore a sound upper bound and the standard Selinger
    estimate. Dropping it instead (the previous behaviour) left every join *above* a
    join with no key NDV, so `_estimate_join` fell back to `max(|L|, |R|)` and could
    not see that an upstream selective join shrinks the pipeline — which is what
    steered TPC-H Q9 into multi-gigabyte `lineitem ⋈ partsupp ⋈ orders` intermediates
    before joining the 5%-selective `part`. `out_rows` is the join's estimated output
    cardinality; when unknown, `ndv` is dropped as before.
    """
    out: dict[str, ColumnStat] = {}
    for o in node.output:
        side = left if o.side == "left" else right
        src = side.columns.get(o.name)
        if src is not None:
            out[o.alias] = dataclasses.replace(
                src.downgrade(Provenance.DEFAULT),
                null_count=None,
                ndv=_join_ndv(src.ndv, out_rows),
                total_sum=None,  # matching duplicates rows, so a recorded sum no longer holds
                mean=None,
                # The measured *width* survives — a join changes which rows are present, not
                # how many bytes one holds — and it is what keeps byte-true memory and
                # broadcast sizing alive above a join. Frequencies (mcv) do not: a join
                # re-weights the value distribution by its match multiplicity.
                mcv=None,
            )
    return out


def asof_join_columns(node: AsofJoin, left: RelStats, right: RelStats) -> dict[str, ColumnStat]:
    """Column stats for an ASOF join's output.

    ASOF is **left-style**: every left row is emitted exactly once, so the left columns'
    values are exactly the left input's — their full stats (EXACT included) carry through
    unchanged. A right column is preserved where matched and NULL where not, and the nearest
    match can repeat or skip right rows, so it survives only as downgraded *bounds* with the
    null count dropped — the same treatment `join_columns` gives a preserved side. Without
    this the estimator returned no columns at all, blinding every operator above an ASOF join
    to the very statistics (bounds, ndv, byte width) that drive its cost and broadcast sizing.
    """
    out: dict[str, ColumnStat] = {}
    for o in node.output:
        if o.side == "left":
            src = left.columns.get(o.name)
            if src is not None:
                out[o.alias] = src  # every left row emitted once → values unchanged
        else:
            src = right.columns.get(o.name)
            if src is not None:
                out[o.alias] = dataclasses.replace(
                    src.downgrade(Provenance.DEFAULT), null_count=None, mcv=None
                )
    return out


def _join_ndv(ndv: float | None, out_rows: float | None) -> float | None:
    """A preserved column's distinct count after a join: `min(ndv, out_rows)`."""
    if ndv is None or out_rows is None:
        return None
    return max(1.0, min(ndv, out_rows))


def _safe_min(values: list):
    try:
        return min(values)
    except TypeError:
        return None


def _safe_max(values: list):
    try:
        return max(values)
    except TypeError:
        return None
