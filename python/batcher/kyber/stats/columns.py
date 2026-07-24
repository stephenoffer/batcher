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
from collections.abc import Iterable, Mapping

from batcher.kyber.stats.aggregate_columns import (
    global_aggregate_columns,
    grouped_aggregate_columns,
)
from batcher.kyber.stats.constants import constant_projection_stat
from batcher.kyber.stats.derived import derived_projection_stat
from batcher.kyber.stats.distribution import (
    distinct_after_selection,
    merge_quantile_grids,
    union_ndv,
)
from batcher.kyber.stats.join_columns import asof_join_columns, join_columns
from batcher.plan.expr_ir import Cast, Col, Lit
from batcher.plan.logical import Projection
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


def filter_columns(
    child: RelStats, max_ndv: float | None = None, shrink_ndv: bool = True
) -> dict[str, ColumnStat]:
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

    `max_ndv` (the operator's surviving row count) bounds each column's distinct count, and
    with `shrink_ndv` the bound is sharpened to the *expected* surviving count rather than
    the trivial `min(ndv, rows)`. Both matter, in the same direction: a learned ndv reflects
    the **unfiltered** source, and a stale-large ndv deflates a downstream join estimate
    (`|L||R|/max(ndv)`) — an under-budget risk. A 1%-selective predicate over 1M distinct
    values leaves ~95K of them, not 1M and not the 100K row cap, and the difference steers
    every group-by, `DISTINCT`, and join above the filter (`distinct_after_selection`).

    `shrink_ndv=False` keeps the plain cap, for a row-shrinking operator whose surviving rows
    are **not** a uniform random subset — a `Limit` takes a *prefix*, which over a sorted or
    clustered relation can hold far fewer distinct values than Yao's random-subset model
    predicts, so only the sound `min(ndv, rows)` upper bound may be claimed there.
    """
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        ndv = stat.ndv
        if max_ndv is not None and ndv is not None:
            if shrink_ndv:
                ndv = distinct_after_selection(ndv, child.rows, max_ndv)
            elif ndv > max_ndv:
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
    distinct count at the surviving rows.

    Deliberately the plain cap and not the random-subset shrinkage a filter gets: a `LIMIT`
    takes the *first* rows, and over a relation sorted or clustered on the column those rows
    can share far fewer distinct values than a uniform sample would, so only the sound
    `min(ndv, rows)` upper bound may be claimed here (see `filter_columns`)."""
    return filter_columns(child, max_ndv, shrink_ndv=False)


def sample_columns(child: RelStats, max_ndv: float | None = None) -> dict[str, ColumnStat]:
    """Sample output column stats: a sample is a row-shrinking operator, so min/max
    survive only as *bounds* and are downgraded from `EXACT` (the sampled subset may
    drop the extremes), exactly like a filter/limit.

    A sample is the one operator whose surviving rows really are a uniform random subset, so
    its distinct count follows Yao's formula exactly rather than the trivial row cap: a 1%
    sample of a column whose values repeat ~100 times keeps ~63% of them, while a 1% sample
    of a unique key keeps one distinct value per row."""
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
    corresponding stat is `EXACT` and present. The distributional statistics merge by the
    identities `UNION ALL` makes available — see `_merge_union_column`.
    """
    out: dict[str, ColumnStat] = {}
    # Resolve each branch's columns in its own output order, aligned by position.
    branch_cols = [list(c.columns.values()) for c in children]
    rows = [c.rows for c in children]
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
        out[name] = _merge_union_column(stats, rows)
    return out


def _merge_union_column(stats: list[ColumnStat], rows: list[float]) -> ColumnStat:
    """Combine one column's per-branch statistics into the `UNION ALL` output's.

    Concatenation is the operator that preserves the most: every branch's rows appear
    unchanged, so several statistics merge *exactly* rather than being dropped.

    * **min/max** — the extremes over the branches.
    * **null_count / total_sum** — additive over branches, exactly.
    * **mean** — the row-weighted mean ``Σ nᵢ μᵢ / Σ nᵢ``, which is the concatenation's true
      mean; averaging the branch means unweighted is only right when the branches are the
      same size.
    * **ndv** — bounded below by the largest branch's and above by the sum; `union_ndv`
      interpolates between those Fréchet bounds under an independent-membership model. It
      carries its **own** `DEFAULT` tag, because it is an estimate even when every branch's
      count is exact — the branches' overlap is unmeasured.
      Dropping it (the previous behaviour) made every join above a partition-union fall back
      to `max(|L|, |R|)`, which is the estimate a partitioned fact table can least afford.
    * **quantiles** — the union's CDF is the row-weighted mixture of the branches' CDFs, an
      exact identity; `merge_quantile_grids` re-grids that mixture.
    * **avg_bytes** — the row-weighted mean width, which is what the concatenation's average
      row actually costs (the widest branch over-charges a union whose wide branch is tiny).
    * **mcv** — dropped. A frequency table merges only if every branch lists the value, and
      absence from a branch's top-k is not evidence of a low frequency there, so any merged
      figure could understate the skew — the one direction that risks an under-sized join.
    """
    prov = weakest(*(s.provenance for s in stats))
    mins = [s.min for s in stats if s.min is not None]
    maxs = [s.max for s in stats if s.max is not None]
    nulls = [s.null_count for s in stats]
    sums = [s.total_sum for s in stats]
    all_min = len(mins) == len(stats)
    all_max = len(maxs) == len(stats)
    all_null = all(n is not None for n in nulls)
    total_rows = sum(r for r in rows if r > 0.0)
    return ColumnStat(
        min=_safe_min(mins) if all_min else None,
        max=_safe_max(maxs) if all_max else None,
        null_count=sum(n for n in nulls if n is not None) if all_null else None,
        total_sum=(
            sum(s for s in sums if s is not None) if all(s is not None for s in sums) else None
        ),
        mean=_row_weighted((s.mean for s in stats), rows, total_rows),
        ndv=union_ndv([s.ndv for s in stats if s.ndv is not None], total_rows or None),
        # The union's distinct count is an *estimate* however exact the branches are — two
        # branches' value sets may overlap by any amount, and nothing here measures which.
        # Without its own tag it would inherit an EXACT bundle provenance and let
        # `count_distinct` answer a `UNION ALL` from a model, which is the one thing the
        # provenance discipline exists to prevent.
        ndv_provenance=Provenance.DEFAULT,
        provenance=prov,
        avg_bytes=_row_weighted((s.avg_bytes for s in stats), rows, total_rows),
        quantiles=merge_quantile_grids([s.quantiles for s in stats], rows),
    )


def _row_weighted(
    values: Iterable[float | None], rows: list[float], total_rows: float
) -> float | None:
    """The row-weighted mean of a per-branch statistic, or None unless every branch has one.

    Requiring every branch is what keeps this sound: a mean or width computed over the
    branches that happen to have measured one is not the concatenation's, it is a different
    relation's.
    """
    paired = [(v, r) for v, r in zip(values, rows, strict=False) if v is not None and r > 0.0]
    if not paired or len(paired) != sum(1 for r in rows if r > 0.0) or total_rows <= 0.0:
        return None
    return sum(v * r for v, r in paired) / total_rows


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
