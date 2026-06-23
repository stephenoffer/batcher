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

from collections.abc import Mapping

from batcher.plan.expr_ir import Col, Lit
from batcher.plan.logical import Aggregate, Projection
from batcher.plan.stats import ColumnStat, Provenance, RelStats, weakest

__all__ = [
    "distinct_columns",
    "filter_columns",
    "global_aggregate_columns",
    "limit_columns",
    "project_columns",
    "scan_columns",
    "union_columns",
]


def scan_columns(
    source_columns: Mapping[str, ColumnStat],
    learned_ndv: Mapping[str, float],
) -> dict[str, ColumnStat]:
    """Seed a `Scan`'s column stats from the connector's declared statistics,
    supplemented by learned (execution-measured) distinct counts.

    Source-declared stats are authoritative and carry their own provenance
    (footer min/max is `EXACT`, byte-truncated string bounds weaker). A learned
    ndv (from an HLL sketch over a past run — never exact) fills in only where it
    cannot mislabel an exact statistic: an `EXACT` footer column is left
    untouched, because a single `ColumnStat` carries one provenance and tagging
    its ndv `EXACT` would let an approximate distinct count wrongly answer
    `count_distinct`. The learned ndv lands only on footerless or already-inexact
    columns, where it informs `approx_count_distinct` and cost.
    """
    cols: dict[str, ColumnStat] = dict(source_columns)
    for name, ndv in learned_ndv.items():
        if ndv <= 0:
            continue
        existing = cols.get(name)
        if existing is None:
            cols[name] = ColumnStat(ndv=float(ndv), provenance=Provenance.SKETCH)
        elif existing.ndv is None and existing.provenance is not Provenance.EXACT:
            cols[name] = ColumnStat(
                min=existing.min,
                max=existing.max,
                null_count=existing.null_count,
                ndv=float(ndv),
                total_sum=existing.total_sum,
                provenance=existing.provenance,
                bloom=existing.bloom,
            )
    return cols


def project_columns(items: tuple[Projection, ...], child: RelStats) -> dict[str, ColumnStat]:
    """Project/select output column stats.

    A `col(x)` output carries `x`'s stats through under its alias (exact stays
    exact — projection touches no values); a literal becomes a constant column
    (`min == max == value`, ndv 1, no nulls, `EXACT`); any other expression is
    dropped (its output distribution is unknown).
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
    return out


def filter_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Filter output column stats: min/max survive as *bounds* (a filter can only
    shrink the value range), but provenance drops to `DEFAULT` because the
    extremes may have been removed. null_count and ndv become unknown/looser."""
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        out[name] = ColumnStat(
            min=stat.min,
            max=stat.max,
            null_count=None,  # filter may drop nulls; count no longer known
            ndv=stat.ndv,  # an upper bound after filtering
            provenance=weakest(stat.provenance, Provenance.DEFAULT),
            bloom=stat.bloom,  # absence in the base bloom persists in any subset
        )
    return out


def limit_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Limit output column stats: like a filter, min/max are retained as bounds
    but downgraded (a prefix of rows may exclude the extremes)."""
    return filter_columns(child)


def distinct_columns(child: RelStats) -> dict[str, ColumnStat]:
    """Distinct output column stats: dedup preserves the exact *value set*, so
    min/max/ndv pass through at their original provenance; null_count is no
    longer known (dedup collapses duplicate nulls to one)."""
    out: dict[str, ColumnStat] = {}
    for name, stat in child.columns.items():
        out[name] = ColumnStat(
            min=stat.min,
            max=stat.max,
            null_count=None,
            ndv=stat.ndv,
            total_sum=None,  # sum changes under dedup
            provenance=stat.provenance,
            bloom=stat.bloom,  # dedup adds no value → absence proof still holds
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
    all_min = len(mins) == len(stats)
    all_max = len(maxs) == len(stats)
    all_null = all(n is not None for n in nulls)
    return ColumnStat(
        min=_safe_min(mins) if all_min else None,
        max=_safe_max(maxs) if all_max else None,
        null_count=sum(n for n in nulls if n is not None) if all_null else None,
        ndv=None,
        provenance=prov,
    )


def global_aggregate_columns(node: Aggregate, child: RelStats) -> dict[str, ColumnStat]:
    """Derive the one output row's column stats for a *global* (no-key) aggregate.

    Each output alias becomes a constant column whose `min == max == <value>`
    when that aggregate is derivable from the child's exact stats:

      - `count(*)`            = child.rows                  (needs exact rows)
      - `count(col)`          = child.rows - null_count(col)(needs exact rows + null_count)
      - `min(col)` / `max(col)` = col.min / col.max         (needs exact col bound)
      - `sum(col)`            = col.total_sum               (needs a recorded sum)
      - `count_distinct(col)` = col.ndv                     (needs *exact* ndv only)

    Anything not exactly derivable is omitted, so a downstream reader sees only
    answerable aggregates (provenance `EXACT`).
    """
    out: dict[str, ColumnStat] = {}
    for spec in node.aggregates:
        value = _derive_scalar_aggregate(spec.agg.func, spec.agg.input, child)
        if value is not None:
            out[spec.alias] = ColumnStat(
                min=value, max=value, null_count=0, ndv=1, provenance=Provenance.EXACT
            )
    return out


def _derive_scalar_aggregate(func: str, input_expr, child: RelStats):
    """The exact scalar value of one global aggregate, or None if not derivable."""
    col_name = input_expr.name if isinstance(input_expr, Col) else None
    if func == "count_star":
        # count(*) = total rows, regardless of any column.
        return int(child.rows) if child.rows_exact else None
    if func == "count":
        # count(col) = rows - nulls(col); needs exact rows and an exact null count.
        if not child.rows_exact or col_name is None:
            return None
        stat = child.columns.get(col_name)
        if stat is None or stat.provenance is not Provenance.EXACT or stat.null_count is None:
            return None
        return int(child.rows - stat.null_count)
    if col_name is None:
        return None
    stat = child.columns.get(col_name)
    if stat is None or stat.provenance is not Provenance.EXACT:
        return None
    if func == "min":
        return stat.min
    if func == "max":
        return stat.max
    if func == "sum":
        return stat.total_sum
    if func == "count_distinct":
        return None if stat.ndv is None else int(stat.ndv)
    return None


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
