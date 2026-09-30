"""Narrow the columns that ascend together with the one a filter constrains.

`filter_columns` models a filter's survivors as a random subset of the rows, which is right
for a column the predicate says nothing about and badly wrong for one that moves with the
filtered column. A date dimension is the everyday case: `d_date_sk`, `d_date`, `d_week_seq` and
`d_year` all ascend together in storage order, so `d_year = 1999` keeps one contiguous run of
rows, and that run holds 52 of the table's 10,436 weeks. The random-subset model predicts 359,
and a join on `d_week_seq` above it — each week is seven dates, so the join fans out sevenfold
— was estimated at 2.5M rows against 16.4M actual in TPC-DS q72. The same filter left
`d_date_sk` spanning the whole table, so a later range predicate the run satisfies entirely
was priced as keeping 3% of it.

When the source says which columns ascend (`SourceStatistics.ascending`) and a filter keeps a
contiguous range of one of them, the kept rows are one run, and under the uniformity the range
estimator already assumes, that run occupies the same share of every other ascending column's
range, starting at the same relative position. This module records that as what the
estimator reads: a distinct count scaled by the kept share, and a quantile grid confined to
the run. **It never touches `min`/`max`**: those stay sound bounds, because rules downstream
may read them as proofs, and a position inferred under a uniformity assumption is not a proof.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from batcher.kyber.stats.selectivity.scalars import comparison_col_side
from batcher.plan.expr_ir import Binary, Expr
from batcher.plan.expr_rewrite import split_conjuncts
from batcher.plan.stats import ColumnStat, RelStats, ordinal_with_axis

__all__ = ["narrow_comonotone", "narrow_to_matched_keys"]

# `col OP literal` operators that bound a column from below or above, as written with the
# column on the left. A column on the right flips them (`5 < col` is `col > 5`).
_LOWER = frozenset({"eq", "ge", "gt"})
_UPPER = frozenset({"eq", "le", "lt"})
_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq"}


def narrow_comonotone(
    by_column: Mapping[str, Expr],
    child: RelStats,
    columns: dict[str, ColumnStat],
    kept_share: Callable[[str, Expr], float | None],
) -> dict[str, ColumnStat]:
    """`columns` with every ascending column narrowed to the run a filter keeps of another.

    Args:
        by_column: Each single-column predicate of the filter, keyed by its column.
        child: The filter's input statistics, carrying `ascending` and the pre-filter bounds.
        columns: The filter's output column statistics, updated and returned.
        kept_share: The fraction of a column's non-null rows its predicate keeps, or None.

    Returns:
        `columns`, with narrowed distinct counts and quantile grids where a run is known.
    """
    if len(child.ascending) < 2:
        return columns
    for name, predicate in by_column.items():
        if name not in child.ascending:
            continue
        run = _kept_run(name, predicate, child, kept_share)
        if run is None:
            continue
        start, share = run
        for other in child.ascending:
            if other == name or other not in columns:
                continue
            narrowed = _narrowed(child.columns.get(other), columns[other], start, share)
            if narrowed is not None:
                columns[other] = narrowed
    return columns


def narrow_to_matched_keys(side: RelStats, key: str, other_key: ColumnStat | None) -> RelStats:
    """`side`, as an inner join on `key` leaves it: confined to the keys the other side holds.

    An inner join keeps only the rows whose key the other side holds, so a dimension joined to
    a fact whose keys are a sliver of the dimension's is filtered to that sliver as surely as
    by a `WHERE`. When the dimension's rows ascend with its key, the columns ascending with it
    keep the same share of their distinct values. The case that needs it: TPC-DS
    `store_sales ⋈ date_dim` keeps 1,823 of 73,049 dates, which is five of `d_year`'s 201
    values, and without this a `GROUP BY ss_customer_sk, d_year` over the join is estimated at
    a 2.1x reduction against a true 14x.

    The share is the join's own containment assumption, `ndv(other key) / ndv(key)`, not the
    other key's *range*: a side a filter has already confined to a run keeps its full
    `min`/`max` (those stay sound bounds), so a range test would narrow it a second time.
    TPC-DS q23's dates, already cut to 2000-2003, read as 37 dates instead of 1,461 that way.
    Only `ndv` moves; the bounds stay what they were.

    Args:
        side: One join input's statistics.
        key: That input's join key.
        other_key: The other input's statistics for its matching key.

    Returns:
        `side`, with the ascending columns' distinct counts scaled by the kept share.
    """
    if len(side.ascending) < 2 or key not in side.ascending or other_key is None:
        return side
    mine = side.columns.get(key)
    if mine is None or not mine.ndv or not other_key.ndv:
        return side
    share = other_key.ndv / mine.ndv
    if not 0.0 < share < 1.0:
        return side  # the other side holds as many keys as this one: the join confines nothing
    columns = dict(side.columns)
    for name in side.ascending:
        stat = columns.get(name)
        if name == key or stat is None or not stat.ndv:
            continue
        columns[name] = dataclasses.replace(stat, ndv=max(1.0, stat.ndv * share))
    return dataclasses.replace(side, columns=columns)


def _kept_run(
    name: str,
    predicate: Expr,
    child: RelStats,
    kept_share: Callable[[str, Expr], float | None],
) -> tuple[float, float] | None:
    """`(start, share)` of the contiguous run `predicate` keeps of column `name`, or None.

    `start` is the run's relative position in the column's range and `share` the fraction of
    rows it holds. Only a conjunction of comparisons against literals keeps a contiguous run:
    `!=`, `IN` or a function of the column can keep scattered rows, and those decline.
    """
    stat = child.columns.get(name)
    if stat is None:
        return None
    span = _axis_span(stat)
    if span is None:
        return None
    axis, lo, hi = span
    lower = lo
    for conjunct in split_conjuncts(predicate):
        bound = _comparison(conjunct)
        if bound is None:
            return None
        op, value = bound
        placed = ordinal_with_axis(value)
        if placed is None or placed[0] != axis:
            return None
        if op in _LOWER:
            lower = max(lower, placed[1])
    share = kept_share(name, predicate)
    if share is None or not 0.0 < share < 1.0:
        return None
    # A discrete column's `max` is where its last value's run *starts*, not where it ends:
    # `d` values over `[lo, hi]` each hold `1/d` of the rows, so value `v` begins at
    # `(v - lo) / (hi - lo) * (1 - 1/d)`. Twenty years read off the bare span put 2005's run
    # at 26% of the table instead of 25%, off by most of a year's width.
    position = max(0.0, (lower - lo) / (hi - lo))
    if stat.ndv is not None and stat.ndv >= 1.0:
        position *= 1.0 - 1.0 / stat.ndv
    return min(position, 1.0 - share), share


def _comparison(expr: Expr) -> tuple[str, Any] | None:
    """`(op, literal)` for a `col OP literal` comparison, normalized to the column on the left."""
    if not isinstance(expr, Binary) or expr.op not in _FLIP:
        return None
    side = comparison_col_side(expr)
    if side is None:
        return None
    _, value, col_on_left = side
    return (expr.op if col_on_left else _FLIP[expr.op]), value


def _axis_span(stat: ColumnStat) -> tuple[str, float, float] | None:
    """`(axis, lo, hi)` of a column's bounds on their number line, or None without a span."""
    if stat.min is None or stat.max is None:
        return None
    lo, hi = ordinal_with_axis(stat.min), ordinal_with_axis(stat.max)
    if lo is None or hi is None or lo[0] != hi[0] or not hi[1] > lo[1]:
        return None
    return lo[0], lo[1], hi[1]


def _narrowed(
    before: ColumnStat | None, after: ColumnStat, start: float, share: float
) -> ColumnStat | None:
    """`after` with its distinct count and quantile grid confined to the run, or None."""
    if before is None:
        return None
    span = _axis_span(before)
    if span is None:
        return None
    axis, lo, hi = span
    run_lo = lo + start * (hi - lo)
    run_hi = lo + (start + share) * (hi - lo)
    ndv = after.ndv
    if before.ndv is not None and before.ndv > 0:
        scaled = max(1.0, before.ndv * share)
        ndv = scaled if ndv is None else min(ndv, scaled)
    return dataclasses.replace(
        after, ndv=ndv, quantiles={"axis": axis, "probs": [0.0, 1.0], "values": [run_lo, run_hi]}
    )
