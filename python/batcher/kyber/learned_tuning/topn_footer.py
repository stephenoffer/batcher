"""First-run top-N bounds, derived from Parquet row-group statistics rather than remembered.

`topn_bound` turns ``ORDER BY x DESC LIMIT k`` into a selective filter by remembering the k-th
best value from the *previous* run. That leaves the first run of every shape reading the whole
relation, which is the run an interactive user and a benchmark both see. DuckDB closes the same
gap from the other end: `RowGroupPruner` (`src/optimizer/row_group_pruner.cpp`) orders a
table's row groups by the sort key's zone map and stops reading once `k` rows are provably in
hand. This module is that technique, expressed the way Batcher already expresses a top-N bound:
as a `Filter` below the `Sort` that predicate pushdown carries to the reader.

## Why the bound is provable, not guessed

Take a descending sort. Every non-null row of a row group is at least that group's footer
`min`. Walk the groups from the largest `min` down, summing each group's non-null row count,
and stop at the first group that brings the sum to `k`; call its `min` **T**. At least `k` rows
are now known to be `>= T`, so the k-th best value is `>= T`, and every row `< T` is strictly
worse than the k-th best row. ``x >= T`` therefore removes only rows that cannot appear in the
answer, *including* rows tied with the k-th value, which it keeps. The ascending case mirrors
it with each group's `max`.

Unlike a remembered bound, nothing here can be stale: the statistics describe the files the scan
is about to read. The conductor still counts the result when it has one, so a footer that lies
about its row counts costs a re-run rather than a wrong answer.

## What it declines, and why each is load-bearing

* **Nulls first.** The filter drops nulls, which a nulls-first ordering ranks ahead of every
  value. The same rule `topn_bound._seedable_key` applies.
* **Floating point.** Parquet omits NaN from a column's min/max while SQL ranks NaN greatest, so
  a footer `max` is not an upper bound on a float column.
* **Strings and binary.** A writer may truncate their statistics.
* **Nanosecond timestamps and unsigned 64-bit integers.** The first has no exact Python literal
  to carry the bound in, the second exceeds what the engine accepts at its boundary.
* **Anything between the `Sort` and the scan but a column-renaming `Project`.** A `Filter` below
  the sort means a group's row count is no longer the number of rows that reach it, and the
  proof above counts rows.
* **A bound that prunes little.** If the groups that could hold a row beyond `T` are most of the
  relation, the filter costs an evaluation per row and saves no I/O. Rejected on the footer's
  own row counts, before anything is read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher.kyber.learned_tuning.topn_bound import (
    _MAX_SEEDED_LIMIT,
    TopNSeed,
    _bound_predicate,
    _seedable_key,
    _topn_shape,
)
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Filter, Limit, LogicalPlan, Project, Scan, Sort

if TYPE_CHECKING:
    from batcher.io.stats import RowGroupBounds

__all__ = ["TopNScanKey", "footer_topn_seed", "topn_scan_key"]

# Above this fraction of the relation surviving the footer prune, the bound is not applied.
#
# The filter is evaluated on every row the reader returns, and the reader only saves work on
# the row groups it skips. A bound that keeps most groups adds that evaluation to a scan that
# is barely smaller, and the in-engine `TopNBound` already skips morsels on the same
# principle once rows arrive. Half is where the skipped I/O is no longer the larger term.
_MAX_SURVIVING_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class TopNScanKey:
    """A top-N whose leading key can be traced straight to one scan column.

    Attributes:
        source_id: The scan's index into the plan's sources.
        column: The key's name **in the source**, after undoing any `Project` renames.
        descending: The leading key's direction.
        k: Rows the top-N asks for.
    """

    source_id: int
    column: str
    descending: bool
    k: int


def topn_scan_key(plan: LogicalPlan) -> TopNScanKey | None:
    """The scan column a footer bound would be computed on, or `None` for any other shape.

    Args:
        plan: The plan as written, before optimization.

    Returns:
        The traced key, or `None` when `plan` is not a seedable top-N directly over a scan.
    """
    shape = _topn_shape(plan)
    if shape is None:
        return None
    sort, k = shape
    if k <= 0 or k > _MAX_SEEDED_LIMIT:
        return None
    keyed = _seedable_key(sort)
    if keyed is None:
        return None
    column, descending = keyed
    node = sort.input
    while isinstance(node, Project):
        source = next((i.expr for i in node.items if i.alias == column), None)
        if not isinstance(source, Col):
            return None
        column, node = source.name, node.input
    if not isinstance(node, Scan):
        return None
    field = node.schema.arrow.field(column) if column in node.schema.arrow.names else None
    if field is None or not _footer_bound_is_exact(field.type):
        return None
    return TopNScanKey(node.source_id, column, descending, k)


def footer_topn_seed(
    plan: LogicalPlan, key: TopNScanKey, bounds: Sequence[RowGroupBounds]
) -> TopNSeed | None:
    """Rewrite a top-N to filter at a bound its source's row-group statistics prove.

    Args:
        plan: The plan `key` was traced from.
        key: What `topn_scan_key` returned for `plan`.
        bounds: The scanned source's per-row-group statistics for `key.column`.

    Returns:
        The seeded plan, or `None` when the statistics cannot prove `k` rows or the bound
        would not prune enough of the relation to pay for itself.
    """
    try:
        threshold = _threshold(key, bounds)
        if threshold is None:
            return None
        total = sum(rg.num_rows for rg in bounds)
        if total <= 0 or _surviving_rows(key, bounds, threshold) > total * _MAX_SURVIVING_FRACTION:
            return None
        shape = _topn_shape(plan)
        if shape is None:
            return None
        sort, _k = shape
        # The predicate names the key as the *sort* sees it, so a renamed column is filtered
        # under its output name, above the Project that renamed it; pushdown undoes the rename.
        guarded = Filter(
            sort.input, _bound_predicate(_sort_column(sort), key.descending, threshold)
        )
        seeded_sort = Sort(guarded, sort.keys, sort.limit)
        seeded: LogicalPlan = (
            seeded_sort if plan is sort else Limit(seeded_sort, plan.n, plan.offset)
        )
        return TopNSeed(plan=seeded, k=key.k, signature="")
    except Exception as exc:  # pragma: no cover - a hint must never break a query
        note_suppressed("kyber", "seed top-n from row-group statistics", exc)
        return None


def _sort_column(sort: Sort) -> str:
    leading = sort.keys[0].expr
    assert isinstance(leading, Col)  # guaranteed by `_seedable_key`
    return leading.name


def _footer_bound_is_exact(arrow_type: pa.DataType) -> bool:
    """Whether a footer min/max of this type is an exact, literal-expressible bound."""
    if pa.types.is_timestamp(arrow_type):
        return arrow_type.unit != "ns"
    if pa.types.is_integer(arrow_type):
        return arrow_type != pa.uint64()
    return pa.types.is_date(arrow_type)


def _threshold(key: TopNScanKey, bounds: Sequence[RowGroupBounds]) -> Any | None:
    """The value `k` rows provably reach, or `None` when the statistics cannot show `k`."""
    # A descending top-N needs rows *at least* some value, which each group's min guarantees;
    # an ascending one needs rows *at most* some value, which each group's max guarantees.
    edges = [
        (edge, rg.num_rows - nulls)
        for rg in bounds
        if (edge := (rg.mins if key.descending else rg.maxs).get(key.column)) is not None
        and (nulls := rg.null_counts.get(key.column)) is not None
    ]
    edges.sort(key=lambda pair: pair[0], reverse=key.descending)
    reached = 0
    for edge, rows in edges:
        reached += rows
        if reached >= key.k:
            return edge
    return None


def _surviving_rows(key: TopNScanKey, bounds: Sequence[RowGroupBounds], threshold: Any) -> int:
    """Rows in the groups a reader could not prune under the bound."""
    total = 0
    for rg in bounds:
        far = (rg.maxs if key.descending else rg.mins).get(key.column)
        if far is None or (far >= threshold if key.descending else far <= threshold):
            total += rg.num_rows
    return total
