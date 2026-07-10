"""Lazy EXACT column statistics over an immutable in-memory Arrow relation.

Split out of `io/source.py` on a responsibility seam: an in-memory source is immutable, so
its per-column min/max/null-count, distinct count, mean, sum, and per-value counts are
exact and constant. Computed once and reused, they are batcher's learned-metadata moat — an
unfiltered ``MIN``/``MAX``/``COUNT(DISTINCT)``/``AVG``/``SUM``/``COUNT(*) WHERE col = v`` is
answered from metadata on every *subsequent* run instead of re-scanning, which a static
optimizer cannot match.

These are pure compute helpers over a `build` callable that materializes one (narrow-int
widened) column; `InMemorySource` owns the batches, supplies `build`, and memoizes each
result per instance (never under its shape-based `identity`, which different data can
share). `pyarrow.compute` is imported at module top, so a source imports this module lazily
(inside the delegating method) to keep `pc`'s load cost off the cheap non-stats path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from batcher.plan.source_stats import SourceStatistics

# Materializes column `name` as one (widened) chunked array/array, or raises if absent.
ColumnBuilder = Callable[[str], "pa.ChunkedArray | pa.Array"]

# The Arrow errors a stat kernel can raise on an unsupported column — treated as
# "not derivable" (None / skip), never propagated: the metadata answer is optional.
_ARROW_ERRORS = (pa.ArrowInvalid, pa.ArrowNotImplementedError, KeyError)


def column_ndv(build: ColumnBuilder, name: str) -> int | None:
    """EXACT distinct count of `name`'s non-null values, or None if not derivable."""
    try:
        return pc.count_distinct(build(name), mode="only_valid").as_py()
    except _ARROW_ERRORS:
        return None


def column_mean(build: ColumnBuilder, name: str) -> float | None:
    """EXACT average of `name`'s non-null values (a float, matched within tolerance)."""
    try:
        return pc.mean(build(name), skip_nulls=True).as_py()
    except _ARROW_ERRORS:
        return None


def column_sum(build: ColumnBuilder, name: str) -> float | int | None:
    """EXACT total of `name`'s non-null values, or None if not exactly representable.

    A float column's sum is float (matches a run within tolerance). An integer column's sum
    is returned only when exactly representable (``|sum| < 2**53``) — then it also fits
    ``i64`` and equals the engine's overflow-checked ``SUM``; a larger integer sum could
    overflow, so None (fall back to execution, which errors identically).
    """
    try:
        col = build(name)
        if pa.types.is_floating(col.type):
            return pc.sum(col, skip_nulls=True).as_py()
        if pa.types.is_integer(col.type):
            approx = pc.sum(pc.cast(col, pa.float64()), skip_nulls=True).as_py()
            if approx is not None and abs(approx) < 2**53:
                return round(approx)  # exact: an integer sum < 2**53 is exact in f64
    except _ARROW_ERRORS:
        return None
    return None


# SQL comparison → the Arrow compute kernel whose truth count IS the surviving row
# count of ``WHERE col <op> value``. Every kernel yields null for a null cell, so summing
# the boolean counts only non-null matches — exactly SQL's three-valued comparison, where a
# null operand is never true (dropped from every comparison, `=` and `<>` alike).
_PREDICATE_KERNEL = {
    "eq": pc.equal,
    "ne": pc.not_equal,
    "lt": pc.less,
    "le": pc.less_equal,
    "gt": pc.greater,
    "ge": pc.greater_equal,
}


def column_predicate_count(build: ColumnBuilder, op: str, name: str, value: object) -> int | None:
    """EXACT count of rows satisfying ``name <op> value`` (nulls excluded, SQL semantics).

    `op` is one of eq/ne/lt/le/gt/ge. The matching Arrow kernel is null for a null cell, so
    the boolean sum counts exactly the non-null rows the SQL comparison keeps — the surviving
    count of ``WHERE col <op> value`` directly, no null-count bookkeeping. None if unsupported.
    """
    kernel = _PREDICATE_KERNEL.get(op)
    if kernel is None:
        return None
    try:
        col = build(name)
        count = pc.sum(kernel(col, pa.scalar(value, col.type)), skip_nulls=True).as_py()
        return int(count) if count is not None else 0
    except (*_ARROW_ERRORS, pa.ArrowTypeError):
        return None


def statistics(build: ColumnBuilder, schema: pa.Schema, rows: int) -> SourceStatistics:
    """EXACT per-column min/max/null-count over the ordered (numeric/temporal) columns.

    A vectorized ``min_max`` per ordered column (strings/nested fall through); an all-null
    column is skipped so a run returns its SQL-NULL ``MIN``/``MAX``. The result is an EXACT
    `SourceStatistics` Kyber answers unfiltered bounds queries from on subsequent runs.
    """
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat, Provenance

    columns: dict[str, ColumnStat] = {}
    for f in schema:
        if not (
            pa.types.is_integer(f.type)
            or pa.types.is_floating(f.type)
            or pa.types.is_temporal(f.type)
        ):
            continue
        try:
            col = build(f.name)
            mm = pc.min_max(col, skip_nulls=True)
            lo, hi = mm["min"].as_py(), mm["max"].as_py()
        except _ARROW_ERRORS:
            continue
        if lo is None:  # all-null column: SQL MIN/MAX is NULL — let a run return it
            continue
        columns[f.name] = ColumnStat(
            min=lo, max=hi, null_count=col.null_count, provenance=Provenance.EXACT
        )
    return SourceStatistics(row_count=rows, columns=columns)
