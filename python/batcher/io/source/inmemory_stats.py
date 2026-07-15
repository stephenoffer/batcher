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
    """EXACT distinct count of `name`'s non-null values, or None if not derivable.

    A float column is canonicalized first, because Arrow's `count_distinct` distinguishes
    `-0.0` from `0.0` (they have different bits) while the engine, SQL, and DuckDB do not
    (they are numerically equal, and the engine's grouping key normalizes the sign). Without
    this, a column holding both spellings of zero reported **two** distinct values from
    metadata and **one** from executing — a `COUNT(DISTINCT)` that the optimization made
    wrong, which is the worst kind of wrong. NaN needs no such treatment: Arrow already
    counts every NaN as one value, as the engine and DuckDB do.
    """
    try:
        return pc.count_distinct(_canonical_zeros(build(name)), mode="only_valid").as_py()
    except _ARROW_ERRORS:
        return None


def _canonical_zeros(col: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    """`col` with any `-0.0` replaced by `0.0`; non-float columns pass through unchanged.

    The one place the metadata layer adopts the engine's view of a signed zero. `-0.0 == 0.0`
    is true, so the two must never count as different values — but they are different *bits*,
    and any statistic computed by hashing raw bits will disagree unless the sign is normalized
    first. `pc.add(col, 0.0)` is that normalization: adding positive zero maps `-0.0` to `0.0`
    and leaves every other value (NaN and the infinities included) exactly as it was.
    """
    if not pa.types.is_floating(col.type):
        return col
    return pc.add(col, pa.scalar(0.0, col.type))


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
        if _float_order_differs(col):
            return None  # see below — this count would not be the count the engine produces
        count = pc.sum(kernel(col, pa.scalar(value, col.type)), skip_nulls=True).as_py()
        return int(count) if count is not None else 0
    except (*_ARROW_ERRORS, pa.ArrowTypeError):
        return None


def _float_order_differs(col: pa.ChunkedArray | pa.Array) -> bool:
    """Whether this column holds a value on which Arrow's comparison and the engine's differ.

    The counts above are computed with **pyarrow's** comparison kernels, which are IEEE. The
    engine compares floats on arrow-rs's *total* order — NaN above every number, `-0.0` below
    `0.0`. For ordinary values the two coincide, but on a NaN or a signed zero they do not, and
    a count taken with one and used to answer the other is a wrong count.

    Rather than guess which is right, this refuses to shortcut the columns where they disagree:
    a column holding a NaN, or holding a zero (the only place `-0.0` can hide). Everything else
    keeps the fast path. Cheap to decide — two vectorized predicates, no per-row Python.

    (The engine's total-order comparison also disagrees with DuckDB — `WHERE f = 0.0` misses
    `-0.0`, `WHERE f > 1` matches NaN — which is the underlying bug, recorded in
    `docs/internals/bug_hunt_ledger.md`. When it is fixed, this guard can go.)
    """
    if not pa.types.is_floating(col.type):
        return False
    has_nan = pc.any(pc.is_nan(col), min_count=0).as_py()
    has_zero = pc.any(pc.equal(col, pa.scalar(0.0, col.type)), min_count=0).as_py()
    return bool(has_nan) or bool(has_zero)


def column_bounds(build: ColumnBuilder, dtype: pa.DataType, name: str):
    """EXACT `ColumnStat` (min/max/null-count) for one column, or `None` if not derivable.

    A single vectorized ``min_max`` pass. Returns `None` for a non-ordered type
    (string/nested), an all-null column (its SQL ``MIN``/``MAX`` is NULL — let a run
    return it), or an unsupported kernel — the same skips [`statistics`] makes per column.
    Float columns take [`_float_bounds`], whose NaN handling `pc.min_max` does not give us.
    """
    if not (
        pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_temporal(dtype)
    ):
        return None
    from batcher.plan.stats import ColumnStat, Provenance

    try:
        col = build(name)
        if pa.types.is_floating(dtype):
            return _float_bounds(col)
        mm = pc.min_max(col, skip_nulls=True)
        lo, hi = mm["min"].as_py(), mm["max"].as_py()
    except _ARROW_ERRORS:
        return None
    if lo is None:  # all-null column
        return None
    return ColumnStat(min=lo, max=hi, null_count=col.null_count, provenance=Provenance.EXACT)


def _float_bounds(col: pa.Array):
    """Truthful float bounds under SQL's total order, where NaN is the **greatest** value.

    `pc.min_max` has no NaN policy we can rely on: over an all-NaN column it hands back the
    kernel's identity element (`+inf`/`-inf`), which is not a value in the column at all —
    so `min(f)` was answered from metadata as `inf` while executing the same query returned
    `nan`. A bound that is not a fact about the data is worse than no bound.

    So: NaN is excluded when computing the *minimum* (it can never be the smallest under a
    total order that makes it the largest), and the *maximum* is reported as NaN whenever any
    NaN is present, which is what it actually is. A column with no non-NaN value has no usable
    bound at all — return `None` and let a real run produce the answer.

    Args:
        col: The column to bound.

    Returns:
        An EXACT `ColumnStat`, or `None` when no sound bound exists.
    """
    from batcher.plan.stats import ColumnStat, Provenance

    non_null = col.drop_null()
    if len(non_null) == 0:  # all-null column — SQL MIN/MAX is NULL; let a run return it
        return None
    nan_mask = pc.is_nan(non_null)
    has_nan = bool(pc.any(nan_mask).as_py())
    finite = pc.filter(non_null, pc.invert(nan_mask))
    if len(finite) == 0:  # every value is NaN — no usable bound
        return None
    mm = pc.min_max(finite)
    return ColumnStat(
        min=mm["min"].as_py(),
        max=float("nan") if has_nan else mm["max"].as_py(),
        null_count=col.null_count,
        provenance=Provenance.EXACT,
    )


def statistics(build: ColumnBuilder, schema: pa.Schema, rows: int) -> SourceStatistics:
    """EXACT bounds where the type is ordered — and an EXACT null count for *every* column.

    A vectorized ``min_max`` per ordered column; strings and nested types have no bound worth
    recording (and an all-null column has none at all — a run returns its SQL-NULL ``MIN``).

    But every column, of every type, still gets its **null count**, and that is deliberate.
    Bounds and null counts are different facts with different requirements, and tying them
    together is what made `n_null("name")` scan a table whose null count was sitting in front of
    us: a string column produced no `ColumnStat` at all, so the one exact thing we knew about it
    was thrown away with the one we didn't. Arrow tracks `null_count` on the array — it is a
    field read, not a pass — so this costs nothing and makes `null_count()`, `count(name)`,
    `has_nulls`, and `dq.not_null` free on the columns most tables are actually made of.
    """
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat, Provenance

    columns: dict[str, ColumnStat] = {}
    for f in schema:
        stat = column_bounds(build, f.type, f.name)
        if stat is not None:
            columns[f.name] = stat
            continue
        nulls = column_null_count(build, f.name)
        if nulls is not None:
            # No trustworthy bounds (a string, a nested type, an all-null column), but an exact
            # null count. `null_count_provenance` is what lets the second ride without the first.
            columns[f.name] = ColumnStat(
                null_count=float(nulls),
                provenance=Provenance.DEFAULT,
                null_count_provenance=Provenance.EXACT,
            )
    # `column_bounds` records NaN as the max when the column holds one (SQL ranks NaN
    # greatest), so unlike a Parquet footer these bounds *are* sound for `max(f)` and
    # for every float fact derived from them — declare it.
    return SourceStatistics(row_count=rows, columns=columns, bounds_include_nan=True)


def column_null_count(build: ColumnBuilder, name: str) -> int | None:
    """EXACT number of nulls in `name`, for a column of any type, or None if not derivable.

    Arrow maintains `null_count` on the array itself, so this is a field read per chunk rather
    than a pass over the values — free for any type, including the string and nested columns
    that have no bounds worth recording.
    """
    try:
        return int(build(name).null_count)
    except _ARROW_ERRORS:
        return None
