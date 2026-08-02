"""Exact aggregate answers over an immutable in-memory Arrow relation.

The metadata moat: an unfiltered ``COUNT(DISTINCT)``/``AVG``/``SUM``/``COUNT(*) WHERE col = v``
is answered from these instead of re-scanning, on every run after the first. An in-memory
source is immutable, so each answer is exact and constant, and `InMemorySource` memoizes it.

Split from `inmemory_stats` on the seam that file grew past: these *answer a query*, while
what remains there *describes a column* for planning (bounds, width, null count). The two had
different consumers all along -- nothing here is read by the optimizer, and nothing there is
returned to a user.

The `build` callable materializes one (narrow-int widened) column; every kernel here sees it
through `_decoded`, since the engine decodes dictionary encoding at the FFI boundary and a
statistic must describe the relation that actually runs.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from batcher.io.source.inmemory_stats import _ARROW_ERRORS, ColumnBuilder, _decoded

__all__ = ["column_mean", "column_ndv", "column_predicate_count", "column_sum"]


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
        return pc.count_distinct(_canonical_zeros(_decoded(build(name))), mode="only_valid").as_py()
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
        return pc.mean(_decoded(build(name)), skip_nulls=True).as_py()
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
        col = _decoded(build(name))
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
        col = _decoded(build(name))
        if _float_order_differs(col) or _decimal_against_float(col, value):
            return None  # see below — this count would not be the count the engine produces
        count = pc.sum(kernel(col, _literal_scalar(value, col.type)), skip_nulls=True).as_py()
        return int(count) if count is not None else 0
    except (*_ARROW_ERRORS, pa.ArrowTypeError):
        return None


def _literal_scalar(value: object, col_type: pa.DataType) -> pa.Scalar:
    """The literal as a scalar that still *means* `value` against a `col_type` column.

    Typing the scalar as the column's type is the fast path and is right whenever the cast
    is lossless. It is not always lossless: `pa.scalar(-2.5, pa.int64())` is `-2`, so a
    count for ``n > -0.5`` was taken as ``n > 0`` and silently lost every row where
    ``n == 0``, while ``n == -2.5`` — which no integer can satisfy — counted the rows equal
    to `-2`. Because this count answers `COUNT(*)` *without executing*, the result
    contradicted the rows the same filter materializes: `count()` said 1 where
    `to_pydict()` returned none.

    When the cast would change the value, the scalar keeps its own type instead and Arrow's
    kernels promote both sides to a common type — which is what the engine does when it
    evaluates the predicate for real, so the count and the rows agree again.

    Args:
        value: The literal the predicate compares against.
        col_type: The Arrow type of the column being compared.

    Returns:
        A scalar equal to `value`, typed as the column where that is exact.
    """
    try:
        typed = pa.scalar(value, col_type)
    except (*_ARROW_ERRORS, pa.ArrowTypeError, pa.ArrowInvalid):
        return pa.scalar(value)
    return typed if typed.as_py() == value else pa.scalar(value)


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
    `docs/architecture/internals/bug_hunt_ledger.md`. When it is fixed, this guard can go.)
    """
    if not pa.types.is_floating(col.type):
        return False
    has_nan = pc.any(pc.is_nan(col), min_count=0).as_py()
    has_zero = pc.any(pc.equal(col, pa.scalar(0.0, col.type)), min_count=0).as_py()
    return bool(has_nan) or bool(has_zero)


def _decimal_against_float(col: pa.ChunkedArray | pa.Array, value: object) -> bool:
    """Whether this is a decimal column compared to a float literal — where the two Arrows differ.

    The counts here run on **pyarrow** (Arrow C++); the engine executes on **arrow-rs**. For a
    decimal column against a float literal the two do not agree, and not in a way either side
    is obviously wrong about: over a `decimal(20,3)` column holding `2.675`, pyarrow counts
    zero rows equal to the float `2.675` and three greater than it, while the engine counts one
    and two. Both are defensible readings of a comparison between an exact value and an
    approximate one; what matters is that a count taken from one cannot answer for the other.

    This is the same disagreement `plan.stats.mismatched_exactness` declines on the bound side,
    reached through a different door: the IR has no decimal literal, so every exact money
    predicate arrives as a float and lands here. Declining costs a scan; answering cost a
    `count()` that contradicted `collect()` on the same plan.

    An integer literal is fine — both implementations widen it into the decimal exactly — and
    so is a `Decimal` literal, which never becomes a float in the first place.

    Args:
        col: The column the predicate compares.
        value: The literal it is compared against.

    Returns:
        ``True`` when the pair is a decimal column and a float literal.
    """
    return pa.types.is_decimal(col.type) and isinstance(value, float)
