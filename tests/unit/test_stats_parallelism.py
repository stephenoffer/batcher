"""Planning statistics must use the machine, and must not change what they report.

Every per-column statistic the optimizer collects before a query — distinct counts, bounds,
widths, quantiles — is an O(rows) pass over a resident source. Two of them were effectively
single-threaded, and on the shape users most often hand Batcher (one contiguous Arrow chunk,
which is what `pa.table(...)` over NumPy arrays and every `combine_chunks()` produce) that
made the *planner* the dominant cost of the query:

* the native sketches parallelize across **batches**, so a single-batch column was sketched
  on one core — measured at 5.53 ns/cell against 0.32 for the same rows in 32 pieces; and
* `pc.min_max` does not parallelize at all, and the float path runs five serial passes.

Together they were 2.5 s of a 5.1 s aggregate over 200M rows. The fix is to cut the column
into per-core slices, which is O(1) and shares buffers.

The risk in that fix is not speed, it is *agreement*: a statistic that changes when the rows
are sliced differently is worse than a slow one, because the optimizer would plan differently
depending on how a caller happened to chunk its input. So these tests are mostly equality
tests, and the timing property is asserted only as a shape (how many pieces), never as a
duration.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest

from batcher.core.stats import _sketch_shards, column_ndv, column_statistics
from batcher.io.source import inmemory_stats as ims

# Comfortably above the four-morsel floor both shard helpers use, so the parallel path is the
# one under test rather than the single-slice short circuit.
_BIG = 4 * 16_384 * 8


def _batches(col: pa.Array, n: int) -> list[pa.RecordBatch]:
    rows = len(col) // n
    return [pa.record_batch({"g": col.slice(i * rows, rows)}) for i in range(n)]


def test_one_big_batch_is_split_for_the_sketch():
    """The regression: a single-batch column must not be sketched on a single core.

    Asserted as the piece count rather than a duration — the cost is linear in pieces, and a
    wall-clock assertion on a shared machine measures the neighbours as much as the change.
    """
    col = pa.array(np.arange(_BIG, dtype=np.int64))
    one = [pa.record_batch({"g": col})]
    sharded = _sketch_shards(one)
    assert len(sharded) > 1, "a single large batch was handed to the sketch whole"
    assert sum(b.num_rows for b in sharded) == _BIG, "slicing must preserve every row"


def test_a_small_batch_is_left_alone():
    """Below the floor, slicing costs more than the parallelism buys — so it does not."""
    col = pa.array(np.arange(1_000, dtype=np.int64))
    one = [pa.record_batch({"g": col})]
    assert _sketch_shards(one) == one


def test_already_batched_input_is_untouched():
    """Nothing to gain, and re-slicing would only add objects."""
    col = pa.array(np.arange(_BIG, dtype=np.int64))
    many = _batches(col, 256)
    assert _sketch_shards(many) is many


@pytest.mark.parametrize("cardinality", [7, 5_000, _BIG])
def test_distinct_count_does_not_depend_on_batching(cardinality):
    """The estimate must be a fact about the data, not about how it was chunked."""
    rng = np.random.default_rng(3)
    col = pa.array(rng.integers(0, cardinality, _BIG, dtype=np.int64))
    got = [column_ndv(_batches(col, n), ["g"])["g"] for n in (1, 4, 64)]
    assert max(got) - min(got) <= 0.02 * min(got), f"ndv moved with batching: {got}"


def test_column_statistics_agree_across_batching():
    """ndv, width and the quantile grid must all survive being measured in more pieces."""
    rng = np.random.default_rng(4)
    col = pa.array(rng.random(_BIG))
    one, many = _batches(col, 1), _batches(col, 64)
    n1, q1, w1 = column_statistics(one, ["g"])
    n2, q2, w2 = column_statistics(many, ["g"])
    assert w1 == w2, "average width is exact and must be identical"
    assert n1["g"] == pytest.approx(n2["g"], rel=0.02)
    assert sorted(q1) == sorted(q2)
    # KLL is approximate, and merging more partials is a different (equally valid) sketch —
    # so the grids are compared on the span they describe, not bit-for-bit.
    v1, v2 = q1["g"]["values"], q2["g"]["values"]
    assert len(v1) == len(v2)
    for a, b in zip(v1, v2, strict=True):
        assert a == pytest.approx(b, abs=0.05), f"quantile grid moved with batching: {v1} {v2}"


# --- exact bounds, which must be exact whichever way the column is cut ------------------


def _serial(col, dtype):
    """The single-slice path — what the parallel one has to reproduce."""
    return ims._bounds_one(col, ims._value_dtype(dtype))


def _parallel(col, dtype):
    return ims.column_bounds(lambda _n: col, dtype, "c")


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None

    def eq(x, y):
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            return True
        return x == y

    return eq(a.min, b.min) and eq(a.max, b.max) and a.null_count == b.null_count


def _cases() -> dict[str, pa.Array]:
    rng = np.random.default_rng(0)
    half_nan = rng.integers(0, 100, _BIG).astype("float64")
    half_nan[: _BIG // 2] = np.nan
    with_nan = rng.random(_BIG)
    with_nan[_BIG // 2] = np.nan
    mask = np.zeros(_BIG, dtype=bool)
    mask[::3] = True
    return {
        "int64": pa.array(rng.integers(-(10**12), 10**12, _BIG, dtype=np.int64)),
        # Beyond f64's exact range: a bound routed through a float would round it.
        "int64 past 2**53": pa.array(np.full(_BIG, 2**62, dtype=np.int64)),
        "float": pa.array(rng.random(_BIG)),
        "float with one NaN": pa.array(with_nan),
        "float all NaN": pa.array(np.full(_BIG, np.nan)),
        "float half NaN": pa.array(half_nan),
        "all null": pa.array([None] * _BIG, type=pa.int64()),
        "int with nulls": pa.array(rng.integers(0, 50, _BIG, dtype=np.int64), mask=mask),
        "empty": pa.array([], type=pa.int64()),
    }


@pytest.mark.parametrize("name", list(_cases()))
def test_parallel_bounds_equal_serial_bounds(name):
    """Cutting a column up must not change its min, max, or null count."""
    col = _cases()[name]
    assert _same(_serial(col, col.type), _parallel(col, col.type)), name


def test_a_nan_in_any_slice_still_decides_the_maximum():
    """The bug the merge had, pinned.

    Under SQL's total order NaN is the greatest value, so a column holding one has NaN as its
    maximum. A slice that is *entirely* NaN has no usable bound and reports none — and the
    first merge dropped it, so a half-NaN column reported the numeric maximum of its other
    half. Measured before the fix: `max` came back 99.0 where a run returns NaN.

    This is the one case where slicing could silently produce a *wrong* bound rather than a
    slower one, and a wrong bound prunes rows a query should return.
    """
    values = np.arange(_BIG, dtype="float64")
    values[: _BIG // 2] = np.nan  # a leading run long enough to fill whole slices
    col = pa.array(values)
    got = _parallel(col, col.type)
    assert got is not None
    assert math.isnan(got.max), f"a NaN slice was dropped from the merge; max={got.max}"
    assert got.min == float(_BIG // 2)
