"""Differential tests for the natural-run sort against DuckDB.

`ops::run_sort` finds the maximal ordered runs of a fixed-width sort key and merges them
instead of radix-sorting from scratch. That is a pure speed change with one way to get it
wrong that no timing would show: the merged permutation must be the **stable** one, and a
descending run may only be reversed when it is *strictly* descending, or two rows sharing a
key come back later-first.

Every case here therefore pins the *sequence*, not the multiset. The payload column ``p`` is
the input row position and Batcher's sort is stable, so ``ORDER BY k, p`` is a total order the
raw output must match row for row — an order-independent comparison would pass while the
merge reversed a tie, which is exactly the bug these exist to catch.

The run structures are the ones real data has: a table written sorted, micro-batches appended
in arrival order, a ``UNION ALL`` of sorted files, and an append-only log with late arrivals.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

# Above `run_sort::MIN_ROWS` (4,096) but below the sample-sort's 131,072, so the whole
# relation reaches run detection in one piece.
SERIAL_ROWS = 40_000

# Above `PARALLEL_SORT_MIN_ROWS`, so the sample-sort routes first and every *range* is what
# run detection sees. Both paths must agree with DuckDB, and they are different code.
PARALLEL_ROWS = 400_000


def _keys(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """A key column with a named run structure."""
    if kind == "random":
        return rng.integers(0, 1 << 40, n, dtype=np.int64)
    if kind == "sorted":
        return np.sort(rng.integers(0, 1 << 40, n, dtype=np.int64))
    if kind == "descending":
        return np.sort(rng.integers(0, 1 << 40, n, dtype=np.int64))[::-1].copy()
    if kind == "descending_with_ties":
        # The stability trap: a non-strictly descending run must NOT be reversed.
        return np.sort(rng.integers(0, 32, n, dtype=np.int64))[::-1].copy()
    if kind == "runs":
        parts = [np.sort(rng.integers(0, 1 << 40, n // 8, dtype=np.int64)) for _ in range(8)]
        return np.concatenate(parts)[:n]
    if kind == "sorted_with_late_arrivals":
        k = np.sort(rng.integers(0, 1 << 40, n, dtype=np.int64))
        late = max(1, n // 100)
        k[rng.choice(n, late, replace=False)] = rng.integers(0, 1 << 40, late)
        return k
    if kind == "duplicates":
        return np.sort(rng.integers(0, 16, n, dtype=np.int64))
    raise ValueError(kind)


SHAPES = (
    "random",
    "sorted",
    "descending",
    "descending_with_ties",
    "runs",
    "sorted_with_late_arrivals",
    "duplicates",
)


def _table(kind: str, n: int, *, nulls: bool = False) -> pa.Table:
    rng = np.random.default_rng(1234)
    keys = _keys(kind, n, rng).tolist()
    if nulls:
        keys = [None if i % 137 == 0 else v for i, v in enumerate(keys)]
    return pa.table(
        {
            "k": pa.array(keys, type=pa.int64()),
            "p": pa.array(range(n), type=pa.int64()),
        }
    )


@pytest.mark.differential
@pytest.mark.parametrize("kind", SHAPES)
@pytest.mark.parametrize("rows", [SERIAL_ROWS, PARALLEL_ROWS])
@pytest.mark.parametrize("descending", [False, True])
def test_run_structured_sort_matches_duckdb(duck, kind, rows, descending):
    """The merged permutation is the stable one, on every run structure and both paths."""
    t = _table(kind, rows)
    duck.register("t", t)
    direction = "DESC" if descending else "ASC"
    out = bt.from_arrow(t).sort("k", descending=descending).collect()
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY k {direction}, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["sorted", "runs", "descending_with_ties"])
def test_run_structured_sort_with_nulls_matches_duckdb(duck, kind):
    """Nulls are partitioned out before run detection, so their placement must survive it."""
    t = _table(kind, SERIAL_ROWS, nulls=True)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC NULLS LAST, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["sorted", "runs", "sorted_with_late_arrivals"])
def test_run_structured_multikey_sort_matches_duckdb(duck, kind):
    """The composite packed key gets run detection too, so it needs its own agreement."""
    rng = np.random.default_rng(99)
    n = PARALLEL_ROWS
    t = pa.table(
        {
            "k": pa.array(_keys(kind, n, rng) % 2_500, type=pa.int64()),
            "s": pa.array(rng.integers(0, 10_000, n, dtype=np.int64), type=pa.int64()),
            "p": pa.array(range(n), type=pa.int64()),
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k", "s").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC, s ASC, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["sorted", "runs"])
def test_run_structured_temporal_sort_matches_duckdb(duck, kind):
    """A date key reaches the same radix as an integer one; `ORDER BY <date>` is the shape
    partly-ordered data most often has, so it is pinned separately rather than assumed."""
    rng = np.random.default_rng(7)
    n = PARALLEL_ROWS
    days = (_keys(kind, n, rng) % 3_000).astype("int32")
    t = pa.table(
        {
            "k": pa.array(days, type=pa.date32()),
            "p": pa.array(range(n), type=pa.int64()),
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["sorted", "runs"])
def test_run_structured_sort_streams_the_same_order(duck, kind):
    """`iter_batches` must produce the same sequence `collect` does.

    The streaming path reaches the sort through a different executor, and a merge that was
    right in one and wrong in the other would pass every test above.
    """
    t = _table(kind, SERIAL_ROWS)
    duck.register("t", t)
    ds = bt.from_arrow(t).sort("k")
    streamed = pa.Table.from_batches(list(ds.iter_batches()), schema=ds.collect().schema)
    assert_same_ordered(streamed, duck.sql("SELECT * FROM t ORDER BY k ASC, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize("rows", [0, 1, 2])
def test_a_degenerate_input_still_sorts(duck, rows):
    """Empty, one row, and two rows — below every threshold, so nothing may fire."""
    t = (
        _table("sorted", rows)
        if rows
        else pa.table({"k": pa.array([], type=pa.int64()), "p": pa.array([], type=pa.int64())})
    )
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["sorted", "runs", "descending", "descending_with_ties"])
def test_run_structured_sort_spills_to_the_same_order(duck, kind):
    """The out-of-core sort must produce the same sequence the in-memory one does.

    `collect(spill=True)` sorts each run separately and merges them, so it calls the shared
    permutation builder over *slices* rather than the whole relation. A merge that was right
    on the whole input and wrong on a slice would pass every other case in this file, and a
    spilled descending sort has already emitted nulls mid-result once in this engine's
    history.
    """
    t = _table(kind, SERIAL_ROWS)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("k").collect(spill=True)
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY k ASC, p ASC"))


@pytest.mark.differential
@pytest.mark.parametrize("kind", ["sorted", "runs"])
def test_run_structured_spilled_sort_matches_the_in_memory_sort(kind):
    """Spilled == in-memory, row for row, without going through DuckDB.

    Held directly against Batcher's own in-memory answer as well as against the oracle above,
    because the property the parallel and spilling paths owe each other is *identity of the
    permutation*, not merely agreement with a correctly sorted reference.
    """
    t = _table(kind, PARALLEL_ROWS)
    ds = bt.from_arrow(t).sort("k", descending=True)
    assert ds.collect(spill=True).to_pydict() == ds.collect().to_pydict()
