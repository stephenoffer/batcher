"""Differential tests for the parallel string sample-sort against DuckDB.

A string ``ORDER BY`` used to decline both sort fast paths (no radix, no range
partitioner) and ran a single-threaded comparison sort. It now range-partitions on
sampled string quantiles and sorts each range in parallel. These tests pin the
*ordering* — the part a parallel range sort can get wrong — above the row count that
engages the parallel path (``PARALLEL_SORT_MIN_ROWS`` = 131_072), and below it for the
serial fallback.

The payload column ``p`` is the input row position, and Batcher's string sort is stable,
so ties come back in ``p`` order. That makes DuckDB's ``ORDER BY s, p`` a *total* order
Batcher's raw output must match row-for-row — which is what makes these assertions
meaningful rather than a multiset comparison.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

# Comfortably over the 131_072-row threshold that turns on the sample-sort.
PARALLEL_ROWS = 200_000


def _string_table(n: int, distinct: int, *, nulls: bool = False) -> pa.Table:
    """Rows whose keys tie heavily (``n // distinct`` duplicates each), with a row-id payload."""
    keys = [
        None if (nulls and i % 97 == 0) else f"str_{(i * 7919) % distinct:05d}" for i in range(n)
    ]
    return pa.table(
        {"s": pa.array(keys, type=pa.string()), "p": pa.array(range(n), type=pa.int64())}
    )


@pytest.mark.differential
@pytest.mark.parametrize("descending", [False, True])
def test_parallel_string_sort_matches_duckdb(duck, descending):
    t = _string_table(PARALLEL_ROWS, distinct=5_000)
    duck.register("t", t)
    direction = "DESC" if descending else "ASC"
    out = bt.from_arrow(t).sort("s", descending=descending).collect()
    assert_same_ordered(out, duck.sql(f"SELECT * FROM t ORDER BY s {direction}, p ASC"))


@pytest.mark.differential
def test_parallel_string_sort_with_nulls_matches_duckdb(duck):
    t = _string_table(PARALLEL_ROWS, distinct=3_000, nulls=True)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("s").collect()
    # Batcher places nulls last on an ascending sort; pin DuckDB to the same placement.
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY s ASC NULLS LAST, p ASC"))


@pytest.mark.differential
def test_parallel_string_sort_is_stable_and_deterministic():
    """Equal keys keep input order, and repeated runs agree.

    The sample-sort sorts each range independently, so a nondeterministic tie order would
    silently make the sequential oracle, the parallel path, and the distributed range sort
    disagree on the same input.
    """
    t = _string_table(PARALLEL_ROWS, distinct=1_000)
    ds = bt.from_arrow(t)
    first = ds.sort("s").collect()
    second = ds.sort("s").collect()
    assert first.column("p").to_pylist() == second.column("p").to_pylist()

    keys = first.column("s").to_pylist()
    payload = first.column("p").to_pylist()
    for i in range(len(keys) - 1):
        if keys[i] == keys[i + 1]:
            assert payload[i] < payload[i + 1], f"tie at row {i} lost input order"


@pytest.mark.differential
def test_serial_string_sort_below_threshold_matches_duckdb(duck):
    """Small inputs decline the sample-sort; the serial path must still agree."""
    t = _string_table(1_000, distinct=50)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("s").collect()
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY s ASC, p ASC"))


@pytest.mark.differential
def test_parallel_string_sort_single_distinct_key(duck):
    """All keys equal: boundaries collapse, the sample-sort declines, order is input order."""
    t = _string_table(PARALLEL_ROWS, distinct=1)
    duck.register("t", t)
    out = bt.from_arrow(t).sort("s").collect()
    assert out.column("p").to_pylist() == list(range(PARALLEL_ROWS))
    assert_same_ordered(out, duck.sql("SELECT * FROM t ORDER BY s ASC, p ASC"))
