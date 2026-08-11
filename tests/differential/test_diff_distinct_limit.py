"""`DISTINCT ... LIMIT k` — the early exit keeps the first `k` distinct rows, in input order.

Why this file does not simply `assert_same` against DuckDB on every case: `SELECT DISTINCT g
FROM t LIMIT k` with no `ORDER BY` is under-determined, so DuckDB is free to return any `k`
distinct values and does (whichever `k` its threads reach first). Comparing sets against it
would be asserting something SQL does not promise, and the test would be flaky rather than
wrong.

So the contracts pinned here are the ones Batcher actually owes:

* the limit is **not binding** when the relation has at most `k` distinct rows — there the
  answer is the whole distinct set and DuckDB agrees exactly, so `assert_same` applies;
* the rows kept are the **first `k` in input order**, which is what makes the answer identical
  on one node and on many (invariant #7) where "whichever `k` won the race" would not be;
* every terminal and every execution path agrees, since the early exit changes *which* rows
  come back and a path that skipped it would silently return different ones.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from _harness import assert_same


def _high_cardinality(n: int = 4000) -> pa.Table:
    """`g` nearly unique, so the fusion's cardinality gate opens and the exit is reachable."""
    return pa.table(
        {
            "g": pa.array(list(range(n)), pa.int64()),
            "v": pa.array([i % 7 for i in range(n)], pa.int64()),
        }
    )


def _first_k_distinct(values: list, k: int) -> list:
    """The first `k` distinct values in order — the answer the operator promises."""
    seen: list = []
    for v in values:
        if v not in seen:
            seen.append(v)
            if len(seen) == k:
                break
    return seen


def test_limit_not_binding_matches_duckdb_exactly(duck):
    """Fewer distinct rows than the limit: the whole distinct set, so DuckDB agrees exactly."""
    t = pa.table({"g": pa.array([3, 1, 3, 1, 2], pa.int64())})
    duck.register("t", t)
    out = bt.from_arrow(t).select("g").distinct().limit(100).collect()
    assert_same(out, duck.sql("SELECT DISTINCT g FROM t LIMIT 100"))


def test_keeps_the_first_k_in_input_order():
    """The kept rows are the input's own first `k` distinct values, not an arbitrary `k`.

    Ordered on purpose: `assert_same` is order-independent and cannot see this property, which
    is the whole point of the change.
    """
    values = [50, 10, 50, 30, 10, 20, 40, 30, 60]
    t = pa.table({"g": pa.array(values, pa.int64())})
    for k in (1, 2, 3, 5):
        out = bt.from_arrow(t).select("g").distinct().limit(k).collect()
        assert out.column("g").to_pylist() == _first_k_distinct(values, k), f"k={k}"


def test_every_terminal_agrees_on_which_rows_survive():
    """`collect`, `iter_batches` and `to_pydict` return the same rows, not merely the same count.

    The early exit changes *which* rows come back, so a terminal that missed the fusion would
    return a different `k` while still returning `k` distinct rows — invisible to a count check.
    """
    t = _high_cardinality()
    k = 5
    ds = bt.from_arrow(t).select("g").distinct().limit(k)
    collected = ds.collect().column("g").to_pylist()
    streamed = [v for batch in ds.iter_batches() for v in batch.column("g").to_pylist()]
    as_dict = ds.to_pydict()["g"]
    expected = _first_k_distinct(t.column("g").to_pylist(), k)
    assert collected == expected
    assert streamed == expected
    assert as_dict == expected


def test_repeated_runs_are_stable():
    """The same query returns the same rows every time.

    Before the fusion this was whichever `k` the parallel dedup's bucket order produced, which
    moved with the shard count; determinism is what the fusion buys.
    """
    t = _high_cardinality()
    ds = bt.from_arrow(t).select("g").distinct().limit(9)
    first = ds.collect().column("g").to_pylist()
    for _ in range(3):
        assert ds.collect().column("g").to_pylist() == first


def test_offset_takes_its_window_from_the_capped_prefix():
    """`LIMIT n OFFSET m` caps the dedup at `m + n` and still returns the right window."""
    values = [50, 10, 50, 30, 10, 20, 40, 30, 60]
    t = pa.table({"g": pa.array(values, pa.int64())})
    out = bt.from_arrow(t).select("g").distinct().limit(2, offset=3).collect()
    assert out.column("g").to_pylist() == _first_k_distinct(values, 5)[3:5]


def test_low_cardinality_key_returns_every_distinct_row(duck):
    """A key with few values: the limit cannot bind, so the answer is the full distinct set.

    This is the shape the cardinality gate deliberately keeps off the early exit, and it must
    still be exactly right when it runs the ordinary dedup.
    """
    t = pa.table({"g": pa.array([i % 4 for i in range(1000)], pa.int64())})
    duck.register("t", t)
    out = bt.from_arrow(t).select("g").distinct().limit(50).collect()
    assert_same(out, duck.sql("SELECT DISTINCT g FROM t LIMIT 50"))


def test_empty_input_keeps_its_schema():
    """An empty relation yields no rows and the right column, not an error."""
    t = pa.table({"g": pa.array([], pa.int64())})
    out = bt.from_arrow(t).select("g").distinct().limit(5).collect()
    assert out.num_rows == 0
    assert out.column_names == ["g"]


def test_nulls_are_one_distinct_row_and_hold_their_position():
    """A null is a value here (SQL `DISTINCT` groups nulls together) and keeps its order slot."""
    values = [None, 7, None, 3, 7]
    t = pa.table({"g": pa.array(values, pa.int64())})
    out = bt.from_arrow(t).select("g").distinct().limit(2).collect()
    assert out.column("g").to_pylist() == [None, 7]


def test_multi_column_distinct_keeps_whole_rows():
    """A whole-row `DISTINCT` over several columns dedups on the combination."""
    t = pa.table(
        {
            "a": pa.array([1, 1, 2, 1, 2], pa.int64()),
            "b": pa.array([9, 9, 8, 7, 8], pa.int64()),
        }
    )
    out = bt.from_arrow(t).distinct().limit(2).collect()
    assert out.to_pydict() == {"a": [1, 2], "b": [9, 8]}


def test_sql_spelling_agrees_with_the_dataframe_spelling():
    """`bt.sql("SELECT DISTINCT ... LIMIT k")` lowers to the same operator."""
    t = _high_cardinality()
    k = 4
    via_sql = bt.sql("SELECT DISTINCT g FROM t LIMIT 4", t=t).collect()
    via_df = bt.from_arrow(t).select("g").distinct().limit(k).collect()
    assert via_sql.column("g").to_pylist() == via_df.column("g").to_pylist()
