"""Empty inputs flow cleanly through every operator (no crash, correct shape)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count


def _empty():
    # A dataset whose filter removes every row, so downstream ops see 0 rows.
    return bt.from_arrow(pa.table({"k": [1, 2, 3], "v": [10, 20, 30]})).filter(col("k") > 100)


def test_empty_filter_then_select():
    out = _empty().select("k", "v").collect()
    assert out.num_rows == 0 and out.column_names == ["k", "v"]


def test_empty_sort():
    assert _empty().sort("v", descending=True).collect().num_rows == 0


def test_empty_distinct():
    assert _empty().distinct().collect().num_rows == 0


def test_empty_grouped_aggregate():
    out = _empty().group_by("k").agg(s=col("v").sum(), n=count()).collect()
    assert out.num_rows == 0  # no groups
    assert set(out.column_names) == {"k", "s", "n"}


def test_empty_global_aggregate():
    # A global aggregate over no rows still yields one row (SUM=NULL, COUNT=0).
    out = _empty().group_by().agg(s=col("v").sum(), n=count()).collect()
    assert out.to_pylist() == [{"s": None, "n": 0}]


def test_empty_join_both_sides():
    left = _empty()
    right = bt.from_arrow(pa.table({"k": [1], "b": [9]}))
    assert left.join(right, on="k").collect().num_rows == 0
    # empty right, non-empty left, inner join → 0 rows
    full_left = bt.from_arrow(pa.table({"k": [1], "a": [1]}))
    empty_right = bt.from_arrow(pa.table({"k": [1], "b": [2]})).filter(col("k") > 100)
    assert full_left.join(empty_right, on="k").collect().num_rows == 0


def test_empty_map_batches():
    def add_one(batch: pa.RecordBatch) -> pa.RecordBatch:
        import pyarrow.compute as pc

        return batch.set_column(1, "v", pc.add(batch.column("v"), 1))

    assert _empty().map_batches(add_one).collect().num_rows == 0


def test_integer_overflow_wraps():
    # Documented behavior: integer arithmetic wraps (arrow/numpy semantics), it
    # does not crash. (DuckDB errors; Batcher wraps — a deliberate, cheap default.)
    m = 2**63 - 1
    out = bt.from_arrow(pa.table({"v": [m, m]})).select(r=col("v") + col("v")).collect()
    assert out.column("r").to_pylist() == [-2, -2]
