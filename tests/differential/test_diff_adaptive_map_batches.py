"""Adaptive re-optimization must not crash — or change the result — on a `map_batches` plan.

`_execute_adaptive` pre-optimizes the whole plan once, which lowers it to the engine IR
to run the rule engine. A `map_batches` operator is opaque to the IR (`to_ir` raises by
design), so that pre-optimize used to raise `NotImplementedError` for any plan that mixed a
Python UDF with a pipeline breaker (join / aggregate) — while the same query with adaptive
off ran fine through the UDF executor. Forcing adaptivity on (the default above ~20M input
rows) therefore turned a working batch-inference-plus-join pipeline into a crash.

The invariant: adaptive on == adaptive off (same result), and both == DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt
from _harness import assert_same


def _double_v(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Identity-shaped UDF: double column ``v`` in place (opaque to the IR)."""
    idx = batch.schema.get_field_index("v")
    return batch.set_column(idx, "v", pc.multiply(batch.column("v"), 2))


def _left() -> bt.Dataset:
    return bt.from_arrow(
        pa.table({"k": [1, 2, 3, None, 2, 2, 5], "v": [10, 20, 30, 40, 50, 60, 70]})
    )


def _right() -> bt.Dataset:
    return bt.from_arrow(pa.table({"k": [2, 2, 3, None, 9], "w": [1, 2, 3, 4, 5]}))


def _rows(table: pa.Table) -> list[tuple]:
    return sorted(
        (tuple(row[c] for c in table.column_names) for row in table.to_pylist()),
        key=lambda r: tuple((v is None, str(v)) for v in r),
    )


def _assert_adaptive_invariant(ds: bt.Dataset) -> None:
    """Adaptive on and off must yield the same rows (and neither may raise)."""
    off = ds.collect(adaptive=False)
    on = ds.collect(adaptive=True)
    assert _rows(on) == _rows(off)


def test_map_then_join_adaptive_does_not_crash() -> None:
    ds = _left().map_batches(_double_v).join(_right(), on="k")
    _assert_adaptive_invariant(ds)


def test_map_then_join_then_aggregate_adaptive() -> None:
    ds = (
        _left().map_batches(_double_v).join(_right(), on="k").group_by("k").agg(s=bt.col("v").sum())
    )
    _assert_adaptive_invariant(ds)


def test_map_then_aggregate_then_join_adaptive() -> None:
    ds = (
        _left().map_batches(_double_v).group_by("k").agg(s=bt.col("v").sum()).join(_right(), on="k")
    )
    _assert_adaptive_invariant(ds)


def test_aggregate_then_map_then_join_adaptive() -> None:
    ds = (
        _left()
        .group_by("k")
        .agg(s=bt.col("v").sum())
        .map_batches(lambda b: b)
        .join(_right(), on="k")
    )
    _assert_adaptive_invariant(ds)


def test_map_join_adaptive_matches_duckdb(duck) -> None:
    left = pa.table({"k": [1, 2, 3, None, 2, 2, 5], "v": [10, 20, 30, 40, 50, 60, 70]})
    right = pa.table({"k": [2, 2, 3, None, 9], "w": [1, 2, 3, 4, 5]})
    duck.register("l", left)
    duck.register("r", right)
    ds = bt.from_arrow(left).map_batches(_double_v).join(bt.from_arrow(right), on="k")
    result = ds.collect(adaptive=True)
    # The UDF doubles v; the join is an inner equi-join on k.
    assert_same(
        result,
        duck.sql("SELECT l.k, l.v * 2 AS v, r.w FROM l JOIN r ON l.k = r.k"),
    )
