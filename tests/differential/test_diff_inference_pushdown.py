"""Differential: pushing a filter below an opaque `map_batches`, and dropping a large
column once the UDF has consumed it, must not change the answer — DuckDB is the oracle.

The optimizer moves a `filter` below a `map_batches` when the predicate reads only columns
the UDF *declares preserved*, so the (in production, GPU) UDF runs on fewer rows. That is a
performance rewrite; here it is proven semantics-preserving against DuckDB, and each test
first asserts the rewrite actually fired so it is the *optimized* plan being checked. The
negative test — a predicate on a column the UDF rewrote and did not declare preserved —
proves the filter stays above the UDF and the result still matches DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.plan.logical import Filter, MapBatches


def _t() -> bt.Dataset:
    return bt.from_arrow(
        pa.table(
            {
                "id": [1, 2, 3, 4, 5, 6],
                "v": [10, 20, 30, 40, 50, 60],
                "big": ["aa", "bbbb", "c", "dddddd", "ee", "f"],
            }
        )
    )


def _register(duck, sql_name: str = "t") -> None:
    duck.execute(
        f"CREATE TABLE {sql_name} AS SELECT * FROM (VALUES "
        "(1,10,'aa'),(2,20,'bbbb'),(3,30,'c'),(4,40,'dddddd'),(5,50,'ee'),(6,60,'f')"
        f") AS s(id, v, big)"
    )


def _add_score(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Append ``score = v * 10`` and leave id, v, big untouched (the declared-preserved set)."""
    return batch.append_column("score", pc.multiply(batch.column("v"), 10))


def _rewrite_v(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Overwrite ``v`` with ``v + 100`` — the case where `v` is NOT preserved."""
    idx = batch.schema.get_field_index("v")
    return batch.set_column(idx, "v", pc.add(batch.column("v"), 100))


def _blob_len(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Append ``blen = length(big)``; big is read but nothing downstream keeps it."""
    return batch.append_column("blen", pc.utf8_length(batch.column("big")).cast(pa.int64()))


def test_filter_on_preserved_column_pushed_and_matches_duckdb(duck):
    ds = (
        _t()
        .ml.map_batches(
            _add_score,
            preserves_columns=["id", "v", "big"],
            output_columns=["id", "v", "big", "score"],
        )
        .filter(col("v") >= 30)
    )

    # The rewrite fired: the filter now sits BELOW the UDF (the UDF runs on fewer rows).
    opt = Optimizer().logical_rewrite(ds._plan)
    assert isinstance(opt, MapBatches)
    assert isinstance(opt.input, Filter)

    _register(duck)
    assert_same(
        ds.collect(),
        duck.sql("SELECT id, v, big, v * 10 AS score FROM t WHERE v >= 30"),
    )


def test_filter_on_rewritten_column_not_pushed_and_matches_duckdb(duck):
    # `v` is rewritten by the UDF and NOT declared preserved, so the filter must stay ABOVE
    # the UDF and see the post-UDF value. DuckDB filters on the same post-rewrite value.
    ds = _t().ml.map_batches(_rewrite_v, output_columns=["id", "v", "big"]).filter(col("v") >= 130)

    opt = Optimizer().logical_rewrite(ds._plan)
    assert isinstance(opt, Filter)  # filter did NOT move below the UDF
    assert isinstance(opt.input, MapBatches)

    _register(duck)
    assert_same(
        ds.collect(),
        duck.sql("SELECT id, v + 100 AS v, big FROM t WHERE v + 100 >= 130"),
    )


def test_large_column_dropped_after_udf_matches_duckdb(duck):
    # The UDF reads `big` to derive `blen`; the final projection keeps only id and blen, so
    # the optimizer frees `big` right above the UDF. The result is unchanged.
    ds = (
        _t()
        .ml.map_batches(
            _blob_len,
            input_columns=["big"],
            output_columns=["id", "v", "big", "blen"],
        )
        .select("id", "blen")
    )

    _register(duck)
    assert_same(
        ds.collect(),
        duck.sql("SELECT id, length(big) AS blen FROM t"),
    )
