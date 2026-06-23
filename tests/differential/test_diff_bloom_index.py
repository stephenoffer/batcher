"""End-to-end bloom data-skipping index vs DuckDB.

Write with the index on → read → an equality/`IN` on an *absent in-range* value is
pruned (the bloom proves absence where min/max can't), and the result still matches
DuckDB. With the index off, the same queries are correct (just not pruned).
"""

from __future__ import annotations

import dataclasses
import os
import tempfile

import pyarrow as pa

import batcher as bt
from batcher import Config, col, config_context


def _written(tmp: str, *, index: bool) -> tuple[str, pa.Table]:
    # Even ids 0..19998 — odd ids in that range are absent but inside [min, max],
    # so only a bloom (not min/max) can prune them.
    vals = list(range(0, 20000, 2))
    table = pa.table({"id": pa.array(vals, pa.int64()), "v": pa.array([i % 7 for i in vals])})
    path = os.path.join(tmp, "t.parquet")
    base = Config()
    cfg = base.replace(optimizer=dataclasses.replace(base.optimizer, build_bloom_index=index))
    with config_context(cfg):
        bt.from_arrow(table).write.parquet(path)
    return path, table


def test_absent_equality_pruned_matches_duckdb(duck):
    from conftest import assert_same

    with tempfile.TemporaryDirectory() as tmp:
        path, table = _written(tmp, index=True)
        duck.register("t", table)
        out = bt.read.parquet(path).filter(col("id") == 7777).select("id", "v").collect()
        assert out.num_rows == 0  # absent in-range value → pruned to empty
        assert_same(out, duck.sql("SELECT id, v FROM t WHERE id = 7777"))


def test_present_equality_matches_duckdb(duck):
    from conftest import assert_same

    with tempfile.TemporaryDirectory() as tmp:
        path, table = _written(tmp, index=True)
        duck.register("t2", table)
        out = bt.read.parquet(path).filter(col("id") == 7778).select("id", "v").collect()
        assert_same(out, duck.sql("SELECT id, v FROM t2 WHERE id = 7778"))


def test_in_list_absent_matches_duckdb(duck):
    from conftest import assert_same

    with tempfile.TemporaryDirectory() as tmp:
        path, table = _written(tmp, index=True)
        duck.register("t3", table)
        out = bt.read.parquet(path).filter(col("id").is_in([7777, 7779, 8001])).collect()
        assert out.num_rows == 0
        assert_same(out, duck.sql("SELECT * FROM t3 WHERE id IN (7777, 7779, 8001)"))


def test_index_off_still_correct(duck):
    from conftest import assert_same

    with tempfile.TemporaryDirectory() as tmp:
        path, table = _written(tmp, index=False)  # no index built
        duck.register("t4", table)
        out = bt.read.parquet(path).filter(col("id") == 7778).select("id", "v").collect()
        assert_same(out, duck.sql("SELECT id, v FROM t4 WHERE id = 7778"))
