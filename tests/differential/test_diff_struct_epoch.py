"""struct.field (structural) + dt.epoch (vs DuckDB) tests."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa

import batcher as bt
from batcher import col


def test_struct_field_extraction():
    fields = pa.struct([("x", pa.int64()), ("y", pa.string())])
    tbl = pa.table({"s": pa.array([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, None], type=fields)})
    out = (
        bt.from_arrow(tbl)
        .select(x=col("s").struct.field("x"), y=col("s").struct.field("y"))
        .collect()
        .to_pydict()
    )
    assert out["x"] == [1, 2, None]  # null struct row → null field
    assert out["y"] == ["a", "b", None]


def test_dt_epoch_vs_duckdb(duck):
    from conftest import assert_same

    ts = [dt.datetime(2021, 1, 1) + dt.timedelta(hours=i * 37) for i in range(20)]
    tbl = pa.table({"ts": pa.array(ts, type=pa.timestamp("us"))})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).select(e=col("ts").dt.epoch()).collect()
    assert_same(out, duck.sql("SELECT epoch(ts)::BIGINT e FROM t"))
