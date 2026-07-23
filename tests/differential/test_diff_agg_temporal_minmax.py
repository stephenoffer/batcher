"""Differential coverage for MIN/MAX over temporal columns (Date/Time/Timestamp).

The engine reduces temporal columns by their underlying chronological (integer) order and
returns the winning row from the original array, so the exact unit and timezone survive.
These must match DuckDB *and* stay identical single-node vs multi-partition (the mergeable
invariant) — the same contract every other reduction owes.

Regression: `min`/`max` over a Date/Timestamp column used to raise "aggregate min is not
supported for column type Date32" while the Parquet footer answered the same query from
metadata — a metadata-vs-engine disagreement (the class already closed for Boolean).
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


def _table():
    return pa.table(
        {
            "g": ["a", "a", "b", "b", "b", "c"],
            "d": pa.array(
                [
                    datetime.date(2024, 1, 3),
                    datetime.date(2024, 1, 1),
                    datetime.date(2021, 6, 6),
                    datetime.date(2025, 12, 31),
                    None,
                    None,
                ],
                type=pa.date32(),
            ),
            "ts": pa.array(
                [datetime.datetime(2020, 1, 1, h) for h in (5, 2, 9, 1, 7, 3)],
                type=pa.timestamp("us"),
            ),
            "tm": pa.array(
                [datetime.time(h, 30) for h in (5, 2, 9, 1, 7, 3)],
                type=pa.time64("us"),
            ),
        }
    )


def test_temporal_minmax_matches_duckdb(duck):
    tbl = _table()
    duck.register("t", tbl)
    out = (
        bt.from_arrow(tbl)
        .group_by("g")
        .agg(
            dmin=col("d").min(),
            dmax=col("d").max(),
            tsmin=col("ts").min(),
            tsmax=col("ts").max(),
            tmmin=col("tm").min(),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, min(d) AS dmin, max(d) AS dmax, min(ts) AS tsmin, "
            "max(ts) AS tsmax, min(tm) AS tmmin FROM t GROUP BY g"
        ),
    )


def test_temporal_min_preserves_timezone(duck):
    tz = pa.timestamp("us", tz="UTC")
    tbl = pa.table(
        {
            "g": ["a", "a", "b"],
            "v": pa.array(
                [
                    datetime.datetime(2020, 1, 1, 5, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2020, 1, 1, 2, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2019, 6, 6, tzinfo=datetime.timezone.utc),
                ],
                type=tz,
            ),
        }
    )
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).group_by("g").agg(r=col("v").min()).collect()
    assert out.schema.field("r").type == tz  # timezone survives the reduction
    assert_same(out, duck.sql("SELECT g, min(v) AS r FROM t GROUP BY g"))


def test_temporal_minmax_single_node_equals_distributed():
    tbl = _table()
    ds = bt.from_arrow(tbl).group_by("g").agg(dmin=col("d").min(), dmax=col("d").max())
    single = {g: (a, b) for g, a, b in zip(*ds.collect().to_pydict().values(), strict=True)}
    dist = ds.collect(distributed=True, num_workers=3).to_pydict()
    multi = {g: (a, b) for g, a, b in zip(*dist.values(), strict=True)}
    assert single == multi
