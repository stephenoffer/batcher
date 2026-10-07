"""`ds.dq.monotonic`, held to the same check written as a DuckDB ``LAG`` query.

The violation count is the number of rows whose value steps against the requested
direction from the previous row in `order_by` order within each `by` group; nulls and the
first row of a group pass. The `order_by` keys here are unique within each group, so the
row order -- and therefore the answer -- is fully determined.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_TABLE = pa.table(
    {
        "device": ["a", "a", "a", "a", "b", "b", "b", "c"],
        "seq": [1, 2, 3, 4, 1, 2, 3, 1],
        "ts": [10, 30, 30, 20, 5, None, 4, 7],
    }
)


@pytest.mark.parametrize("by", [None, "device"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("decreasing", [False, True])
def test_violation_count_matches_a_lag_query(duck, by, strict, decreasing):
    order = "device, seq" if by is None else "seq"
    partition = "" if by is None else "PARTITION BY device"
    op = {(False, False): ">=", (True, False): ">", (False, True): "<=", (True, True): "<"}[
        (strict, decreasing)
    ]
    duck.register("t", _TABLE)
    expected = duck.sql(
        f"SELECT count(*) FROM (SELECT ts, lag(ts) OVER ({partition} ORDER BY {order}) AS p "
        f"FROM t) WHERE ts IS NOT NULL AND p IS NOT NULL AND NOT (ts {op} p)"
    ).fetchone()[0]
    order_by = ["device", "seq"] if by is None else "seq"
    report = (
        bt.from_arrow(_TABLE)
        .dq.monotonic("ts", order_by=order_by, by=by, strict=strict, decreasing=decreasing)
        .validate()
    )
    assert list(report.violations.values()) == ([expected] if expected else [])
    assert report.ok == (expected == 0)


def test_drop_removes_exactly_the_rows_where_the_sequence_broke():
    ds = bt.from_arrow(_TABLE)
    kept = ds.dq.monotonic("ts", order_by="seq", by="device").drop().sort("device", "seq")
    assert kept.to_pydict()["ts"] == [10, 30, 30, 5, None, 4, 7]


def test_order_by_is_required():
    with pytest.raises(bt.PlanError, match="order_by"):
        bt.from_arrow(_TABLE).dq.monotonic("ts", order_by=[])
