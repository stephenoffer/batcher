"""Workstream C — session windows match DuckDB.

A session groups consecutive per-key events whose gap is below the timeout; a larger
gap starts a new session. Batcher composes this from the window engine (lag + a
running-sum session id) + group-by — no new operator. DuckDB computes the same
session id with the identical SQL window formulation as the independent oracle.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

_BASE = dt.datetime(2024, 1, 1, 0, 0, 0)


def _data() -> pa.Table:
    # Two keys with several sessions each; 10-minute gap boundary.
    recs = [
        ("a", 0, 1),
        ("a", 2, 2),
        ("a", 5, 3),  # session 1 (gaps ≤ 10m)
        ("a", 40, 4),
        ("a", 45, 5),  # session 2
        ("b", 0, 6),  # session 1 (lone)
        ("b", 30, 7),
        ("b", 31, 8),  # session 2
    ]
    ts = [_BASE + dt.timedelta(minutes=m) for _, m, _ in recs]
    return pa.table(
        {
            "k": pa.array([k for k, _, _ in recs], pa.string()),
            "ts": pa.array(ts, pa.timestamp("us")),
            "v": pa.array([v for _, _, v in recs], pa.int64()),
        }
    )


@pytest.mark.parametrize("gap", ["10m", "1m"])
def test_session_window_matches_duckdb(duck, gap):
    from conftest import assert_same

    tbl = _data()
    got = (
        bt.from_arrow(tbl)
        .session_window("ts", gap, partition_by=["k"], total=col("v").sum(), n=col("v").count())
        .collect()
    )

    gap_secs = {"10m": 600, "1m": 60}[gap]
    duck.register("t", tbl)
    # Independent oracle: the same session-id formulation in SQL (lag → new-session
    # marker → running sum), then group by (k, session_id).
    rel = duck.sql(
        f"""
        WITH marked AS (
            SELECT *,
                CASE WHEN epoch_us(ts) - lag(epoch_us(ts)) OVER w > {gap_secs} * 1000000
                          OR lag(epoch_us(ts)) OVER w IS NULL
                     THEN 1 ELSE 0 END AS new_session
            FROM t WINDOW w AS (PARTITION BY k ORDER BY ts)
        ),
        sessioned AS (
            SELECT *, sum(new_session) OVER (PARTITION BY k ORDER BY ts) AS sid FROM marked
        )
        SELECT k, min(ts) AS session_start, max(ts) AS session_end,
               sum(v) AS total, count(v) AS n
        FROM sessioned GROUP BY k, sid
        """
    )
    assert_same(got, rel)
