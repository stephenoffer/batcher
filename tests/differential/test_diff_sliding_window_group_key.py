"""A sliding `window(...)` may not be a group key directly — it is a list, not a start.

A row belongs to several overlapping sliding windows, so `window(ts, width, slide)` is the
*list* of the starts containing it. Grouping by that list groups by the array: rows whose
overlap sets happen to match land together, and each row is counted once instead of once
per window it belongs to. It returned rows, and the rows were wrong.

The engine now refuses it and names the fix. The correct spelling — explode, then group —
is checked against DuckDB here so the right answer stays pinned while the wrong one stays
rejected.
"""

from __future__ import annotations

from datetime import datetime

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential


@pytest.fixture
def events(duck):
    t = pa.table(
        {
            "ts": pa.array(
                [
                    datetime(2024, 1, 1, 23, 40),
                    datetime(2024, 1, 1, 23, 50),
                    datetime(2024, 1, 2, 0, 10),
                    datetime(2024, 1, 2, 0, 45),
                ],
                pa.timestamp("us"),
            ),
            "v": [1, 2, 3, 4],
        }
    )
    duck.register("events", t)
    return t


def test_sliding_window_as_a_group_key_is_rejected(events):
    ds = bt.from_arrow(events)
    with pytest.raises(PlanError, match="sliding window"):
        ds.group_by(w=bt.window(col("ts"), "1h", "30m")).agg(n=col("v").sum())


def test_tumbling_window_still_groups_directly(duck, events):
    got = (
        bt.from_arrow(events).group_by(w=bt.window(col("ts"), "1h")).agg(n=col("v").sum()).collect()
    )
    expected = duck.sql(
        "SELECT time_bucket(INTERVAL '1 hour', ts) AS w, SUM(v) AS n FROM events GROUP BY w"
    )
    assert_same(got, expected)


def test_exploded_sliding_window_matches_duckdb(duck, events):
    # The supported spelling: fan out to one row per window, then group.
    got = (
        bt.from_arrow(events)
        .select(w=bt.window(col("ts"), "1h", "30m"), v=col("v"))
        .explode("w")
        .group_by("w")
        .agg(n=col("v").sum())
        .collect()
    )
    # Oracle: every 30-minute-aligned start s where s <= ts < s + 1h.
    expected = duck.sql(
        """
        SELECT s AS w, SUM(v) AS n
        FROM events,
             generate_series(
                 time_bucket(INTERVAL '30 minutes', ts - INTERVAL '30 minutes'),
                 time_bucket(INTERVAL '30 minutes', ts),
                 INTERVAL '30 minutes'
             ) AS g(s)
        WHERE s <= ts AND ts < s + INTERVAL '1 hour'
        GROUP BY s
        """
    )
    assert_same(got, expected)
