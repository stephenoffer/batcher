"""Differential coverage for GROUPS and peer-RANGE window frames vs DuckDB.

`ROWS` frames count physical rows; `GROUPS`/`RANGE` count peer groups (ties in the
ORDER BY key), so they differ from `ROWS` exactly when the order key has ties.
"""

from __future__ import annotations

import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _data():
    # Order key `t` has ties (1,1 / 2,2), so peer-based frames differ from ROWS.
    return bt.from_pydict(
        {
            "g": ["a", "a", "a", "a", "a", "b", "b"],
            "t": [1, 1, 2, 2, 3, 1, 2],
            "v": [10, 20, 30, 40, 50, 5, 7],
        }
    )


def test_groups_frame_matches_duckdb(duck):
    ds = _data()
    duck.register("t", ds.collect())
    got = ds.window(
        partition_by=["g"], order_by=["t"], functions={"s": ("sum", "v")}, frame=(-1, 0, "groups")
    ).collect()
    want = duck.sql(
        "SELECT *, sum(v) OVER (PARTITION BY g ORDER BY t "
        "GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM t"
    )
    assert_same(got, want)


def test_range_peer_frame_matches_duckdb(duck):
    ds = _data()
    duck.register("t", ds.collect())
    got = ds.window(
        partition_by=["g"], order_by=["t"], functions={"s": ("sum", "v")}, frame=(None, 0, "range")
    ).collect()
    want = duck.sql(
        "SELECT *, sum(v) OVER (PARTITION BY g ORDER BY t "
        "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM t"
    )
    assert_same(got, want)


def test_numeric_range_offset_is_honoured_not_silently_downgraded(duck):
    """A numeric ``RANGE`` offset is value-based, and the engine now computes it.

    This case is the one that proves it is not being quietly served by some other frame.
    For ``t = [10, 12, 13, 20, 21, 30]`` and ``RANGE BETWEEN 5 PRECEDING AND 5 FOLLOWING``
    DuckDB sums the rows whose *value* is within 5 -> ``[6, 6, 6, 9, 9, 6]``, while the
    default running frame gives the cumulative ``[1, 3, 6, 10, 15, 21]``. The engine once
    returned the second answer for the first query, silently; it was then made to raise,
    and now it returns the first. Asserting that the two frames genuinely disagree keeps
    the check honest — without it, a downgrade would pass.
    """
    ds = bt.from_pydict({"g": ["a"] * 6, "t": [10, 12, 13, 20, 21, 30], "v": [1, 2, 3, 4, 5, 6]})
    duck.register("t", ds.collect())
    want = [
        r[0]
        for r in duck.sql(
            "SELECT sum(v) OVER (PARTITION BY g ORDER BY t "
            "RANGE BETWEEN 5 PRECEDING AND 5 FOLLOWING) FROM t ORDER BY t"
        ).fetchall()
    ]
    running = [
        r[0]
        for r in duck.sql(
            "SELECT sum(v) OVER (PARTITION BY g ORDER BY t "
            "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t ORDER BY t"
        ).fetchall()
    ]
    assert want != running  # the two frames genuinely disagree here
    got = (
        ds.window(
            partition_by=["g"],
            order_by=["t"],
            functions={"s": ("sum", "v")},
            frame=(-5, 5, "range"),
        )
        .sort("t")
        .to_pydict()["s"]
    )
    assert got == want


def test_a_range_offset_over_an_unmeasurable_key_is_declined_not_approximated(duck):
    """The one shape the engine still refuses, and it refuses rather than substituting.

    A string order key sorts perfectly well, so a peer-bounded ``RANGE`` frame over it is
    fine — but ``2 PRECEDING`` would have to subtract two strings. Falling back to the peer
    frame there would answer a different question with no error, so it errors.
    """
    ds = bt.from_pydict({"g": ["a"] * 3, "t": ["x", "y", "z"], "v": [1, 2, 3]})
    peer = ds.window(
        partition_by=["g"], order_by=["t"], functions={"s": ("sum", "v")}, frame=(None, 0, "range")
    )
    assert peer.sort("t").to_pydict()["s"] == [1, 3, 6]
    with pytest.raises(Exception, match="numeric or temporal"):
        ds.window(
            partition_by=["g"],
            order_by=["t"],
            functions={"s": ("sum", "v")},
            frame=(-2, 0, "range"),
        ).collect()


def test_groups_current_row_is_peer_sum(duck):
    ds = _data()
    duck.register("t", ds.collect())
    got = ds.window(
        partition_by=["g"], order_by=["t"], functions={"s": ("sum", "v")}, frame=(0, 0, "groups")
    ).collect()
    want = duck.sql(
        "SELECT *, sum(v) OVER (PARTITION BY g ORDER BY t "
        "GROUPS BETWEEN CURRENT ROW AND CURRENT ROW) AS s FROM t"
    )
    assert_same(got, want)
