"""`join_stream` against DuckDB, and the streaming driver against the bounded path.

The interval is part of the join *condition*, not a filter above it. For an inner join
those are the same query; for an outer join they are different answers, and the two paths
disagreed on exactly that case — a left row whose key matched but whose event time was
outside the window was deleted by the bounded path and emitted null-padded by the
streaming one.

So this checks two equivalences at once: the bounded path against DuckDB, which is the
oracle for what an interval outer join means, and the streaming driver against the bounded
path, which is invariant #7 (single-node == distributed, and here bounded == unbounded).
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_BASE = datetime.datetime(2024, 1, 1)
_WITHIN_MINUTES = 10
_WITHIN_US = _WITHIN_MINUTES * 60 * 1_000_000

_LEFT_SCHEMA = pa.schema([("k", pa.int64()), ("lt", pa.timestamp("us")), ("lv", pa.string())])
_RIGHT_SCHEMA = pa.schema([("k", pa.int64()), ("rt", pa.timestamp("us")), ("rv", pa.string())])

#: Deliberately covers every case the join has to distinguish: a pair matching on both key
#: and interval, a pair matching on key but *not* interval, a left-only key, a right-only
#: key, and a duplicated key on both sides.
_LEFT = {
    "k": [1, 2, 3, 4, 4],
    "lt": [_BASE, _BASE, _BASE, _BASE, _BASE + datetime.timedelta(minutes=1)],
    "lv": ["a", "b", "c", "d1", "d2"],
}
_RIGHT = {
    "k": [1, 2, 9, 4, 4],
    "rt": [
        _BASE + datetime.timedelta(minutes=5),
        _BASE + datetime.timedelta(minutes=500),
        _BASE,
        _BASE,
        _BASE + datetime.timedelta(minutes=2),
    ],
    "rv": ["A", "B", "Z", "D1", "D2"],
}

_SQL = """
SELECT l.k AS k, l.lt AS lt, l.lv AS lv, r.k AS k_right, r.rt AS rt, r.rv AS rv
FROM l {how} JOIN r
  ON l.k = r.k
 AND abs(epoch_us(l.lt) - epoch_us(r.rt)) <= {within}
"""


def _batcher(how: str):
    left = bt.from_pydict(_LEFT)
    right = bt.from_pydict(_RIGHT)
    return left.join_stream(right, on="k", left_time="lt", right_time="rt", within="10m", how=how)


def _duck(con, how: str):
    con.register("l", pa.table(_LEFT, schema=_LEFT_SCHEMA))
    con.register("r", pa.table(_RIGHT, schema=_RIGHT_SCHEMA))
    sql_how = {"inner": "INNER", "left": "LEFT", "right": "RIGHT", "full": "FULL OUTER"}[how]
    return con.sql(_SQL.format(how=sql_how, within=_WITHIN_US))


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_bounded_interval_join_matches_duckdb(duck, how):
    shared = ["lt", "lv", "rt", "rv"]
    got = _batcher(how).collect().select(shared)
    # DuckDB keeps both key columns under their own names; Batcher's inner-shaped output
    # keeps the left key and suffixes the right one. Compare the columns both agree on.
    want = _duck(duck, how).project(", ".join(shared))
    assert_same(got, want)


#: The equivalence dataset. Every event time sits inside one window, so the watermark
#: never has cause to discard a row that a bounded run would still have matched — which is
#: the only condition under which "bounded == unbounded" can hold at all. A far-future
#: event on either side legitimately closes the window for everything older, and that
#: divergence is semantics, not a defect (`_LEFT`/`_RIGHT` above carry one on purpose, for
#: the DuckDB comparison, which has no watermark).
_WM_LEFT = {
    "k": [1, 2, 4, 4],
    "lt": [_BASE, _BASE, _BASE, _BASE + datetime.timedelta(minutes=1)],
    "lv": ["a", "b", "d1", "d2"],
}
_WM_RIGHT = {
    "k": [1, 9, 4, 4],
    "rt": [
        _BASE + datetime.timedelta(minutes=5),
        _BASE + datetime.timedelta(minutes=2),
        _BASE,
        _BASE + datetime.timedelta(minutes=2),
    ],
    "rv": ["A", "Z", "D1", "D2"],
}


def _stream(how: str) -> list[dict]:
    def feed(rows, schema):
        def gen():
            for i in range(len(rows["k"])):
                yield pa.record_batch({c: [v[i]] for c, v in rows.items()}, schema=schema)

        return gen

    left = bt.from_batches(feed(_WM_LEFT, _LEFT_SCHEMA), _LEFT_SCHEMA, bounded=False)
    right = bt.from_batches(feed(_WM_RIGHT, _RIGHT_SCHEMA), _RIGHT_SCHEMA, bounded=False)
    joined = left.join_stream(right, on="k", left_time="lt", right_time="rt", within="10m", how=how)
    rows: list[dict] = []
    for batch in joined.iter_batches():
        rows.extend(batch.to_pylist())
    return rows


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
def test_the_streaming_driver_agrees_with_the_bounded_path(how):
    """Bounded == unbounded. The whole point of one API over both."""
    streamed = sorted((str(r["lv"]), str(r["rv"])) for r in _stream(how))
    bounded = (
        bt.from_pydict(_WM_LEFT)
        .join_stream(
            bt.from_pydict(_WM_RIGHT),
            on="k",
            left_time="lt",
            right_time="rt",
            within="10m",
            how=how,
        )
        .to_pydict()
    )
    expected = sorted((str(a), str(b)) for a, b in zip(bounded["lv"], bounded["rv"], strict=True))
    assert streamed == expected


def test_a_key_match_outside_the_interval_is_not_a_match(duck):
    """The case the two paths disagreed on, pinned on its own: an outer join must emit
    the row null-padded, not delete it."""
    left = bt.from_pydict({"k": [1], "lt": [_BASE], "lv": ["a"]})
    right = bt.from_pydict({"k": [1], "rt": [_BASE + datetime.timedelta(minutes=500)], "rv": ["A"]})
    out = left.join_stream(
        right, on="k", left_time="lt", right_time="rt", within="10m", how="left"
    ).to_pydict()
    assert out["lv"] == ["a"]
    assert out["rv"] == [None]


def test_an_unknown_how_is_refused_by_name():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="unknown how"):
        bt.from_pydict({"k": [1], "lt": [_BASE]}).join_stream(
            bt.from_pydict({"k": [1], "rt": [_BASE]}),
            on="k",
            left_time="lt",
            right_time="rt",
            within="10m",
            how="semi",
        )


def test_a_null_key_survives_an_outer_join_and_not_an_inner_one(duck):
    """A null key matches nothing — which for an inner join means "drop it" and for an
    outer join means "this is exactly the row to emit null-padded". Kyber's null-key
    rejection knew only the first reading, so it deleted the row from both."""
    left = {"k": [1, None], "lt": [_BASE, _BASE], "lv": ["a", "orphan"]}
    right = {"k": [1], "rt": [_BASE], "rv": ["A"]}
    con = duck
    con.register("nl", pa.table(left, schema=_LEFT_SCHEMA))
    con.register("nr", pa.table(right, schema=_RIGHT_SCHEMA))

    for how, sql_how in (("inner", "INNER"), ("left", "LEFT")):
        got = (
            bt.from_pydict(left)
            .join_stream(
                bt.from_pydict(right),
                on="k",
                left_time="lt",
                right_time="rt",
                within="10m",
                how=how,
            )
            .collect()
            .select(["lv", "rv"])
        )
        want = con.sql(
            f"SELECT l.lv AS lv, r.rv AS rv FROM nl l {sql_how} JOIN nr r "
            f"ON l.k = r.k AND abs(epoch_us(l.lt) - epoch_us(r.rt)) <= {_WITHIN_US}"
        )
        assert_same(got, want)
