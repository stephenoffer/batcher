"""Value-based `RANGE` window frames, checked against DuckDB.

`RANGE BETWEEN 300000000 PRECEDING AND CURRENT ROW` over a microsecond timestamp is "the
last five minutes". Unlike the `ROWS` frame of the same shape, its row count varies with how
densely the series was sampled, which is exactly why time-series work reaches for it: a
five-minute average means the same thing whether the sensor reported twice a minute or two
hundred times.

DuckDB implements the same SQL frame, so it is the oracle throughout — including for the
cases a naive implementation gets wrong: ties on the order key (every tied row shares one
window), gaps larger than the window (the window collapses to the row itself), descending
order (`PRECEDING` still means *larger* values, not earlier rows), nulls in the order key
(they frame only their own peers), and `FOLLOWING` bounds on both sides.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

# Order-key sequences with the shapes that break a naive value search: dense, tied, a gap
# far wider than any window, a single row, and negative values.
_KEYS = {
    "dense": [1, 2, 3, 4, 5],
    "ties": [1, 1, 2, 2, 2, 5],
    "gap": [1, 2, 100, 101, 500],
    "single": [7],
    "negative": [-5, -3, 0, 2, 9],
}


def _table(keys, vals=None):
    vals = vals or [float(i + 1) for i in range(len(keys))]
    return pa.table(
        {
            "t": pa.array(keys, type=pa.int64()),
            "v": pa.array(vals, type=pa.float64()),
            "rid": pa.array(list(range(len(keys))), type=pa.int64()),
        }
    )


def _duck(duck, table, agg: str, start: str, end: str, *, descending=False, part=""):
    duck.register("s", table)
    order = "t DESC" if descending else "t"
    partition = f"PARTITION BY {part} " if part else ""
    sql = f"""
        SELECT {agg}(v) OVER (
                   {partition}ORDER BY {order}
                   RANGE BETWEEN {start} AND {end}
               ) AS w
        FROM s ORDER BY rid
    """
    return [r[0] for r in duck.sql(sql).fetchall()]


def _batcher(table, agg: str, frame, *, descending=False, part=()):
    order = [("t", True)] if descending else ["t"]
    w = getattr(bt.col("v"), agg)().over(partition_by=list(part), order_by=order, frame=frame)
    return bt.from_arrow(table).with_columns(w=w).sort("rid").to_pydict()["w"]


def _close(got, want, what):
    assert len(got) == len(want), f"{what}: length {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if g is None or w is None:
            assert g is None and w is None, f"{what}[{i}]: {g!r} vs {w!r}"
        else:
            assert abs(float(g) - float(w)) < 1e-9, f"{what}[{i}]: {g} vs {w}"


@pytest.mark.parametrize("shape", sorted(_KEYS))
@pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
@pytest.mark.parametrize("width", [0, 1, 2, 3, 10])
def test_a_trailing_range_window_matches_duckdb(duck, shape, agg, width):
    table = _table(_KEYS[shape])
    duck_agg = "avg" if agg == "mean" else agg
    _close(
        _batcher(table, agg, (-width, 0, "range")),
        _duck(duck, table, duck_agg, f"{width} PRECEDING", "CURRENT ROW"),
        f"{agg} RANGE {width} PRECEDING [{shape}]",
    )


@pytest.mark.parametrize("shape", sorted(_KEYS))
@pytest.mark.parametrize(
    ("frame", "start", "end"),
    [
        ((0, 2, "range"), "CURRENT ROW", "2 FOLLOWING"),
        ((-2, 2, "range"), "2 PRECEDING", "2 FOLLOWING"),
        ((-1, -1, "range"), "1 PRECEDING", "1 PRECEDING"),
        ((1, 3, "range"), "1 FOLLOWING", "3 FOLLOWING"),
        ((None, 2, "range"), "UNBOUNDED PRECEDING", "2 FOLLOWING"),
        ((-2, None, "range"), "2 PRECEDING", "UNBOUNDED FOLLOWING"),
    ],
    ids=["ahead", "centered", "lagged", "future_only", "unbounded_start", "unbounded_end"],
)
def test_every_range_bound_combination_matches_duckdb(duck, shape, frame, start, end):
    table = _table(_KEYS[shape])
    _close(
        _batcher(table, "sum", frame),
        _duck(duck, table, "sum", start, end),
        f"RANGE {start}..{end} [{shape}]",
    )


@pytest.mark.parametrize("shape", sorted(_KEYS))
def test_a_descending_order_still_measures_preceding_as_larger_values(duck, shape):
    """`PRECEDING` means "earlier in the ordering", which under DESC is a *larger* value.

    A search that assumed ascending keys would return the mirror-image window here, and it
    would still look plausible.
    """
    table = _table(_KEYS[shape])
    _close(
        _batcher(table, "sum", (-2, 0, "range"), descending=True),
        _duck(duck, table, "sum", "2 PRECEDING", "CURRENT ROW", descending=True),
        f"DESC RANGE [{shape}]",
    )


def test_a_range_window_restarts_at_every_partition(duck):
    table = pa.table(
        {
            "g": pa.array(["a", "a", "a", "b", "b"]),
            "t": pa.array([1, 2, 3, 1, 2], type=pa.int64()),
            "v": pa.array([1.0, 2.0, 3.0, 10.0, 20.0]),
            "rid": pa.array(list(range(5)), type=pa.int64()),
        }
    )
    _close(
        _batcher(table, "sum", (-1, 0, "range"), part=("g",)),
        _duck(duck, table, "sum", "1 PRECEDING", "CURRENT ROW", part="g"),
        "partitioned RANGE",
    )


def test_a_null_order_key_frames_only_its_own_peers(duck):
    table = pa.table(
        {
            "t": pa.array([None, None, 1, 2, 3], type=pa.int64()),
            "v": pa.array([1.0, 2.0, 4.0, 8.0, 16.0]),
            "rid": pa.array(list(range(5)), type=pa.int64()),
        }
    )
    _close(
        _batcher(table, "sum", (-1, 0, "range")),
        _duck(duck, table, "sum", "1 PRECEDING", "CURRENT ROW"),
        "null order key",
    )


def test_an_interval_window_over_timestamps_matches_duckdb(duck):
    base = dt.datetime(2024, 3, 1, 9, 0, 0)
    stamps = [base + dt.timedelta(minutes=m) for m in (0, 1, 2, 30, 31, 120)]
    table = pa.table(
        {
            "t": pa.array(stamps),
            "v": pa.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
            "rid": pa.array(list(range(6)), type=pa.int64()),
        }
    )
    duck.register("s", table)
    want = [
        r[0]
        for r in duck.sql(
            """
            SELECT sum(v) OVER (
                       ORDER BY t
                       RANGE BETWEEN INTERVAL 5 MINUTE PRECEDING AND CURRENT ROW
                   ) AS w
            FROM s ORDER BY rid
            """
        ).fetchall()
    ]
    # Five minutes is 300 million microseconds, which is the unit the engine normalizes
    # every temporal order key to.
    _close(_batcher(table, "sum", (-300_000_000, 0, "range")), want, "interval frame")
    # ...and the ergonomic spelling of the same window agrees.
    got = (
        bt.from_arrow(table)
        .with_columns(w=bt.col("v").rolling_sum_by("t", "5m"))
        .sort("rid")
        .to_pydict()["w"]
    )
    _close(got, want, "rolling_sum_by")


def test_the_sql_spelling_of_an_interval_range_frame_matches_duckdb(duck):
    base = dt.datetime(2024, 3, 1, 9, 0, 0)
    stamps = [base + dt.timedelta(minutes=m) for m in (0, 1, 2, 30, 31, 120)]
    table = pa.table(
        {
            "t": pa.array(stamps),
            "v": pa.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
            "rid": pa.array(list(range(6)), type=pa.int64()),
        }
    )
    duck.register("s", table)
    sql = """
        SELECT rid, sum(v) OVER (
                   ORDER BY t
                   RANGE BETWEEN INTERVAL 5 MINUTE PRECEDING AND CURRENT ROW
               ) AS w
        FROM s
    """
    want = [r[1] for r in duck.sql(sql + " ORDER BY rid").fetchall()]
    got = bt.sql(sql, s=bt.from_arrow(table)).sort("rid").to_pydict()["w"]
    _close(got, want, "SQL interval RANGE frame")


def test_a_groups_frame_is_now_reachable_from_sql_too(duck):
    """`GROUPS` was rejected alongside `RANGE` even though the engine executed it."""
    table = _table(_KEYS["ties"])
    duck.register("s", table)
    sql = """
        SELECT rid, sum(v) OVER (
                   ORDER BY t GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW
               ) AS w
        FROM s
    """
    want = [r[1] for r in duck.sql(sql + " ORDER BY rid").fetchall()]
    got = bt.sql(sql, s=bt.from_arrow(table)).sort("rid").to_pydict()["w"]
    _close(got, want, "SQL GROUPS frame")


def test_rolling_by_min_periods_nulls_the_thin_windows():
    ds = bt.from_pydict({"t": [0, 60, 120, 1800], "v": [1.0, 2.0, 3.0, 4.0]})
    got = ds.with_columns(r=bt.col("v").rolling_mean_by("t", 300, min_periods=2)).to_pydict()["r"]
    # The first row's window holds one value, and the last row's gap isolates it.
    assert got[0] is None and got[3] is None
    _close(got[1:3], [1.5, 2.0], "min_periods")


def test_a_range_frame_matches_single_node_when_distributed():
    n = 600
    table = pa.table(
        {
            "g": pa.array([f"s{i % 7}" for i in range(n)]),
            "t": pa.array([(i * 13) % 400 for i in range(n)], type=pa.int64()),
            "v": pa.array([float((i * 31) % 17) for i in range(n)]),
            "rid": pa.array(list(range(n)), type=pa.int64()),
        }
    )
    ds = bt.from_arrow(table)
    w = bt.col("v").sum().over(partition_by=["g"], order_by=["t"], frame=(-20, 0, "range"))
    one = ds.with_columns(w=w).sort("rid").to_pydict()["w"]
    many = ds.repartition(8).with_columns(w=w).sort("rid").to_pydict()["w"]
    _close(one, many, "repartitioned RANGE frame")


def test_a_range_offset_needs_a_measurable_single_order_key():
    ds = bt.from_pydict({"a": ["x", "y"], "b": [1, 2], "v": [1.0, 2.0]})
    w = bt.col("v").sum().over(order_by=["a"], frame=(-1, 0, "range"))
    with pytest.raises(Exception, match="numeric or temporal"):
        ds.with_columns(w=w).to_pydict()
    two = bt.col("v").sum().over(order_by=["b", "a"], frame=(-1, 0, "range"))
    with pytest.raises(Exception, match="exactly one ORDER BY key"):
        ds.with_columns(w=two).to_pydict()
