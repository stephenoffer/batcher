"""`interpolate` and `rle_id` — the whole-prefix series recurrences, checked against DuckDB.

Neither has a DuckDB *function*, but both have an exact DuckDB *expression*, which is a
better oracle than a hand-computed constant: it is derived independently of the kernel and
it exercises the same null and partition edges.

    rle_id      == sum(CASE WHEN x IS DISTINCT FROM lag(x) OVER w THEN 1 ELSE 0 END) OVER w
    interpolate == the point on the segment between the nearest non-null value at or before
                   the row and the nearest at or after it, weighted by ordered position —
                   spelled with `IGNORE NULLS` value functions over the two half-frames.

The cases are the ones that break naive implementations: leading and trailing null runs
(which have no bracket and must stay null), an all-null partition, partition boundaries a
run must not cross, and rows whose physical order differs from the `order_by` key.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

# The nearest non-null value at or before / at or after the current row, and the ordered
# position it sits at. Together these are the two ends of the segment to interpolate along.
_PREV = "last_value(x IGNORE NULLS) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
_PREV_POS = (
    "last_value(CASE WHEN x IS NOT NULL THEN p END IGNORE NULLS) "
    "OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
)
_NEXT = "first_value(x IGNORE NULLS) OVER (w ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)"
_NEXT_POS = (
    "first_value(CASE WHEN x IS NOT NULL THEN p END IGNORE NULLS) "
    "OVER (w ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)"
)


def _table(x, t=None, g=None, dtype=None):
    dtype = dtype or pa.float64()
    cols = {"t": pa.array(t or list(range(len(x))), type=pa.int64())}
    if g is not None:
        cols["g"] = pa.array(g, type=pa.string())
    cols["x"] = pa.array(x, type=dtype)
    return pa.table(cols)


def _duck_interpolate(duck, table, partition: bool) -> list[float | None]:
    duck.register("s", table)
    part = "PARTITION BY g " if partition else ""
    final = "ORDER BY t, g" if partition else "ORDER BY t"
    # `p` is the ordered position, which is what the interpolation weight is measured in.
    sql = f"""
        WITH pos AS (
            SELECT *, row_number() OVER ({part}ORDER BY t) AS p FROM s
        ), ends AS (
            SELECT t, {"g," if partition else ""} x, p,
                   {_PREV} AS pv, {_PREV_POS} AS pp,
                   {_NEXT} AS nv, {_NEXT_POS} AS np
            FROM pos WINDOW w AS ({part}ORDER BY t)
        )
        SELECT CASE
                 WHEN x IS NOT NULL THEN x::DOUBLE
                 WHEN pv IS NULL OR nv IS NULL THEN NULL
                 ELSE pv + (nv - pv) * (p - pp)::DOUBLE / (np - pp)
               END AS i
        FROM ends {final}
    """
    return [r[0] for r in duck.sql(sql).fetchall()]


def _batcher_interpolate(table, partition: bool) -> list[float | None]:
    ds = bt.from_arrow(table)
    w = bt.col("x").interpolate().over(partition_by=["g"] if partition else [], order_by=["t"])
    keys = ["t", "g"] if partition else ["t"]
    return ds.with_columns(i=w).sort(*keys).to_pydict()["i"]


def _close(got, want, what):
    assert len(got) == len(want), f"{what}: length {len(got)} != {len(want)}"
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        if g is None or w is None:
            assert g is None and w is None, f"{what}[{i}]: {g!r} vs {w!r}"
        else:
            assert abs(g - w) < 1e-9, f"{what}[{i}]: {g} vs {w}"


@pytest.mark.parametrize(
    "x",
    [
        [10.0, None, None, 40.0],  # one interior gap
        [None, None, 30.0, None],  # leading and trailing runs: no bracket, stay null
        [1.0, None, 3.0, None, 5.0],  # alternating
        [None, None, None, None],  # all null
        [1.0, 2.0, 3.0, 4.0],  # dense: interpolation is the identity
        [5.0, None, 5.0, None, 5.0],  # flat series: a gap must not drift
        [10.0, None, None, None, 2.0],  # a descending segment
    ],
    ids=["interior", "edges", "alternating", "all_null", "dense", "flat", "descending"],
)
def test_interpolate_matches_the_duckdb_segment_expression(duck, x):
    table = _table(x)
    _close(_batcher_interpolate(table, False), _duck_interpolate(duck, table, False), "interp")


def test_interpolate_never_crosses_a_partition_boundary(duck):
    table = _table(
        [1.0, None, None, 100.0, None, 300.0],
        t=[0, 1, 2, 0, 1, 2],
        g=["a", "a", "a", "b", "b", "b"],
    )
    _close(
        _batcher_interpolate(table, True),
        _duck_interpolate(duck, table, True),
        "partitioned interp",
    )


def test_interpolate_follows_order_by_not_arrival_order(duck):
    # The rows arrive with `t` descending, so a kernel that walked physical row order
    # would interpolate along the reverse of the requested order and disagree.
    table = _table([40.0, None, None, 10.0], t=[3, 2, 1, 0])
    _close(_batcher_interpolate(table, False), _duck_interpolate(duck, table, False), "unsorted")


def test_interpolate_widens_an_integer_column(duck):
    table = _table([0, None, 1], dtype=pa.int64())
    got = _batcher_interpolate(table, False)
    _close(got, [0.0, 0.5, 1.0], "int widen")
    _close(got, _duck_interpolate(duck, table, False), "int widen vs duckdb")


def _duck_rle(duck, table, partition: bool) -> list[int]:
    duck.register("s", table)
    part = "PARTITION BY g " if partition else ""
    final = "ORDER BY t, g" if partition else "ORDER BY t"
    # Two steps, because DuckDB forbids a window call inside another: mark each row that
    # differs from its predecessor, then running-sum the marks. `IS DISTINCT FROM` is what
    # makes a null run one run rather than a change at every row.
    sql = f"""
        WITH marked AS (
            SELECT *,
                   row_number() OVER ({part}ORDER BY t) AS rn,
                   CASE WHEN x IS DISTINCT FROM lag(x) OVER ({part}ORDER BY t)
                        THEN 1 ELSE 0 END AS chg
            FROM s
        )
        SELECT sum(CASE WHEN rn = 1 THEN 0 ELSE chg END) OVER (
                   {part}ORDER BY t ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS r
        FROM marked {final}
    """
    return [r[0] for r in duck.sql(sql).fetchall()]


def _batcher_rle(table, partition: bool) -> list[int]:
    ds = bt.from_arrow(table)
    w = bt.col("x").rle_id().over(partition_by=["g"] if partition else [], order_by=["t"])
    keys = ["t", "g"] if partition else ["t"]
    return ds.with_columns(r=w).sort(*keys).to_pydict()["r"]


@pytest.mark.parametrize(
    ("x", "dtype"),
    [
        (["on", "on", "off", "on"], pa.string()),
        (["on", None, None, "on"], pa.string()),  # a null run is one run of its own
        ([None, None, None], pa.string()),  # all null: one run
        ([1, 1, 1, 1], pa.int64()),  # one run over the whole partition
        ([1, 2, 3, 4], pa.int64()),  # a run per row
        ([1.0, 1.0, None, 1.0], pa.float64()),
        ([True, True, False, None, False], pa.bool_()),
    ],
    ids=["states", "gap", "all_null", "constant", "distinct", "floats", "bools"],
)
def test_rle_id_matches_the_duckdb_change_counter(duck, x, dtype):
    table = _table(x, dtype=dtype)
    assert _batcher_rle(table, False) == _duck_rle(duck, table, False)


def test_rle_id_restarts_at_every_partition(duck):
    table = _table(
        ["a", "a", "b", "x", "y", "y"],
        t=[0, 1, 2, 0, 1, 2],
        g=["p", "p", "p", "q", "q", "q"],
        dtype=pa.string(),
    )
    assert _batcher_rle(table, True) == _duck_rle(duck, table, True)


def test_rle_id_segments_a_series_into_groupable_runs(duck):
    """The point of a run id: collapse each run to a row and measure how long it lasted."""
    table = _table(
        ["idle", "idle", "run", "run", "run", "idle"],
        t=[0, 1, 2, 3, 4, 5],
        dtype=pa.string(),
    )
    duck.register("s", table)
    ds = bt.from_arrow(table)
    runs = (
        ds.with_columns(r=bt.col("x").rle_id().over(order_by=["t"]))
        .group_by("r")
        .agg(state=bt.col("x").min(), started=bt.col("t").min(), n=bt.col("t").count())
        .sort("r")
    )
    got = runs.to_pydict()
    assert got["state"] == ["idle", "run", "idle"]
    assert got["started"] == [0, 2, 5]
    assert got["n"] == [2, 3, 1]


@pytest.mark.parametrize("method", ["interpolate", "rle_id"])
def test_a_series_recurrence_requires_an_order(method):
    ds = bt.from_pydict({"x": [1.0, None, 3.0]})
    with pytest.raises(PlanError, match="requires order_by"):
        ds.with_columns(v=getattr(bt.col("x"), method)().over()).to_pydict()


@pytest.mark.parametrize("method", ["interpolate", "rle_id"])
def test_a_series_recurrence_matches_single_node_when_distributed(method):
    """The mergeable contract: partitioning by the window key must not change a row."""
    n = 400
    table = pa.table(
        {
            "g": pa.array([f"s{i % 7}" for i in range(n)]),
            "t": pa.array(list(range(n)), type=pa.int64()),
            "x": pa.array(
                [None if i % 5 == 0 else float(i % 13) for i in range(n)], type=pa.float64()
            ),
        }
    )
    ds = bt.from_arrow(table)
    w = getattr(bt.col("x"), method)().over(partition_by=["g"], order_by=["t"])
    one = ds.with_columns(v=w).sort("g", "t").to_pydict()["v"]
    many = ds.repartition(8).with_columns(v=w).sort("g", "t").to_pydict()["v"]
    _close(
        [None if v is None else float(v) for v in one],
        [None if v is None else float(v) for v in many],
        f"{method} repartitioned",
    )
