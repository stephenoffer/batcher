"""`forward_fill` / `backward_fill` — the time-series gap filler, checked against DuckDB.

DuckDB spells these as framed value functions with null-skipping:

    forward_fill  == last_value(x IGNORE NULLS)
                     OVER (… ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    backward_fill == first_value(x IGNORE NULLS)
                     OVER (… ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)

Batcher makes the frame implicit — a fill has exactly one sensible frame — so there is
nothing to pass and nothing to get wrong. The semantics must still be identical, which
is what these assert, over the edges that break naive implementations: leading nulls with
nothing to carry, all-null partitions, partition boundaries, and rows that arrive out of
order relative to the `order_by` key.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

_FORWARD = "last_value(x IGNORE NULLS) OVER w"
_BACKWARD = "first_value(x IGNORE NULLS) OVER w"
_FRAME_FWD = "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
_FRAME_BWD = "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"


def _table(x: list[int | None], t: list[int] | None = None, g: list[str] | None = None):
    cols = {"t": pa.array(t or list(range(len(x))), type=pa.int64())}
    if g is not None:
        cols["g"] = pa.array(g, type=pa.string())
    cols["x"] = pa.array(x, type=pa.int64())
    return pa.table(cols)


def _duck(duck, table, direction: str, partition: bool) -> list[int | None]:
    duck.register("s", table)
    value = _FORWARD if direction == "forward" else _BACKWARD
    frame = _FRAME_FWD if direction == "forward" else _FRAME_BWD
    part = "PARTITION BY g " if partition else ""
    # `t` alone ties across partitions, so the final ORDER BY must include `g` too —
    # otherwise the two engines emit the same multiset in different orders.
    final = "ORDER BY t, g" if partition else "ORDER BY t"
    sql = f"SELECT t, {value} AS f FROM s WINDOW w AS ({part}ORDER BY t {frame}) {final}"
    return [r[1] for r in duck.sql(sql).fetchall()]


def _batcher(table, direction: str, partition: bool) -> list[int | None]:
    ds = bt.from_arrow(table)
    fill = bt.col("x").forward_fill() if direction == "forward" else bt.col("x").backward_fill()
    windowed = fill.over(partition_by=["g"] if partition else [], order_by=["t"])
    keys = ["t", "g"] if partition else ["t"]
    return ds.with_columns(f=windowed).sort(*keys).to_pydict()["f"]


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize(
    "x",
    [
        [10, None, None, 40],  # interior gaps
        [None, None, 30, None],  # leading nulls (forward has nothing to carry)
        [1, None, None, None],  # trailing nulls (backward has nothing to carry)
        [None, None, None, None],  # all null
        [1, 2, 3, 4],  # no nulls: the fill is the identity
        [None, 2, None, 4],  # alternating
    ],
    ids=["interior", "leading", "trailing", "all_null", "dense", "alternating"],
)
def test_it_matches_duckdbs_ignore_nulls_value_function(duck, direction, x):
    table = _table(x)
    assert _batcher(table, direction, partition=False) == _duck(duck, table, direction, False)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_a_fill_never_crosses_a_partition_boundary(duck, direction):
    """Two interleaved series. One device's reading must not leak into another's gap."""
    table = _table(
        x=[10, None, None, 20, None, 30],
        t=[0, 0, 1, 1, 2, 2],
        g=["a", "b", "a", "b", "a", "b"],
    )
    assert _batcher(table, direction, partition=True) == _duck(duck, table, direction, True)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_the_fill_follows_the_order_key_not_arrival_order(duck, direction):
    """Rows arrive shuffled; the carried value must be the neighbour in `t`, not in the scan."""
    table = _table(x=[None, 7, None, None, 9], t=[4, 0, 2, 1, 3])
    assert _batcher(table, direction, partition=False) == _duck(duck, table, direction, False)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_it_agrees_across_batch_boundaries(duck, direction):
    """A morselized scan must not restart the carry at each batch."""
    x = [i if i % 4 == 0 else None for i in range(200)]
    table = _table(x)
    ds = bt.from_arrow(table.to_batches(max_chunksize=7))
    fill = bt.col("x").forward_fill() if direction == "forward" else bt.col("x").backward_fill()
    got = ds.with_columns(f=fill.over(order_by=["t"])).sort("t").to_pydict()["f"]
    assert got == _duck(duck, table, direction, False)


def test_a_string_column_fills_too(duck):
    """The fill is type-generic — it selects a row, it does not compute a value."""
    table = pa.table(
        {
            "t": pa.array([0, 1, 2, 3], type=pa.int64()),
            "x": pa.array(["a", None, None, "d"], type=pa.string()),
        }
    )
    duck.register("s", table)
    expected = [
        r[1]
        for r in duck.sql(
            f"SELECT t, {_FORWARD} AS f FROM s WINDOW w AS (ORDER BY t {_FRAME_FWD}) ORDER BY t"
        ).fetchall()
    ]
    ds = bt.from_arrow(table)
    got = ds.with_columns(f=bt.col("x").forward_fill().over(order_by=["t"])).sort("t").to_pydict()
    assert got["f"] == expected


# --- the Dataset-level strategy ------------------------------------------------------


def test_fill_null_strategy_forward_matches_the_expression(duck):
    table = _table([10, None, None, 40])
    got = bt.from_arrow(table).fill_null(strategy="forward", order_by=["t"]).to_pydict()["x"]
    assert got == _duck(duck, table, "forward", False)


def test_fill_null_strategy_respects_partitions(duck):
    table = _table(x=[10, None, None, 20], t=[0, 0, 1, 1], g=["a", "b", "a", "b"])
    got = (
        bt.from_arrow(table)
        .fill_null(strategy="forward", order_by=["t"], partition_by=["g"])
        .sort("t", "g")
        .to_pydict()["x"]
    )
    assert got == [10, None, 10, 20]


def test_the_order_and_partition_keys_are_never_filled():
    """They are the frame of reference; filling them would redefine the order mid-fill."""
    table = pa.table(
        {
            "t": pa.array([0, 1, 2], type=pa.int64()),
            "g": pa.array(["a", None, "a"], type=pa.string()),
            "x": pa.array([5, None, None], type=pa.int64()),
        }
    )
    got = bt.from_arrow(table).fill_null(strategy="forward", order_by=["t"], partition_by=["g"])
    out = got.to_pydict()
    assert out["g"] == ["a", None, "a"], "the partition key must survive untouched"
    assert out["t"] == [0, 1, 2]


def test_a_carrying_strategy_without_order_by_is_an_error():
    ds = bt.from_pydict({"t": [1, 2], "x": [1, None]})
    with pytest.raises(PlanError, match="requires `order_by`"):
        ds.fill_null(strategy="forward")
    with pytest.raises(PlanError, match="requires `order_by`"):
        ds.fill_null(strategy="backward")


def test_the_bare_window_expression_also_demands_an_order():
    ds = bt.from_pydict({"x": [1, None]})
    with pytest.raises(PlanError, match="requires order_by"):
        ds.with_columns(f=bt.col("x").forward_fill().over()).to_pydict()


def test_an_unknown_order_key_is_reported():
    ds = bt.from_pydict({"t": [1], "x": [1]})
    with pytest.raises(PlanError, match="unknown order_by/partition_by"):
        ds.fill_null(strategy="forward", order_by=["nope"])


def test_median_is_still_rejected_with_an_actionable_message():
    ds = bt.from_pydict({"x": [1, None, 3]})
    with pytest.raises(PlanError, match="median is not a window aggregate"):
        ds.fill_null(strategy="median")


def test_the_fill_composes_with_a_downstream_aggregate(duck):
    """A window is a pipeline breaker; the plan must keep flowing through it."""
    table = _table([10, None, None, 40])
    total = (
        bt.from_arrow(table)
        .fill_null(strategy="forward", order_by=["t"])
        .agg(s=bt.col("x").sum())
        .to_pydict()["s"][0]
    )
    assert total == 10 + 10 + 10 + 40
