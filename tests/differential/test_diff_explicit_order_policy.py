"""One row-order policy: every order-dependent expression needs an explicit ``order_by``.

Batcher keeps no arrival order across a parallel or distributed scan, so an expression
whose answer depends on which row came first -- ``shift``, ``diff``, ``pct_change``,
``cum_*``, ``rolling_*``, ``first``/``last``, the fills, ``interpolate``, ``rle_id``, the
EWMs, ``is_first_distinct``/``is_last_distinct``, ``peak_*``, ``row_number`` -- is refused
without one, with a message naming ``order_by=`` and ``.with_row_index("_row")``. Every one
is accepted inside ``.over(order_by=...)``, and the ones that take ``order_by=`` directly
accept it there too. Order-*independent* windows (a whole-partition aggregate,
``is_duplicated``, ``rank`` over its own value) still need none.

Once ordered by ``_row`` the answers match DuckDB's window SQL over the same row number,
across nulls, an empty input, one row, duplicates, and a descending order; and match
Polars 1.40, whose implicit row order is the one ``with_row_index`` makes explicit.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential

col = bt.col

#: Every order-dependent expression, built without an order.
UNORDERED = {
    "shift": lambda: col("x").shift(1),
    "shift_lead": lambda: col("x").shift(-1),
    "diff": lambda: col("x").diff(),
    "pct_change": lambda: col("x").pct_change(),
    "cum_sum": lambda: col("x").cum_sum(),
    "cum_min": lambda: col("x").cum_min(),
    "cum_max": lambda: col("x").cum_max(),
    "cum_prod": lambda: col("x").cum_prod(),
    "cum_count": lambda: col("x").cum_count(),
    "rolling_sum": lambda: col("x").rolling_sum(2),
    "rolling_mean": lambda: col("x").rolling_mean(2),
    "rolling_std": lambda: col("x").rolling_std(2),
    "first": lambda: col("x").first(),
    "last": lambda: col("x").last(),
    "forward_fill": lambda: col("x").forward_fill(),
    "backward_fill": lambda: col("x").backward_fill(),
    "interpolate": lambda: col("x").interpolate(),
    "rle_id": lambda: col("x").rle_id(),
    "ewm_mean": lambda: col("x").ewm_mean(alpha=0.5),
    "ewm_std": lambda: col("x").ewm_std(alpha=0.5),
    "is_first_distinct": lambda: col("x").is_first_distinct(),
    "is_last_distinct": lambda: col("x").is_last_distinct(),
    "peak_max": lambda: col("x").peak_max(),
    "row_number": lambda: bt.row_number(),
    "lag": lambda: bt.lag(col("x")),
    "first_value": lambda: bt.first_value(col("x")),
}

_TABLE = pa.table({"x": pa.array([3.0, None, 1.0, 1.0, 5.0], pa.float64())})


@pytest.mark.parametrize("name", sorted(UNORDERED))
def test_an_unordered_order_dependent_expression_is_refused(name):
    ds = bt.from_arrow(_TABLE)
    with pytest.raises(PlanError, match=r"requires order_by") as err:
        ds.with_columns(out=UNORDERED[name]()).collect()
    assert "order_by=" in str(err.value)
    assert 'with_row_index("_row")' in str(err.value)


@pytest.mark.parametrize("name", sorted(UNORDERED))
def test_every_one_is_accepted_inside_over(name):
    ds = bt.from_arrow(_TABLE).with_row_index("_row")
    out = ds.with_columns(out=UNORDERED[name]().over(order_by="_row")).collect()
    assert out.num_rows == _TABLE.num_rows


def test_grouped_first_without_order_is_refused():
    ds = bt.from_arrow(_TABLE)
    with pytest.raises(PlanError, match=r"first depends on row order"):
        ds.group_by().agg(col("x").first())


def test_order_independent_windows_need_no_order():
    ds = bt.from_arrow(_TABLE)
    out = ds.select(
        total=col("x").sum().over(),
        dup=col("x").is_duplicated(),
        rank=col("x").rank(),
    ).to_pydict()
    assert out["dup"] == [False, False, True, True, False]
    assert out["total"] == [10.0] * 5


_TABLES = {
    "nulls": pa.table({"x": pa.array([3.0, None, 1.0, None, 5.0, 2.0], pa.float64())}),
    "empty": pa.table({"x": pa.array([], pa.float64())}),
    "one_row": pa.table({"x": pa.array([4.0], pa.float64())}),
    "duplicates": pa.table({"x": pa.array([2.0, 2.0, 2.0, 1.0], pa.float64())}),
}


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("fixture", sorted(_TABLES))
def test_ordered_by_row_index_matches_duckdb(duck, fixture, descending):
    table = _TABLES[fixture]
    order = {"order_by": "_row", "descending": descending}
    ours = (
        bt.from_arrow(table)
        .with_row_index("_row")
        .with_columns(
            prev=col("x").shift(1).over(**order),
            delta=col("x").diff().over(**order),
            running=col("x").cum_sum().over(**order),
            roll=col("x").rolling_sum(2).over(**order),
            ffill=col("x").forward_fill().over(**order),
            first=col("x").is_first_distinct().over(**order),
        )
        .collect()
        # Ordered by pyarrow rather than by the engine's `sort`, deliberately: this file tests
        # the windows, and an engine sort after a partitioned window is currently eliminated
        # on a source an earlier query in the same process read (a Kyber defect reproduced at
        # the base commit), which would fail the comparison for a reason unrelated to it.
        .sort_by("_row")
    )
    duck.register("src", table)
    direction = "DESC" if descending else "ASC"
    w = f"ORDER BY _row {direction}"
    expected = duck.sql(
        "WITH t AS (SELECT row_number() OVER () - 1 AS _row, x FROM src) "
        f"SELECT _row, x, lag(x) OVER ({w}) AS prev, x - lag(x) OVER ({w}) AS delta, "
        f"sum(x) OVER ({w} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running, "
        f"sum(x) OVER ({w} ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS roll, "
        f"last_value(x IGNORE NULLS) OVER ({w} ROWS BETWEEN UNBOUNDED PRECEDING "
        "AND CURRENT ROW) AS ffill, "
        f"row_number() OVER (PARTITION BY x {w}) = 1 AS first FROM t ORDER BY _row"
    )
    assert_same_ordered(ours, expected)


def test_polars_implicit_order_is_the_row_index():
    pl = pytest.importorskip("polars")
    table = _TABLES["duplicates"]
    theirs = pl.from_arrow(table).select(
        pl.col("x").shift(1).alias("prev"),
        pl.col("x").cum_max().alias("running"),
        pl.col("x").is_last_distinct().alias("last"),
        pl.col("x").first().alias("first"),
    )
    ours = (
        bt.from_arrow(table)
        .with_row_index("_row")
        .select(
            "_row",
            prev=col("x").shift(1).over(order_by="_row"),
            running=col("x").cum_max(order_by="_row"),
            last=col("x").is_last_distinct("_row"),
            first=col("x").first().over(order_by="_row"),
        )
        .collect()
        .sort_by("_row")
        .drop(["_row"])
    )
    assert ours.to_pydict() == theirs.to_dict(as_series=False)
