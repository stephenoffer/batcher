"""A window aggregate admits every column type the same aggregate admits under `GROUP BY`.

`OVER (PARTITION BY ...)` and `GROUP BY` are the same aggregate over the same groups, so the
set of column types each accepts must be the same set. It was not: the window kernels are
written against `Int64`/`Float64`/`Utf8`/`Boolean`, and everything else raised
`window function <f> is not supported for column type <t>` while the identical `GROUP BY`
answered. `COUNT(DISTINCT order_date) OVER (PARTITION BY customer)` — an ordinary analytics
query — was one of the shapes that failed.

Two rules close the gap, and this module pins both against DuckDB:

* An all-null column carries Arrow's `Null` type, which widens to an all-null `Int64` exactly
  as `agg::coerce_null_call_inputs` already widened it under a `GROUP BY`.
* `COUNT(DISTINCT x)` only has to tell two values apart, so it keys on the `RowConverter`
  encoding for any type outside the four above — the same encoding `MIN`/`MAX` already use.

The parity itself is asserted directly (window result == group-by result broadcast back), so
a future kernel that accepts a type on one path and not the other fails here rather than in a
user's query.
"""

from __future__ import annotations

import datetime as dt
import decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

# One column per type family the window kernels do not read natively, each with a repeat and
# a null so DISTINCT, the null-skipping and the group-collapsing all have something to do.
_COLUMNS: dict[str, pa.Array] = {
    "int32": pa.array([1, 1, 2, None], pa.int32()),
    "float32": pa.array([1.5, 1.5, 2.5, None], pa.float32()),
    "uint64": pa.array([1, 1, 2, None], pa.uint64()),
    "decimal": pa.array(
        [decimal.Decimal("1.00"), decimal.Decimal("1.00"), decimal.Decimal("2.50"), None],
        pa.decimal128(10, 2),
    ),
    "date32": pa.array(
        [dt.date(2020, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 6, 30), None], pa.date32()
    ),
    "timestamp": pa.array(
        [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 1), dt.datetime(2020, 6, 30), None],
        pa.timestamp("us"),
    ),
    "time64": pa.array([dt.time(1, 0), dt.time(1, 0), dt.time(2, 30), None], pa.time64("us")),
    "large_string": pa.array(["a", "a", "b", None], pa.large_string()),
    "null": pa.array([None, None, None, None], pa.null()),
}

# Two partitions, so a per-partition answer is distinguishable from a whole-relation one.
_KEYS = pa.array(["p", "p", "q", "q"])
#: An ORDER BY column, so the *running* frame can be exercised as well as the whole partition.
#: They are separate kernels and diverged separately.
_ORDER = pa.array([1, 2, 1, 2])


def _table(name: str) -> pa.Table:
    return pa.table({"v": _COLUMNS[name], "k": _KEYS, "o": _ORDER})


@pytest.mark.differential
@pytest.mark.parametrize("name", sorted(_COLUMNS))
def test_count_distinct_over_matches_duckdb(duck, name):
    """`COUNT(DISTINCT v) OVER (PARTITION BY k)` for every type family."""
    tbl = _table(name)
    got = bt.from_arrow(tbl).with_columns(n=col("v").n_unique().over("k")).collect()

    duck.register("t", tbl)
    assert_same(
        got,
        duck.sql("select v, k, o, count(distinct v) over (partition by k) as n from t"),
    )


@pytest.mark.differential
@pytest.mark.parametrize("func", ["sum", "mean", "min", "max", "count", "n_unique"])
def test_every_aggregate_over_an_all_null_column_matches_duckdb(duck, func):
    """An all-null (`Null`-typed) column: the window form raised where `GROUP BY` answered."""
    tbl = _table("null")
    got = bt.from_arrow(tbl).with_columns(r=getattr(col("v"), func)().over("k")).collect()

    sql = {"mean": "avg(v)", "n_unique": "count(distinct v)"}.get(func, f"{func}(v)")
    duck.register("t", tbl)
    assert_same(got, duck.sql(f"select v, k, o, {sql} over (partition by k) as r from t"))


@pytest.mark.differential
@pytest.mark.parametrize("name", sorted(_COLUMNS))
@pytest.mark.parametrize("func", ["min", "max", "count", "n_unique"])
def test_window_admits_exactly_what_group_by_admits(name, func):
    """The parity itself: same aggregate, same groups, therefore the same answer.

    `min`/`max`/`count`/`n_unique` are the aggregates defined for every type here; `sum` and
    `mean` are numeric-only, so they are covered by the all-null case above instead.
    """
    tbl = _table(name)
    windowed = bt.from_arrow(tbl).with_columns(r=getattr(col("v"), func)().over("k")).to_pydict()
    grouped = bt.from_arrow(tbl).group_by("k").agg(r=getattr(col("v"), func)()).to_pydict()

    # Broadcast the group-by answer back over the rows and compare per partition.
    by_key = dict(zip(grouped["k"], grouped["r"], strict=True))
    assert [by_key[k] for k in windowed["k"]] == windowed["r"]


@pytest.mark.differential
@pytest.mark.parametrize("func", ["count", "n_unique", "min", "max"])
def test_running_frame_over_an_all_null_column_matches_duckdb(duck, func):
    """The `ORDER BY` (running) frame is a different kernel and diverged separately.

    `COUNT(v) OVER (PARTITION BY k ORDER BY o)` counted 1, 2, 3, ... over an all-null column
    where the answer is 0 at every row.
    """
    tbl = _table("null")
    got = (
        bt.from_arrow(tbl)
        .with_columns(r=getattr(col("v"), func)().over("k", order_by="o"))
        .collect()
    )

    sql = {"n_unique": "count(distinct v)"}.get(func, f"{func}(v)")
    duck.register("t", tbl)
    assert_same(
        got,
        duck.sql(f"select v, k, o, {sql} over (partition by k order by o) as r from t"),
    )


@pytest.mark.differential
@pytest.mark.parametrize("name", sorted(_COLUMNS))
def test_running_count_distinct_matches_duckdb(duck, name):
    """`COUNT(DISTINCT v)` under a running frame, for every type family."""
    tbl = _table(name)
    got = bt.from_arrow(tbl).with_columns(n=col("v").n_unique().over("k", order_by="o")).collect()
    duck.register("t", tbl)
    assert_same(
        got,
        duck.sql("select v, k, o, count(distinct v) over (partition by k order by o) as n from t"),
    )
