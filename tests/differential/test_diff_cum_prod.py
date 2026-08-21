"""`cum_prod` against DuckDB's `product() OVER (... ROWS UNBOUNDED PRECEDING)`.

`cum_prod` completes the running family beside `cum_sum`/`cum_min`/`cum_max`/`cum_count`, and
it is the member with the two behaviours worth pinning against an oracle rather than against a
hand-written list: it returns `Float64` for an integer input (a running product overflows an
`Int64` far sooner than a running sum, and wrapping is the wrong answer for a compounding
factor), and it *skips* nulls rather than propagating them.

The cases below are the ones that make a running aggregate fragile: nulls inside a partition,
an all-null partition, a single row, an empty relation, negative and zero factors (zero is the
absorbing element, so a later value cannot revive the product), and a partitioned + ordered
form where the emitted order differs from the row order.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

_ORDERED = (
    "product(v) OVER (PARTITION BY g ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
)
_ROWS = "product(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"


def _tbl(rows):
    return pa.table(
        {
            "g": pa.array([r[0] for r in rows], pa.string()),
            "o": pa.array([r[1] for r in rows], pa.int64()),
            "v": pa.array([r[2] for r in rows], pa.float64()),
        }
    )


@pytest.mark.differential
@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([("a", 1, 2.0), ("a", 2, 3.0), ("b", 1, 4.0)], id="plain"),
        pytest.param([("a", 1, 2.0), ("a", 2, None), ("a", 3, 3.0)], id="null-inside"),
        pytest.param([("a", 1, None), ("a", 2, None)], id="all-null-partition"),
        pytest.param([("a", 1, 5.0)], id="single-row"),
        pytest.param([("a", 1, -2.0), ("a", 2, -3.0), ("a", 3, 2.0)], id="negatives"),
        pytest.param([("a", 1, 0.0), ("a", 2, 7.0)], id="zero-absorbs"),
        pytest.param([("a", 2, 3.0), ("a", 1, 2.0), ("b", 9, 1.5)], id="order-differs-from-rows"),
    ],
)
def test_cum_prod_partitioned_matches_duckdb(rows, duck):
    """The partitioned + ordered running product must equal DuckDB's window form."""
    table = _tbl(rows)
    duck.register("t", table)
    got = (
        bt.from_arrow(table)
        .with_columns(cp=bt.col("v").cum_prod(partition_by="g", order_by="o"))
        .collect()
    )
    assert_same(got, duck.sql(f"SELECT g, o, v, {_ORDERED} AS cp FROM t"))


@pytest.mark.differential
def test_cum_prod_over_row_order_matches_duckdb(duck):
    """With no partition or order key the accumulation follows row order, as DuckDB's does."""
    table = _tbl([("a", 1, 2.0), ("b", 2, 3.0), ("a", 3, 4.0), ("b", 4, 0.5)])
    duck.register("t", table)
    got = bt.from_arrow(table).with_columns(cp=bt.col("v").cum_prod()).collect()
    assert_same(got, duck.sql(f"SELECT g, o, v, {_ROWS} AS cp FROM t"))


@pytest.mark.differential
def test_cum_prod_widens_an_integer_input(duck):
    """An `Int64` input yields `Float64`, so a long run cannot silently wrap."""
    table = pa.table({"g": pa.array(["a"] * 6), "o": pa.array(range(6)), "v": pa.array([7] * 6)})
    duck.register("t", table)
    got = bt.from_arrow(table).with_columns(cp=bt.col("v").cum_prod(order_by="o")).collect()
    assert got.schema.field("cp").type == pa.float64()
    assert_same(
        got,
        duck.sql(
            "SELECT g, o, v, product(v) OVER (ORDER BY o ROWS BETWEEN UNBOUNDED PRECEDING "
            "AND CURRENT ROW) AS cp FROM t"
        ),
    )


@pytest.mark.differential
def test_cumprod_alias_is_the_same_expression(duck):
    """The pandas spelling must lower to the identical plan, not a second implementation."""
    table = _tbl([("a", 1, 2.0), ("a", 2, 3.0)])
    ds = bt.from_arrow(table)
    assert (
        ds.with_columns(cp=bt.col("v").cumprod()).explain()
        == ds.with_columns(cp=bt.col("v").cum_prod()).explain()
    )


@pytest.mark.differential
def test_cum_prod_on_an_empty_relation(duck):
    """An empty input produces an empty result rather than raising."""
    table = _tbl([]).slice(0, 0)
    got = bt.from_arrow(table).with_columns(cp=bt.col("v").cum_prod()).collect()
    assert got.num_rows == 0
