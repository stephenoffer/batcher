"""The Python constructors lifted out of the SQL front end, against DuckDB where it has them.

`bt.pi`, `.list.append`, `.list.prepend` and `.list.has_any` have a DuckDB function, and so do
the two existing names Spark calls are mapped to: `bt.partition_days` for `unix_date` and `%`
for `try_mod`. DuckDB is the oracle for all of them. The Spark-only names (`pmod`, `elt`,
`find_in_set`, `parse_url`, `months_between`, ...) have no DuckDB counterpart and are pinned
to Spark's documented examples in `tests/unit/test_sql_kernel_constructors.py` instead.

Each fixture carries nulls, an empty list or string, and a row id, so a row that goes
missing or a null that turns into a value is visible to the order-independent comparison.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

col = bt.col


@pytest.fixture
def lists(duck):
    tbl = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "a": pa.array([[1, 2], [], None, [1, None], [None], [3]], pa.list_(pa.int64())),
            "b": pa.array([[2, 5], [1], [1], [2, None], [None], None], pa.list_(pa.int64())),
            "v": pa.array([3, None, 7, 1, 0, -1], pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_list_append_matches_list_append(duck, lists):
    out = bt.from_arrow(lists).select("id", r=col("a").list.append(col("v"))).collect()
    assert_same(out, duck.sql("SELECT id, list_append(a, v) AS r FROM t"))


def test_list_append_of_a_literal_matches(duck, lists):
    out = bt.from_arrow(lists).select("id", r=col("a").list.append(9)).collect()
    assert_same(out, duck.sql("SELECT id, list_append(a, 9) AS r FROM t"))


def test_list_prepend_matches_duckdbs_element_first_order(duck, lists):
    out = bt.from_arrow(lists).select("id", r=col("a").list.prepend(col("v"))).collect()
    assert_same(out, duck.sql("SELECT id, list_prepend(v, a) AS r FROM t"))


def test_list_has_any_matches_including_null_elements_on_both_sides(duck, lists):
    # Row 4 holds a null on both sides and row 5 is nothing but nulls: DuckDB never pairs
    # two nulls, and `intersect` does, which is what `has_any` must not inherit.
    out = bt.from_arrow(lists).select("id", r=col("a").list.has_any(col("b"))).collect()
    assert_same(out, duck.sql("SELECT id, list_has_any(a, b) AS r FROM t"))


def test_the_spark_overlap_differs_from_duckdb_only_on_null_elements(duck, lists):
    # The positive control for the parameter: the two readings disagree on exactly the
    # rows where no element is shared and a null is present, and agree everywhere else.
    ds = bt.from_arrow(lists).select(
        "id",
        duck_rule=col("a").list.has_any(col("b")),
        spark_rule=col("a").list.has_any(col("b"), propagate_nulls=True),
    )
    rows = ds.sort("id").to_pydict()
    assert rows["duck_rule"] == [True, False, None, False, False, None]
    assert rows["spark_rule"] == [True, False, None, None, None, None]


def test_pi_matches(duck, lists):
    out = bt.from_arrow(lists).select("id", r=bt.pi() * col("v")).collect()
    assert_same(out, duck.sql("SELECT id, pi() * v AS r FROM t"))


def test_percent_is_the_zero_safe_remainder_try_mod_maps_to(duck):
    tbl = pa.table({"a": [-7, 7, 7, None, 0], "b": [2, -2, 0, 3, 5]})
    duck.register("m", tbl)
    out = bt.from_arrow(tbl).select("a", "b", r=col("a") % col("b")).collect()
    assert_same(out, duck.sql("SELECT a, b, a % b AS r FROM m"))


def test_partition_days_is_sparks_unix_date_and_matches_duckdb(duck):
    tbl = pa.table(
        {
            "d": pa.array(
                [dt.date(1970, 1, 2), dt.date(2022, 1, 2), dt.date(1969, 12, 31), None],
                pa.date32(),
            ),
            "ts": pa.array(
                [
                    dt.datetime(1970, 1, 1, 23, 59),
                    dt.datetime(1969, 12, 31, 0, 0, 1),
                    dt.datetime(2000, 2, 29, 12),
                    None,
                ],
                pa.timestamp("us"),
            ),
        }
    )
    duck.register("u", tbl)
    out = bt.from_arrow(tbl).select(
        "d", "ts", dd=bt.partition_days("d"), td=bt.partition_days("ts")
    )
    want = duck.sql(
        "SELECT d, ts, date_diff('day', DATE '1970-01-01', d) AS dd, "
        "date_diff('day', DATE '1970-01-01', CAST(ts AS DATE)) AS td FROM u"
    )
    assert_same(out.collect(), want)


def test_empty_input_keeps_the_declared_types(duck, lists):
    empty = bt.from_arrow(lists).filter(col("id") < 0)
    out = empty.select(
        r=col("a").list.append(col("v")), h=col("a").list.has_any(col("b"))
    ).collect()
    assert out.num_rows == 0
    assert out.schema.field("r").type == pa.list_(pa.int64())
    assert out.schema.field("h").type == pa.bool_()
