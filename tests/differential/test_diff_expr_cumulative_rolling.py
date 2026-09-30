"""Cumulative, rolling, expanding and distinct-marker expressions against DuckDB.

Every expression here is a window over an explicit ``order_by``, so each comparison is
row-for-row in that order (`assert_same_ordered`) rather than as a multiset: a running
value in the wrong row is exactly the bug a multiset comparison would hide. Each case runs
over a partitioned fixture carrying nulls and ties, a single row, and an empty input.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential

_SCHEMA = pa.schema([("i", pa.int64()), ("g", pa.string()), ("x", pa.float64())])

_TABLES = {
    "partitioned_with_nulls": pa.table(
        {
            "i": list(range(10)),
            "g": ["a", "a", "b", "a", "b", "b", "a", "b", "a", "b"],
            "x": [3.0, None, 5.0, -1.0, 5.0, None, 2.5, 7.0, 3.0, -4.0],
        },
        schema=_SCHEMA,
    ),
    "one_row": pa.table({"i": [0], "g": ["a"], "x": [2.0]}, schema=_SCHEMA),
    "empty": _SCHEMA.empty_table(),
}

_W = "PARTITION BY g ORDER BY i"
_UNB = f"OVER ({_W} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
_R3 = f"OVER ({_W} ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)"

# (batcher expression factory, the DuckDB expression it must equal)
_CASES = {
    "cum_min": (lambda: bt.col("x").cum_min(partition_by=["g"], order_by=["i"]), f"min(x) {_UNB}"),
    "cum_max": (lambda: bt.col("x").cum_max(partition_by=["g"], order_by=["i"]), f"max(x) {_UNB}"),
    "cum_count": (
        lambda: bt.col("x").cum_count(partition_by=["g"], order_by=["i"]),
        f"count(x) {_UNB}",
    ),
    "rolling_min": (
        lambda: bt.col("x").rolling_min(3, partition_by=["g"], order_by=["i"]),
        f"min(x) {_R3}",
    ),
    "rolling_max": (
        lambda: bt.col("x").rolling_max(3, partition_by=["g"], order_by=["i"]),
        f"max(x) {_R3}",
    ),
    "rolling_count": (
        lambda: bt.col("x").rolling_count(3, partition_by=["g"], order_by=["i"]),
        f"count(x) {_R3}",
    ),
    "rolling_std_sample": (
        lambda: bt.col("x").rolling_std(3, partition_by=["g"], order_by=["i"]),
        f"stddev_samp(x) {_R3}",
    ),
    "rolling_std_population": (
        lambda: bt.col("x").rolling_std(3, ddof=0, partition_by=["g"], order_by=["i"]),
        f"stddev_pop(x) {_R3}",
    ),
    "rolling_max_min_periods": (
        lambda: bt.col("x").rolling_max(3, min_periods=2, partition_by=["g"], order_by=["i"]),
        f"CASE WHEN count(x) {_R3} >= 2 THEN max(x) {_R3} END",
    ),
    # A numeric `by` of 3 is RANGE BETWEEN 3 PRECEDING AND CURRENT ROW: both ends included.
    "rolling_max_by": (
        lambda: bt.col("x").rolling_max_by("i", 3, partition_by=["g"]),
        "max(x) OVER (PARTITION BY g ORDER BY i RANGE BETWEEN 3 PRECEDING AND CURRENT ROW)",
    ),
    "rolling_min_by": (
        lambda: bt.col("x").rolling_min_by("i", 3, partition_by=["g"]),
        "min(x) OVER (PARTITION BY g ORDER BY i RANGE BETWEEN 3 PRECEDING AND CURRENT ROW)",
    ),
    "is_duplicated": (
        lambda: bt.col("x").is_duplicated(),
        "count(*) OVER (PARTITION BY x) > 1",
    ),
    "is_first_distinct": (
        lambda: bt.col("x").is_first_distinct(order_by=bt.col("i")),
        "row_number() OVER (PARTITION BY x ORDER BY i) = 1",
    ),
    "is_last_distinct": (
        lambda: bt.col("x").is_last_distinct(order_by=bt.col("i")),
        "row_number() OVER (PARTITION BY x ORDER BY i DESC) = 1",
    ),
}


def _run(table: pa.Table, expr) -> pa.Table:
    return bt.from_arrow(table).with_columns(r=expr).sort("i").select("i", "r").to_arrow()


@pytest.mark.parametrize("table_name", list(_TABLES))
@pytest.mark.parametrize("case", list(_CASES))
def test_window_expression_matches_duckdb_row_for_row(duck, case, table_name):
    table = _TABLES[table_name]
    make, sql = _CASES[case]
    duck.register("t", table)
    got = _run(table, make())
    assert got.num_rows == table.num_rows
    assert_same_ordered(got, duck.sql(f"SELECT i, {sql} AS r FROM t ORDER BY i"))


def test_the_fixture_exercises_nulls_ties_and_partial_frames():
    """Positive control: the main fixture holds the inputs the cases above claim to cover."""
    rows = _TABLES["partitioned_with_nulls"].to_pydict()
    assert None in rows["x"]
    present = [v for v in rows["x"] if v is not None]
    assert len(present) > len(set(present)), "needs a duplicated value for the markers"
    assert len(set(rows["g"])) == 2


def _expanding_oracle(duck, fn: str) -> list:
    return [r[0] for r in duck.execute(f"SELECT {fn}(x) {_UNB} FROM t ORDER BY i").fetchall()]


@pytest.mark.parametrize(
    ("method", "ddof", "duck_fn"),
    [
        ("expanding_var", 1, "var_samp"),
        ("expanding_var", 0, "var_pop"),
        ("expanding_std", 1, "stddev_samp"),
        ("expanding_std", 0, "stddev_pop"),
    ],
)
def test_expanding_moments_match_duckdb(duck, method, ddof, duck_fn):
    """Equal to DuckDB's running moment, except where the docstring promises NaN.

    `expanding_var` documents that a frame holding a single value "is undefined and yields
    NaN" with the Bessel correction, where SQL answers NULL. That is the only row class
    allowed to differ, and it is asserted as NaN rather than skipped.
    """
    table = pa.table({"i": list(range(6)), "g": ["a"] * 6, "x": [2.0, 4.0, 4.0, 5.0, 7.0, 9.0]})
    duck.register("t", table)
    expr = getattr(bt.col("x"), method)(partition_by=["g"], order_by=["i"], ddof=ddof)
    got = _run(table, expr).column("r").to_pylist()
    want = _expanding_oracle(duck, duck_fn)
    assert len(got) == len(want) == 6
    for k, (b, d) in enumerate(zip(got, want, strict=True)):
        if d is None:
            assert ddof == 1 and k == 0, "only a one-value sample frame is undefined"
            assert b is not None and math.isnan(b), f"row {k}: documented NaN, got {b!r}"
        else:
            assert b == pytest.approx(d, rel=1e-9, abs=1e-12), f"row {k}: {b} vs {d}"
