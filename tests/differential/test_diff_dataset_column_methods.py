"""Dataset column and scalar-terminal options vs DuckDB.

The filter mapping form against ``WHERE k1 = v1 AND k2 = v2``; ``isna``/``notna`` with
``nan=True`` against ``x IS NULL OR isnan(x)``; the forwarded scalar-terminal options against
the DuckDB function that spells the same meaning (``stddev_pop`` for ``std(ddof=0)``,
``quantile_disc`` for ``quantile(interpolation="equiprobable")``, ``coalesce(sum(x), 0)`` for
``sum(empty_value=0)``); and per-column standardization through one bound selector against
``(x - avg(x) OVER ()) / stddev(x) OVER ()``. Every input carries nulls, duplicates, a one-row
and an empty shape, and NaN where the column is a float.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_NAN = float("nan")

_ROWS = {
    "batch_size": [5, 5, 6, None, 5, 5],
    "num_workers": [2, 3, 2, 2, None, 2],
    "x": [1.0, _NAN, None, 4.0, 4.0, -0.0],
    "s": ["a", None, "b", "a", "a", "c"],
}


def _shapes() -> dict[str, pa.Table]:
    full = pa.table(_ROWS)
    return {"full": full, "one": full.slice(0, 1), "empty": full.slice(0, 0)}


@pytest.fixture(params=["full", "one", "empty"])
def table(request) -> pa.Table:
    return _shapes()[request.param]


def test_filter_mapping_matches_where_equalities(duck, table: pa.Table) -> None:
    duck.register("t", table)
    got = bt.from_arrow(table).filter({"batch_size": 5, "num_workers": 2}).collect()
    assert_same(got, duck.sql("SELECT * FROM t WHERE batch_size = 5 AND num_workers = 2"))


def test_filter_mapping_on_a_string_column_matches(duck, table: pa.Table) -> None:
    duck.register("t", table)
    got = bt.from_arrow(table).filter({"s": "a"}, bt.col("batch_size") == 5).collect()
    assert_same(got, duck.sql("SELECT * FROM t WHERE s = 'a' AND batch_size = 5"))


def test_isna_nan_matches_is_null_or_isnan(duck, table: pa.Table) -> None:
    duck.register("t", table)
    got = bt.from_arrow(table).isna(nan=True).collect()
    want = duck.sql(
        "SELECT batch_size IS NULL AS batch_size, num_workers IS NULL AS num_workers, "
        "(x IS NULL OR isnan(x)) AS x, s IS NULL AS s FROM t"
    )
    assert_same(got, want)


def test_notna_nan_matches_the_complement(duck, table: pa.Table) -> None:
    duck.register("t", table)
    got = bt.from_arrow(table).notna(nan=True).collect()
    want = duck.sql(
        "SELECT batch_size IS NOT NULL AS batch_size, num_workers IS NOT NULL AS num_workers, "
        "NOT (x IS NULL OR isnan(x)) AS x, s IS NOT NULL AS s FROM t"
    )
    assert_same(got, want)


def test_isna_default_is_null_only(duck, table: pa.Table) -> None:
    duck.register("t", table)
    got = bt.from_arrow(table).select("x").isna().collect()
    assert_same(got, duck.sql("SELECT x IS NULL AS x FROM t"))


def test_isna_nan_on_an_all_null_float_column(duck) -> None:
    table = pa.table({"x": pa.array([None, None], pa.float64())})
    duck.register("t", table)
    got = bt.from_arrow(table).isna(nan=True).collect()
    assert_same(got, duck.sql("SELECT (x IS NULL OR isnan(x)) AS x FROM t"))


def _scalar(duck, sql: str):
    return duck.sql(sql).fetchone()[0]


def _close(got, want) -> None:
    if want is None:
        assert got is None
    elif isinstance(want, float) and math.isnan(want):
        assert isinstance(got, float) and math.isnan(got)
    else:
        # DuckDB computes the moment forms in a less exact order (kurtosis_pop is off at 1e-11).
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9)


_TERMINALS = [
    (lambda ds: ds.std("batch_size", ddof=0), "stddev_pop(batch_size)"),
    (lambda ds: ds.var("batch_size", ddof=0), "var_pop(batch_size)"),
    (lambda ds: ds.std("batch_size"), "stddev_samp(batch_size)"),
    (
        lambda ds: ds.quantile("batch_size", 0.4, interpolation="equiprobable"),
        "quantile_disc(batch_size, 0.4)",
    ),
    (lambda ds: ds.quantile("batch_size", 0.4), "quantile_cont(batch_size, 0.4)"),
    (lambda ds: ds.sum("batch_size", empty_value=0), "coalesce(sum(batch_size), 0)"),
    (lambda ds: ds.sum("batch_size"), "sum(batch_size)"),
    (lambda ds: ds.product("num_workers", empty_value=1.0), "coalesce(product(num_workers), 1)"),
    (
        lambda ds: ds.count_distinct("batch_size", count_nulls=True),
        "count(DISTINCT batch_size) + (count(*) > count(batch_size))::INT",
    ),
    (lambda ds: ds.max("x"), "max(x)"),
    (
        lambda ds: ds.max("x", nan_policy="ignore"),
        "coalesce(max(x) FILTER (WHERE NOT isnan(x)), max(x))",
    ),
    (lambda ds: ds.kurtosis("batch_size", bias=True), "kurtosis_pop(batch_size)"),
]


@pytest.mark.parametrize(("terminal", "sql"), _TERMINALS, ids=[s for _, s in _TERMINALS])
def test_scalar_terminal_options_match_duckdb(duck, table: pa.Table, terminal, sql: str) -> None:
    duck.register("t", table)
    _close(terminal(bt.from_arrow(table)), _scalar(duck, f"SELECT {sql} FROM t"))


def test_population_skew_matches_the_moment_formula(duck) -> None:
    table = pa.table({"v": [1.0, 2.0, 2.0, 3.0, 10.0, None]})
    duck.register("t", table)
    want = _scalar(
        duck,
        "SELECT avg(power(v - m, 3)) / power(avg(power(v - m, 2)), 1.5) "
        "FROM t, (SELECT avg(v) AS m FROM t)",
    )
    _close(bt.from_arrow(table).skew("v", bias=True), want)


def test_standardize_every_numeric_column_with_one_bound_selector(duck) -> None:
    table = pa.table(
        {"a": [1.0, 2.0, 2.0, None, 7.0], "b": [3, 3, 9, 1, None], "s": ["x", "y", "z", "w", "v"]}
    )
    duck.register("t", table)
    n = bt.numeric()
    got = bt.from_arrow(table).with_columns((n - n.mean()) / n.std()).collect()
    want = duck.sql(
        "SELECT (a - avg(a) OVER ()) / stddev(a) OVER () AS a, "
        "(b - avg(b) OVER ()) / stddev(b) OVER () AS b, s FROM t"
    )
    assert_same(got, want)
