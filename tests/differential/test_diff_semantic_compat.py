"""The semantic-compatibility table in the migration guide, executed against every engine it names.

`docs/getting-started/migration/differences.md` tabulates how Batcher answers the questions a
port most often gets wrong without an error: whether NaN counts as null, what `count(col)`
and an all-null `sum` return, whether an integer column survives a null, division by zero,
where nulls sort, and whether a null group key is kept. Each row there makes a claim about
Batcher, pandas, Polars and SQL (DuckDB), and each claim is one assertion here, run against the
installed library rather than remembered. If a library changes a default, this fails and the
table has to change with it.

Spark is not a column because the suite has no JVM to run it on.
"""

from __future__ import annotations

import math

import duckdb
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

NAN = float("nan")


def _same(got: list, want: list) -> bool:
    """List equality where NaN equals NaN, so a NaN claim can be asserted exactly."""
    return len(got) == len(want) and all(
        (isinstance(g, float) and isinstance(w, float) and math.isnan(g) and math.isnan(w))
        or g == w
        for g, w in zip(got, want, strict=True)
    )


@pytest.fixture
def sql() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


_FLOATS = [1.0, None, NAN]


def test_nan_is_not_null(sql) -> None:
    """Batcher, Polars and SQL keep NaN distinct from null; pandas treats NaN as missing."""
    ds = bt.from_pydict({"x": _FLOATS})
    got = ds.select(n=bt.col("x").is_null(), m=bt.col("x").is_nan()).to_pydict()
    assert got == {"n": [False, True, False], "m": [False, None, True]}
    assert sql.sql("SELECT isnan(NULL::DOUBLE)").fetchone() == (None,)  # null, as above
    assert pl.DataFrame({"x": _FLOATS})["x"].is_null().to_list() == [False, True, False]
    assert sql.sql("SELECT 'nan'::DOUBLE IS NULL").fetchone() == (False,)
    assert pd.Series(_FLOATS).isna().tolist() == [False, True, True]


def test_fill_null_leaves_nan() -> None:
    """Filling nulls leaves NaN in place everywhere except pandas, whose fillna replaces it."""
    got = bt.from_pydict({"x": _FLOATS}).fill_null(0.0).to_pydict()["x"]
    assert _same(got, [1.0, 0.0, NAN])
    assert _same(pl.DataFrame({"x": _FLOATS}).fill_null(0.0)["x"].to_list(), [1.0, 0.0, NAN])
    assert pd.Series(_FLOATS).fillna(0.0).tolist() == [1.0, 0.0, 0.0]


def test_count_skips_nulls_but_counts_nan(sql) -> None:
    """`count(col)` skips nulls and counts NaN; pandas skips both. The row count keeps all."""
    ds = bt.from_pydict({"x": _FLOATS})
    assert ds.agg(c=bt.col("x").count(), n=bt.count()).to_pydict() == {"c": [2], "n": [3]}
    assert ds.count() == 3
    assert pl.DataFrame({"x": _FLOATS}).select(pl.col("x").count()).item() == 2
    t = pa.table({"x": pa.array(_FLOATS, pa.float64())})  # noqa: F841 - read by DuckDB
    assert sql.sql("SELECT count(x), count(*) FROM t").fetchone() == (2, 3)
    assert pd.Series(_FLOATS).count() == 1


def test_nan_propagates_through_mean_and_sum(sql) -> None:
    """A NaN makes `mean`/`sum` NaN (nulls are skipped); pandas skips NaN as missing."""
    got = bt.from_pydict({"x": _FLOATS}).agg(m=bt.col("x").mean(), s=bt.col("x").sum())
    assert all(math.isnan(v[0]) for v in got.to_pydict().values())
    out = pl.DataFrame({"x": _FLOATS}).select(m=pl.col("x").mean(), s=pl.col("x").sum())
    assert all(math.isnan(v[0]) for v in out.to_dict(as_series=False).values())
    t = pa.table({"x": pa.array(_FLOATS, pa.float64())})  # noqa: F841 - read by DuckDB
    assert all(math.isnan(v) for v in sql.sql("SELECT avg(x), sum(x) FROM t").fetchone())
    assert (pd.Series(_FLOATS).mean(), pd.Series(_FLOATS).sum()) == (1.0, 1.0)


def test_sum_of_nothing_is_null(sql) -> None:
    """An all-null or empty `sum` is null in Batcher and SQL, and 0 in pandas and Polars."""
    nulls = pa.table({"x": pa.array([None, None], pa.int64())})
    assert bt.from_arrow(nulls).agg(s=bt.col("x").sum()).to_pydict() == {"s": [None]}
    empty = bt.from_pydict({"x": [1]}).filter(bt.col("x") > 5)
    assert empty.agg(s=bt.col("x").sum()).to_pydict() == {"s": [None]}
    assert sql.sql("SELECT sum(x) FROM nulls").fetchone() == (None,)
    assert pl.from_arrow(nulls).select(pl.col("x").sum()).item() == 0
    assert pd.Series([None, None], dtype="float64").sum() == 0


def test_integer_column_survives_a_null(sql) -> None:
    """A null keeps an integer column integer; pandas turns it into float64."""
    assert bt.from_pydict({"x": [1, None]}).schema.field("x").type == pa.int64()
    rows = sql.sql("SELECT DISTINCT typeof(x) FROM (VALUES (1), (NULL)) t(x)").fetchall()
    assert rows == [("INTEGER",)]
    assert pl.DataFrame({"x": [1, None]}).schema["x"] == pl.Int64
    assert pd.DataFrame({"x": [1, None]})["x"].dtype == "float64"


def test_division_by_zero(sql) -> None:
    """`/` gives inf and NaN; integer `//` by zero gives null, except in pandas (inf, NaN)."""
    ds = bt.from_pydict({"a": [1, 0], "b": [0, 0]})
    got = ds.select(q=bt.col("a") / bt.col("b"), f=bt.col("a") // bt.col("b")).to_pydict()
    assert _same(got["q"], [math.inf, NAN]) and got["f"] == [None, None]
    polars = pl.DataFrame({"a": [1, 0], "b": [0, 0]}).select(
        q=pl.col("a") / pl.col("b"), f=pl.col("a") // pl.col("b")
    )
    assert _same(polars["q"].to_list(), [math.inf, NAN]) and polars["f"].to_list() == [None, None]
    assert _same(list(sql.sql("SELECT 1 / 0, 0 / 0").fetchone()), [math.inf, NAN])
    assert sql.sql("SELECT 1 // 0").fetchone() == (None,)
    floor = (pd.Series([1, 0]) // pd.Series([0, 0])).tolist()
    assert _same(floor, [math.inf, NAN])


@pytest.mark.parametrize("descending", [False, True])
def test_nulls_sort_last(sql, descending: bool) -> None:
    """Nulls sort last both ways in Batcher, SQL and pandas; Polars puts them first."""
    values = [2, None, 1]
    ordered = [2, 1] if descending else [1, 2]
    got = bt.from_pydict({"x": values}).sort("x", descending=descending).to_pydict()["x"]
    assert got == [*ordered, None]
    direction = "DESC" if descending else "ASC"
    rows = sql.sql(f"SELECT x FROM (VALUES (2), (NULL), (1)) t(x) ORDER BY x {direction}")
    assert [r[0] for r in rows.fetchall()] == [*ordered, None]
    pandas = pd.Series(values).sort_values(ascending=not descending).tolist()
    assert _same(pandas, [*map(float, ordered), NAN])
    polars = pl.Series(values).sort(descending=descending).to_list()
    assert polars == [None, *ordered]


def test_null_group_key_is_kept(sql) -> None:
    """Rows with a null key form one group; pandas' groupby drops them by default."""
    data = {"k": ["a", None, None, "a"], "v": [1, 2, 3, 4]}
    got = bt.from_pydict(data).group_by("k").agg(s=bt.col("v").sum()).sort("k").to_pydict()
    assert got == {"k": ["a", None], "s": [5, 5]}
    polars = pl.DataFrame(data).group_by("k").agg(pl.col("v").sum()).sort("k", nulls_last=True)
    assert polars.to_dict(as_series=False) == {"k": ["a", None], "v": [5, 5]}
    t = pa.table(data)  # noqa: F841 - read by DuckDB
    rows = sql.sql("SELECT k, sum(v) FROM t GROUP BY k ORDER BY k NULLS LAST").fetchall()
    assert rows == [("a", 5), (None, 5)]
    assert pd.DataFrame(data).groupby("k")["v"].sum().to_dict() == {"a": 5}


def test_pandas_groupby_sorts_its_keys() -> None:
    """The contrast the page draws: pandas orders groups by key unless told not to."""
    frame = pd.DataFrame({"k": ["b", "a", "b"], "v": [1, 2, 3]})
    assert frame.groupby("k")["v"].sum().index.tolist() == ["a", "b"]


def test_count_distinct_ignores_null(sql) -> None:
    """`count_distinct` skips null, as SQL and pandas do; Polars' `n_unique` counts it."""
    values = [1, 1, None, 2]
    got = bt.from_pydict({"x": values}).agg(n=bt.col("x").count_distinct()).to_pydict()
    assert got == {"n": [2]}
    t = pa.table({"x": values})  # noqa: F841 - read by DuckDB
    assert sql.sql("SELECT count(DISTINCT x) FROM t").fetchone() == (2,)
    assert pd.Series(values).nunique() == 2
    assert pl.Series(values).n_unique() == 3
