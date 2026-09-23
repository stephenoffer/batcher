"""Edge cases of the data-inspection surface, each checked against DuckDB where one applies.

`test_diff_describe.py` covers the happy path of ``describe``/``corr_matrix``/``cov_matrix``.
This file holds the shapes that audit found wrong: a user column whose name collides with a
label column, decimal columns that were silently skipped, a correlation diagonal that missed
``1.0`` by one ulp, ``approx_quantile`` answering a number for a timestamp, and the small
printing terminals (``show``/``glimpse``/``info``). Every cell that has a SQL counterpart is
compared with the matching DuckDB scalar aggregate rather than a hand-written constant.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential


def _stat(d: dict, label: str, column: str):
    return d[column][d["statistic"].index(label)]


def _duck_row(duck, sql: str) -> dict:
    return duck.sql(sql).to_arrow_table().to_pylist()[0]


def _undefined(value: object) -> bool:
    """A statistic with no value: SQL NULL, or DuckDB's NaN for a zero-variance correlation.

    DuckDB answers ``corr`` over a constant column (or a single row) with NaN; Batcher
    answers null, which is what `corr_matrix` documents. Both mean "no correlation exists",
    so the comparison accepts either spelling on the DuckDB side and requires null on ours.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def _decimal_table() -> pa.Table:
    return pa.table(
        {
            "d": pa.array(
                [Decimal("1.50"), Decimal("2.25"), None, Decimal("7.00"), Decimal("-3.10")],
                type=pa.decimal128(10, 2),
            ),
            "x": pa.array([1.0, 2.0, 3.0, 5.0, 4.0], type=pa.float64()),
        }
    )


# ---------------------------------------------------------------- label-column collisions


def test_describe_rejects_a_column_named_statistic():
    ds = bt.from_pydict({"statistic": [1, 2], "y": [3, 4]})
    with pytest.raises(PlanError, match="statistic"):
        ds.describe()
    # The actionable fix the message names works.
    d = ds.rename({"statistic": "statistic_"}).describe().to_pydict()
    assert d["statistic"][0] == "count"
    assert _stat(d, "max", "statistic_") == 2.0


@pytest.mark.parametrize("method", ["corr_matrix", "cov_matrix"])
def test_pairwise_matrix_rejects_a_numeric_column_named_column(method):
    ds = bt.from_pydict({"column": [1.0, 2.0, 3.0], "y": [3.0, 4.0, 1.0]})
    with pytest.raises(PlanError, match="column"):
        getattr(ds, method)()
    with pytest.raises(PlanError, match="column"):
        getattr(ds, method)(["column", "y"])


def test_pairwise_matrix_ignores_a_non_numeric_column_named_column():
    # A skipped column never reaches the output, so it cannot collide with the label.
    ds = bt.from_pydict({"column": ["p", "q", "r"], "a": [1.0, 2.0, 4.0], "b": [2.0, 1.0, 0.0]})
    assert ds.corr_matrix().to_pydict()["column"] == ["a", "b"]


@pytest.mark.parametrize("method", ["corr_matrix", "cov_matrix"])
def test_pairwise_matrix_rejects_duplicate_column_requests(method):
    ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    with pytest.raises(PlanError, match="more than once"):
        getattr(ds, method)(["a", "a"])


# ------------------------------------------------------------------------- decimal columns


def test_describe_includes_decimal_columns(duck):
    t = _decimal_table()
    duck.register("t", t)
    d = bt.from_arrow(t).describe().to_pydict()
    # DuckDB's `quantile_cont` over a DECIMAL keeps the input scale (1.875 -> 1.87), while
    # `describe` reports Float64; interpolating over the DOUBLE cast is the like-for-like.
    exp = _duck_row(
        duck,
        "SELECT count(d)::DOUBLE cnt, avg(d)::DOUBLE mean, stddev_samp(d)::DOUBLE std, "
        "min(d)::DOUBLE lo, max(d)::DOUBLE hi, quantile_cont(d::DOUBLE, 0.25) q25, "
        "quantile_cont(d::DOUBLE, 0.5) q50, quantile_cont(d::DOUBLE, 0.75) q75 FROM t",
    )
    for label, key in [
        ("count", "cnt"),
        ("mean", "mean"),
        ("std", "std"),
        ("min", "lo"),
        ("25%", "q25"),
        ("50%", "q50"),
        ("75%", "q75"),
        ("max", "hi"),
    ]:
        assert _stat(d, label, "d") == pytest.approx(exp[key]), label
    assert _stat(d, "null_count", "d") == 1.0


@pytest.mark.parametrize(("method", "fn"), [("corr_matrix", "corr"), ("cov_matrix", "covar_samp")])
def test_pairwise_matrix_includes_decimal_columns(duck, method, fn):
    t = _decimal_table()
    duck.register("t", t)
    got = getattr(bt.from_arrow(t), method)().to_pydict()
    assert got["column"] == ["d", "x"]
    for i, a in enumerate(got["column"]):
        for b in got["column"]:
            expected = duck.sql(f"SELECT {fn}({a}::DOUBLE, {b}::DOUBLE) FROM t").fetchone()[0]
            assert got[b][i] == pytest.approx(expected), (a, b)


# ------------------------------------------------------------------- correlation diagonal


def test_corr_matrix_diagonal_is_exactly_one():
    # Without the clamp this data's diagonal came back 0.9999999999999998.
    got = bt.from_pydict({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 5.0, 3.0]}).corr_matrix()
    d = got.to_pydict()
    assert d["a"][0] == 1.0
    assert d["b"][1] == 1.0


def test_corr_matrix_constant_column_diagonal_stays_null(duck):
    t = pa.table({"a": [1.0, 2.0, 3.0], "k": [5.0, 5.0, 5.0]})
    duck.register("t", t)
    d = bt.from_arrow(t).corr_matrix().to_pydict()
    assert _undefined(duck.sql("SELECT corr(k, k) FROM t").fetchone()[0])
    assert d["k"][1] is None
    assert d["a"][0] == 1.0


# ------------------------------------------------------------- empty / one-row / all-null


def _edge_tables() -> dict[str, pa.Table]:
    base = pa.table({"a": [1.0, 2.0, 4.0], "b": pa.array([3, 1, 2], type=pa.int64())})
    return {
        "one_row": base.slice(0, 1),
        "all_null": pa.table(
            {
                "a": pa.array([None, None, None], type=pa.float64()),
                "b": pa.array([3, 1, 2], type=pa.int64()),
            }
        ),
    }


@pytest.mark.parametrize("shape", ["empty", "one_row", "all_null"])
def test_describe_edge_shapes_match_duckdb(duck, shape):
    if shape == "empty":
        t = pa.table({"a": [1.0, 2.0], "b": pa.array([3, 1], type=pa.int64())})
        ds = bt.from_arrow(t).filter(col("b") > 100)
        duck.register("src", t)
        duck.sql("CREATE VIEW t AS SELECT * FROM src WHERE b > 100")
    else:
        t = _edge_tables()[shape]
        ds = bt.from_arrow(t)
        duck.register("t", t)
    d = ds.describe().to_pydict()
    for c in ("a", "b"):
        exp = _duck_row(
            duck,
            f"SELECT count({c})::DOUBLE cnt, (count(*) - count({c}))::DOUBLE nn, "
            f"avg({c})::DOUBLE mean, stddev_samp({c})::DOUBLE std, min({c})::DOUBLE lo, "
            f"max({c})::DOUBLE hi, quantile_cont({c}, 0.5)::DOUBLE q50 FROM t",
        )
        for label, key in [
            ("count", "cnt"),
            ("null_count", "nn"),
            ("mean", "mean"),
            ("std", "std"),
            ("min", "lo"),
            ("50%", "q50"),
            ("max", "hi"),
        ]:
            got = _stat(d, label, c)
            if exp[key] is None:
                assert got is None, (c, label)
            else:
                assert got == pytest.approx(exp[key]), (c, label)


@pytest.mark.parametrize(("method", "fn"), [("corr_matrix", "corr"), ("cov_matrix", "covar_samp")])
@pytest.mark.parametrize("shape", ["one_row", "all_null"])
def test_pairwise_matrix_edge_shapes_match_duckdb(duck, method, fn, shape):
    t = _edge_tables()[shape]
    duck.register("t", t)
    got = getattr(bt.from_arrow(t), method)().to_pydict()
    assert got["column"] == ["a", "b"]
    for i, a in enumerate(got["column"]):
        for b in got["column"]:
            expected = duck.sql(f"SELECT {fn}({a}, {b}) FROM t").fetchone()[0]
            if _undefined(expected):
                assert got[b][i] is None, (a, b)
            else:
                assert got[b][i] == pytest.approx(expected), (a, b)


# ---------------------------------------------------------------------------- null_count


def test_null_count_counts_nulls_not_nan(duck):
    t = pa.table({"f": pa.array([1.0, float("nan"), None, None], type=pa.float64())})
    duck.register("t", t)
    got = bt.from_arrow(t).null_count().to_pydict()
    assert got == {"f": [duck.sql("SELECT count(*) - count(f) FROM t").fetchone()[0]]}
    assert got == {"f": [2]}
    # The docstring used to promise pandas `isnull().sum()` parity, which counts NaN too.
    assert "NaN" in (bt.Dataset.null_count.__doc__ or "")


# ------------------------------------------------------------------------ approx_quantile


def _mixed_types() -> bt.Dataset:
    return bt.from_arrow(
        pa.table(
            {
                "ts": pa.array([dt.datetime(2020, 1, 1), dt.datetime(2021, 1, 1)]),
                "day": pa.array([dt.date(2020, 1, 1), dt.date(2021, 1, 1)]),
                "flag": [True, False],
                "s": ["x", "y"],
                "d": pa.array([Decimal("1.50"), Decimal("2.50")], type=pa.decimal128(10, 2)),
                "i": [10, 20],
            }
        )
    )


@pytest.mark.parametrize("column", ["ts", "day", "flag", "s"])
def test_approx_quantile_is_none_for_non_numeric(column):
    ds = _mixed_types()
    assert ds.approx_quantile(column, 0.5) is None
    assert ds.approx_median(column) is None
    assert ds.approx_percentile(column, 90) is None


@pytest.mark.parametrize("column", ["d", "i"])
def test_approx_quantile_answers_numeric_and_decimal(column):
    ds = _mixed_types()
    lo, hi = ds.min(column), ds.max(column)
    got = ds.approx_median(column)
    assert got is not None
    assert float(lo) <= got <= float(hi)


def test_approx_quantile_still_validates_the_column():
    with pytest.raises(PlanError):
        _mixed_types().approx_quantile("nope", 0.5)


# ------------------------------------------------------------------- printing terminals


def test_show_rejects_a_negative_limit():
    with pytest.raises(PlanError, match="limit"):
        bt.from_pydict({"x": [1, 2]}).show(-1)


def test_glimpse_inflects_the_column_count(capsys):
    bt.from_pydict({"x": [1, 2]}).glimpse()
    assert capsys.readouterr().out.splitlines()[0] == "Dataset: 1 column"
    bt.from_pydict({"x": [1, 2], "y": ["a", "b"]}).glimpse()
    assert capsys.readouterr().out.splitlines()[0] == "Dataset: 2 columns"


def test_info_reports_rows_types_and_non_null_counts(duck, capsys):
    t = pa.table({"x": pa.array([1, None, 3], type=pa.int64()), "s": ["a", "b", None]})
    duck.register("t", t)
    bt.from_arrow(t).info()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Dataset: 3 rows x 2 columns"
    body = {line.split()[0]: line.split() for line in lines[3:5]}
    exp = _duck_row(duck, "SELECT count(x) x, count(s) s FROM t")
    assert body["x"][1:] == ["int64", str(exp["x"])]
    assert body["s"][1:] == ["string", str(exp["s"])]
    assert lines[-1].startswith("estimated size: ")

    bt.from_pydict({"x": [1]}).info()
    assert capsys.readouterr().out.splitlines()[0] == "Dataset: 1 row x 1 column"


# ------------------------------------------------------------------- query inspection


def test_stats_measures_a_map_batches_pipeline():
    ds = bt.from_pydict({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    stats = ds.map_batches(lambda b: b).group_by("k").agg(s=col("v").sum()).stats()
    kinds = {op.kind.lower(): op for op in stats.ops}
    assert "mapbatches" in kinds
    assert kinds["mapbatches"].rows_out == 3
    assert stats.rows_out == 2
    # The docstring used to say this shape raised `BackendError`.
    assert "BackendError" not in (bt.Dataset.stats.__doc__ or "")


def test_explain_json_is_a_json_string():
    out = bt.from_pydict({"x": [1, 2, 3]}).filter(col("x") > 1).explain(format="json")
    assert isinstance(out, str)
    assert isinstance(json.loads(out), dict)
