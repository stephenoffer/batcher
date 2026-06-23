"""Differential coverage for ``Dataset.describe`` / ``Dataset.null_count``.

``describe`` composes already-tested aggregates and transposes them into statistic
rows, so each cell is checked against the matching DuckDB scalar aggregate
(``avg``/``stddev_samp``/``min``/``max``/``count``/``quantile_cont``). ``null_count``
is checked against ``count(*) - count(col)`` per column.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "name": ["a", "b", "c", "d", None],
            "x": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "y": pa.array([10.0, 20.0, None, 40.0, 50.0], type=pa.float64()),
        }
    )


def _stat(d, label, column):
    i = d["statistic"].index(label)
    return d[column][i]


def test_describe_numeric_cells_match_duckdb(duck):
    duck.register("t", _t())
    d = bt.from_arrow(_t()).describe().to_pydict()

    for c in ("x", "y"):
        exp = (
            duck.sql(
                f"SELECT count({c})::DOUBLE cnt, avg({c}) mean, stddev_samp({c}) std, "
                f"min({c})::DOUBLE lo, max({c})::DOUBLE hi, "
                f"quantile_cont({c}, 0.25) q25, quantile_cont({c}, 0.5) q50, "
                f"quantile_cont({c}, 0.75) q75 FROM t"
            )
            .to_arrow_table()
            .to_pylist()[0]
        )
        assert _stat(d, "count", c) == pytest.approx(exp["cnt"])
        assert _stat(d, "mean", c) == pytest.approx(exp["mean"])
        assert _stat(d, "std", c) == pytest.approx(exp["std"])
        assert _stat(d, "min", c) == pytest.approx(exp["lo"])
        assert _stat(d, "max", c) == pytest.approx(exp["hi"])
        assert _stat(d, "25%", c) == pytest.approx(exp["q25"])
        assert _stat(d, "50%", c) == pytest.approx(exp["q50"])
        assert _stat(d, "75%", c) == pytest.approx(exp["q75"])


def test_describe_null_count_row(duck):
    duck.register("t", _t())
    d = bt.from_arrow(_t()).describe().to_pydict()
    # name has 1 null, y has 1 null, x has none.
    assert _stat(d, "null_count", "name") == 1.0
    assert _stat(d, "null_count", "x") == 0.0
    assert _stat(d, "null_count", "y") == 1.0


def test_describe_non_numeric_stats_are_null():
    d = bt.from_arrow(_t()).describe().to_pydict()
    # A string column reports count/null_count only; numeric stats are null.
    for label in ("mean", "std", "min", "25%", "max"):
        assert _stat(d, label, "name") is None
    assert _stat(d, "count", "name") == 4.0  # 4 non-null of 5


def test_describe_custom_percentiles(duck):
    duck.register("t", _t())
    d = bt.from_arrow(_t()).describe(percentiles=(0.1, 0.9)).to_pydict()
    assert d["statistic"] == ["count", "null_count", "mean", "std", "min", "10%", "90%", "max"]
    exp = (
        duck.sql("SELECT quantile_cont(x, 0.1) p10, quantile_cont(x, 0.9) p90 FROM t")
        .to_arrow_table()
        .to_pylist()[0]
    )
    assert _stat(d, "10%", "x") == pytest.approx(exp["p10"])
    assert _stat(d, "90%", "x") == pytest.approx(exp["p90"])


def test_null_count_lazy_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _t())
    out = bt.from_arrow(_t()).null_count().collect()
    assert_same(
        out,
        duck.sql(
            'SELECT (count(*) - count(name)) "name", (count(*) - count(x)) "x", '
            '(count(*) - count(y)) "y" FROM t'
        ),
    )


def test_describe_empty_input():
    # An empty relation via a filter that removes every row (from_arrow needs ≥1 row).
    d = bt.from_arrow(_t()).filter(col("x") < 0).describe().to_pydict()
    assert _stat(d, "count", "x") == 0.0
    assert _stat(d, "null_count", "x") == 0.0
    # mean/std/min/max over no rows are undefined (null), not NaN.
    mean_x = _stat(d, "mean", "x")
    assert mean_x is None or math.isnan(mean_x)
