"""percentile_cont / quantile_cont (continuous quantile aggregate) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table({"g": [1, 1, 1, 1, 2, 2, 2], "v": [10.0, 20, 30, 40, 5, 15, 25]})
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_quantile_grouped(duck, t, p):
    from conftest import assert_same

    q = f"SELECT g, quantile_cont(v, {p}) q FROM t GROUP BY g"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.99])
def test_quantile_global(duck, t, p):
    from conftest import assert_same

    q = f"SELECT quantile_cont(v, {p}) q FROM t"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_quantile_dataframe_roundtrip(duck, t):
    """col.quantile(p) round-trips through the IR to the engine."""
    from batcher import col
    from conftest import assert_same

    out = bt.from_arrow(t).group_by("g").agg(q=col("v").quantile(0.25)).collect()
    assert_same(out, duck.sql("SELECT g, quantile_cont(v, 0.25) q FROM t GROUP BY g"))


def test_quantile_out_of_range():
    from batcher import col
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        col("v").quantile(1.5)
