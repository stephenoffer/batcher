"""percentile_cont / quantile_cont (continuous quantile aggregate) vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


@pytest.fixture
def t(duck):
    tbl = pa.table({"g": [1, 1, 1, 1, 2, 2, 2], "v": [10.0, 20, 30, 40, 5, 15, 25]})
    duck.register("t", tbl)
    return tbl


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_quantile_grouped(duck, t, p):
    q = f"SELECT g, quantile_cont(v, {p}) q FROM t GROUP BY g"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.99])
def test_quantile_global(duck, t, p):
    q = f"SELECT quantile_cont(v, {p}) q FROM t"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_quantile_dataframe_roundtrip(duck, t):
    """col.quantile(p) round-trips through the IR to the engine."""
    from batcher import col

    out = bt.from_arrow(t).group_by("g").agg(q=col("v").quantile(0.25)).collect()
    assert_same(out, duck.sql("SELECT g, quantile_cont(v, 0.25) q FROM t GROUP BY g"))


def test_quantile_out_of_range():
    from batcher import col
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        col("v").quantile(1.5)


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_bare_quantile_is_duckdbs_discrete_one(duck, t, p):
    """`quantile(x, p)` is DuckDB's `quantile_disc`, not `bt.quantile`'s continuous one.

    The name exists in both front ends and means two different things: `bt.quantile(col, p)`
    interpolates, DuckDB's `quantile` picks an element. SQL follows SQL, which is why this
    is pinned against DuckDB itself rather than against the DataFrame API -- and the
    continuous one keeps its own SQL spelling, `quantile_cont`, tested above.
    """
    q = f"SELECT g, quantile(v, {p}) q FROM t GROUP BY g"
    assert_same(bt.sql(q, t=t).collect(), duck.sql(q))


def test_quantile_and_quantile_cont_are_not_the_same_function(duck, t):
    """The positive control for the case above: the two spellings really do differ here."""
    # 0.25, not 0.5: this fixture has seven values, so the median falls *on* an element
    # and the two definitions agree there. A control that cannot tell them apart is not a
    # control, and the median is exactly the quantile where they cannot.
    disc = bt.sql("SELECT quantile(v, 0.25) q FROM t", t=t).to_pydict()["q"]
    cont = bt.sql("SELECT quantile_cont(v, 0.25) q FROM t", t=t).to_pydict()["q"]
    assert disc != cont
