"""Differential + mergeability coverage for corr / covar / skewness / kurtosis.

All use a sum-of-powers partial state, so each must match DuckDB *and* be identical
single-node vs multi-partition (the mergeable-algebra invariant). Skewness/kurtosis
are checked on asymmetric data — the symmetric case alone wouldn't catch a
convention error.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, corr, covar_pop, covar_samp

pytestmark = pytest.mark.differential

_X = [1.0, 1.0, 2.0, 3.0, 10.0, 1.0, 2.0, 1.0, 50.0, 2.0]
_Y = [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 1.0, 3.0]


def _data():
    return pa.table({"x": _X, "y": _Y})


def test_corr_covar_match_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _data())
    out = (
        bt.from_arrow(_data())
        .agg(
            c=corr(col("x"), col("y")),
            cp=covar_pop(col("x"), col("y")),
            cs=covar_samp(col("x"), col("y")),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT corr(x,y) AS c, covar_pop(x,y) AS cp, covar_samp(x,y) AS cs FROM t"),
    )


def test_skewness_kurtosis_match_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _data())
    out = bt.from_arrow(_data()).agg(s=col("x").skewness(), k=col("x").kurtosis()).collect()
    assert_same(out, duck.sql("SELECT skewness(x) AS s, kurtosis(x) AS k FROM t"))


def test_stats_aggregates_single_node_equals_distributed():
    g = {
        "g": ["a", "a", "a", "b", "b", "b", "b"],
        "x": [1.0, 2.0, 3.0, 1.0, 5.0, 2.0, 8.0],
        "y": [2.0, 4.0, 5.0, 1.0, 6.0, 3.0, 9.0],
    }
    ds = bt.from_pydict(g).group_by("g").agg(c=corr(col("x"), col("y")), s=col("x").skewness())
    sd = ds.collect().to_pydict()
    single = {
        k: (round(c, 9), round(s, 9)) for k, c, s in zip(sd["g"], sd["c"], sd["s"], strict=True)
    }
    dd = ds.collect(distributed=True, num_workers=3).to_pydict()
    multi = {
        k: (round(c, 9), round(s, 9)) for k, c, s in zip(dd["g"], dd["c"], dd["s"], strict=True)
    }
    assert single == multi
