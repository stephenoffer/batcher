"""`.list.l2_norm` and `.list.flatten` — list algebra for vectors / nested lists.

`l2_norm` is checked against DuckDB (`sqrt(list_aggregate(... ))`-equivalent via
`list_dot_product` is unavailable, so against a hand-built reference); `flatten`
is structural (DuckDB `flatten` over list literals).
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def test_list_l2_norm_values():
    t = pa.table({"v": [[3.0, 4.0], [1.0, 2.0, 2.0], [], None]})
    out = bt.from_arrow(t).select(n=col("v").list.l2_norm()).to_pydict()["n"]
    assert out[0] == pytest.approx(5.0)
    assert out[1] == pytest.approx(3.0)
    assert out[2] is None  # empty list → null
    assert out[3] is None  # null row → null


def test_list_l2_norm_matches_duckdb(duck):
    t = pa.table({"v": [[3.0, 4.0], [5.0, 12.0], [1.0, 1.0, 1.0, 1.0]]})
    duck.register("t", t)
    got = bt.from_arrow(t).select(n=col("v").list.l2_norm()).to_pydict()["n"]
    exp = (
        duck.sql("SELECT sqrt(list_aggregate(list_transform(v, x -> x*x), 'sum')) n FROM t")
        .to_arrow_table()
        .to_pydict()["n"]
    )
    for a, b in zip(got, exp, strict=True):
        assert a == pytest.approx(b)


def test_list_flatten_structural():
    t = pa.table({"xs": [[[1, 2], [3]], [[4]], [None, [5, 6]], None, []]})
    out = bt.from_arrow(t).select(f=col("xs").list.flatten()).to_pydict()["f"]
    assert out == [[1, 2, 3], [4], [5, 6], None, []]


def test_list_flatten_matches_duckdb(duck):
    t = pa.table({"xs": [[[1, 2], [3]], [[4], [5, 6]]]})
    duck.register("t", t)
    got = bt.from_arrow(t).select(f=col("xs").list.flatten()).to_pydict()["f"]
    exp = duck.sql("SELECT flatten(xs) f FROM t").to_arrow_table().to_pydict()["f"]
    assert got == exp


def test_list_norm_then_cosine_pattern():
    # l2_norm is the building block for cosine similarity: emb / ||emb||.
    t = pa.table({"v": [[3.0, 4.0]]})
    n = bt.from_arrow(t).select(n=col("v").list.l2_norm()).to_pydict()["n"][0]
    assert math.isclose(n, 5.0)
