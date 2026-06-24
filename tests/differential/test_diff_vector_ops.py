"""`.list.dot` / `cosine_similarity` / `l2_distance` — vector ops for RAG / search.

Checked against DuckDB's native `list_dot_product` / `list_cosine_similarity` /
`list_distance` where available, plus a numeric reference for the edge cases.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import array, col, lit

pytestmark = pytest.mark.differential


def _vecs():
    return pa.table(
        {
            "a": [[1.0, 0.0], [1.0, 2.0], [3.0, 4.0]],
            "b": [[0.0, 1.0], [2.0, 4.0], [4.0, 3.0]],
        }
    )


def test_dot_matches_duckdb(duck):
    t = _vecs()
    duck.register("t", t)
    got = bt.from_arrow(t).select(d=col("a").list.dot(col("b"))).to_pydict()["d"]
    exp = duck.sql("SELECT list_dot_product(a, b) d FROM t").to_arrow_table().to_pydict()["d"]
    for x, y in zip(got, exp, strict=True):
        assert x == pytest.approx(y)


def test_cosine_matches_duckdb(duck):
    t = _vecs()
    duck.register("t", t)
    got = bt.from_arrow(t).select(c=col("a").list.cosine_similarity(col("b"))).to_pydict()["c"]
    exp = duck.sql("SELECT list_cosine_similarity(a, b) c FROM t").to_arrow_table().to_pydict()["c"]
    for x, y in zip(got, exp, strict=True):
        assert x == pytest.approx(y)


def test_l2_distance_matches_duckdb(duck):
    t = _vecs()
    duck.register("t", t)
    got = bt.from_arrow(t).select(d=col("a").list.l2_distance(col("b"))).to_pydict()["d"]
    exp = duck.sql("SELECT list_distance(a, b) d FROM t").to_arrow_table().to_pydict()["d"]
    for x, y in zip(got, exp, strict=True):
        assert x == pytest.approx(y)


def test_normalize_produces_unit_vectors():
    t = pa.table({"a": [[3.0, 4.0], [1.0, 2.0, 2.0], [0.0, 0.0]]})
    got = bt.from_arrow(t).select(n=col("a").list.normalize()).to_pydict()["n"]
    assert got[0] == pytest.approx([0.6, 0.8])
    assert math.isclose(sum(x * x for x in got[1]), 1.0)  # unit length
    assert got[2] == [0.0, 0.0]  # zero vector → zeros (no div-by-zero)


def test_normalize_makes_dot_equal_cosine(duck):
    # dot(normalize(a), normalize(b)) == cosine_similarity(a, b): cross-checks the new
    # op against the existing cosine implementation (and DuckDB's).
    t = _vecs()
    duck.register("t", t)
    got = (
        bt.from_arrow(t)
        .select(d=col("a").list.normalize().list.dot(col("b").list.normalize()))
        .to_pydict()["d"]
    )
    exp = duck.sql("SELECT list_cosine_similarity(a, b) c FROM t").to_arrow_table().to_pydict()["c"]
    for x, y in zip(got, exp, strict=True):
        assert x == pytest.approx(y)


def test_cosine_zero_norm_is_null():
    t = pa.table({"a": [[0.0, 0.0]], "b": [[1.0, 1.0]]})
    out = bt.from_arrow(t).select(c=col("a").list.cosine_similarity(col("b"))).to_pydict()["c"]
    assert out == [None]


def test_query_vector_via_array_literal():
    # The RAG pattern: similarity of each row's embedding to a fixed query vector,
    # broadcast through `array(...)`.
    t = pa.table({"emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]})
    out = (
        bt.from_arrow(t)
        .select(sim=col("emb").list.cosine_similarity(array(lit(1.0), lit(0.0))))
        .to_pydict()["sim"]
    )
    assert out[0] == pytest.approx(1.0)  # identical direction
    assert out[1] == pytest.approx(0.0)  # orthogonal
    assert out[2] == pytest.approx(1.0 / math.sqrt(2))
