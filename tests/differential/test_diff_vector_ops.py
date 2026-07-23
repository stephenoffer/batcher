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


def _fsl_table(dim: int = 2, value_type=None):
    value_type = value_type or pa.float32()
    # A `FixedSizeList` column is the physical type behind `arrow.fixed_shape_tensor`,
    # i.e. every embedding column the engine produces (`.image.to_tensor`, `ds.ml.embed`).
    a = pa.array([[1.0, 0.0], [1.0, 2.0], [3.0, 4.0]], type=pa.list_(value_type, dim))
    b = pa.array([[0.0, 1.0], [2.0, 4.0], [4.0, 3.0]], type=pa.list_(value_type, dim))
    return pa.table({"a": a, "b": b})


def test_vector_ops_accept_fixed_size_list():
    # The kernels used to reject `FixedSizeList` outright, so cosine/dot/l2 could not run
    # on the one Arrow type designed for embeddings. They now normalize it to a list.
    t = _fsl_table()
    got = bt.from_arrow(t).select(
        dot=col("a").list.dot(col("b")),
        cos=col("a").list.cosine_similarity(col("b")),
        l2=col("a").list.l2_distance(col("b")),
    )
    d = got.to_pydict()
    assert d["dot"] == pytest.approx([0.0, 10.0, 24.0])
    assert d["cos"][0] == pytest.approx(0.0)  # orthogonal
    assert d["l2"][0] == pytest.approx(math.sqrt(2.0))


def test_fixed_size_list_matches_var_list_bitwise():
    # The f32 fast path must equal the f64 var-list path exactly, not approximately —
    # f32->f64 is lossless and accumulation is f64 in both.
    fsl = bt.from_arrow(_fsl_table(value_type=pa.float32()))
    var = bt.from_arrow(
        pa.table(
            {
                "a": [[1.0, 0.0], [1.0, 2.0], [3.0, 4.0]],
                "b": [[0.0, 1.0], [2.0, 4.0], [4.0, 3.0]],
            }
        )
    )
    for op in ("dot", "cosine_similarity", "l2_distance"):
        f = fsl.select(v=getattr(col("a").list, op)(col("b"))).to_pydict()["v"]
        v = var.select(v=getattr(col("a").list, op)(col("b"))).to_pydict()["v"]
        assert f == v  # exact equality, not approx


def test_elementwise_vector_arithmetic():
    t = pa.table({"a": [[1.0, 2.0], [3.0, 4.0]], "b": [[10.0, 20.0], [1.0, 1.0]]})
    ds = bt.from_arrow(t)
    assert ds.select(r=col("a").list.add(col("b"))).to_pydict()["r"] == [[11.0, 22.0], [4.0, 5.0]]
    assert ds.select(r=col("a").list.subtract(col("b"))).to_pydict()["r"] == [
        [-9.0, -18.0],
        [2.0, 3.0],
    ]
    assert ds.select(r=col("a").list.multiply(col("b"))).to_pydict()["r"] == [
        [10.0, 40.0],
        [3.0, 4.0],
    ]


def test_vector_arithmetic_broadcasts_a_literal_and_survives_pushdown():
    # A bias vector via `array(...)`, and a join so projection pushdown must carry both
    # sides' columns through the ListZip node (regression: `remap_columns`/`referenced_columns`
    # must traverse it, else a column is pruned → "unknown column").
    left = bt.from_arrow(pa.table({"id": [1, 2], "emb": [[1.0, 1.0], [2.0, 2.0]]}))
    right = bt.from_arrow(pa.table({"id": [1, 2], "other": [[10.0, 0.0], [0.0, 10.0]]}))
    joined = left.join(right, on="id")
    out = joined.select("id", s=col("emb").list.add(col("other"))).to_pydict()
    by_id = dict(zip(out["id"], out["s"], strict=True))
    assert by_id[1] == [11.0, 1.0]
    assert by_id[2] == [2.0, 12.0]


def test_vector_arithmetic_length_mismatch_raises():
    t = pa.table({"a": [[1.0, 2.0]], "b": [[1.0, 2.0, 3.0]]})
    with pytest.raises(Exception):  # noqa: B017 - typed engine error on mismatch
        bt.from_arrow(t).select(r=col("a").list.add(col("b"))).to_pydict()


def test_list_softmax_is_a_probability_distribution():
    import math

    t = pa.table({"logits": [[0.0, 0.0], [1.0, 2.0, 3.0]]})
    out = bt.from_arrow(t).select(p=col("logits").list.softmax()).to_pydict()["p"]
    assert out[0] == pytest.approx([0.5, 0.5])
    # Stable softmax of [1,2,3].
    denom = sum(math.exp(x - 3) for x in [1.0, 2.0, 3.0])
    assert out[1] == pytest.approx([math.exp(x - 3) / denom for x in [1.0, 2.0, 3.0]])
    assert all(math.isclose(sum(row), 1.0, abs_tol=1e-9) for row in out)


def test_list_softmax_on_fixed_size_list():
    import math

    t = pa.table({"l": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32(), 2))})
    out = bt.from_arrow(t).select(p=col("l").list.softmax()).to_pydict()["p"]
    denom = math.exp(1 - 2) + 1
    assert out[0] == pytest.approx([math.exp(1 - 2) / denom, 1 / denom])


def test_list_arg_sort_ranks_scores():
    t = pa.table({"scores": [[0.3, 0.9, 0.1], [5.0, 5.0, 1.0]]})
    out = bt.from_arrow(t).select(r=col("scores").list.arg_sort()).to_pydict()["r"]
    assert out[0] == [2, 0, 1]  # ascending: 0.1(idx2), 0.3(idx0), 0.9(idx1)
    assert out[1] == [2, 0, 1]  # stable on ties: 1.0(idx2), then 5.0(idx0), 5.0(idx1)


def test_list_arg_sort_top_k_via_reverse():
    # The canonical use: descending rank of a per-row score vector → top-k indices.
    t = pa.table({"scores": [[0.3, 0.9, 0.1, 0.5]]})
    top = bt.from_arrow(t).select(r=col("scores").list.arg_sort().list.reverse()).to_pydict()["r"]
    assert top[0][:2] == [1, 3]  # highest 0.9(idx1), then 0.5(idx3)


def test_list_l1_norm():
    t = pa.table({"v": [[3.0, -4.0], [1.0, 1.0, 1.0]]})
    out = bt.from_arrow(t).select(n=col("v").list.l1_norm()).to_pydict()["n"]
    assert out == pytest.approx([7.0, 3.0])
    # On a fixed-size-list (tensor) column too.
    ft = pa.table({"v": pa.array([[3.0, -4.0]], type=pa.list_(pa.float32(), 2))})
    fout = bt.from_arrow(ft).select(n=col("v").list.l1_norm()).to_pydict()["n"]
    assert fout == pytest.approx([7.0])


def test_list_max_abs():
    t = pa.table({"v": [[1.0, -5.0, 3.0], [-2.0, 2.0], [0.5]]})
    out = bt.from_arrow(t).select(m=col("v").list.max_abs()).to_pydict()["m"]
    # The MaxAbs-scaling divisor: the largest magnitude in each row.
    assert out == pytest.approx([5.0, 2.0, 0.5])


def test_list_diff_first_difference():
    t = pa.table({"v": [[1.0, 2.0, 4.0, 7.0], [5.0]]})
    out = bt.from_arrow(t).select(d=col("v").list.diff()).to_pydict()["d"]
    # Leading null (no predecessor), then consecutive differences.
    assert out[0] == [None, 1.0, 2.0, 3.0]
    assert out[1] == [None]


def test_list_cum_sum_accumulates():
    t = pa.table({"xs": [[1.0, 2.0, 3.0], [10.0, -5.0, 5.0]]})
    out = bt.from_arrow(t).select(r=col("xs").list.cum_sum()).to_pydict()["r"]
    assert out[0] == [1.0, 3.0, 6.0]
    assert out[1] == [10.0, 5.0, 10.0]
    # Divide by the total for a cumulative distribution — a common downstream use.
    cdf = bt.from_arrow(pa.table({"w": [[1.0, 1.0, 2.0]]}))
    c = cdf.select(r=col("w").list.cum_sum()).to_pydict()["r"][0]
    assert c == [1.0, 2.0, 4.0]


def test_list_reduction_accepts_fixed_size_list():
    # The unary reductions (l2_norm/normalize/mean/sum) take a tensor column too.
    t = pa.table({"a": pa.array([[3.0, 4.0]], type=pa.list_(pa.float32(), 2))})
    out = bt.from_arrow(t).select(n=col("a").list.l2_norm()).to_pydict()["n"]
    assert out[0] == pytest.approx(5.0)


def test_l1_distance_matches_reference():
    t = pa.table({"a": [[0.0, 0.0], [1.0, 2.0]], "b": [[3.0, 4.0], [1.0, -1.0]]})
    got = bt.from_arrow(t).select(d=col("a").list.l1_distance(col("b"))).to_pydict()["d"]
    assert got[0] == pytest.approx(7.0)  # |0-3| + |0-4|
    assert got[1] == pytest.approx(3.0)  # |1-1| + |2-(-1)|


def test_l1_matches_duckdb_when_available(duck):
    # DuckDB has list_distance (L2) but not a native L1; cross-check against a numeric ref
    # here and reuse the shared fixture so the column shapes match the other cases.
    t = _vecs()
    got = bt.from_arrow(t).select(d=col("a").list.l1_distance(col("b"))).to_pydict()["d"]
    exp = [
        sum(abs(x - y) for x, y in zip(ra, rb, strict=True))
        for ra, rb in zip(t["a"].to_pylist(), t["b"].to_pylist(), strict=True)
    ]
    for x, y in zip(got, exp, strict=True):
        assert x == pytest.approx(y)


def test_hamming_distance_counts_differing_positions():
    t = pa.table({"a": [[1, 0, 1, 1], [1, 1, 1, 1]], "b": [[1, 1, 0, 1], [0, 0, 0, 0]]})
    got = bt.from_arrow(t).select(h=col("a").list.hamming_distance(col("b"))).to_pydict()["h"]
    assert got[0] == pytest.approx(2.0)
    assert got[1] == pytest.approx(4.0)


def test_new_metrics_require_equal_dims():
    t = pa.table({"a": [[1.0, 2.0]], "b": [[1.0, 2.0, 3.0]]})
    for op in ("l1_distance", "hamming_distance"):
        with pytest.raises(Exception):  # noqa: B017 - engine raises a typed error on mismatch
            bt.from_arrow(t).select(r=getattr(col("a").list, op)(col("b"))).to_pydict()
