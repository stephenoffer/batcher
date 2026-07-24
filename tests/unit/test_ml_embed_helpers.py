"""`ds.ml` embedding post-processing helpers: Matryoshka truncation and degenerate drop.

Both compose native `.list` ops, so these verify the semantics (prefix + renormalize;
null/zero removal) against small hand-checked cases — no model or GPU.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def test_truncate_keeps_the_prefix_and_renormalizes():
    ds = bt.from_pydict({"emb": [[3.0, 4.0, 1.0, 1.0]]})
    out = ds.ml.truncate_embeddings("emb", 2).to_pydict()["emb"][0]
    assert out == [0.6, 0.8]
    assert math.isclose(math.hypot(*out), 1.0, rel_tol=1e-9)


def test_truncate_without_normalize_is_a_raw_slice():
    ds = bt.from_pydict({"emb": [[3.0, 4.0, 9.0]]})
    out = ds.ml.truncate_embeddings("emb", 2, normalize=False).to_pydict()["emb"][0]
    assert out == [3.0, 4.0]


def test_truncate_can_write_a_new_column():
    ds = bt.from_pydict({"emb": [[3.0, 4.0, 0.0]]})
    out = ds.ml.truncate_embeddings("emb", 2, output_column="small").to_pydict()
    assert out["small"][0] == [0.6, 0.8]
    assert out["emb"][0] == [3.0, 4.0, 0.0]  # original untouched


def test_truncate_rejects_a_non_positive_dim():
    ds = bt.from_pydict({"emb": [[1.0, 2.0]]})
    with pytest.raises(PlanError):
        ds.ml.truncate_embeddings("emb", 0)


def test_drop_degenerate_removes_zero_and_null_vectors():
    ds = bt.from_pydict({"id": [1, 2, 3, 4], "emb": [[1.0, 0.0], [0.0, 0.0], None, [0.0, 1.0]]})
    kept = ds.ml.drop_degenerate_embeddings("emb").to_pydict()
    assert kept["id"] == [1, 4]


def test_drop_degenerate_is_a_no_op_when_all_are_healthy():
    ds = bt.from_pydict({"id": [1, 2], "emb": [[1.0, 0.0], [0.0, 1.0]]})
    assert ds.ml.drop_degenerate_embeddings("emb").count() == 2


def test_embed_dedup_encodes_each_distinct_text_once():
    """embed(dedup=True) runs the encoder once per distinct text and gathers back in order."""
    import numpy as np

    from batcher.ml import embed

    calls = []

    def factory():
        def enc(texts):
            calls.append(list(texts))
            return np.array([[float(len(t)), 0.0] for t in texts])

        return enc

    batch = pa.RecordBatch.from_pydict({"t": ["a", "a", "bb", "a"]})
    out = list(embed([batch], factory, text_column="t", dedup=True, output_type="fixed_size_list"))
    assert calls == [["a", "bb"]]  # two distinct texts encoded, not four rows
    assert out[0].column("embedding").to_pylist() == [
        [1.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [1.0, 0.0],
    ]


def test_embed_unique_matches_encoding_every_row():
    import numpy as np

    from batcher.ml._embed_dedup import embed_unique

    texts = ["x", "y", "x", "z", "y"]
    ref = np.array([[float(len(t))] for t in texts])
    got = embed_unique(texts, lambda ts: np.array([[float(len(t))] for t in ts]))
    assert np.array_equal(got, ref)


def test_reciprocal_rank_fusion_ranks_shared_ids_highest():
    dense = bt.from_pydict({"id": [1, 2, 3], "score": [0.9, 0.5, 0.1]})
    lexical = bt.from_pydict({"id": [2, 3, 4], "score": [0.8, 0.7, 0.6]})
    fused = dense.ml.reciprocal_rank_fusion(lexical, key="id", score="score")
    # 2 and 3 appear in both lists, so they lead; 1 and 4 (one list each) follow.
    assert fused.to_pydict()["id"] == [2, 3, 1, 4]


def test_reciprocal_rank_fusion_fuses_three_lists():
    a = bt.from_pydict({"id": [1, 2], "s": [0.9, 0.1]})
    b = bt.from_pydict({"id": [2, 3], "s": [0.9, 0.1]})
    c = bt.from_pydict({"id": [2, 4], "s": [0.9, 0.1]})
    fused = a.ml.reciprocal_rank_fusion(b, c, key="id", score="s")
    assert fused.to_pydict()["id"][0] == 2  # ranked top by all three


def test_reciprocal_rank_fusion_rejects_non_positive_k():
    a = bt.from_pydict({"id": [1], "s": [0.5]})
    with pytest.raises(PlanError):
        a.ml.reciprocal_rank_fusion(a, key="id", score="s", k=0)


def test_nearest_neighbors_supports_l1_and_hamming_metrics():
    ds = bt.from_pydict({"id": [1, 2, 3], "emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]})
    assert ds.ml.nearest_neighbors([1.0, 0.0], column="emb", k=2, metric="l1").to_pydict()[
        "id"
    ] == [
        1,
        3,
    ]
    codes = bt.from_pydict({"id": [1, 2], "c": [[1, 0, 1, 1], [0, 1, 0, 0]]})
    hit = codes.ml.nearest_neighbors([1, 0, 1, 1], column="c", k=1, metric="hamming")
    assert hit.to_pydict()["id"] == [1]


def test_similarity_to_l1_is_negative_distance():
    ds = bt.from_pydict({"id": [1, 2], "emb": [[1.0, 0.0], [0.0, 1.0]]})
    out = ds.ml.similarity_to([1.0, 0.0], column="emb", metric="l1").to_pydict()["score"]
    assert out == [0.0, -2.0]  # larger (0) is nearer than -2


def test_batched_nearest_neighbors_returns_top_k_per_query():
    corpus = bt.from_pydict({"cid": [1, 2, 3], "emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]})
    queries = bt.from_pydict({"qid": [10, 11], "qv": [[1.0, 0.05], [0.0, 1.0]]})
    hits = corpus.ml.batched_nearest_neighbors(
        queries, query_key="qid", query_column="qv", corpus_key="cid", column="emb", k=2
    )
    d = hits.to_pydict()
    assert sorted(zip(d["qid"], d["cid"], strict=True)) == [(10, 1), (10, 3), (11, 2), (11, 3)]


def test_batched_nearest_neighbors_handles_colliding_column_names():
    corpus = bt.from_pydict({"cid": [1, 2], "embedding": [[1.0, 0.0], [0.0, 1.0]]})
    queries = bt.from_pydict({"qid": [9], "embedding": [[1.0, 0.0]]})
    hits = corpus.ml.batched_nearest_neighbors(
        queries,
        query_key="qid",
        query_column="embedding",
        corpus_key="cid",
        column="embedding",
        k=1,
    )
    assert hits.to_pydict()["cid"] == [1]


def test_batched_nearest_neighbors_rejects_bad_k_and_metric():
    corpus = bt.from_pydict({"cid": [1], "emb": [[1.0]]})
    queries = bt.from_pydict({"qid": [1], "qv": [[1.0]]})
    kw = {"query_key": "qid", "query_column": "qv", "corpus_key": "cid", "column": "emb"}
    with pytest.raises(PlanError):
        corpus.ml.batched_nearest_neighbors(queries, k=0, **kw)
    with pytest.raises(PlanError):
        corpus.ml.batched_nearest_neighbors(queries, metric="nope", **kw)


def test_binarize_embeddings_produces_a_sign_code():
    ds = bt.from_pydict({"emb": [[0.5, -0.2, 0.9, -0.1], [-1.0, 0.3, -0.4, 0.8]]})
    out = ds.ml.binarize_embeddings("emb", output_column="code").to_pydict()["code"]
    assert out == [[1, 0, 1, 0], [0, 1, 0, 1]]


def test_binarized_codes_hamming_search():
    ds = bt.from_pydict({"id": [1, 2], "emb": [[0.5, -0.2, 0.9, -0.1], [-1.0, 0.3, -0.4, 0.8]]})
    coded = ds.ml.binarize_embeddings("emb", output_column="code")
    hit = coded.ml.nearest_neighbors([1, 0, 1, 0], column="code", k=1, metric="hamming")
    assert hit.to_pydict()["id"] == [1]


def test_recall_at_k_scores_retrieval():
    retrieved = bt.from_pydict({"qid": [1, 1, 2, 2], "cid": [10, 11, 20, 21]})
    relevant = bt.from_pydict({"qid": [1, 2, 2], "cid": [10, 20, 22]})
    # q1: retrieved {10,11} vs relevant {10} -> 1.0; q2: {20,21} vs {20,22} -> 0.5; mean 0.75
    assert round(retrieved.ml.recall_at_k(relevant, query_key="qid", corpus_key="cid"), 3) == 0.75


def test_recall_at_k_edges():
    r = bt.from_pydict({"qid": [1], "cid": [10]})
    assert r.ml.recall_at_k(r, query_key="qid", corpus_key="cid") == 1.0
    miss = bt.from_pydict({"qid": [1], "cid": [99]})
    assert r.ml.recall_at_k(miss, query_key="qid", corpus_key="cid") == 0.0


def test_mrr_scores_first_relevant_rank():
    ranked = bt.from_pydict({"qid": [1, 1, 2, 2], "cid": [10, 11, 20, 21], "rank": [1, 2, 1, 2]})
    relevant = bt.from_pydict({"qid": [1, 2], "cid": [10, 21]})
    # q1 first-relevant at rank 1 -> 1.0; q2 at rank 2 -> 0.5; mean 0.75
    assert ranked.ml.mrr(relevant, query_key="qid", corpus_key="cid") == 0.75
