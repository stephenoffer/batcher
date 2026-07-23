"""`ds.ml.nearest_neighbors` — exact brute-force top-k vector retrieval.

Composes the `.list` distance kernels + sort + limit, so this checks the ranking is
correct across metrics, works on both `list` and fixed-shape-tensor (`FixedSizeList`)
embedding columns, and validates its arguments.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.integration


def _ds():
    return bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0]],
        }
    )


def test_cosine_returns_k_nearest_first():
    hits = _ds().ml.nearest_neighbors([1.0, 0.0], column="embedding", k=2)
    out = hits.to_pydict()
    assert out["id"] == [1, 3]  # exact direction, then near-direction
    assert out["distance"][0] == pytest.approx(0.0)  # identical direction → 0 distance


def test_l2_and_dot_metrics():
    # l2 ranks by smallest Euclidean distance; dot by largest inner product.
    l2 = _ds().ml.nearest_neighbors([1.0, 0.0], metric="l2", k=1).to_pydict()
    assert l2["id"] == [1]
    dot = _ds().ml.nearest_neighbors([1.0, 1.0], metric="dot", k=2).to_pydict()
    # [1,1]·[0.9,0.1]=1.0, ·[1,0]=1.0, ·[0,1]=1.0 — ties; the top-2 are among the positives.
    assert set(dot["id"]) <= {1, 2, 3}
    assert len(dot["id"]) == 2


def test_works_on_fixed_size_list_tensor_embeddings():
    t = pa.table(
        {
            "id": [1, 2, 3],
            "embedding": pa.array(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], type=pa.list_(pa.float32(), 2)
            ),
        }
    )
    hits = bt.from_arrow(t).ml.nearest_neighbors([1.0, 0.0], k=2).to_pydict()
    assert hits["id"] == [1, 3]


def test_distance_column_is_configurable_and_correct():
    hits = _ds().ml.nearest_neighbors([1.0, 0.0], k=4, distance_column="d").to_pydict()
    # cosine_distance of [0,1] to [1,0] is 1.0 (orthogonal); of [-1,0] is 2.0 (opposite).
    by_id = dict(zip(hits["id"], hits["d"], strict=True))
    assert by_id[2] == pytest.approx(1.0)
    assert by_id[4] == pytest.approx(2.0)
    assert by_id[1] == pytest.approx(0.0)
    # Sorted ascending: nearest first, farthest last.
    assert hits["d"] == sorted(hits["d"])


def test_invalid_arguments_raise():
    with pytest.raises(PlanError):
        _ds().ml.nearest_neighbors([1.0, 0.0], k=0)
    with pytest.raises(PlanError):
        _ds().ml.nearest_neighbors([1.0, 0.0], metric="manhattan")


def test_k_larger_than_dataset_returns_all():
    hits = _ds().ml.nearest_neighbors([1.0, 0.0], k=100).to_pydict()
    assert len(hits["id"]) == 4
    assert math.isclose(min(hits["distance"]), 0.0, abs_tol=1e-9)


def test_normalize_embeddings_unit_length():
    ds = bt.from_pydict({"emb": [[3.0, 4.0], [0.0, 0.0]]})
    out = ds.ml.normalize_embeddings("emb").to_pydict()["emb"]
    assert out[0] == pytest.approx([0.6, 0.8])
    assert out[1] == [0.0, 0.0]  # zero vector stays zero, no div-by-zero
    # out-of-place preserves the source
    two = ds.ml.normalize_embeddings("emb", output_column="unit").to_pydict()
    assert two["emb"][0] == [3.0, 4.0]
    assert two["unit"][0] == pytest.approx([0.6, 0.8])


def test_similarity_to_scores_every_row():
    ds = bt.from_pydict({"id": [1, 2, 3], "emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]})
    cos = ds.ml.similarity_to([1.0, 0.0], column="emb").to_pydict()["score"]
    assert cos[0] == pytest.approx(1.0)
    assert cos[1] == pytest.approx(0.0)
    assert cos[2] == pytest.approx(1.0 / math.sqrt(2))
    # l2 metric is negated so larger is still nearer; the exact match scores highest.
    l2 = ds.ml.similarity_to([1.0, 0.0], column="emb", metric="l2").to_pydict()["score"]
    assert l2[0] == max(l2)


def test_similarity_to_rejects_bad_metric():
    with pytest.raises(PlanError):
        _ds().ml.similarity_to([1.0, 0.0], column="embedding", metric="jaccard")
