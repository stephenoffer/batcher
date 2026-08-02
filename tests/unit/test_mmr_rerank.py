"""Maximal marginal relevance reranking of a retrieved candidate set.

The test that carries the argument is the contrast: at `lambda_mult=1.0` the reranker returns
the relevance ranking it was given, twins and all, and lowering it displaces the duplicate. If
that difference disappeared, the whole thing would be an expensive no-op that still looked
plausible in a result set.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import mmr_rerank_udf

pytestmark = pytest.mark.unit

# Two near-identical vectors and one orthogonal, with the twins scoring highest. Pure
# relevance takes both twins; any diversity weight displaces one of them.
_TWINS = {
    "vecs": [[[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]],
    "docs": [["twin a", "twin b", "different"]],
    "scores": [[0.9, 0.89, 0.5]],
}


def _rerank(data, **kwargs):
    udf = mmr_rerank_udf(
        embedding_column="vecs",
        score_column=kwargs.pop("score_column", "scores"),
        rerank_columns=kwargs.pop("rerank_columns", ("docs", "scores")),
        **kwargs,
    )
    return bt.from_pydict(data).ml.map_batches(udf).to_pydict()


def test_pure_relevance_returns_the_ranking_it_was_given():
    got = _rerank(_TWINS, k=2, lambda_mult=1.0)
    assert got["docs"] == [["twin a", "twin b"]]


def test_a_diversity_weight_displaces_the_duplicate():
    """The whole point: the second slot goes to new information, not to a near-copy."""
    got = _rerank(_TWINS, k=2, lambda_mult=0.7)
    assert got["docs"] == [["twin a", "different"]]


def test_the_most_relevant_candidate_is_always_first():
    for lam in (0.0, 0.3, 0.7, 1.0):
        got = _rerank(_TWINS, k=2, lambda_mult=lam)
        assert got["docs"][0][0] == "twin a"


def test_every_reranked_column_stays_aligned():
    got = _rerank(_TWINS, k=2, lambda_mult=0.7)
    assert got["docs"] == [["twin a", "different"]]
    assert got["scores"] == [[0.9, 0.5]]
    assert got["vecs"] == [[[1.0, 0.0], [0.0, 1.0]]]


def test_a_column_outside_the_rerank_set_passes_through_unchanged():
    data = {**_TWINS, "query_id": [42]}
    got = _rerank(data, k=2, lambda_mult=0.7)
    assert got["query_id"] == [42]


def test_k_larger_than_the_candidate_set_keeps_everything():
    got = _rerank(_TWINS, k=10, lambda_mult=0.7)
    assert sorted(got["docs"][0]) == ["different", "twin a", "twin b"]


def test_a_single_candidate_is_returned_as_is():
    data = {"vecs": [[[1.0, 0.0]]], "docs": [["only"]], "scores": [[0.5]]}
    got = _rerank(data, k=3, lambda_mult=0.5)
    assert got["docs"] == [["only"]]


def test_an_empty_candidate_list_stays_empty():
    data = {"vecs": [[]], "docs": [[]], "scores": [[]]}
    got = _rerank(data, k=3, lambda_mult=0.5)
    assert got["docs"] == [[]]


def test_without_a_score_column_the_existing_order_is_the_ranking():
    """A search result is already ranked, so the reranker must not need a score column."""
    data = {"vecs": _TWINS["vecs"], "docs": _TWINS["docs"]}
    got = _rerank(data, score_column=None, rerank_columns=("docs",), k=2, lambda_mult=1.0)
    assert got["docs"] == [["twin a", "twin b"]]


def test_unnormalized_embeddings_give_the_same_selection_as_normalized_ones():
    """Similarity is cosine, so scaling a candidate's vector must not change the answer."""
    scaled = {
        "vecs": [[[10.0, 0.0], [9.9, 0.1], [0.0, 0.5]]],
        "docs": _TWINS["docs"],
        "scores": _TWINS["scores"],
    }
    assert _rerank(scaled, k=2, lambda_mult=0.7)["docs"] == [["twin a", "different"]]


def test_a_zero_vector_does_not_produce_a_nan_score():
    data = {
        "vecs": [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        "docs": [["zero", "x", "y"]],
        "scores": [[0.1, 0.9, 0.8]],
    }
    got = _rerank(data, k=3, lambda_mult=0.5)
    assert len(got["docs"][0]) == 3


def test_several_rows_are_reranked_independently():
    data = {
        "vecs": [_TWINS["vecs"][0], [[0.0, 1.0], [1.0, 0.0]]],
        "docs": [_TWINS["docs"][0], ["p", "q"]],
        "scores": [_TWINS["scores"][0], [0.4, 0.3]],
    }
    got = _rerank(data, k=2, lambda_mult=0.7)
    assert got["docs"][0] == ["twin a", "different"]
    assert got["docs"][1] == ["p", "q"]


def test_a_null_score_is_treated_as_the_lowest_relevance():
    data = {
        "vecs": [[[1.0, 0.0], [0.0, 1.0]]],
        "docs": [["scored", "unscored"]],
        "scores": [[0.9, None]],
    }
    got = _rerank(data, k=1, lambda_mult=1.0)
    assert got["docs"] == [["scored"]]


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"lambda_mult": 1.5}, {"lambda_mult": -0.1}])
def test_an_invalid_setting_is_rejected_when_the_udf_is_built(kwargs):
    with pytest.raises(PlanError):
        mmr_rerank_udf(embedding_column="vecs", **kwargs)


def test_a_missing_column_names_the_ones_that_exist():
    udf = mmr_rerank_udf(embedding_column="absent", k=2)
    with pytest.raises(ColumnNotFoundError):
        bt.from_pydict({"vecs": [[[1.0]]]}).ml.map_batches(udf).to_pydict()


def test_ungrouped_candidates_are_rejected_by_name():
    """The natural mistake, and the message it used to produce.

    MMR reranks *within* a query, so its input is one row per query holding that query's whole
    candidate set (`list<list<double>>`). Handing it the retrieval output before grouping — one
    candidate per row, `list<double>` — is the same column name and element type, one nesting
    level short. NumPy answered with `AxisError: axis 1 is out of bounds for array of dimension
    1` from three frames inside the UDF, naming neither the column nor the shape it wanted.
    """
    flat = bt.from_pydict(
        {"emb": [[1.0, 0.0], [0.0, 1.0]], "score": [1.0, 2.0]},
    )
    udf = mmr_rerank_udf(embedding_column="emb", score_column="score", k=2)
    with pytest.raises(PlanError, match="candidate set"):
        flat.ml.map_batches(udf).collect()


def test_a_non_list_embedding_column_is_rejected():
    scalar = bt.from_pydict({"emb": [1.0, 2.0], "score": [1.0, 2.0]})
    udf = mmr_rerank_udf(embedding_column="emb", score_column="score", k=2)
    with pytest.raises(PlanError, match="as a list"):
        scalar.ml.map_batches(udf).collect()
