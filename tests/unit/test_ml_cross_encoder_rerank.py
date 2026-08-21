"""Cross-encoder reranking — one model call per batch, and an ordering that stays aligned."""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import PlanError
from batcher.ml import cross_encoder_rerank_udf
from batcher.ml.retrieval.rerank import _activation, _flatten_pairs, _per_row_order

pytestmark = pytest.mark.unit


def _batch(queries, passages, ids=None):
    arrays = [pa.array(queries), pa.array(passages)]
    names = ["q", "docs"]
    if ids is not None:
        arrays.append(pa.array(ids))
        names.append("ids")
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _keyword_scorer(word):
    calls = []

    def factory():
        def score(pairs):
            calls.append(len(pairs))
            return [float(word in passage) for _, passage in pairs]

        return score

    return factory, calls


def test_every_pair_in_the_batch_reaches_the_model_in_one_call():
    # The whole reason this is a batch UDF: 2 rows x 3 candidates must be one forward of 6
    # pairs, not two forwards of 3 that leave the device mostly idle.
    factory, calls = _keyword_scorer("yes")
    udf = cross_encoder_rerank_udf(factory, query_column="q", document_column="docs")
    udf()(_batch(["a", "b"], [["yes 1", "no", "no"], ["no", "yes 2", "no"]]))
    assert calls == [6]


def test_candidates_are_reordered_best_first_and_truncated_to_k():
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(factory, query_column="q", document_column="docs", k=1)
    out = udf()(_batch(["a"], [["miss", "hit", "miss"]]))
    assert out.column("docs").to_pylist() == [["hit"]]


def test_aligned_columns_are_reordered_with_the_passages():
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(
        factory, query_column="q", document_column="docs", rerank_columns=("ids",), k=2
    )
    out = udf()(_batch(["a"], [["miss", "hit"]], ids=[["m", "h"]]))
    assert out.column("docs").to_pylist() == [["hit", "miss"]]
    assert out.column("ids").to_pylist() == [["h", "m"]]


def test_the_reranker_score_column_lines_up_with_the_reordered_candidates():
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(factory, query_column="q", document_column="docs")
    out = udf()(_batch(["a"], [["miss", "hit"]]))
    assert out.column("rerank_score").to_pylist() == [[1.0, 0.0]]


def test_the_score_column_can_be_declined():
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(
        factory, query_column="q", document_column="docs", score_column=None
    )
    out = udf()(_batch(["a"], [["miss", "hit"]]))
    assert "rerank_score" not in out.schema.names


def test_rescoring_into_an_existing_score_column_replaces_it():
    # Arrow permits duplicate field names, so appending would leave two columns of one name:
    # to_pydict keeps the last and every expression resolves the first, and nothing raises.
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(
        factory, query_column="q", document_column="docs", score_column="ids"
    )
    out = udf()(_batch(["a"], [["miss", "hit"]], ids=[["m", "h"]]))
    assert out.schema.names.count("ids") == 1


def test_rows_with_different_candidate_counts_stay_in_their_own_rows():
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(factory, query_column="q", document_column="docs", k=1)
    out = udf()(_batch(["a", "b"], [["miss", "hit"], ["hit", "miss", "miss"]]))
    assert out.column("docs").to_pylist() == [["hit"], ["hit"]]


def test_an_empty_candidate_list_survives_as_an_empty_list():
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(factory, query_column="q", document_column="docs")
    out = udf()(_batch(["a", "b"], [[], ["hit"]]))
    assert out.column("docs").to_pylist() == [[], ["hit"]]


def test_a_null_query_or_candidate_does_not_fail_the_batch():
    # A None reaching a tokenizer fails the whole batch over one bad row, which is why the
    # embedding and prompt paths render it as "" too.
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(factory, query_column="q", document_column="docs")
    out = udf()(_batch([None], [["hit", None]]))
    assert out.column("docs").to_pylist() == [["hit", None]]


def test_a_scorer_returning_the_wrong_count_is_refused_rather_than_misaligned():
    def factory():
        return lambda pairs: [0.0]

    udf = cross_encoder_rerank_udf(factory, query_column="q", document_column="docs")
    with pytest.raises(PlanError, match="one score per pair"):
        udf()(_batch(["a"], [["one", "two"]]))


def test_a_missing_column_names_what_the_batch_does_have():
    factory, _ = _keyword_scorer("hit")
    udf = cross_encoder_rerank_udf(factory, query_column="q", document_column="nope")
    with pytest.raises(Exception, match="nope"):
        udf()(_batch(["a"], [["x"]]))


def test_k_below_one_is_refused_at_construction():
    with pytest.raises(PlanError, match="at least 1"):
        cross_encoder_rerank_udf(lambda: None, query_column="q", document_column="d", k=0)


def test_ties_keep_the_first_stages_order():
    # A rerank that reorders equal scores arbitrarily is not reproducible.
    order, _ = _per_row_order([1.0, 1.0, 1.0], [0, 0, 0], 1, None)
    assert order == [[0, 1, 2]]


def test_flattening_records_which_row_each_pair_came_from():
    pairs, owners = _flatten_pairs(["a", "b"], [["x"], ["y", "z"]])
    assert pairs == [("a", "x"), ("b", "y"), ("b", "z")]
    assert owners == [0, 1, 1]


def test_sigmoid_is_numerically_stable_for_a_confident_logit():
    import numpy as np

    values = _activation("sigmoid")(np.array([1000.0, -1000.0, 0.0]))
    assert not np.isnan(values).any()
    assert values.tolist() == pytest.approx([1.0, 0.0, 0.5])


def test_an_unknown_activation_is_refused():
    with pytest.raises(PlanError, match="sigmoid"):
        _activation("softmax")


def test_no_activation_leaves_the_raw_logits():
    assert _activation(None)([1.0, 2.0]) == [1.0, 2.0]
