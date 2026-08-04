"""`CountVectorizer` / `TfidfVectorizer` / `HashingVectorizer` against scikit-learn.

scikit-learn is the oracle here for the same reason DuckDB is the oracle for the relational
surface: the semantics of a bag of words are not obvious (what counts as a token, whether
`min_df` is inclusive, which IDF smoothing is meant), and matching a widely-used
implementation is the only claim worth making. The hashing vectorizer is checked against
invariants rather than sklearn, because the two use different hash functions and so cannot
agree on which feature index a term lands on.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    CountVectorizer,
    HashingVectorizer,
    Preprocessor,
    TfidfVectorizer,
)

pytestmark = pytest.mark.unit

sklearn_text = pytest.importorskip("sklearn.feature_extraction.text")

DOCS = [
    "red car red",
    "blue bike",
    "red bike blue car",
    "the quick brown fox jumps",
    "",
    "RED CAR",
]


def _ds(docs: list[str | None] = DOCS) -> bt.Dataset:
    return bt.from_pydict({"t": docs})


def _dense(pre: Preprocessor, ds: bt.Dataset) -> np.ndarray:
    return np.array(pre.transform(ds).to_pydict()["features"], dtype=float)


COUNT_CASES = [
    {},
    {"ngram_range": (1, 2)},
    {"ngram_range": (2, 3)},
    {"stop_words": "english"},
    {"min_df": 2},
    {"max_df": 2},
    {"binary": True},
    {"lowercase": False},
    {"max_features": 2},
]


@pytest.mark.parametrize("options", COUNT_CASES, ids=lambda o: str(sorted(o)) or "defaults")
def test_count_vectorizer_matches_sklearn(options: dict) -> None:
    ours = CountVectorizer("t", dense=True, **options).fit(_ds())
    theirs = sklearn_text.CountVectorizer(**options).fit(DOCS)
    assert ours.vocabulary_ == sorted(theirs.vocabulary_)
    np.testing.assert_allclose(_dense(ours, _ds()), theirs.transform(DOCS).toarray())


TFIDF_CASES = [
    {},
    {"norm": "l1"},
    {"norm": None},
    {"sublinear_tf": True},
    {"smooth_idf": False},
    {"use_idf": False},
    {"ngram_range": (1, 2), "min_df": 2},
]


@pytest.mark.parametrize("options", TFIDF_CASES, ids=lambda o: str(sorted(o)) or "defaults")
def test_tfidf_vectorizer_matches_sklearn(options: dict) -> None:
    ours = TfidfVectorizer("t", dense=True, **options).fit(_ds())
    theirs = sklearn_text.TfidfVectorizer(**options).fit(DOCS)
    assert ours.vocabulary_ == sorted(theirs.vocabulary_)
    if options.get("use_idf", True):
        np.testing.assert_allclose(ours.idf_, theirs.idf_)
    np.testing.assert_allclose(_dense(ours, _ds()), theirs.transform(DOCS).toarray())


def test_sparse_and_dense_describe_the_same_matrix() -> None:
    """The index/value pair and the fixed-width column must be the same row, spelled twice."""
    ds = _ds()
    dense = _dense(TfidfVectorizer("t", dense=True).fit(ds), ds)
    sparse = TfidfVectorizer("t").fit(ds).transform(ds).to_pydict()
    rebuilt = np.zeros_like(dense)
    for row, (indices, values) in enumerate(
        zip(sparse["features_indices"], sparse["features_values"], strict=True)
    ):
        rebuilt[row, indices] = values
    np.testing.assert_allclose(rebuilt, dense)


def test_a_null_document_is_an_empty_one_not_a_dropped_row() -> None:
    """A null must keep its row and produce no features, never vanish from the output."""
    ds = _ds(["red car", None, ""])
    out = CountVectorizer("t").fit_transform(ds).to_pydict()
    assert out["features_indices"] == [[0, 1], [], []]
    assert out["features_values"] == [[1.0, 1.0], [], []]


def test_an_unseen_term_is_ignored_rather_than_bucketed() -> None:
    """The fitted vocabulary is the feature space; a new word adds no feature."""
    fitted = CountVectorizer("t").fit(_ds(["red car"]))
    out = fitted.transform(_ds(["red bicycle helmet"])).to_pydict()
    assert fitted.vocabulary_ == ["car", "red"]
    assert out["features_indices"] == [[1]]
    assert out["features_values"] == [[1.0]]


def test_transform_preserves_row_count_and_the_other_columns() -> None:
    ds = bt.from_pydict({"t": DOCS, "label": list(range(len(DOCS)))})
    out = CountVectorizer("t").fit_transform(ds)
    assert out.count() == len(DOCS)
    assert out.to_pydict()["label"] == list(range(len(DOCS)))


def test_fit_is_independent_of_partitioning() -> None:
    """The vocabulary is a mergeable aggregate, so batch boundaries cannot move it."""
    one = CountVectorizer("t").fit(_ds())
    many = CountVectorizer("t").fit(bt.from_pydict({"t": DOCS}).repartition(4))
    assert one.vocabulary_ == many.vocabulary_
    assert one.document_frequencies_ == many.document_frequencies_
    assert one.document_count_ == many.document_count_


def test_iter_batches_agrees_with_collect() -> None:
    """The transform streams, so reading it batch by batch must give the same rows."""
    ds = _ds()
    fitted = TfidfVectorizer("t").fit(ds)
    collected = fitted.transform(ds).to_pydict()["features_values"]
    streamed = [
        row
        for batch in fitted.transform(ds).iter_batches(batch_size=2)
        for row in batch.to_pydict()["features_values"]
    ]
    assert collected == streamed


def test_hashing_needs_no_fit_and_stays_stable() -> None:
    ds = _ds()
    first = HashingVectorizer("t", n_features=256, norm=None).fit_transform(ds).to_pydict()
    second = HashingVectorizer("t", n_features=256, norm=None).fit_transform(ds).to_pydict()
    assert first["features_indices"] == second["features_indices"]
    counts = [sum(v) for v in first["features_values"]]
    assert counts == [3.0, 2.0, 4.0, 5.0, 0.0, 2.0]


def test_hashing_normalizes_rows_and_stays_inside_the_feature_space() -> None:
    out = HashingVectorizer("t", n_features=32).fit_transform(_ds()).to_pydict()
    for indices, values in zip(out["features_indices"], out["features_values"], strict=True):
        assert all(0 <= i < 32 for i in indices)
        norm = float(np.sqrt(np.sum(np.square(values))))
        assert norm == pytest.approx(1.0) or not values


def test_a_fitted_vectorizer_round_trips_through_save(tmp_path) -> None:
    ds = _ds()
    fitted = TfidfVectorizer("t", ngram_range=(1, 2), min_df=2).fit(ds)
    target = str(tmp_path / "tfidf.json")
    fitted.save(target)
    restored = Preprocessor.load(target)
    assert restored.vocabulary_ == fitted.vocabulary_
    assert restored.idf_ == fitted.idf_
    assert (
        restored.transform(ds).to_pydict()["features_values"]
        == fitted.transform(ds).to_pydict()["features_values"]
    )


def test_transform_before_fit_names_the_class() -> None:
    with pytest.raises(PlanError, match="CountVectorizer must be fitted"):
        CountVectorizer("t").transform(_ds())


def test_an_unbounded_vocabulary_fails_with_a_way_out() -> None:
    ds = bt.from_pydict({"t": [f"word{i} shared" for i in range(50)]})
    with pytest.raises(PlanError, match="max_features"):
        CountVectorizer("t", max_vocabulary=10).fit(ds)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"ngram_range": (2, 1)}, "min_n <= max_n"),
        ({"stop_words": "spanish"}, "stop_words must be"),
        ({"max_features": 0}, "max_features must be"),
        ({"min_df": 5, "max_df": 2}, "No term can satisfy both"),
    ],
)
def test_bad_configuration_is_rejected_with_the_reason(options: dict, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        CountVectorizer("t", **options).fit(_ds())


def test_norm_is_validated_on_both_vectorizers() -> None:
    for klass in (TfidfVectorizer, HashingVectorizer):
        with pytest.raises(PlanError, match="norm must be"):
            klass("t", norm="l3")
