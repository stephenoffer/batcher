"""Text vectorizers: scikit-learn-compatible hashing, and a null-document policy.

scikit-learn is the oracle. `HashingVectorizer(hash_function="murmur3", alternate_sign=True)`
must reproduce its indices, signs and values exactly, so a model fitted on scikit-learn's
hashed features scores Batcher's; the default FNV hash is documented as not doing so.
`null_documents="null"` must keep missing text missing, and leave it out of a fitted IDF,
which scikit-learn (with no nulls) defines by the corpus of the present documents.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import CountVectorizer, HashingVectorizer, TfidfVectorizer

sk_text = pytest.importorskip("sklearn.feature_extraction.text")

DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "fox fox fox",
    "",
    "naïve café résumé déjà vu",
    "a b c",
    "hello, world! hello again world world",
]


@pytest.mark.parametrize("n_features", [8, 2**10])
@pytest.mark.parametrize("ngram_range", [(1, 1), (1, 2)])
@pytest.mark.parametrize("norm", ["l2", "l1", None])
@pytest.mark.parametrize("binary", [False, True])
def test_murmur3_with_alternate_sign_matches_sklearn(n_features, ngram_range, norm, binary):
    expected = (
        sk_text.HashingVectorizer(
            n_features=n_features, ngram_range=ngram_range, norm=norm, binary=binary
        )
        .transform(DOCS)
        .toarray()
    )
    got = (
        HashingVectorizer(
            "t",
            n_features=n_features,
            ngram_range=ngram_range,
            norm=norm,
            binary=binary,
            hash_function="murmur3",
            alternate_sign=True,
            dense=True,
        )
        .fit_transform(bt.from_pydict({"t": DOCS}))
        .to_pydict()["features"]
    )
    np.testing.assert_allclose(np.array(got), expected, rtol=0, atol=1e-12)


def test_the_sparse_form_carries_the_same_indices_and_signs_as_sklearn():
    expected = sk_text.HashingVectorizer(n_features=2**12, norm=None).transform(DOCS)
    out = (
        HashingVectorizer(
            "t", n_features=2**12, norm=None, hash_function="murmur3", alternate_sign=True
        )
        .fit_transform(bt.from_pydict({"t": DOCS}))
        .to_pydict()
    )
    for row, (indices, values) in enumerate(
        zip(out["features_indices"], out["features_values"], strict=True)
    ):
        reference = expected.getrow(row)
        assert dict(zip(indices, values, strict=True)) == pytest.approx(
            dict(zip(reference.indices.tolist(), reference.data.tolist(), strict=True))
        )


def test_the_default_hash_is_not_sklearns():
    """The documented default: equivalent feature spaces, different indices."""
    expected = sk_text.HashingVectorizer(n_features=2**12, alternate_sign=False, norm=None)
    got = HashingVectorizer("t", n_features=2**12, norm=None).fit_transform(
        bt.from_pydict({"t": ["fox"]})
    )
    assert got.to_pydict()["features_indices"][0] != expected.transform(["fox"]).indices.tolist()


def test_an_unknown_hash_function_is_refused():
    with pytest.raises(PlanError, match="hash_function"):
        HashingVectorizer("t", hash_function="md5")


@pytest.mark.parametrize(
    "make",
    [
        lambda **kw: CountVectorizer("t", **kw),
        lambda **kw: TfidfVectorizer("t", **kw),
        lambda **kw: HashingVectorizer("t", n_features=16, **kw),
    ],
    ids=["count", "tfidf", "hashing"],
)
@pytest.mark.parametrize("dense", [False, True])
def test_null_documents_null_keeps_missing_text_missing(make, dense):
    ds = bt.from_pydict({"t": ["red car", None, "", "red bike"]})
    out = make(null_documents="null", dense=dense).fit_transform(ds).to_pydict()
    columns = ["features"] if dense else ["features_indices", "features_values"]
    for name in columns:
        assert out[name][1] is None, name
        # An empty document is still an observed, empty bag.
        assert out[name][2] is not None, name
    default = make(dense=dense).fit_transform(ds).to_pydict()
    assert all(default[name][1] is not None for name in columns)


def test_a_null_document_is_left_out_of_the_idf_under_the_null_policy():
    present = ["red car", "", "red bike"]
    ds = bt.from_pydict({"t": ["red car", None, "", "red bike"]})
    fitted = TfidfVectorizer("t", null_documents="null").fit(ds)
    reference = sk_text.TfidfVectorizer().fit(present)
    assert fitted.document_count_ == 3
    assert fitted.vocabulary_ == sorted(reference.vocabulary_)
    assert fitted.idf_ == pytest.approx(list(reference.idf_))


def test_an_unknown_null_policy_is_refused():
    with pytest.raises(PlanError, match="null_documents"):
        CountVectorizer("t", null_documents="drop")
