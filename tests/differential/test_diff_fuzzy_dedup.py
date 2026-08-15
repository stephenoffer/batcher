"""Fuzzy deduplication: MinHash signatures, `list.jaccard`, and LSH near-duplicate removal.

A MinHash signature has no oracle in DuckDB — but the quantity it *estimates* does, and
that is the thing worth checking. `exact_jaccard` below computes the true Jaccard over
the documents' character-shingle sets directly, and the tests assert the signature's
agreement rate tracks it within the estimator's own error bound (`1/sqrt(num_perm)`).
That is a real check: a broken permutation, a byte-wise shingler, or a signature that
lost precision through `Float64` would all fail it.

The dedup semantics are asserted as invariants, because they are what silently corrupts a
training corpus when wrong:

* every returned pair **clears the threshold** — candidate generation is a recall filter,
  never a precision one, so a false positive must not survive;
* pairs are **canonical** (`key_a < key_b`) and **unique**, so a pair colliding in several
  LSH bands is not counted several times;
* `drop_near_duplicates` leaves **no two rows that are near-duplicates of each other**,
  and leaves **at least one representative** of every cluster — dropping a whole cluster
  would silently delete data.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import col

pytestmark = pytest.mark.differential

_NGRAM = 5


def _shingles(text: str, width: int = _NGRAM) -> set[str]:
    chars = list(text)
    if len(chars) <= width:
        return {text}
    return {"".join(chars[i : i + width]) for i in range(len(chars) - width + 1)}


def exact_jaccard(a: str, b: str) -> float:
    """The quantity MinHash estimates — computed directly, as the oracle."""
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / len(sa | sb)


def _signatures(docs: list[str], num_perm: int = 128) -> list[list[int]]:
    ds = bt.from_pydict({"t": docs})
    return ds.select(s=col("t").str.minhash(num_perm, _NGRAM)).to_pydict()["s"]


def _jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    pair = bt.from_pydict({"a": [sig_a], "b": [sig_b]})
    return pair.select(j=col("a").list.jaccard(col("b"))).to_pydict()["j"][0]


# --- the signature -----------------------------------------------------------


def test_signature_has_the_requested_length_and_fits_in_32_bits():
    (sig,) = _signatures(["the quick brown fox"], num_perm=64)
    assert len(sig) == 64
    assert all(0 <= v < 2**32 for v in sig), "values must be exact as float64 for jaccard"


def test_minhash_infers_a_list_of_int64():
    ds = bt.from_pydict({"t": ["abc"]})
    assert str(ds.select(s=col("t").str.minhash(8)).schema).startswith("s: list<item: int64>")


def test_identical_documents_have_identical_signatures():
    a, b = _signatures(["hello world", "hello world"])
    assert a == b


def test_a_null_document_has_a_null_signature():
    ds = bt.from_pydict({"t": ["abc", None]})
    got = ds.select(s=col("t").str.minhash(8)).to_pydict()["s"]
    assert got[1] is None


def test_a_document_shorter_than_a_shingle_still_has_a_signature():
    short, other = _signatures(["ab", "cd"], num_perm=16)
    assert len(short) == 16
    assert short != other


def test_signatures_are_partition_independent():
    docs = [f"document number {i} with some shared prefix text" for i in range(64)]
    one = _signatures(docs)
    chunked = bt.from_arrow(bt.from_pydict({"t": docs}).to_arrow().to_batches(max_chunksize=7))
    many = chunked.select(s=col("t").str.minhash(128, _NGRAM)).to_pydict()["s"]
    assert many == one


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog",
        ),
        (
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog!",
        ),
        (
            "the quick brown fox jumps over the lazy dog",
            "the quick brown cat jumps over the lazy dog",
        ),
        ("the quick brown fox jumps over the lazy dog", "an entirely different sentence about sql"),
    ],
)
def test_the_estimate_tracks_the_true_jaccard(a, b):
    """128 permutations give a standard error of ~0.09; 0.15 is a wide but real bound."""
    sig_a, sig_b = _signatures([a, b])
    estimate, exact = _jaccard(sig_a, sig_b), exact_jaccard(a, b)
    assert abs(estimate - exact) < 0.15, f"estimate {estimate:.3f} vs exact {exact:.3f}"


def test_more_permutations_estimate_more_accurately():
    """The estimator's error shrinks as 1/sqrt(num_perm) — pin the direction, not the rate."""
    a = "the quick brown fox jumps over the lazy dog and keeps running"
    b = "the quick brown cat jumps over the lazy dog and keeps walking"
    exact = exact_jaccard(a, b)
    coarse = abs(_jaccard(*_signatures([a, b], num_perm=16)) - exact)
    fine = abs(_jaccard(*_signatures([a, b], num_perm=512)) - exact)
    assert fine <= coarse + 0.02


def test_identical_signatures_have_jaccard_one():
    (sig,) = _signatures(["anything at all"])
    assert _jaccard(sig, sig) == 1.0


def test_unrelated_documents_have_jaccard_near_zero():
    a, b = _signatures(
        ["the quick brown fox jumps over the lazy dog", "SELECT * FROM t WHERE x > 1"]
    )
    assert _jaccard(a, b) < 0.05


def test_unicode_shingles_on_character_boundaries():
    a, b = _signatures(["héllo wörld → unicode text", "héllo wörld → unicode text"])
    assert a == b


@pytest.mark.parametrize(("num_perm", "ngram"), [(0, 5), (-1, 5), (8, 0), (8, -1)])
def test_minhash_rejects_degenerate_parameters(num_perm, ngram):
    with pytest.raises(PlanError, match=r"str\.minhash"):
        col("t").str.minhash(num_perm, ngram)


# --- near-duplicate pairs ----------------------------------------------------


@pytest.fixture
def corpus():
    return bt.from_pydict(
        {
            "id": [0, 1, 2, 3, 4],
            "text": [
                "the quick brown fox jumps over the lazy dog",
                "the quick brown fox jumps over the lazy dog!",  # near-dup of 0
                "The quick brown fox jumps over the lazy dog.",  # near-dup of 0, 1
                "completely unrelated text about databases",
                "a treatise on the migratory habits of geese",
            ],
        }
    )


def test_every_returned_pair_clears_the_threshold(corpus):
    """Banding is a recall filter; a false positive must be verified away."""
    pairs = corpus.ml.near_duplicates("text", threshold=0.7, ngram=_NGRAM).to_pydict()
    assert pairs["jaccard"], "the near-duplicate cluster must be found"
    assert all(j >= 0.7 for j in pairs["jaccard"])


def test_pairs_are_canonical_and_unique(corpus):
    pairs = corpus.ml.near_duplicates("text", threshold=0.5, ngram=_NGRAM).to_pydict()
    keys = list(zip(pairs["key_a"], pairs["key_b"], strict=True))
    assert all(a < b for a, b in keys), "a pair must be emitted once, ordered"
    assert len(keys) == len(set(keys)), "a pair colliding in several bands is one pair"


def test_a_stricter_threshold_returns_fewer_pairs(corpus):
    loose = corpus.ml.near_duplicates("text", threshold=0.5, ngram=_NGRAM).count()
    strict = corpus.ml.near_duplicates("text", threshold=0.95, ngram=_NGRAM).count()
    assert strict <= loose


def test_unrelated_documents_produce_no_pairs():
    ds = bt.from_pydict({"t": ["alpha beta gamma delta", "SELECT * FROM t", "geese migrate south"]})
    assert ds.ml.near_duplicates("t", threshold=0.5).count() == 0


def test_near_duplicates_accepts_an_explicit_key(corpus):
    pairs = corpus.ml.near_duplicates("text", threshold=0.5, key="id").to_pydict()
    assert all(a < b for a, b in zip(pairs["key_a"], pairs["key_b"], strict=True))
    assert set(pairs["key_a"]) | set(pairs["key_b"]) <= {0, 1, 2}


# --- dropping near-duplicates ------------------------------------------------


def test_drop_keeps_one_representative_per_cluster(corpus):
    kept = corpus.ml.drop_near_duplicates("text", threshold=0.7, ngram=_NGRAM).to_pydict()["text"]
    assert len(kept) == 3, kept
    assert "completely unrelated text about databases" in kept
    assert "a treatise on the migratory habits of geese" in kept


def test_drop_leaves_no_two_rows_that_are_near_duplicates(corpus):
    kept = corpus.ml.drop_near_duplicates("text", threshold=0.7, ngram=_NGRAM).to_pydict()["text"]
    for i, a in enumerate(kept):
        for b in kept[i + 1 :]:
            assert exact_jaccard(a, b) < 0.7, f"{a!r} and {b!r} survived together"


def test_drop_never_deletes_a_whole_cluster(corpus):
    """At least one member of the duplicate cluster must survive."""
    kept = set(
        corpus.ml.drop_near_duplicates("text", threshold=0.7, ngram=_NGRAM).to_pydict()["text"]
    )
    cluster = {
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog!",
        "The quick brown fox jumps over the lazy dog.",
    }
    assert kept & cluster


def test_drop_collapses_byte_identical_documents():
    """The default key is the text's digest, so exact copies share a key; they must not
    all survive together when that key wins."""
    docs = ["same text here exactly", "same text here exactly", "different entirely zzz"]
    ds = bt.from_pydict({"id": [0, 1, 2], "text": docs})
    kept = ds.ml.drop_near_duplicates("text", threshold=0.8).to_pydict()["text"]
    assert sorted(kept) == ["different entirely zzz", "same text here exactly"]


def test_drop_preserves_the_input_schema(corpus):
    assert corpus.ml.drop_near_duplicates("text").columns == ["id", "text"]


def test_drop_on_a_corpus_with_no_duplicates_keeps_everything():
    ds = bt.from_pydict({"t": ["alpha beta gamma", "SELECT * FROM t", "geese migrate"]})
    assert ds.ml.drop_near_duplicates("t", threshold=0.5).count() == 3


def test_drop_stays_lazy():
    ds = bt.from_pydict({"t": ["a b c d e"]})
    assert isinstance(ds.ml.drop_near_duplicates("t"), bt.Dataset)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": 0.0}, "threshold must be in"),
        ({"threshold": 1.5}, "threshold must be in"),
        ({"bands": 7}, "bands must divide num_perm"),
        ({"num_perm": 0}, "must be >= 1"),
        # The shared `require_columns` check reports the name and the alternatives, and
        # appends the caller's hint ("...groups rows by this key column"), rather than the
        # bare "unknown key column" the inline copy used to raise.
        ({"key": "nope"}, r"Unknown column 'nope'.*key column"),
    ],
)
def test_invalid_parameters_are_rejected(corpus, kwargs, match):
    with pytest.raises(PlanError, match=match):
        corpus.ml.near_duplicates("text", **kwargs)


def test_unknown_column_is_rejected(corpus):
    # The canonical `unknown_message` shape: what failed, the offending value, the
    # alternatives, the next action. Asserted on the value and the parameter rather than on
    # the prose, so rewording the message does not fail this while a *wrong* message would.
    with pytest.raises(PlanError, match=r"Unknown column 'nope'.*Available columns.*column"):
        corpus.ml.near_duplicates("nope")
