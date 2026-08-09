"""`.list.simhash` and `ds.ml.similarity_join` — vector LSH, checked against brute force.

DuckDB has no SimHash, so the oracle is the thing the LSH is an approximation *of*: the
exact cosine similarity over every pair, computed by a cross join. That gives two
properties worth far more than a value comparison:

* **Precision is exact.** Every pair `similarity_join` returns must appear in the
  brute-force set with the same similarity. Banding decides which pairs are *looked at*,
  never which are *returned* — if a below-threshold pair ever escaped, that would be a
  correctness bug, not a tuning issue.
* **Recall is high but not guaranteed.** A pair above the threshold can miss every band.
  That is inherent to LSH, so it is asserted as a bound, not an equality — and the
  bound is checked against `bands`, the dial that trades recall for cost.

The signature itself is pinned by its defining probability, `P(bits agree) = 1 - θ/π`,
which is what makes the agreement rate an estimator of the angle.
"""

from __future__ import annotations

import math
import random

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.differential


def _agreement(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v, strict=True))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv)


# --- the signature ------------------------------------------------------------------


def test_the_signature_length_is_num_bits_and_the_values_are_bits():
    ds = bt.from_pydict({"v": [[1.0, 2.0, 3.0]]})
    sig = ds.select(s=bt.col("v").list.simhash(32)).to_pydict()["s"][0]
    assert len(sig) == 32
    assert set(sig) <= {0, 1}


def test_the_signature_sees_only_direction_not_magnitude():
    ds = bt.from_pydict({"v": [[1.0, 2.0], [1000.0, 2000.0]]})
    sig = ds.select(s=bt.col("v").list.simhash(64)).to_pydict()["s"]
    assert sig[0] == sig[1]


def test_agreement_estimates_the_angle_as_the_theory_says(duck):
    """`P(bits agree) = 1 - θ/π`. Check it against `acos` of the exact cosine, computed
    by DuckDB, over a spread of angles. 2048 bits keeps the sampling error ~1%."""
    rng = random.Random(11)
    base = [1.0, 0.0]
    vectors = [base] + [[math.cos(t), math.sin(t)] for t in (0.1, 0.5, 1.0, 2.0, 3.0)]
    table = pa.table({"i": list(range(len(vectors))), "v": vectors})
    duck.register("t", table)

    sigs = bt.from_arrow(table).select(
        "i", s=bt.col("v").list.simhash(2048, seed=rng.randint(0, 99))
    )
    sigs = sigs.sort("i").to_pydict()["s"]

    for i, vec in enumerate(vectors):
        exact = duck.sql(
            f"SELECT list_cosine_similarity({base}::DOUBLE[], {vec}::DOUBLE[])"
        ).fetchone()[0]
        theta = math.acos(max(-1.0, min(1.0, exact)))
        predicted = 1.0 - theta / math.pi
        observed = _agreement(sigs[0], sigs[i])
        assert abs(observed - predicted) < 0.04, (
            f"vector {i}: predicted {predicted:.3f}, observed {observed:.3f}"
        )


def test_a_null_or_empty_vector_has_no_direction_and_yields_null():
    ds = bt.from_arrow(pa.table({"v": pa.array([None, [], [1.0]], type=pa.list_(pa.float64()))}))
    sig = ds.select(s=bt.col("v").list.simhash(8)).to_pydict()["s"]
    assert sig[0] is None
    assert sig[1] is None
    assert sig[2] is not None


def test_the_signature_is_partition_independent():
    """A signature computed on one node must equal one computed on another."""
    rows = pa.table({"v": [[float(i), float(i * 2 + 1)] for i in range(200)]})
    one = bt.from_arrow(rows).select(s=bt.col("v").list.simhash(32)).to_pydict()["s"]
    many = bt.from_arrow(rows.to_batches(max_chunksize=13))
    assert many.select(s=bt.col("v").list.simhash(32)).to_pydict()["s"] == one


def test_the_seed_selects_different_hyperplanes():
    ds = bt.from_pydict({"v": [[1.0, 2.0, -3.0]]})
    a = ds.select(s=bt.col("v").list.simhash(64, seed=0)).to_pydict()["s"][0]
    b = ds.select(s=bt.col("v").list.simhash(64, seed=1)).to_pydict()["s"][0]
    assert a != b


def test_an_out_of_range_bit_count_is_rejected_at_the_api_edge():
    for bits in (0, -1, 4097):
        with pytest.raises(PlanError, match=r"num_bits must be in \[1, 4096\]"):
            bt.col("v").list.simhash(bits)


def test_projection_pushdown_sees_through_simhash():
    from batcher.plan.expr_ir import referenced_columns

    assert referenced_columns(bt.col("v").list.simhash(8)) == {"v"}


# --- the join -----------------------------------------------------------------------


def _brute_force(left, right, threshold):
    """Every pair whose exact cosine clears the threshold. The oracle."""
    out = set()
    for ka, u in left:
        for kb, v in right:
            if _cosine(u, v) >= threshold:
                out.add((ka, kb))
    return out


def _random_vectors(rng, n, dim):
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def test_every_returned_pair_is_a_true_match_precision_is_exact():
    """Banding governs recall, never precision: no false positive may ever be returned."""
    rng = random.Random(7)
    lv, rv = _random_vectors(rng, 40, 8), _random_vectors(rng, 40, 8)
    left = bt.from_pydict({"id": list(range(40)), "v": lv})
    right = bt.from_pydict({"id": list(range(100, 140)), "v": rv})

    threshold = 0.5
    got = left.ml.similarity_join(
        right,
        left_on="v",
        threshold=threshold,
        num_bits=128,
        bands=32,
        left_key="id",
        right_key="id",
    ).to_pydict()

    truth = _brute_force(
        list(zip(range(40), lv, strict=True)),
        list(zip(range(100, 140), rv, strict=True)),
        threshold,
    )
    for ka, kb, sim in zip(got["key_a"], got["key_b"], got["similarity"], strict=True):
        assert (ka, kb) in truth, f"false positive: ({ka}, {kb}) at {sim}"
        assert sim >= threshold
        assert abs(sim - _cosine(lv[ka], rv[kb - 100])) < 1e-9, (
            "similarity must be the exact cosine"
        )


def test_recall_is_high_with_enough_bands():
    """LSH may lose a true pair. With many bands it should find nearly all of them."""
    rng = random.Random(3)
    lv, rv = _random_vectors(rng, 40, 6), _random_vectors(rng, 40, 6)
    left = bt.from_pydict({"id": list(range(40)), "v": lv})
    right = bt.from_pydict({"id": list(range(100, 140)), "v": rv})

    threshold = 0.8
    truth = _brute_force(
        list(zip(range(40), lv, strict=True)),
        list(zip(range(100, 140), rv, strict=True)),
        threshold,
    )
    got = left.ml.similarity_join(
        right,
        left_on="v",
        threshold=threshold,
        num_bits=256,
        bands=64,
        left_key="id",
        right_key="id",
    ).to_pydict()
    found = set(zip(got["key_a"], got["key_b"], strict=True))

    assert found <= truth, "precision must still be exact"
    if truth:
        assert len(found) / len(truth) >= 0.8, f"recall {len(found)}/{len(truth)} too low"


def test_identical_vectors_match_at_similarity_one():
    v = [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]
    left = bt.from_pydict({"id": [1, 2], "v": v})
    right = bt.from_pydict({"ref": [10, 20], "v": v})
    got = (
        left.ml.similarity_join(right, left_on="v", threshold=0.999, left_key="id", right_key="ref")
        .sort("key_a")
        .to_pydict()
    )
    assert got["key_a"] == [1, 2]
    assert got["key_b"] == [10, 20]
    assert all(abs(s - 1.0) < 1e-9 for s in got["similarity"])


def test_an_orthogonal_corpus_yields_no_pairs():
    left = bt.from_pydict({"id": [1], "v": [[1.0, 0.0]]})
    right = bt.from_pydict({"ref": [9], "v": [[0.0, 1.0]]})
    got = left.ml.similarity_join(right, left_on="v", threshold=0.5, left_key="id", right_key="ref")
    assert got.count() == 0


def test_rows_with_a_null_vector_never_match_and_do_not_blow_up_the_candidate_set():
    left = bt.from_arrow(
        pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "v": pa.array([None, None, [1.0, 0.0]], type=pa.list_(pa.float64())),
            }
        )
    )
    right = bt.from_arrow(
        pa.table(
            {
                "ref": pa.array([9, 8], type=pa.int64()),
                "v": pa.array([None, [1.0, 0.0]], type=pa.list_(pa.float64())),
            }
        )
    )
    got = left.ml.similarity_join(
        right, left_on="v", threshold=0.9, left_key="id", right_key="ref"
    ).to_pydict()
    assert got["key_a"] == [3]
    assert got["key_b"] == [8]


def test_the_two_sides_may_name_their_vector_columns_differently():
    left = bt.from_pydict({"id": [1], "emb": [[1.0, 0.0]]})
    right = bt.from_pydict({"ref": [9], "vector": [[1.0, 0.01]]})
    got = left.ml.similarity_join(
        right, left_on="emb", right_on="vector", threshold=0.9, left_key="id", right_key="ref"
    ).to_pydict()
    assert got["key_a"] == [1] and got["key_b"] == [9]


def test_without_explicit_keys_the_vector_is_its_own_key():
    left = bt.from_pydict({"v": [[1.0, 0.0]]})
    right = bt.from_pydict({"v": [[1.0, 0.02]]})
    got = left.ml.similarity_join(right, left_on="v", threshold=0.9).to_pydict()
    assert len(got["similarity"]) == 1
    assert got["key_a"] != got["key_b"], "different vectors get different digests"


def test_bad_arguments_are_rejected():
    left = bt.from_pydict({"v": [[1.0]]})
    right = bt.from_pydict({"v": [[1.0]]})
    # Which *argument* was wrong is the discriminating fact here — all three of these name a
    # column called "nope" — so the assertion is on the parameter the message points at,
    # which is what `unknown_message`'s hint carries.
    with pytest.raises(PlanError, match=r"Unknown column 'nope'.*left_on"):
        left.ml.similarity_join(right, left_on="nope")
    with pytest.raises(PlanError, match=r"Unknown column 'nope'.*right_on"):
        left.ml.similarity_join(right, left_on="v", right_on="nope")
    with pytest.raises(PlanError, match="threshold must be in"):
        left.ml.similarity_join(right, left_on="v", threshold=2.0)
    with pytest.raises(PlanError, match="bands must divide num_bits"):
        left.ml.similarity_join(right, left_on="v", num_bits=64, bands=7)
    with pytest.raises(PlanError, match=r"Unknown column 'nope'.*left_key"):
        left.ml.similarity_join(right, left_on="v", left_key="nope")


def test_the_join_is_lazy_and_composes():
    left = bt.from_pydict({"id": [1, 2], "v": [[1.0, 0.0], [0.0, 1.0]]})
    right = bt.from_pydict({"ref": [9], "v": [[1.0, 0.0]]})
    pairs = left.ml.similarity_join(
        right, left_on="v", threshold=0.9, left_key="id", right_key="ref"
    )
    assert isinstance(pairs, bt.Dataset)
    assert pairs.filter(bt.col("similarity") > 0.99).count() == 1
