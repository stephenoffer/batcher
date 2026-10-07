"""`Dataset.sample(weights=)` and `Dataset.sample(key=)`.

A weighted sample is Efraimidis-Spirakis over the same seeded row hash the plain sampler
uses, so it is checked statistically (inclusion tracks weight) and for determinism. A
keyed sample is checked for the two properties that are its whole point: a key is kept or
dropped whole, and the choice does not move when a non-key column changes.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def _weighted() -> bt.Dataset:
    # 3 classes x 2,000 distinct rows, weighted 1 : 2 : 7. Drawing 300 of 6,000 keeps the
    # without-replacement correction small, so class shares sit near 10% / 20% / 70%.
    n = 2_000
    return bt.from_pydict(
        {
            "cls": [0] * n + [1] * n + [2] * n,
            "row": list(range(3 * n)),
            "w": [1.0] * n + [2.0] * n + [7.0] * n,
        }
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_inclusion_tracks_the_weights(seed: int) -> None:
    got = Counter(_weighted().sample(n=300, weights="w", seed=seed).to_pydict()["cls"])
    assert sum(got.values()) == 300
    for cls, share in ((0, 0.1), (1, 0.2), (2, 0.7)):
        expected = 300 * share
        sigma = math.sqrt(300 * share * (1 - share))
        assert abs(got[cls] - expected) < 5 * sigma, (cls, got)


def test_a_weighted_sample_is_deterministic_for_a_seed() -> None:
    a = sorted(_weighted().sample(n=50, weights="w", seed=11).to_pydict()["row"])
    b = sorted(_weighted().sample(n=50, weights="w", seed=11).to_pydict()["row"])
    c = sorted(_weighted().sample(n=50, weights="w", seed=12).to_pydict()["row"])
    assert a == b
    assert a != c


def test_null_and_zero_weights_are_never_chosen() -> None:
    ds = bt.from_pydict({"x": [1, 2, 3, 4, 5], "w": [0.0, None, 3.0, 0.0, 1.0]})
    for seed in range(5):
        assert sorted(ds.sample(n=5, weights="w", seed=seed).to_pydict()["x"]) == [3, 5]


def test_integer_weights_and_an_empty_input() -> None:
    ds = bt.from_pydict({"x": [1, 2, 3], "w": [1, 0, 2]})
    assert sorted(ds.sample(n=2, weights="w", seed=4).to_pydict()["x"]) == [1, 3]
    assert ds.filter(bt.col("x") > 9).sample(n=2, weights="w").count() == 0


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"x": [1, 2], "w": [-1.0, 2.0]}, "negative weight"),
        ({"x": [1, 2], "w": [0.0, 0.0]}, "total zero"),
        ({"x": [1, 2], "w": ["a", "b"]}, "must be numeric"),
    ],
)
def test_bad_weights_are_refused(data: dict, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        bt.from_pydict(data).sample(n=1, weights="w", seed=1)


def test_weights_need_a_count_and_a_known_column() -> None:
    ds = bt.from_pydict({"x": [1, 2], "w": [1.0, 2.0]})
    with pytest.raises(PlanError, match="needs a row count"):
        ds.sample(0.5, weights="w")
    with pytest.raises(PlanError, match="unknown column 'nope'"):
        ds.sample(n=1, weights="nope")


def _events() -> bt.Dataset:
    return bt.from_pydict({"user": [i % 200 for i in range(4_000)], "v": list(range(4_000))})


def test_a_keyed_sample_keeps_whole_keys() -> None:
    kept = _events().sample(0.3, key="user", seed=5)
    sizes = kept.group_by("user").agg(n=bt.count()).to_pydict()["n"]
    assert sizes, "the sample kept no key at all"
    assert set(sizes) == {20}  # every kept user keeps all 20 of its rows
    assert 30 <= len(sizes) <= 90  # 30% of 200 keys, binomial


def test_a_keyed_sample_ignores_non_key_columns() -> None:
    before = set(_events().sample(0.3, key="user", seed=5).to_pydict()["user"])
    changed = _events().with_columns(v=bt.col("v") * 7 + 1)
    after = set(changed.sample(0.3, key="user", seed=5).to_pydict()["user"])
    assert before == after


def test_keyed_fraction_bounds_and_composite_keys() -> None:
    ds = _events().with_columns(day=bt.col("v") % 3)
    assert ds.sample(0.0, key="user", seed=1).count() == 0
    assert ds.sample(1.0, key=["user", "day"], seed=1).count() == 4_000
    with pytest.raises(PlanError, match="takes a fraction, not n"):
        ds.sample(n=3, key="user")
    with pytest.raises(PlanError, match="unknown column"):
        ds.sample(0.5, key="nope")
