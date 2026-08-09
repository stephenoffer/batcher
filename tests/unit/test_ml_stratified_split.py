"""`stratified_split`, and `train_test_split(stratify=...)` on top of it.

A hash split is proportional in expectation and nothing more. On 200 rows with ten positives
and a quarter held out, the test half should get two or three; across six seeds it gets
between one and four. One positive in the test half makes precision, recall and AUC
meaningless, and nothing anywhere says so - the split succeeded, the model fitted, the number
came back.

The first test here pins that variance so the rest are not defending a fix to a problem
nobody demonstrated.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml.splitting import stratified_split

pytestmark = pytest.mark.unit

SEEDS = range(6)


def _imbalanced(n: int = 200, every: int = 20) -> bt.Dataset:
    """`n` rows with one positive in every `every`, so positives are rare."""
    return bt.from_pydict(
        {
            "x": [float(i) for i in range(n)],
            "y": [1 if i % every == 0 else 0 for i in range(n)],
        }
    )


def test_the_unstratified_split_really_does_vary_by_seed() -> None:
    """The premise. Without it the tests below assert a fix to nothing."""
    ds = _imbalanced()
    counts = {
        sum(ds.ml.train_test_split(test_size=0.25, seed=seed)[1].to_pydict()["y"]) for seed in SEEDS
    }
    assert len(counts) > 1, f"expected the hash split to vary across seeds, got {counts}"


def test_the_stratified_split_does_not_vary_by_seed() -> None:
    ds = _imbalanced()
    counts = {
        sum(stratified_split(ds, "y", test_size=0.25, seed=seed)[1].to_pydict()["y"])
        for seed in SEEDS
    }
    assert counts == {3}, f"the positive count should be fixed by the cut, got {counts}"


@pytest.mark.parametrize("test_size", [0.2, 0.25, 0.33, 0.5])
def test_the_test_fraction_is_met(test_size: float) -> None:
    ds = _imbalanced(n=300, every=10)
    train, test = stratified_split(ds, "y", test_size=test_size)
    assert test.count() / 300 == pytest.approx(test_size, abs=0.05)
    assert train.count() + test.count() == 300


@pytest.mark.parametrize("test_size", [0.2, 0.25, 0.33, 0.5])
def test_each_label_keeps_its_share(test_size: float) -> None:
    ds = _imbalanced(n=300, every=10)
    _, test = stratified_split(ds, "y", test_size=test_size)
    held = test.to_pydict()["y"]
    assert sum(held) / 30 == pytest.approx(test_size, abs=0.07), "positive share"
    assert (len(held) - sum(held)) / 270 == pytest.approx(test_size, abs=0.06), "negative share"


def test_the_halves_are_disjoint_and_cover_every_row() -> None:
    ds = _imbalanced()
    train, test = stratified_split(ds, "y", test_size=0.25)
    left = set(train.to_pydict()["x"])
    right = set(test.to_pydict()["x"])
    assert left & right == set()
    assert len(left | right) == 200


def test_a_rare_class_still_reaches_both_halves() -> None:
    """Three rows of a class, and neither half may be left without one."""
    ds = bt.from_pydict(
        {
            "x": [float(i) for i in range(103)],
            "y": ["rare" if i < 3 else "common" for i in range(103)],
        }
    )
    train, test = stratified_split(ds, "y", test_size=0.25)
    assert "rare" in train.to_pydict()["y"]
    assert "rare" in test.to_pydict()["y"]


def test_a_single_row_class_goes_to_train() -> None:
    """It cannot be in both, and a model that never saw the class is the worse outcome."""
    ds = bt.from_pydict(
        {"x": [float(i) for i in range(51)], "y": ["only" if i == 0 else "rest" for i in range(51)]}
    )
    train, test = stratified_split(ds, "y", test_size=0.25)
    assert "only" in train.to_pydict()["y"]
    assert "only" not in test.to_pydict()["y"]


def test_the_split_does_not_depend_on_row_order() -> None:
    """Content-hash ordering, so a repartitioned or reordered input splits identically."""
    ds = _imbalanced()
    table = ds.to_pydict()
    order = list(range(200))[::-1]
    shuffled = bt.from_pydict({k: [v[i] for i in order] for k, v in table.items()})
    _, first = stratified_split(ds, "y", test_size=0.25)
    _, second = stratified_split(shuffled, "y", test_size=0.25)
    assert set(first.to_pydict()["x"]) == set(second.to_pydict()["x"])


def test_a_union_of_partitions_splits_the_same_way() -> None:
    ds = _imbalanced()
    table = ds.to_pydict()
    left = bt.from_pydict({k: v[:100] for k, v in table.items()})
    right = bt.from_pydict({k: v[100:] for k, v in table.items()})
    _, whole = stratified_split(ds, "y", test_size=0.25)
    _, parted = stratified_split(left.union(right), "y", test_size=0.25)
    assert set(whole.to_pydict()["x"]) == set(parted.to_pydict()["x"])


def test_the_key_argument_keeps_the_split_stable_across_a_new_column() -> None:
    """Hashing the identifier means recomputing a feature does not reshuffle the halves."""
    ds = _imbalanced()
    _, before = stratified_split(ds, "y", test_size=0.25, key="x")
    widened = ds.with_columns(extra=bt.col("x") * bt.lit(3.0))
    _, after = stratified_split(widened, "y", test_size=0.25, key="x")
    assert set(before.to_pydict()["x"]) == set(after.to_pydict()["x"])


def test_a_string_label_stratifies_too() -> None:
    ds = bt.from_pydict(
        {
            "x": [float(i) for i in range(120)],
            "y": ["a" if i % 3 == 0 else ("b" if i % 3 == 1 else "c") for i in range(120)],
        }
    )
    _, test = stratified_split(ds, "y", test_size=0.25)
    held = test.to_pydict()["y"]
    assert set(held) == {"a", "b", "c"}
    for label in "abc":
        assert held.count(label) == pytest.approx(10, abs=4)


# --------------------------------------------------------------------------------------
# Through the Dataset surface
# --------------------------------------------------------------------------------------


def test_the_dataset_method_stratifies_when_asked() -> None:
    ds = _imbalanced()
    counts = {
        sum(ds.ml.train_test_split(test_size=0.25, seed=seed, stratify="y")[1].to_pydict()["y"])
        for seed in SEEDS
    }
    assert counts == {3}


def test_the_dataset_method_is_unchanged_without_stratify() -> None:
    """The default path must stay the plain row-wise filter it was."""
    ds = _imbalanced()
    train, test = ds.ml.train_test_split(test_size=0.25, seed=0)
    assert train.count() + test.count() == 200


def test_the_dataset_method_passes_the_key_through() -> None:
    ds = _imbalanced()
    _, first = ds.ml.train_test_split(test_size=0.25, stratify="y", key="x")
    _, second = ds.with_columns(extra=bt.lit(1.0)).ml.train_test_split(
        test_size=0.25, stratify="y", key="x"
    )
    assert set(first.to_pydict()["x"]) == set(second.to_pydict()["x"])


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_a_test_size_outside_the_unit_interval_is_rejected(value: float) -> None:
    with pytest.raises(PlanError, match="test_size"):
        stratified_split(_imbalanced(), "y", test_size=value)


def test_a_missing_label_is_named() -> None:
    with pytest.raises(ColumnNotFoundError):
        stratified_split(_imbalanced(), "nope")
