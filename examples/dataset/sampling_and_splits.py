"""Sampling and splitting: reproducible subsets that do not leak.

Every one of these takes a seed, because an unseeded split is a split you cannot reproduce
when the result looks wrong. ``stratified_split`` preserves class balance; a plain random
split does not, and on an imbalanced problem that matters.

    python examples/dataset/sampling_and_splits.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def main() -> None:
    data = bt.from_pydict(
        {
            "id": list(range(100)),
            "label": [0] * 80 + [1] * 20,
            "grp": [f"g{i % 10}" for i in range(100)],
        }
    )

    # A fraction of the rows, reproducibly.
    sampled = data.sample(fraction=0.2, seed=0)
    n = sampled.count()
    print("sampled rows:", n)
    assert 5 <= n <= 40
    # The same seed gives the same rows.
    again = data.sample(fraction=0.2, seed=0).to_pydict()["id"]
    assert again == sampled.to_pydict()["id"]

    # A different seed gives a different sample.
    other = data.sample(fraction=0.2, seed=1).to_pydict()["id"]
    assert other != again

    # Train/validation/test in one call. The first argument is the *key* the split is
    # grouped by, so every row sharing a key lands in the same split -- that is what
    # stops a group leaking across the boundary.
    train, val, test = data.train_val_test_split("id", 0.2, 0.2, seed=0)
    sizes = (train.count(), val.count(), test.count())
    print("split sizes:", sizes)
    assert sum(sizes) == 100
    # The splits are disjoint.
    ids = set(train.to_pydict()["id"]) | set(val.to_pydict()["id"]) | set(test.to_pydict()["id"])
    assert len(ids) == 100

    # Stratified: the label proportion survives the split. `test_size` is the holdout.
    s_train, s_test = data.stratified_split("label", 0.25, seed=0)

    def positive_rate(ds: bt.Dataset) -> float:
        d = ds.to_pydict()
        return sum(d["label"]) / len(d["label"])

    print("rates:", positive_rate(s_train), positive_rate(s_test))
    assert abs(positive_rate(s_train) - 0.2) < 0.1
    assert abs(positive_rate(s_test) - 0.2) < 0.15

    # Per-group sampling, for a balanced panel.
    per_group = data.sample_per_group("grp", n=2).to_pydict()
    print("per-group rows:", len(per_group["id"]))
    assert len(per_group["id"]) == 20  # 10 groups x 2

    # Class balancing as a Dataset verb.
    balanced = data.balance_classes("label").to_pydict()
    counts = {c: balanced["label"].count(c) for c in set(balanced["label"])}
    print("balanced:", counts)
    assert counts[0] == counts[1]

    # A shuffle, and a deterministic random column for custom splits.
    shuffled = data.shuffle(seed=0).to_pydict()["id"]
    assert shuffled != list(range(100))
    assert sorted(shuffled) == list(range(100))

    with_rand = data.with_random("r", seed=0).to_pydict()
    assert all(0.0 <= v <= 1.0 for v in with_rand["r"])
    # Which makes a hand-rolled split trivial and reproducible.
    manual = data.with_random("r", seed=0).filter(col("r") < 0.5).count()
    assert 0 < manual < 100


if __name__ == "__main__":
    main()
