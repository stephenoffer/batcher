"""Class imbalance: measure it, then resample or reweight.

Resampling changes the data; weighting changes the loss. Prefer weights when the model
supports them, because oversampling duplicates rows (and any leakage in them) while
undersampling throws information away.

    python examples/ml/imbalance_and_sampling.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    # 18 negatives to 6 positives: a 3:1 imbalance.
    data = bt.from_pydict(
        {
            "x": list(range(24)),
            "label": [0] * 18 + [1] * 6,
        }
    )

    # Measure first.
    counts = ml.class_counts(data, "label")
    print("counts:", counts)
    assert counts == {0: 18, 1: 6}

    weights = ml.class_weights(data, "label")
    print("weights:", weights)
    # The rare class carries the larger weight.
    assert weights[1] > weights[0]

    # Attach a per-row weight column for a weighted loss.
    weighted = ml.sample_weights(data, "label").to_pydict()
    assert "sample_weight" in weighted
    pos = [w for w, y in zip(weighted["sample_weight"], weighted["label"], strict=True) if y == 1]
    neg = [w for w, y in zip(weighted["sample_weight"], weighted["label"], strict=True) if y == 0]
    assert pos[0] > neg[0]

    # Resampling, all seeded so a run is reproducible.
    over = ml.oversample(data, "label", seed=0)
    over_counts = ml.class_counts(over, "label")
    print("oversampled:", over_counts)
    assert over_counts[1] == over_counts[0] == 18

    under = ml.undersample(data, "label", seed=0)
    under_counts = ml.class_counts(under, "label")
    print("undersampled:", under_counts)
    assert under_counts[0] == under_counts[1] == 6

    balanced = ml.balanced_sample(data, "label", seed=0)
    bc = ml.class_counts(balanced, "label")
    assert bc[0] == bc[1]

    # Stratified sampling keeps the original proportions in a smaller sample.
    strat = ml.stratified_sample(data, "label", fraction=0.5, seed=0)
    sc = ml.class_counts(strat, "label")
    print("stratified:", sc)
    assert sum(sc.values()) < 24
    # The 3:1 ratio survives, unlike a naive random sample.
    assert sc[0] > sc[1]


if __name__ == "__main__":
    main()
