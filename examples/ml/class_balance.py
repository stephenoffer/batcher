"""Measuring and correcting class imbalance.

Resampling changes the prior the model learns, so it changes what a predicted probability
means. That is fine as long as it is deliberate — the failure mode is rebalancing the
training set and then reading the probabilities as if they were calibrated.

    python examples/ml/class_balance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col, ml


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice", "o_orderstatus")

    balance = orders.class_balance("o_orderstatus").to_pydict()
    print(balance)
    counts = orders.value_counts("o_orderstatus").to_pydict()
    assert sum(counts["count"]) == orders.count()

    # `class_weights` returns a Dataset of (class, weight), not a dict.
    weight_rows = orders.class_weights("o_orderstatus").to_pydict()
    weights = dict(zip(weight_rows["o_orderstatus"], weight_rows["weight"], strict=True))
    print("class weights:", weights)
    # A rarer class gets a larger weight.
    by_count = dict(zip(counts["o_orderstatus"], counts["count"], strict=True))
    rarest = min(by_count, key=lambda key: by_count[key])
    commonest = max(by_count, key=lambda key: by_count[key])
    assert weights[rarest] > weights[commonest]

    # Resampling to an even split.
    # Deterministic by construction: it undersamples to the smallest class in a
    # defined order rather than drawing at random, so it needs no seed.
    balanced = orders.balance_classes("o_orderstatus", order_by="o_orderkey")
    after = balanced.value_counts("o_orderstatus").to_pydict()
    print("after balancing:", after)
    spread = max(after["count"]) - min(after["count"])
    assert spread <= 1

    # Balancing changes the prior, which is the thing to be deliberate about.
    before_share = by_count[rarest] / orders.count()
    after_share = (
        dict(zip(after["o_orderstatus"], after["count"], strict=True))[rarest] / balanced.count()
    )
    print(f"{rarest}: {before_share:.4f} -> {after_share:.4f}")
    assert after_share > before_share

    # A stratified sample keeps the prior instead, which is what you want for evaluation.
    sample = ml.stratified_sample(orders, "o_orderstatus", 0.1, seed=5)
    sampled = sample.value_counts("o_orderstatus").to_pydict()
    sampled_share = (
        dict(zip(sampled["o_orderstatus"], sampled["count"], strict=True))[rarest] / sample.count()
    )
    assert abs(sampled_share - before_share) < 0.02
    assert bt is not None
    assert col is not None


if __name__ == "__main__":
    main()
