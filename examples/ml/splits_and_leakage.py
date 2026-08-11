"""Splitting data without leaking: random, stratified, grouped, and by time.

A random split is wrong whenever rows are related. Orders from one customer must not
straddle the split, and a time series must not be shuffled at all. Each of these is a
different function because they are different questions, not different flavours.

    python examples/ml/splits_and_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select(
        "o_orderkey", "o_custkey", "o_orderdate", "o_totalprice", "o_orderstatus"
    )

    # Random: fine when rows are independent.
    train, test = orders.ml.train_test_split(test_size=0.2, seed=1)
    assert train.count() + test.count() == orders.count()
    print("random:", train.count(), test.count())

    # Three-way, when you need a validation set as well as a test set. The first
    # argument is the column to stratify on, so all three parts keep the same mix.
    a, b, c = orders.train_val_test_split("o_orderstatus", 0.15, 0.15, seed=1)
    assert a.count() + b.count() + c.count() == orders.count()
    print("three-way:", a.count(), b.count(), c.count())

    # Stratified: keep the class balance identical in both halves.
    strat_train, strat_test = orders.stratified_split("o_orderstatus", test_size=0.2, seed=1)
    train_mix = strat_train.value_counts("o_orderstatus").sort("o_orderstatus").to_pydict()
    test_mix = strat_test.value_counts("o_orderstatus").sort("o_orderstatus").to_pydict()
    train_share = [value / strat_train.count() for value in train_mix["count"]]
    test_share = [value / strat_test.count() for value in test_mix["count"]]
    print("train mix:", [round(v, 4) for v in train_share])
    print("test mix: ", [round(v, 4) for v in test_share])
    assert all(
        abs(left - right) < 0.01 for left, right in zip(train_share, test_share, strict=True)
    )

    # Time-based: everything in the test set happens after everything in the training
    # set. The cutoff comes from a sorted scan rather than a quantile, because quantile
    # aggregates are defined on numeric columns, not on Date32.
    ordered = orders.select("o_orderdate").sort("o_orderdate").to_pydict()["o_orderdate"]
    cutoff = ordered[int(len(ordered) * 0.8)]
    past = orders.filter(col("o_orderdate") <= cutoff)
    future = orders.filter(col("o_orderdate") > cutoff)
    print("time split:", past.count(), future.count())
    assert past.count() + future.count() == orders.count()
    latest_past = past.agg(m=col("o_orderdate").max()).to_pydict()["m"][0]
    earliest_future = future.agg(m=col("o_orderdate").min()).to_pydict()["m"][0]
    assert latest_past < earliest_future


if __name__ == "__main__":
    main()
