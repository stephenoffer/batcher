"""Training-serving parity: the same transformations over one row and over millions.

A fitted chain applied to a single row must produce what it produced for that row during
training. That is the property that breaks when the serving path reimplements the features,
and it is checkable in three lines.

    python examples/ml/pipeline_serving_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col, ml


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_acctbal", "c_nationkey", "c_mktsegment")
    train, holdout = customer.ml.train_test_split(test_size=0.2, seed=21)

    chain = ml.Chain(
        ml.SimpleImputer("c_acctbal", strategy="median"),
        ml.StandardScaler("c_acctbal"),
        ml.OrdinalEncoder("c_mktsegment"),
    ).fit(train)

    # Batch: the whole holdout at once.
    batch = chain.transform(holdout).sort("c_custkey")
    batch_rows = batch.to_pydict()
    print("batch rows:", batch.count())

    # Serving: one row at a time, through the same fitted chain.
    first_key = batch_rows["c_custkey"][0]
    one_row = holdout.filter(col("c_custkey") == first_key)
    assert one_row.count() == 1

    served = chain.transform(one_row).to_pydict()
    print("served:", served)

    # The single-row result must equal the batch result for that row.
    assert abs(served["c_acctbal"][0] - batch_rows["c_acctbal"][0]) < 1e-12
    assert served["c_mktsegment"][0] == batch_rows["c_mktsegment"][0]

    # And for a handful more, so it is not one lucky row.
    for key in batch_rows["c_custkey"][:5]:
        index = batch_rows["c_custkey"].index(key)
        single = chain.transform(holdout.filter(col("c_custkey") == key)).to_pydict()
        assert abs(single["c_acctbal"][0] - batch_rows["c_acctbal"][index]) < 1e-12

    # The chain uses the *training* statistics either way, which is why the single-row
    # path cannot recompute a mean from one row.
    train_mean = chain.transform(train).agg(m=col("c_acctbal").mean()).to_pydict()["m"][0]
    assert abs(train_mean) < 1e-6
    assert bt is not None


if __name__ == "__main__":
    main()
