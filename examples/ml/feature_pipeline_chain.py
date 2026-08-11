"""Composing preprocessors into one fitted Chain.

A `Chain` fits its stages in order, each on the output of the last, and then applies the
whole thing as a unit. That matters because it is the only way to be sure the exact same
transformations, with the exact same fitted statistics, reach production.

    python examples/ml/feature_pipeline_chain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col, ml


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_acctbal", "c_nationkey", "c_mktsegment")

    train, holdout = customer.ml.train_test_split(test_size=0.2, seed=11)

    chain = ml.Chain(
        ml.SimpleImputer("c_acctbal", strategy="median"),
        ml.StandardScaler("c_acctbal"),
        ml.OneHotEncoder("c_mktsegment"),
    ).fit(train)

    transformed = chain.transform(train)
    print(transformed.columns)

    # The scaler centred the training column.
    mean = transformed.agg(m=col("c_acctbal").mean()).to_pydict()["m"][0]
    print("training mean after scaling:", round(mean, 9))
    assert abs(mean) < 1e-6

    # One-hot expanded the segment into indicator columns.
    indicators = [name for name in transformed.columns if name.startswith("c_mktsegment")]
    print("indicator columns:", indicators)
    assert len(indicators) == train.n_unique("c_mktsegment")

    # The same fitted chain applies to unseen rows, and does *not* re-fit: the holdout
    # mean is near zero but not exactly zero, which is the proof it used the training
    # statistics rather than its own.
    applied = chain.transform(holdout)
    assert applied.columns == transformed.columns
    holdout_mean = applied.agg(m=col("c_acctbal").mean()).to_pydict()["m"][0]
    print("holdout mean after scaling:", round(holdout_mean, 6))
    assert holdout_mean != 0.0
    assert abs(holdout_mean) < 0.2


if __name__ == "__main__":
    main()
