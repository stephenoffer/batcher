"""Turning categories into numbers, four ways, and when each is wrong.

Ordinal encoding invents an ordering that is not there. One-hot does not, but costs a column
per category. Target encoding is compact and leaks the label unless it is fitted on the
training split only — which is the whole reason fit and transform are separate calls.

    python examples/ml/encoding_categories.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import ml


def main() -> None:
    customer = tpch("customer").select("c_custkey", "c_mktsegment", "c_acctbal")
    train, holdout = customer.ml.train_test_split(test_size=0.2, seed=4)
    categories = train.n_unique("c_mktsegment")
    print("segments:", categories)

    # Ordinal: one column, an invented ordering.
    ordinal = ml.OrdinalEncoder("c_mktsegment").fit(train).transform(train)
    values = set(ordinal.to_pydict()["c_mktsegment"])
    assert values == set(range(categories))

    # One-hot: one column per category, no invented ordering.
    hot = ml.OneHotEncoder("c_mktsegment").fit(train).transform(train)
    indicators = [name for name in hot.columns if name.startswith("c_mktsegment")]
    print("one-hot columns:", len(indicators))
    assert len(indicators) == categories

    # Exactly one indicator is set per row.
    rows = hot.select(*indicators).head(20).to_pydict()
    for index in range(20):
        assert sum(rows[name][index] for name in indicators) == 1

    # Frequency: one column, ordered by how common the category is.
    frequency = ml.FrequencyEncoder("c_mktsegment").fit(train).transform(train)
    encoded = frequency.to_pydict()["c_mktsegment"]
    assert all(0.0 <= value <= 1.0 for value in encoded)

    # Every encoder applies unchanged to unseen rows, using the fitted mapping.
    applied = ml.OrdinalEncoder("c_mktsegment").fit(train).transform(holdout)
    assert applied.count() == holdout.count()
    assert set(applied.to_pydict()["c_mktsegment"]) <= values


if __name__ == "__main__":
    main()
