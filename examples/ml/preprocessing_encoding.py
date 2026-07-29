"""Turning categories into numbers, and picking the encoder by cardinality.

One-hot is fine for a handful of categories and catastrophic for a million user ids. The
alternatives trade information for width: ordinal keeps one column, frequency and target
encoding keep one column carrying signal, and hashing bounds the width outright.

    python examples/ml/preprocessing_encoding.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    train = bt.from_pydict(
        {
            "city": ["nyc", "sf", "nyc", "la", "sf", "nyc"],
            "converted": [1, 0, 1, 0, 0, 1],
        }
    )

    # One column per category. `drop_first` avoids the collinear dummy.
    onehot = ml.OneHotEncoder("city").fit(train).transform(train).to_pydict()
    print(sorted(onehot))
    assert len([c for c in onehot if c.startswith("city_")]) == 3
    dropped = ml.OneHotEncoder("city", drop_first=True).fit(train).transform(train).to_pydict()
    assert len([c for c in dropped if c.startswith("city_")]) == 2

    # One integer column. Unseen categories map to `unknown_value`.
    ordinal = ml.OrdinalEncoder("city").fit(train)
    assert set(ordinal.transform(train).to_pydict()["city"]) == {0, 1, 2}
    unseen = ordinal.transform(bt.from_pydict({"city": ["berlin"]})).to_pydict()
    print("unseen ->", unseen["city"])
    assert unseen["city"] == [-1]

    # How often the category occurs -- cheap and surprisingly strong.
    freq = ml.FrequencyEncoder("city").fit(train).transform(train).to_pydict()
    print("frequency:", freq["city"])
    assert abs(freq["city"][0] - 0.5) < 1e-9  # nyc is 3 of 6 rows

    # The target's mean per category, smoothed toward the global mean.
    target = ml.TargetEncoder("city", "converted", smoothing=1.0).fit(train).transform(train)
    te = target.to_pydict()
    print("target:", te["city"])
    assert te["city"][0] > te["city"][3]  # nyc converts, la does not

    # Weight of evidence, for binary targets.
    woe = ml.WOEEncoder("city", "converted", positive=1).fit(train).transform(train).to_pydict()
    assert len(woe["city"]) == 6

    # Bound the width regardless of cardinality.
    hashed = ml.HashingEncoder("city", n_buckets=4).fit(train).transform(train).to_pydict()
    assert all(0 <= v < 4 for v in hashed["city"])

    # Fold the long tail into one bucket before encoding.
    rare = ml.RareCategoryEncoder("city", min_frequency=0.3).fit(train).transform(train).to_pydict()
    print("rare-folded:", rare["city"])
    assert "__rare__" in rare["city"]

    # A single label column to integers, and to one-hot columns.
    le = ml.LabelEncoder("city").fit(train).transform(train).to_pydict()
    assert set(le["city"]) == {0, 1, 2}
    lb = ml.LabelBinarizer("city").fit(train).transform(train).to_pydict()
    assert len(lb) > 1

    # Binary encoding: log2(cardinality) columns instead of one per category.
    be = ml.BinaryEncoder("city").fit(train).transform(train).to_pydict()
    print(sorted(be))
    assert len(be) >= 2


if __name__ == "__main__":
    main()
