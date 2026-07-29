"""Scaling numeric features, and why the choice of scaler matters.

Every scaler follows the ``fit`` / ``transform`` split for a reason: the statistics come
from the training set and are then *applied* to validation and production data. Fitting on
everything is the classic leak, and the API makes the correct thing the easy thing.

    python examples/ml/preprocessing_scaling.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    train = bt.from_pydict({"x": [0.0, 5.0, 10.0, 5.0], "y": [1.0, 2.0, 3.0, 100.0]})
    holdout = bt.from_pydict({"x": [2.5, 7.5], "y": [1.5, 2.5]})

    # Zero mean, unit variance.
    std = ml.StandardScaler("x").fit(train)
    scaled = std.transform(train).to_pydict()
    print("standard:", scaled["x"])
    assert abs(sum(scaled["x"]) / 4) < 1e-9

    # The fitted statistics carry over to unseen data -- no refitting.
    applied = std.transform(holdout).to_pydict()
    print("applied to holdout:", applied["x"])
    assert len(applied["x"]) == 2

    # Squash into a fixed range.
    mm = ml.MinMaxScaler("x").fit(train).transform(train).to_pydict()
    assert min(mm["x"]) == 0.0 and max(mm["x"]) == 1.0
    custom = ml.MinMaxScaler("x", feature_range=(-1.0, 1.0)).fit(train).transform(train).to_pydict()
    assert min(custom["x"]) == -1.0 and max(custom["x"]) == 1.0

    # Robust scaling uses the median and IQR, so the 100.0 outlier barely moves it.
    rob = ml.RobustScaler("y").fit(train).transform(train).to_pydict()
    std_y = ml.StandardScaler("y").fit(train).transform(train).to_pydict()
    print("robust y:", rob["y"])
    print("standard y:", std_y["y"])
    # The three clustered values stay close together under robust scaling.
    assert max(rob["y"][:3]) - min(rob["y"][:3]) > max(std_y["y"][:3]) - min(std_y["y"][:3])

    # Divide by the largest magnitude, preserving sign and sparsity.
    ma = ml.MaxAbsScaler("y").fit(train).transform(train).to_pydict()
    assert max(abs(v) for v in ma["y"]) == 1.0

    # Normalizer works across a row's columns rather than down a column.
    norm = ml.Normalizer(["x", "y"], norm="l2").fit(train).transform(train).to_pydict()
    lengths = [(a * a + b * b) ** 0.5 for a, b in zip(norm["x"], norm["y"], strict=True)]
    print("row norms:", lengths)
    assert all(abs(v - 1.0) < 1e-9 for v in lengths)


if __name__ == "__main__":
    main()
