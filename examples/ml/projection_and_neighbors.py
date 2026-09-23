"""Project correlated features with PCA or TruncatedSVD, then predict with k nearest neighbours.

k-NN measures distance across every feature alike, so redundant columns count twice and noise
columns blur every neighbourhood. Projecting onto a few principal components first keeps the
directions that carry the variance, which is where the neighbours are worth measuring.

Both steps are expressions over the feature columns: the projection is a fitted linear map,
and a k-NN prediction folds the (bounded) reference set in as literals, so scoring is one
projection with no join and no per-row Python.

    python examples/ml/projection_and_neighbors.py
"""

from __future__ import annotations

import math

import numpy as np

import batcher as bt
from batcher import ml


def _frame(points: np.ndarray, **extra: list) -> bt.Dataset:
    columns = {f"f{i}": points[:, i].tolist() for i in range(points.shape[1])}
    return bt.from_pydict(columns | extra)


def main() -> None:
    rng = np.random.default_rng(7)
    # Two informative directions, copied into four more correlated columns plus noise.
    latent = rng.normal(size=(160, 2))
    mixing = rng.normal(size=(2, 6))
    points = latent @ mixing + rng.normal(scale=0.05, size=(160, 6))
    target = latent[:, 0] * 3.0 - latent[:, 1]
    label = np.where(target > 0, "up", "down")
    train_rows, test_rows = slice(0, 120), slice(120, 160)
    names = [f"f{i}" for i in range(6)]
    train = _frame(
        points[train_rows], y=target[train_rows].tolist(), label=label[train_rows].tolist()
    )
    test = _frame(points[test_rows], y=target[test_rows].tolist(), label=label[test_rows].tolist())

    # PCA: two components hold almost all of the variance of six mixed columns.
    pca = ml.PCA(names, n_components=2).fit(train)
    print("explained variance ratio", [round(r, 4) for r in pca.explained_variance_ratio_])
    assert sum(pca.explained_variance_ratio_) > 0.99
    projected_train = pca.transform(train)
    projected_test = pca.transform(test)
    assert {"pc1", "pc2"} <= set(projected_train.columns)
    assert not set(names) & set(projected_train.columns)  # the originals are replaced

    # TruncatedSVD does not center, so it suits sparse or count-like inputs; on this dense
    # data it finds the same two-dimensional subspace.
    svd = ml.TruncatedSVD(names, n_components=2).fit(train)
    assert sum(svd.explained_variance_ratio_) > 0.95

    # Regression on the projected features. weights="distance" weights each of the k
    # neighbours by 1 / Euclidean distance, as scikit-learn does.
    regressor = ml.KNeighborsRegressor(["pc1", "pc2"], "y", k=5, weights="distance")
    predicted = regressor.fit(projected_train).predict(projected_test).to_pydict()
    errors = [p - t for p, t in zip(predicted["prediction"], predicted["y"], strict=True)]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    print(f"k-NN regression RMSE on held-out rows: {rmse:.3f} (target std {target.std():.3f})")
    assert rmse < 0.5 * float(target.std())

    # Classification: the nearest five vote, a closer row counting for more.
    classifier = ml.KNeighborsClassifier(["pc1", "pc2"], "label", k=5, weights="distance")
    voted = classifier.fit(projected_train).predict(projected_test).to_pydict()
    accuracy = sum(p == t for p, t in zip(voted["prediction"], voted["label"], strict=True)) / len(
        voted["label"]
    )
    print(f"k-NN classification accuracy: {accuracy:.2f}", "classes", classifier.classes_)
    assert classifier.classes_ == ["down", "up"]
    assert accuracy >= 0.85

    # A query that coincides with a training row takes that row's target exactly.
    first = projected_train.limit(1)
    exact = regressor.predict(first).to_pydict()
    assert math.isclose(exact["prediction"][0], exact["y"][0], rel_tol=1e-9)


if __name__ == "__main__":
    main()
