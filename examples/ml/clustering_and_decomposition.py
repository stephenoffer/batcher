"""Unsupervised models: KMeans, Gaussian mixtures, PCA, and truncated SVD.

Clustering appends a label column; decomposition appends component columns. Both are
transformations of the Dataset, so the result composes with everything else -- you can
cluster, then group by the cluster, in one chain.

    python examples/ml/clustering_and_decomposition.py
"""

from __future__ import annotations

import batcher as bt
from batcher import ml


def main() -> None:
    # Two obvious blobs.
    points = bt.from_pydict(
        {
            "x": [0.0, 0.2, 0.1, 0.3, 10.0, 10.2, 9.8, 10.1],
            "y": [0.0, 0.1, 0.3, 0.2, 10.0, 9.9, 10.3, 10.1],
        }
    )

    # KMeans: `seed` makes the run reproducible.
    km = ml.KMeans(["x", "y"], n_clusters=2, seed=0).fit(points)
    labelled = km.predict(points).to_pydict()
    print(labelled["cluster"])

    assert set(labelled["cluster"]) == {0, 1}
    # The two blobs land in different clusters.
    assert len(set(labelled["cluster"][:4])) == 1
    assert len(set(labelled["cluster"][4:])) == 1
    assert labelled["cluster"][0] != labelled["cluster"][4]

    # Cluster, then aggregate per cluster, in one chain.
    per_cluster = (
        km.predict(points)
        .group_by("cluster")
        .agg(n=bt.count(), cx=bt.mean("x"))
        .sort("cluster")
        .to_pydict()
    )
    print(per_cluster)
    assert per_cluster["n"] == [4, 4]

    # A Gaussian mixture gives soft components rather than hard assignments.
    gm = ml.GaussianMixture(["x", "y"], n_components=2, seed=0).fit(points)
    comps = gm.predict(points).to_pydict()
    assert len(set(comps["component"])) == 2

    # PCA is a *preprocessor*, so it exposes `transform`/`fit_transform` rather than
    # the `predict` an estimator has.
    pca = ml.PCA(["x", "y"], n_components=2).fit(points)
    print("explained variance ratio:", pca.explained_variance_ratio_)
    projected = pca.transform(points).to_pydict()
    print(sorted(projected))
    # Component columns are 1-based: `pc1`, `pc2`, ...
    assert "pc1" in projected and "pc2" in projected
    # The blobs separate along the first component.
    assert abs(projected["pc1"][0] - projected["pc1"][4]) > 1.0

    # Keep the source columns alongside the components when you still need them.
    kept = ml.PCA(["x", "y"], n_components=1, keep_original=True).fit_transform(points)
    cols = kept.to_pydict()
    assert "x" in cols and "pc1" in cols

    # Truncated SVD is the same idea without mean-centering, for sparse-ish data.
    svd = ml.TruncatedSVD(["x", "y"], n_components=2).fit(points)
    sv = svd.transform(points).to_pydict()
    assert "svd1" in sv and "svd2" in sv


if __name__ == "__main__":
    main()
