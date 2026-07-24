"""Internal clustering-quality scores — how good a clustering is with no reference labeling.

The agreement metrics in `clustering` need a ground-truth labeling to compare against. These
score a clustering on its own geometry instead: a good clustering has compact clusters that sit
far apart, and both measures here turn that into a number, so they work when no true labels exist
(the usual case) and are what an elbow-style search over the cluster count optimizes.

Both are built from per-cluster aggregates — a centroid, a count, a dispersion — so they cost a
scan or two, not the pairwise blow-up a silhouette needs. Each is checked against scikit-learn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["calinski_harabasz_score", "davies_bouldin_score"]


def calinski_harabasz_score(ds: Dataset, columns: Sequence[str], labels: str) -> float:
    """The Calinski-Harabasz score — between-cluster over within-cluster dispersion.

    The variance-ratio criterion: the spread *between* cluster centroids divided by the spread
    *within* clusters, scaled by the degrees of freedom. Higher is better — compact, well-separated
    clusters — so it is what to maximize when choosing the number of clusters. Cheap, because both
    dispersions are group-wise sums of squares.

    Args:
        ds: The dataset holding the feature columns and the cluster label.
        columns: The numeric feature columns the clustering was built on.
        labels: The cluster-label column.

    Returns:
        The Calinski-Harabasz score; higher is a better-separated clustering.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import calinski_harabasz_score
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 0.1, 9.0, 9.1], "y": [0.0, 0.1, 9.0, 9.1], "c": [0, 0, 1, 1]}
            ... )
            >>> calinski_harabasz_score(ds, ["x", "y"], "c") > 100
            True
    """
    import numpy as np

    from batcher.plan.functions.aggregate import mean as mean_
    from batcher.plan.functions.statistics import var_pop

    require_columns(ds, *columns, labels)
    names = list(columns)
    grand_row = ds.agg(**{f"g{i}": mean_(col(name)) for i, name in enumerate(names)}).collect()
    grand = np.array([float(grand_row.column(f"g{i}")[0].as_py()) for i in range(len(names))])
    aggregates: dict[str, object] = {"__bt_n": col(labels).count()}
    for i, name in enumerate(names):
        aggregates[f"m{i}"] = mean_(col(name))
        aggregates[f"v{i}"] = var_pop(col(name))
    grouped = ds.group_by(labels).agg(**aggregates).collect()
    n = 0
    ss_between = 0.0
    ss_within = 0.0
    for r in range(grouped.num_rows):
        count = int(grouped.column("__bt_n")[r].as_py())
        n += count
        centroid = np.array([float(grouped.column(f"m{i}")[r].as_py()) for i in range(len(names))])
        ss_between += count * float(np.sum((centroid - grand) ** 2))
        ss_within += count * sum(
            float(grouped.column(f"v{i}")[r].as_py() or 0.0) for i in range(len(names))
        )
    k = grouped.num_rows
    if k <= 1 or ss_within == 0.0:
        return float("nan")
    return (ss_between / ss_within) * (n - k) / (k - 1)


def davies_bouldin_score(ds: Dataset, columns: Sequence[str], labels: str) -> float:
    """The Davies-Bouldin score — the average worst-case cluster overlap.

    For each cluster it finds the other cluster it is most confusable with — the one maximizing
    ``(spread_i + spread_j) / distance(centroid_i, centroid_j)`` — and averages that worst case
    over all clusters. Lower is better: 0 would mean every cluster is a point infinitely far from
    the rest. Unlike `calinski_harabasz_score` it is a ratio of distances rather than variances, so
    it responds differently to elongated clusters, and the two are worth reading together.

    Args:
        ds: The dataset holding the feature columns and the cluster label.
        columns: The numeric feature columns the clustering was built on.
        labels: The cluster-label column.

    Returns:
        The Davies-Bouldin score; lower is a better-separated clustering.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import davies_bouldin_score
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 0.1, 9.0, 9.1], "y": [0.0, 0.1, 9.0, 9.1], "c": [0, 0, 1, 1]}
            ... )
            >>> davies_bouldin_score(ds, ["x", "y"], "c") < 0.1
            True
    """
    import numpy as np

    from batcher.plan.functions.aggregate import mean as mean_

    require_columns(ds, *columns, labels)
    names = list(columns)
    centroid_row = (
        ds.group_by(labels).agg(**{name: mean_(col(name)) for name in names}).sort(labels).collect()
    )
    cluster_labels = [centroid_row.column(labels)[r].as_py() for r in range(centroid_row.num_rows)]
    centroids = {
        cluster_labels[r]: np.array([float(centroid_row.column(name)[r].as_py()) for name in names])
        for r in range(centroid_row.num_rows)
    }
    # Second pass: the mean distance of each cluster's rows to its own centroid.
    distance = lit(0.0)
    for label, centroid in centroids.items():
        squared = lit(0.0)
        for i, name in enumerate(names):
            delta = col(name) - lit(float(centroid[i]))
            squared = squared + delta * delta
        distance = when(col(labels) == lit(label)).then(squared.sqrt()).otherwise(distance)
    spread_row = (
        ds.with_columns(__bt_d=distance)
        .group_by(labels)
        .agg(__bt_s=col("__bt_d").mean())
        .sort(labels)
        .collect()
    )
    spread = {
        spread_row.column(labels)[r].as_py(): float(spread_row.column("__bt_s")[r].as_py())
        for r in range(spread_row.num_rows)
    }
    k = len(cluster_labels)
    if k <= 1:
        return float("nan")
    total = 0.0
    for i in cluster_labels:
        worst = 0.0
        for j in cluster_labels:
            if j == i:
                continue
            separation = float(np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2)))
            if separation > 0:
                worst = max(worst, (spread[i] + spread[j]) / separation)
        total += worst
    return total / k
