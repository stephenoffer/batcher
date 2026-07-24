"""Unsupervised clustering — grouping rows by similarity, with no labels.

Clustering is the unsupervised half of tabular machine learning: segment customers, quantize a
feature, find the natural groups in a dataset nobody has labeled. K-means is the workhorse, and
it maps cleanly onto the engine — each Lloyd iteration is one *assign* step (a per-row nearest-
centroid expression that lowers to Rust) and one *update* step (a `group_by` mean), so the fit
is a handful of scans and the assignment is a single streaming pass with no per-row Python.

The centroids live on the driver as a small ``k x d`` table between iterations, which is the
only state; everything touching a row is an expression or an aggregate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir.core import Expr

__all__ = ["KMeans", "NearestCentroid"]


def _squared_distance(columns: Sequence[str], centroid: Sequence[float]) -> Expr:
    """The squared Euclidean distance from each row to one centroid."""
    total: Expr | None = None
    for name, coordinate in zip(columns, centroid, strict=True):
        delta = col(name) - lit(coordinate)
        term = delta * delta
        total = term if total is None else total + term
    assert total is not None
    return total


def _assignment(columns: Sequence[str], centroids: Sequence[Sequence[float]]) -> Expr:
    """The index of the nearest centroid per row, as a nested-conditional expression."""
    distances = [_squared_distance(columns, c) for c in centroids]
    cluster: Expr = lit(0)
    best = distances[0]
    for index in range(1, len(centroids)):
        closer = distances[index] < best
        cluster = when(closer).then(lit(index)).otherwise(cluster)
        best = when(closer).then(distances[index]).otherwise(best)
    return cluster


class KMeans:
    """Partition rows into `n_clusters` groups by Lloyd's k-means algorithm.

    The standard centroid-based clusterer: it alternates assigning each row to its nearest
    centroid and moving each centroid to the mean of its rows, until the assignment stops
    changing or `max_iter` is reached. `fit` learns the centroids in a handful of scans (each
    iteration is an assignment expression plus a `group_by` mean), and `predict` labels any
    dataset in a single streaming pass. The learned `inertia_` — the total squared distance from
    each row to its centroid — is the quantity an elbow plot uses to choose `n_clusters`.

    Initialization is a seeded content-hash sample of `n_clusters` rows, so a fit is reproducible
    from its seed and identical however the data is partitioned. An empty cluster keeps its
    previous centroid rather than vanishing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.cluster import KMeans
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 0.1, 10.0, 10.1], "y": [0.0, 0.1, 10.0, 10.1]}
            ... )
            >>> km = KMeans(["x", "y"], n_clusters=2, seed=1).fit(ds)
            >>> labels = km.predict(ds).to_pydict()["cluster"]
            >>> labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
            True

    Args:
        columns: The numeric feature columns to cluster on.
        n_clusters: How many clusters to find.
        max_iter: The maximum number of Lloyd iterations.
        output_column: The name of the cluster-label column `predict` appends.
        seed: Seed for the centroid initialization.
    """

    __slots__ = (
        "centroids_",
        "columns",
        "inertia_",
        "max_iter",
        "n_clusters",
        "n_iter_",
        "output_column",
        "seed",
    )

    def __init__(
        self,
        columns: Sequence[str],
        *,
        n_clusters: int = 8,
        max_iter: int = 100,
        output_column: str = "cluster",
        seed: int = 0,
    ) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("KMeans needs at least one feature column.")
        if n_clusters < 1:
            raise PlanError(f"n_clusters must be at least 1, got {n_clusters}.")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.output_column = output_column
        self.seed = seed
        self.centroids_: list[list[float]] = []
        self.inertia_: float = float("nan")
        self.n_iter_: int = 0

    def _initial_centroids(self, ds: Dataset) -> list[list[float]]:
        """Seed the centroids from a deterministic content-hash sample of `n_clusters` rows."""
        from batcher.api.dataset._build import split_key

        sample = (
            ds.with_columns(__bt_hash=split_key(ds, None, self.seed))
            .sort("__bt_hash")
            .select(*self.columns)
            .limit(self.n_clusters)
            .to_pydict()
        )
        rows = len(next(iter(sample.values()))) if sample else 0
        if rows < self.n_clusters:
            raise PlanError(f"KMeans needs at least n_clusters={self.n_clusters} rows, got {rows}.")
        return [[float(sample[name][i]) for name in self.columns] for i in range(self.n_clusters)]

    def fit(self, ds: Dataset) -> KMeans:
        """Learn the centroids by alternating assignment and mean-update to convergence.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.cluster import KMeans
                >>> ds = bt.from_pydict({"x": [0.0, 0.0, 9.0, 9.0], "y": [0.0, 1.0, 9.0, 8.0]})
                >>> km = KMeans(["x", "y"], n_clusters=2, seed=0).fit(ds)
                >>> len(km.centroids_)
                2

        Args:
            ds: The dataset to cluster.

        Returns:
            ``self``, fitted.
        """
        from batcher.plan.functions.aggregate import mean as mean_

        centroids = self._initial_centroids(ds)
        aggregates = {name: mean_(col(name)) for name in self.columns}
        for iteration in range(self.max_iter):
            assigned = ds.with_columns(__bt_cluster=_assignment(self.columns, centroids))
            updated = (
                assigned.group_by("__bt_cluster")
                .agg(**aggregates, __bt_n=col(self.columns[0]).count())
                .collect()
            )
            found = {
                int(updated.column("__bt_cluster")[i].as_py()): [
                    float(updated.column(name)[i].as_py()) for name in self.columns
                ]
                for i in range(updated.num_rows)
            }
            new_centroids = [found.get(j, centroids[j]) for j in range(self.n_clusters)]
            self.n_iter_ = iteration + 1
            if new_centroids == centroids:
                centroids = new_centroids
                break
            centroids = new_centroids
        self.centroids_ = centroids
        self.inertia_ = self._inertia(ds)
        return self

    def _inertia(self, ds: Dataset) -> float:
        """The total squared distance from each row to its nearest centroid."""
        from batcher.plan.functions.aggregate import sum as sum_

        best = _squared_distance(self.columns, self.centroids_[0])
        for centroid in self.centroids_[1:]:
            distance = _squared_distance(self.columns, centroid)
            best = when(distance < best).then(distance).otherwise(best)
        value = ds.agg(__bt_inertia=sum_(best)).collect().column("__bt_inertia")[0].as_py()
        return float("nan") if value is None else float(value)

    def predict(self, ds: Dataset) -> Dataset:
        """Append a cluster-label column assigning each row to its nearest fitted centroid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.cluster import KMeans
                >>> ds = bt.from_pydict({"x": [0.0, 10.0], "y": [0.0, 10.0]})
                >>> km = KMeans(["x", "y"], n_clusters=2, seed=0).fit(ds)
                >>> "cluster" in km.predict(ds).columns
                True

        Args:
            ds: The dataset to label.

        Returns:
            A new lazy `Dataset` with the cluster-label column appended.
        """
        if not self.centroids_:
            raise PlanError("KMeans must be fitted before predict.")
        return ds.with_columns(**{self.output_column: _assignment(self.columns, self.centroids_)})

    def fit_predict(self, ds: Dataset) -> Dataset:
        """Fit the centroids and return the labeled dataset in one call.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.cluster import KMeans
                >>> ds = bt.from_pydict({"x": [0.0, 0.0, 9.0, 9.0], "y": [0.0, 1.0, 9.0, 8.0]})
                >>> out = KMeans(["x", "y"], n_clusters=2, seed=0).fit_predict(ds)
                >>> sorted(set(out.to_pydict()["cluster"]))
                [0, 1]

        Args:
            ds: The dataset to cluster and label.

        Returns:
            A new lazy `Dataset` with the cluster-label column appended.
        """
        return self.fit(ds).predict(ds)


class NearestCentroid:
    """Classify each row by the nearest class centroid — the simplest distance-based classifier.

    Fits one centroid per class (the mean of that class's rows) in a single ``group_by(target)``
    aggregate, then labels a row with the class whose centroid is closest in Euclidean distance.
    It is the supervised cousin of `KMeans`: same centroids-and-assignment machinery, but the
    centroids are the known classes rather than discovered clusters. Reproduces scikit-learn's
    ``NearestCentroid`` predictions.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.cluster import NearestCentroid
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 0.5, 5.0, 5.5], "y": ["a", "a", "b", "b"]}
            ... )
            >>> model = NearestCentroid(["x"], "y").fit(ds)
            >>> model.predict(bt.from_pydict({"x": [0.2, 5.2]})).to_pydict()["prediction"]
            ['a', 'b']

    Args:
        features: The numeric feature columns.
        target: The class label column.
        output_column: The name of the predicted-class column `predict` appends.
    """

    __slots__ = ("centroids_", "classes_", "features", "output_column", "target")

    def __init__(
        self, features: Sequence[str], target: str, *, output_column: str = "prediction"
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("NearestCentroid needs at least one feature column.")
        self.target = target
        self.output_column = output_column
        self.classes_: list[object] = []
        self.centroids_: list[list[float]] = []

    def fit(self, ds: Dataset) -> NearestCentroid:
        """Learn one centroid per class in a single grouped aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.cluster import NearestCentroid
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 9.0, 10.0], "y": [0, 0, 1, 1]})
                >>> sorted(NearestCentroid(["x"], "y").fit(ds).classes_)
                [0, 1]

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        from batcher.plan.functions.aggregate import mean as mean_

        for name in (*self.features, self.target):
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        grouped = (
            ds.group_by(self.target)
            .agg(**{name: mean_(col(name)) for name in self.features})
            .collect()
        )
        self.classes_ = [grouped.column(self.target)[i].as_py() for i in range(grouped.num_rows)]
        self.centroids_ = [
            [float(grouped.column(name)[i].as_py()) for name in self.features]
            for i in range(grouped.num_rows)
        ]
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Append the label of the nearest class centroid for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.cluster import NearestCentroid
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 9.0, 10.0], "y": [0, 0, 1, 1]})
                >>> model = NearestCentroid(["x"], "y").fit(ds)
                >>> model.predict(bt.from_pydict({"x": [0.5, 9.5]})).to_pydict()["prediction"]
                [0, 1]

        Args:
            ds: The dataset to classify.

        Returns:
            A new lazy `Dataset` with the predicted-class column appended.
        """
        if not self.centroids_:
            raise PlanError("NearestCentroid must be fitted before predict.")
        distances = [_squared_distance(self.features, c) for c in self.centroids_]
        prediction = lit(self.classes_[0])
        best = distances[0]
        for index in range(1, len(self.centroids_)):
            closer = distances[index] < best
            prediction = when(closer).then(lit(self.classes_[index])).otherwise(prediction)
            best = when(closer).then(distances[index]).otherwise(best)
        return ds.with_columns(**{self.output_column: prediction})
