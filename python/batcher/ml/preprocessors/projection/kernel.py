"""Kernel approximation — the accuracy of an RBF kernel at the cost of a linear model.

A kernel SVM is often the best model on a medium tabular problem and the worst thing you
can put in a pipeline: it needs the full pairwise kernel matrix, so it is quadratic in rows
in both time and memory, and nothing about that shape distributes.

The standard escape is to approximate the kernel with an explicit feature map. Project each
row into a few hundred new columns such that an ordinary *dot product* in that space
approximates the kernel value, then fit a plain linear model on those columns. The linear
model streams, spills and distributes like any other, and gets most of the kernel's accuracy.

Two ways to build the map:

`RBFSampler`
    Random Fourier features. Needs no data at all — the map is a seeded draw — so it works
    on a stream and cannot skew between training and serving.
`Nystroem`
    Picks actual rows as landmarks and measures similarity to them. Data-dependent, so it
    usually needs fewer components for the same accuracy, at the cost of a fit pass.

Both lower to plain arithmetic over the source columns, so the transform runs column-wise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir import Expr, col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["Nystroem", "RBFSampler"]


def _require_columns(ds: Dataset, columns: list[str]) -> None:
    """Raise a `ColumnNotFoundError` naming any column the frame does not have."""
    require_columns(ds, *columns, hint="Pass numeric columns.")


class RBFSampler(Preprocessor):
    """Approximate an RBF kernel with random Fourier features.

    Each output column is ``sqrt(2/D) * cos(w . x + b)`` for a random frequency ``w`` drawn
    from the Gaussian whose spread is set by `gamma`, and a random phase ``b``. Bochner's
    theorem is what makes this work: the dot product of two such feature vectors converges
    to ``exp(-gamma * ||x - y||^2)`` as the width grows.

    Nothing is learned from the data — `fit` only needs the column count — so this composes
    into a streaming pipeline and has no train/serve skew.

    Scale the inputs first. `gamma` is a distance in the feature space, so a column measured
    in millions and one measured in fractions cannot share a sensible value; put a
    {py:class}`StandardScaler <batcher.ml.StandardScaler>` in front.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import RBFSampler
            >>> ds = bt.from_pydict({"a": [0.0, 1.0], "b": [1.0, 0.0]})
            >>> out = RBFSampler(["a", "b"], n_components=4, seed=0).fit_transform(ds)
            >>> sorted(c for c in out.columns if c.startswith("rbf"))
            ['rbf0', 'rbf1', 'rbf2', 'rbf3']

    Args:
        columns: The numeric columns to map.
        n_components: How many features to generate. More is a closer approximation.
        gamma: The RBF kernel's width parameter; larger means a narrower kernel.
        seed: Seed for the draw, so the map is reproducible.
        output_prefix: The prefix of the emitted column names.
        drop_original: Remove the source columns after mapping them.
    """

    numeric_only = True

    __slots__ = (
        "columns",
        "drop_original",
        "gamma",
        "n_components",
        "offsets_",
        "output_prefix",
        "seed",
        "weights_",
    )

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        n_components: int = 100,
        gamma: float = 1.0,
        seed: int = 0,
        output_prefix: str = "rbf",
        drop_original: bool = False,
    ) -> None:
        self.columns = columns_arg(columns, what="RBFSampler")
        if n_components < 1:
            raise PlanError(f"RBFSampler: n_components must be at least 1, got {n_components}")
        if gamma <= 0:
            raise PlanError(f"RBFSampler: gamma must be positive, got {gamma!r}")
        self.n_components = n_components
        self.gamma = gamma
        self.seed = seed
        self.output_prefix = output_prefix
        self.drop_original = drop_original
        self.weights_: list[list[float]] = []
        self.offsets_: list[float] = []

    def fit(self, ds: Dataset) -> RBFSampler:
        """Draw the random frequencies and phases. No data is read.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RBFSampler
                >>> ds = bt.from_pydict({"a": [0.0], "b": [1.0]})
                >>> pre = RBFSampler(["a", "b"], n_components=3).fit(ds)
                >>> len(pre.weights_), len(pre.offsets_)
                (3, 3)

        Args:
            ds: The dataset, read only for its column names.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        self._check_numeric(ds)
        import numpy as np

        _require_columns(ds, self.columns)
        rng = np.random.default_rng(self.seed)
        # The frequency spread is sqrt(2 * gamma): the Fourier transform of the RBF kernel
        # exp(-gamma ||d||^2) is a Gaussian with that standard deviation.
        scale = np.sqrt(2.0 * self.gamma)
        weights = rng.normal(0.0, scale, size=(self.n_components, len(self.columns)))
        offsets = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        self.weights_ = [[float(v) for v in row] for row in weights]
        self.offsets_ = [float(v) for v in offsets]
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Append the random Fourier feature columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RBFSampler
                >>> ds = bt.from_pydict({"a": [0.0], "b": [1.0]})
                >>> out = RBFSampler(["a", "b"], n_components=2).fit_transform(ds)
                >>> len(out.to_pydict()["rbf0"])
                1

        Args:
            ds: The dataset to map.

        Returns:
            A new lazy `Dataset` with the mapped columns appended.
        """
        self._require_fitted()
        import math

        amplitude = math.sqrt(2.0 / self.n_components)
        projections: dict[str, Expr] = {}
        for index, (weights, offset) in enumerate(zip(self.weights_, self.offsets_, strict=True)):
            angle = lit(offset)
            for weight, name in zip(weights, self.columns, strict=True):
                angle = angle + lit(weight) * col(name).cast("float64")
            projections[f"{self.output_prefix}{index}"] = lit(amplitude) * angle.cos()
        out = ds.with_columns(**projections)
        return out.drop(*self.columns) if self.drop_original else out


class Nystroem(Preprocessor):
    """Approximate an RBF kernel from a sample of actual rows.

    `fit` picks `n_components` rows as landmarks, computes the kernel matrix between them,
    and stores its inverse square root. `transform` writes each row's similarity to every
    landmark, combined through that matrix so that a dot product in the output space
    approximates the kernel.

    Being data-dependent, it usually needs fewer components than `RBFSampler` for the same
    accuracy — the landmarks sit where the data actually is. The cost is that it needs a
    fit pass, and that the landmarks are held on the driver.

    Scale the inputs first, for the same reason `RBFSampler` needs it.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import Nystroem
            >>> ds = bt.from_pydict({"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]})
            >>> out = Nystroem(["a", "b"], n_components=3, seed=0).fit_transform(ds)
            >>> sorted(c for c in out.columns if c.startswith("ny"))
            ['ny0', 'ny1', 'ny2']

    Args:
        columns: The numeric columns to map.
        n_components: How many landmark rows to sample.
        gamma: The RBF kernel's width parameter.
        seed: Seed for the landmark sample.
        output_prefix: The prefix of the emitted column names.
        drop_original: Remove the source columns after mapping them.
    """

    numeric_only = True

    __slots__ = (
        "columns",
        "components_",
        "drop_original",
        "gamma",
        "landmarks_",
        "n_components",
        "output_prefix",
        "seed",
    )

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        n_components: int = 100,
        gamma: float = 1.0,
        seed: int = 0,
        output_prefix: str = "ny",
        drop_original: bool = False,
    ) -> None:
        self.columns = columns_arg(columns, what="Nystroem")
        if n_components < 1:
            raise PlanError(f"Nystroem: n_components must be at least 1, got {n_components}")
        if gamma <= 0:
            raise PlanError(f"Nystroem: gamma must be positive, got {gamma!r}")
        self.n_components = n_components
        self.gamma = gamma
        self.seed = seed
        self.output_prefix = output_prefix
        self.drop_original = drop_original
        self.landmarks_: list[list[float]] = []
        self.components_: list[list[float]] = []

    def fit(self, ds: Dataset) -> Nystroem:
        """Sample the landmarks and factor their kernel matrix.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Nystroem
                >>> ds = bt.from_pydict({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]})
                >>> pre = Nystroem(["a", "b"], n_components=2, seed=0).fit(ds)
                >>> len(pre.landmarks_)
                2

        Args:
            ds: The dataset to sample landmarks from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the dataset has no complete rows to use as landmarks.
            ColumnNotFoundError: If a named column is missing.
        """
        self._check_numeric(ds)
        import numpy as np

        _require_columns(ds, self.columns)
        selected = ds.select(*self.columns)
        for name in self.columns:
            selected = selected.filter(col(name).is_not_null())
        # The landmark sample is a deterministic content-hash order, not a shuffle, so the
        # same rows are chosen however the data is partitioned — the fitted map has to be
        # reproducible or serving sees a different feature space from training.
        table = selected.sample(fraction=1.0, seed=self.seed).limit(self.n_components).collect()
        if table.num_rows == 0:
            raise PlanError(
                "Nystroem: no complete rows to use as landmarks. Every row has a null in at "
                "least one of the projected columns; impute first."
            )
        landmarks = np.array(
            [table.column(name).to_pylist() for name in self.columns], dtype=float
        ).T
        squared = ((landmarks[:, None, :] - landmarks[None, :, :]) ** 2).sum(axis=2)
        kernel = np.exp(-self.gamma * squared)
        # K^{-1/2} via the symmetric eigendecomposition. A landmark set with duplicates
        # makes the matrix singular, so tiny eigenvalues are floored rather than inverted.
        values, vectors = np.linalg.eigh(kernel)
        values = np.maximum(values, 1e-12)
        inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
        self.landmarks_ = [[float(v) for v in row] for row in landmarks]
        self.components_ = [[float(v) for v in row] for row in inverse_root]
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Append each row's kernel similarity to the landmarks, combined and whitened.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Nystroem
                >>> ds = bt.from_pydict({"a": [0.0, 1.0], "b": [1.0, 0.0]})
                >>> out = Nystroem(["a", "b"], n_components=2, seed=0).fit_transform(ds)
                >>> len(out.to_pydict()["ny0"])
                2

        Args:
            ds: The dataset to map.

        Returns:
            A new lazy `Dataset` with the mapped columns appended.
        """
        self._require_fitted()
        similarity: list[Expr] = []
        for landmark in self.landmarks_:
            squared = lit(0.0)
            for value, name in zip(landmark, self.columns, strict=True):
                difference = col(name).cast("float64") - lit(value)
                squared = squared + difference * difference
            similarity.append((lit(-self.gamma) * squared).exp())
        projections: dict[str, Expr] = {}
        for index in range(len(self.landmarks_)):
            total = lit(0.0)
            for position, kernel_value in enumerate(similarity):
                weight = self.components_[position][index]
                if weight:
                    total = total + lit(weight) * kernel_value
            projections[f"{self.output_prefix}{index}"] = total
        out = ds.with_columns(**projections)
        return out.drop(*self.columns) if self.drop_original else out
