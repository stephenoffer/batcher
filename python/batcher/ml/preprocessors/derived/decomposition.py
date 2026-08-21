"""Dimensionality reduction — projecting many correlated columns onto a few components.

A wide feature table is usually low-rank: the columns move together, so a handful of directions
carry most of the variance. Principal component analysis finds those directions and projects
onto them, which shrinks the table for a downstream model, kills multicollinearity, and turns a
correlated block into orthogonal features.

The whole fit is one scan: the mean and the covariance matrix are mergeable aggregates, and the
only driver-side work is the eigendecomposition of that small ``d x d`` matrix. The transform is
then a set of linear-combination expressions, one per component, so it lowers to Rust and runs
in a single streaming pass exactly as every other preprocessor's does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["PCA", "TruncatedSVD"]


def _empty_fit(what: str, column: str) -> PlanError:
    """The error raised when a fitted column carries no value to decompose.

    A decomposition learns its directions from second moments, so a column that is empty or
    entirely null contributes no variance and leaves the covariance undefined. Reaching
    ``float(None)`` instead surfaced a bare ``TypeError`` from three frames down that named
    neither the preprocessor nor the column.

    Args:
        what: The preprocessor's class name.
        column: The column with nothing to learn from.

    Returns:
        A `PlanError` naming the column and the way out.
    """
    return PlanError(
        f"{what}: column {column!r} has no non-null values, so there is no variance to "
        "decompose. Drop the column, or fill it first with SimpleImputer."
    )


class PCA(Preprocessor):
    """Project a block of numeric columns onto their top principal components.

    The standard linear dimensionality reducer. It learns the mean and covariance of the fitted
    columns, takes the `n_components` eigenvectors of the covariance with the largest eigenvalues,
    and projects each row onto them — replacing the input columns with ``{output_prefix}1`` ..
    ``{output_prefix}{n_components}``, which are uncorrelated and ordered by the variance they
    carry. The learned `explained_variance_ratio_` says how much of the total variance each kept
    component accounts for, which is how you choose `n_components`.

    Reproduces scikit-learn's ``PCA`` projection up to the per-component sign that PCA is free to
    flip. The fit is a single scan (mean and covariance are aggregates); only the small
    eigendecomposition runs on the driver.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import PCA
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0],
            ...      "c": [4.0, 3.0, 2.0, 1.0]}
            ... )
            >>> out = PCA(["a", "b", "c"], n_components=1).fit_transform(ds)
            >>> out.columns
            ['pc1']

    Args:
        columns: The numeric columns to reduce (replaced by the components).
        n_components: How many principal components to keep.
        output_prefix: The stem of the output column names (``pc1``, ``pc2``, ...).
        keep_original: Keep the input columns alongside the components instead of dropping them.
    """

    numeric_only = True

    __slots__ = (
        "columns",
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "keep_original",
        "mean_",
        "n_components",
        "output_prefix",
    )

    def __init__(
        self,
        columns: Sequence[str],
        *,
        n_components: int = 2,
        output_prefix: str = "pc",
        keep_original: bool = False,
    ) -> None:
        self.columns = columns_arg(columns, what="PCA")
        if n_components < 1 or n_components > len(self.columns):
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"n_components must be between 1 and {len(self.columns)}, got {n_components}."
            )
        self.n_components = n_components
        self.output_prefix = output_prefix
        self.keep_original = keep_original
        self.mean_: list[float] = []
        self.components_: list[list[float]] = []
        self.explained_variance_: list[float] = []
        self.explained_variance_ratio_: list[float] = []

    def fit(self, ds: Dataset) -> PCA:
        """Learn the mean, the principal components, and the variance each explains.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import PCA
                >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
                >>> pre = PCA(["a", "b"], n_components=1).fit(ds)
                >>> round(pre.explained_variance_ratio_[0], 6)
                1.0

        Args:
            ds: The dataset to learn from.

        Returns:
            ``self``, fitted.
        """
        self._check_numeric(ds)
        import numpy as np

        from batcher.ml.stats.multivariate import covariance_matrix
        from batcher.plan.functions.aggregate import mean as mean_

        means = ds.agg(**{name: mean_(col(name)) for name in self.columns}).collect()
        self.mean_ = []
        for name in self.columns:
            centre = means.column(name)[0].as_py()
            if centre is None:
                raise _empty_fit("PCA", name)
            self.mean_.append(float(centre))
        covariance = covariance_matrix(ds, self.columns).to_pydict()
        matrix = np.array([covariance[name] for name in self.columns], dtype=float).T
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        total = float(values.sum())
        kept = self.n_components
        self.components_ = [vectors[:, j].tolist() for j in range(kept)]
        self.explained_variance_ = [float(values[j]) for j in range(kept)]
        self.explained_variance_ratio_ = [
            (float(values[j]) / total if total else 0.0) for j in range(kept)
        ]
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Project each row onto the fitted components, replacing the input columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import PCA
                >>> ds = bt.from_pydict(
                ...     {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "id": [0, 1, 2]}
                ... )
                >>> PCA(["a", "b"], n_components=1).fit_transform(ds).columns
                ['id', 'pc1']

        Args:
            ds: The dataset to project.

        Returns:
            A new lazy `Dataset` with the input columns replaced by ``{output_prefix}k`` columns.
        """
        self._require_fitted()
        centered = [col(name) - lit(self.mean_[index]) for index, name in enumerate(self.columns)]
        projections = {}
        for k, component in enumerate(self.components_, start=1):
            expression = lit(0.0)
            for weight, term in zip(component, centered, strict=True):
                expression = expression + lit(weight) * term
            projections[f"{self.output_prefix}{k}"] = expression
        out = ds.with_columns(**projections)
        if self.keep_original:
            return out
        return out.drop(*self.columns)


class TruncatedSVD(Preprocessor):
    """Reduce dimensionality by truncated SVD — like `PCA` but without centering the columns.

    Projects the feature block onto the top `n_components` right singular vectors of the data
    matrix itself, rather than of its centered covariance. Dropping the centering step is what
    makes it the reducer for data that should not be shifted — a non-negative or already-centered
    feature block, or the sparse count matrix a bag-of-words produces (where centering would
    destroy the sparsity). On centered data it coincides with `PCA`. Reproduces scikit-learn's
    ``TruncatedSVD`` projection up to the per-component sign, and its `explained_variance_ratio_`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import TruncatedSVD
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0],
            ...      "c": [1.0, 3.0, 2.0, 4.0]}
            ... )
            >>> out = TruncatedSVD(["a", "b", "c"], n_components=2).fit_transform(ds)
            >>> out.columns
            ['svd1', 'svd2']

    Args:
        columns: The numeric columns to reduce (replaced by the components).
        n_components: How many singular components to keep.
        output_prefix: The stem of the output column names (``svd1``, ``svd2``, ...).
        keep_original: Keep the input columns alongside the components instead of dropping them.
    """

    numeric_only = True

    __slots__ = (
        "columns",
        "components_",
        "explained_variance_ratio_",
        "keep_original",
        "n_components",
        "output_prefix",
    )

    def __init__(
        self,
        columns: Sequence[str],
        *,
        n_components: int = 2,
        output_prefix: str = "svd",
        keep_original: bool = False,
    ) -> None:
        self.columns = columns_arg(columns, what="TruncatedSVD")
        if n_components < 1 or n_components > len(self.columns):
            from batcher._internal.errors import PlanError

            raise PlanError(
                f"n_components must be between 1 and {len(self.columns)}, got {n_components}."
            )
        self.n_components = n_components
        self.output_prefix = output_prefix
        self.keep_original = keep_original
        self.components_: list[list[float]] = []
        self.explained_variance_ratio_: list[float] = []

    def fit(self, ds: Dataset) -> TruncatedSVD:
        """Learn the top singular components from the data's second-moment matrix.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import TruncatedSVD
                >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
                >>> len(TruncatedSVD(["a", "b"], n_components=1).fit(ds).components_)
                1

        Args:
            ds: The dataset to learn from.

        Returns:
            ``self``, fitted.
        """
        self._check_numeric(ds)
        import numpy as np

        from batcher.plan.functions.aggregate import sum as sum_
        from batcher.plan.functions.statistics import var_pop

        names = self.columns
        d = len(names)
        gram_aggs = {}
        for i in range(d):
            for j in range(i, d):
                gram_aggs[f"g{i}_{j}"] = sum_(col(names[i]) * col(names[j]))
        gram_row = ds.agg(**gram_aggs).collect()
        gram = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                value = gram_row.column(f"g{i}_{j}")[0].as_py()
                if value is None:
                    if i == j:
                        raise _empty_fit("TruncatedSVD", names[i])
                    raise PlanError(
                        f"TruncatedSVD: columns {names[i]!r} and {names[j]!r} share no row "
                        "where both are non-null, so their gram entry is undefined. Fill "
                        "them first with SimpleImputer, or fit on the complete rows only."
                    )
                gram[i, j] = gram[j, i] = float(value)
        values, vectors = np.linalg.eigh(gram)
        order = np.argsort(values)[::-1]
        vectors = vectors[:, order]
        self.components_ = [vectors[:, k].tolist() for k in range(self.n_components)]
        # explained_variance_ratio_ = variance of each projection over the total feature variance.
        projected = self.transform(ds, _skip_fit_check=True)
        variance_row = projected.agg(
            **{
                f"{self.output_prefix}{k + 1}": var_pop(col(f"{self.output_prefix}{k + 1}"))
                for k in range(self.n_components)
            }
        ).collect()
        total_row = ds.agg(
            **{f"t{i}": var_pop(col(name)) for i, name in enumerate(names)}
        ).collect()
        total = sum(float(total_row.column(f"t{i}")[0].as_py() or 0.0) for i in range(d))
        self.explained_variance_ratio_ = [
            float(variance_row.column(f"{self.output_prefix}{k + 1}")[0].as_py() or 0.0) / total
            if total
            else 0.0
            for k in range(self.n_components)
        ]
        self._fitted = True
        return self

    def transform(self, ds: Dataset, *, _skip_fit_check: bool = False) -> Dataset:
        """Project each row onto the fitted components, replacing the input columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import TruncatedSVD
                >>> ds = bt.from_pydict(
                ...     {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "id": [0, 1, 2]}
                ... )
                >>> TruncatedSVD(["a", "b"], n_components=1).fit_transform(ds).columns
                ['id', 'svd1']

        Args:
            ds: The dataset to project.

        Returns:
            A new lazy `Dataset` with the input columns replaced by ``{output_prefix}k`` columns.
        """
        if not _skip_fit_check:
            self._require_fitted()
        projections = {}
        for k, component in enumerate(self.components_, start=1):
            expression = lit(0.0)
            for weight, name in zip(component, self.columns, strict=True):
                expression = expression + lit(weight) * col(name)
            projections[f"{self.output_prefix}{k}"] = expression
        out = ds.with_columns(**projections)
        if self.keep_original:
            return out
        return out.drop(*self.columns)
