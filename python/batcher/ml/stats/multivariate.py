"""Multivariate association — correlation and covariance across many columns at once.

A pairwise statistic answers "do these two columns relate"; a modelling decision usually needs
the whole picture — which of thirty features move together, and whether an apparent link
between two survives controlling for a third. This module gives the matrix form of correlation
and covariance and the partial correlation that removes a confounder.

Every entry is one mergeable aggregate, so the whole matrix is built from a single scan over
the data. The only driver-side work is assembling the small result table and, for a partial
correlation controlling for several variables, inverting the (tiny) correlation matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir.constructors import col
from batcher.plan.functions.aggregate import corr, covar_samp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = [
    "correlation_matrix",
    "covariance_matrix",
    "partial_correlation",
    "variance_inflation_factor",
]


def _pairwise_matrix(ds: Dataset, columns: Sequence[str], kind: str) -> Dataset:
    """Build a labeled square matrix of a pairwise aggregate over `columns` in one pass."""
    import batcher as bt

    require_columns(ds, *columns)
    names = list(columns)
    aggregates = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                continue
            if kind == "correlation":
                aggregates[f"__c_{i}_{j}"] = corr(col(a), col(b))
            else:
                aggregates[f"__c_{i}_{j}"] = covar_samp(col(a), col(b))
    row = ds.agg(**aggregates).collect()

    def cell(i: int, j: int) -> float:
        key = f"__c_{i}_{j}" if j >= i else f"__c_{j}_{i}"
        value = row.column(key)[0].as_py()
        return float("nan") if value is None else float(value)

    table: dict[str, list] = {"column": names}
    for j, b in enumerate(names):
        table[b] = [cell(i, j) for i in range(len(names))]
    return bt.from_pydict(table)


def correlation_matrix(ds: Dataset, columns: Sequence[str]) -> Dataset:
    """The Pearson correlation between every pair of `columns`, as a labeled square matrix.

    The whole correlation structure of a feature set in one scan: one row and one value column
    per input column, with a leading ``"column"`` label column naming each row. Reading down a
    column shows what a feature correlates with, which is what flags redundant features before a
    linear model and multicollinearity before trusting its coefficients.

    Args:
        ds: The dataset holding the columns.
        columns: The numeric columns to correlate.

    Returns:
        A `Dataset` with a ``"column"`` label column and one value column per input, the
        ``[i, j]`` cell being the correlation of column `i` with column `j`.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import correlation_matrix
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "c": [3.0, 2.0, 1.0]}
            ... )
            >>> m = correlation_matrix(ds, ["a", "b", "c"]).to_pydict()
            >>> m["column"], [round(v, 1) for v in m["a"]]
            (['a', 'b', 'c'], [1.0, 1.0, -1.0])
    """
    return _pairwise_matrix(ds, columns, "correlation")


def covariance_matrix(ds: Dataset, columns: Sequence[str]) -> Dataset:
    """The sample covariance between every pair of `columns`, as a labeled square matrix.

    The unnormalized cousin of `correlation_matrix`, in the columns' own units. The diagonal is
    each column's variance, so it is the input a Gaussian model, a Mahalanobis distance, or a
    PCA needs. Use `correlation_matrix` to *compare* associations across differently-scaled
    columns and this when the scale carries meaning.

    Args:
        ds: The dataset holding the columns.
        columns: The numeric columns to cross-covary.

    Returns:
        A `Dataset` with a ``"column"`` label column and one value column per input, the
        ``[i, j]`` cell being the covariance of column `i` with column `j`.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import covariance_matrix
            >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
            >>> m = covariance_matrix(ds, ["a", "b"]).to_pydict()
            >>> m["column"], m["a"]
            (['a', 'b'], [1.0, 2.0])
    """
    return _pairwise_matrix(ds, columns, "covariance")


def partial_correlation(ds: Dataset, x: str, y: str, controlling: str | Sequence[str]) -> float:
    """The correlation between `x` and `y` after removing the effect of `controlling`.

    The confounder killer: two features can correlate only because both track a third, and the
    partial correlation is what is left of their association once that third is held fixed. It
    is computed from the correlation matrix over ``x``, ``y``, and the controls, so it needs one
    scan; controlling for several variables inverts that small matrix on the driver.

    Args:
        ds: The dataset holding every named column.
        x: The first column.
        y: The second column.
        controlling: The column (or columns) to control for.

    Returns:
        The partial correlation of `x` and `y` given `controlling`, in ``[-1, 1]``.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import partial_correlation
            >>> # x and y both track z, so they correlate strongly on their own...
            >>> ds = bt.from_pydict(
            ...     {
            ...         "x": [1.0, 2.5, 2.0, 4.5, 5.0, 5.5, 7.5, 8.0],
            ...         "y": [1.5, 1.5, 3.5, 3.5, 5.5, 6.5, 6.5, 8.5],
            ...         "z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ...     }
            ... )
            >>> round(partial_correlation(ds, "x", "y", "z"), 4)  # ...but not once z is held fixed
            -0.7801
    """
    import numpy as np

    controls = [controlling] if isinstance(controlling, str) else list(controlling)
    variables = [x, y, *controls]
    require_columns(ds, *variables)
    matrix = correlation_matrix(ds, variables).to_pydict()
    r = np.array([matrix[name] for name in variables], dtype=float)
    if len(controls) == 1:
        rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
        denominator = np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))
        return float("nan") if denominator == 0 else float((rxy - rxz * ryz) / denominator)
    precision = np.linalg.pinv(r)
    denominator = np.sqrt(precision[0, 0] * precision[1, 1])
    return float("nan") if denominator == 0 else float(-precision[0, 1] / denominator)


def variance_inflation_factor(ds: Dataset, columns: Sequence[str]) -> dict[str, float]:
    """The variance inflation factor of each column — how much multicollinearity inflates it.

    ``VIF_j = 1 / (1 - R^2_j)`` where ``R^2_j`` is from regressing column `j` on all the others, so
    it measures how well the *rest* of the feature set predicts each column. A VIF of 1 means the
    column is uncorrelated with the others; a large VIF (a rule of thumb is above 5 or 10) means it
    is nearly a linear combination of them, which makes a linear model's coefficient on it unstable
    and its sign untrustworthy. It is read straight off the diagonal of the inverted correlation
    matrix, so the whole set of VIFs is one scan plus a tiny driver-side inverse.

    Args:
        ds: The dataset holding the columns.
        columns: The numeric columns to check for mutual collinearity.

    Returns:
        A ``{column: vif}`` dict, each at least 1.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.stats import variance_inflation_factor
            >>> # a and b track each other closely; c is on its own
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ...      "b": [1.1, 1.9, 3.2, 3.8, 5.1, 5.9, 7.2, 7.8],
            ...      "c": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]}
            ... )
            >>> vif = variance_inflation_factor(ds, ["a", "b", "c"])
            >>> vif["a"] > 10 and vif["c"] < 2  # the collinear pair inflates, the loner does not
            True
    """
    import numpy as np

    require_columns(ds, *columns)
    names = list(columns)
    matrix = correlation_matrix(ds, names).to_pydict()
    correlation = np.array([matrix[name] for name in names], dtype=float).T
    diagonal = np.diag(np.linalg.pinv(correlation))
    return {name: float(diagonal[i]) for i, name in enumerate(names)}
