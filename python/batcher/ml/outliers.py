"""Outlier detection — finding the rows a model should not be trained on, at scale.

An outlier is not a bug to be deleted reflexively; it is a row that does not belong to the
same process as the rest, and finding it is a decision the data forces. What is universal is
the *rule*, and there are three that cover almost every case, in increasing order of how
skewed a column they tolerate:

`z-score`
    More than `threshold` standard deviations from the mean. Correct for a roughly normal
    column, and wrong for a skewed one — the mean and the standard deviation are themselves
    dragged by the outliers, so a heavy tail hides its own extremes.
`iqr`
    Outside ``[q1 - k*IQR, q3 + k*IQR]``, Tukey's rule. Robust, distribution-free, and the
    right default: the quartiles are not moved by the tail they are trying to detect.
`mad`
    More than `threshold` scaled median-absolute-deviations from the median. The most robust
    of the three — it tolerates up to half the data being arbitrary — and the one to use on a
    genuinely heavy-tailed column.

Each rule is a per-column bound learned in one aggregate and applied as a lazy predicate, so
`flag_outliers` marks them, `count_outliers` tallies them, and `OutlierClipper` clamps them —
all without a per-row pass and all identical single-node or distributed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = [
    "METHODS",
    "OutlierClipper",
    "count_outliers",
    "flag_outliers",
    "mahalanobis_distance",
    "outlier_bounds",
]

#: The outlier rules `outlier_bounds` and its callers understand.
METHODS = ("iqr", "zscore", "mad")


def _require(ds: Dataset, *names: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column for any missing name."""
    for name in names:
        if name not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, ds.columns, hint="Pass an existing column.")
            )


def outlier_bounds(
    ds: Dataset, column: str, *, method: str = "iqr", threshold: float = 1.5
) -> tuple[float, float]:
    """The lower and upper cut points for one column under a chosen rule.

    Exposed because the bounds are often what you want directly — to report the acceptable
    range, to reuse across datasets, or to clamp a *different* frame to a reference frame's
    limits. `flag_outliers`, `count_outliers`, and `OutlierClipper` all derive from these.

    Args:
        ds: The dataset to learn the bounds from.
        column: The numeric column.
        method: ``"iqr"`` (Tukey), ``"zscore"``, or ``"mad"`` (robust).
        threshold: The rule's width — the IQR multiplier (1.5 is Tukey's fence), or the
            number of standard deviations / scaled MADs (3.0 is the usual choice).

    Returns:
        The ``(lower, upper)`` bounds; a value outside them is an outlier.

    Raises:
        PlanError: On an unknown `method`.
        ColumnNotFoundError: If `column` is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.outliers import outlier_bounds
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
            >>> outlier_bounds(ds, "x", method="iqr", threshold=1.5)
            (-1.0, 7.0)
    """
    import batcher as bt

    _require(ds, column)
    if method not in METHODS:
        from batcher._internal.errors import suggestion

        hint = suggestion(method, METHODS)
        tail = f" {hint}" if hint else ""
        raise PlanError(f"method must be one of {sorted(METHODS)}, got {method!r}.{tail}")
    if method == "zscore":
        row = ds.agg(m=bt.mean(col(column)), s=bt.std(col(column))).collect()
        center = float(row.column("m")[0].as_py() or 0.0)
        spread = float(row.column("s")[0].as_py() or 0.0)
        return center - threshold * spread, center + threshold * spread
    if method == "mad":
        from batcher.ml.stats import median_abs_deviation

        center = float(ds.agg(m=col(column).median()).collect().column("m")[0].as_py() or 0.0)
        spread = median_abs_deviation(ds, column)
        return center - threshold * spread, center + threshold * spread
    row = ds.agg(q1=col(column).quantile(0.25), q3=col(column).quantile(0.75)).collect()
    q1 = float(row.column("q1")[0].as_py() or 0.0)
    q3 = float(row.column("q3")[0].as_py() or 0.0)
    iqr = q3 - q1
    return q1 - threshold * iqr, q3 + threshold * iqr


def _outlier_predicate(column: str, low: float, high: float):
    """The boolean expression true on a row outside ``[low, high]``."""
    return (col(column) < lit(low)) | (col(column) > lit(high))


def flag_outliers(
    ds: Dataset,
    columns: str | Sequence[str],
    *,
    method: str = "iqr",
    threshold: float = 1.5,
    suffix: str = "_outlier",
) -> Dataset:
    """Append a boolean flag per column marking the rows outside its outlier bounds.

    Flags rather than drops, deliberately: whether an outlier is an error to remove or a rare
    event to keep is a judgment the data cannot make, so this surfaces them and leaves the
    decision to a downstream `filter`. The flag is also a feature in its own right — "this row
    was anomalous" often predicts the target.

    Args:
        ds: The dataset to check.
        columns: The numeric columns to flag.
        method: The outlier rule (see `outlier_bounds`).
        threshold: The rule's width.
        suffix: Appended to each column name to build its flag column.

    Returns:
        A new lazy `Dataset` with one boolean flag column appended per input column.

    Raises:
        PlanError: On an unknown `method`.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.outliers import flag_outliers
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 100.0]})
            >>> flag_outliers(ds, "x").to_pydict()["x_outlier"]
            [False, False, False, True]
    """
    names = columns_arg(columns, what="flag_outliers")
    projections = {}
    for name in names:
        low, high = outlier_bounds(ds, name, method=method, threshold=threshold)
        projections[f"{name}{suffix}"] = _outlier_predicate(name, low, high)
    return ds.with_columns(**projections)


def count_outliers(
    ds: Dataset, columns: str | Sequence[str], *, method: str = "iqr", threshold: float = 1.5
) -> dict[str, int]:
    """How many outliers each column has under the chosen rule.

    The triage number before deciding what to do with them: 3 outliers in a million rows is a
    filter, 300,000 is a distribution you have misread. One aggregate per column.

    Args:
        ds: The dataset to check.
        columns: The numeric columns to check.
        method: The outlier rule.
        threshold: The rule's width.

    Returns:
        A ``{column: outlier_count}`` dict.

    Raises:
        PlanError: On an unknown `method`.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.outliers import count_outliers
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 100.0]})
            >>> count_outliers(ds, "x")
            {'x': 1}
    """
    from batcher.plan.functions.aggregate import count_if

    names = columns_arg(columns, what="count_outliers")
    aggregates = {}
    bounds = {}
    for name in names:
        bounds[name] = outlier_bounds(ds, name, method=method, threshold=threshold)
        aggregates[name] = count_if(_outlier_predicate(name, *bounds[name]))
    row = ds.agg(**aggregates).collect()
    return {name: int(row.column(name)[0].as_py() or 0) for name in names}


class OutlierClipper(Preprocessor):
    """Clamp each column to its learned outlier bounds — winsorize by an outlier rule.

    A fitted preprocessor rather than a one-shot function, so the *training* bounds apply to
    the serving data: a new record-breaking value at serving time is clamped to the training
    limit rather than passed through as an extreme the model never saw. Nothing is dropped, so
    the row count and every join key survive.

    The difference from `Clipper` is the rule: `Clipper` clamps at fixed quantiles, this clamps
    at a statistical outlier boundary (IQR, z-score, or MAD), which adapts its width to the
    column's spread rather than fixing a percentile.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.outliers import OutlierClipper
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
            >>> clipped = OutlierClipper("x", method="iqr").fit_transform(ds).to_pydict()["x"]
            >>> max(clipped) < 100.0
            True

    Args:
        columns: The numeric columns to clamp (replaced in place).
        method: The outlier rule (see `outlier_bounds`).
        threshold: The rule's width.
    """

    __slots__ = ("bounds_", "columns", "method", "threshold")

    def __init__(
        self, columns: str | Sequence[str], *, method: str = "iqr", threshold: float = 1.5
    ) -> None:
        self.columns = columns_arg(columns, what="OutlierClipper")
        if method not in METHODS:
            raise PlanError(f"method must be one of {sorted(METHODS)}, got {method!r}")
        self.method = method
        self.threshold = threshold
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, ds: Dataset) -> OutlierClipper:
        """Learn each column's outlier bounds under the chosen rule.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.outliers import OutlierClipper
                >>> pre = OutlierClipper("x").fit(bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}))
                >>> pre.bounds_["x"]
                (-1.0, 7.0)

        Args:
            ds: The dataset to learn the bounds from.

        Returns:
            ``self``, fitted.
        """
        for name in self.columns:
            self.bounds_[name] = outlier_bounds(
                ds, name, method=self.method, threshold=self.threshold
            )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Clamp each fitted column to its learned bounds.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.outliers import OutlierClipper
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
                >>> pre = OutlierClipper("x").fit(ds)
                >>> pre.transform(bt.from_pydict({"x": [-50.0, 50.0]})).to_pydict()["x"]
                [-1.0, 7.0]

        Args:
            ds: The dataset to clamp.

        Returns:
            A new lazy `Dataset` with the fitted columns clamped.
        """
        self._require_fitted()
        projections = {
            name: col(name).clip(lit(low), lit(high)) for name, (low, high) in self.bounds_.items()
        }
        return ds.with_columns(**projections)


def mahalanobis_distance(
    ds: Dataset, columns: Sequence[str], *, output_column: str = "mahalanobis"
) -> Dataset:
    """Append each row's Mahalanobis distance from the multivariate center.

    The multivariate outlier score the univariate rules cannot give: a row can be unremarkable
    on every column on its own yet be a clear outlier in the *joint* distribution — a tall person
    who weighs very little. The Mahalanobis distance measures how far a row sits from the mean in
    units that account for the columns' correlations and scales, so it catches exactly that.
    Under a multivariate normal its square is chi-squared with ``len(columns)`` degrees of
    freedom, which is how a threshold is chosen (`chi2_sf` gives the tail probability).

    The mean and covariance are learned from `ds` itself in one scan; only the covariance
    inversion, over a tiny ``d x d`` matrix, runs on the driver. The distance is then a single
    quadratic-form expression, so scoring every row is one streaming pass.

    Args:
        ds: The dataset to score.
        columns: The numeric columns defining the space.
        output_column: The name of the distance column to append.

    Returns:
        A new lazy `Dataset` with the Mahalanobis distance appended.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.outliers import mahalanobis_distance
            >>> ds = bt.from_pydict(
            ...     {"x": [1.0, 2.0, 3.0, 4.0, 50.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
            ... )
            >>> scored = mahalanobis_distance(ds, ["x", "y"]).to_pydict()["mahalanobis"]
            >>> scored[4] == max(scored)  # the (50, 5) row is the joint outlier
            True
    """
    import numpy as np

    from batcher.ml.stats.multivariate import covariance_matrix

    _require(ds, *columns)
    names = list(columns)
    from batcher.plan.functions.aggregate import mean as mean_

    means = ds.agg(**{name: mean_(col(name)) for name in names}).collect()
    center = [float(means.column(name)[0].as_py()) for name in names]
    covariance = covariance_matrix(ds, names).to_pydict()
    matrix = np.array([covariance[name] for name in names], dtype=float).T
    inverse = np.linalg.pinv(matrix)
    centered = [col(name) - lit(center[index]) for index, name in enumerate(names)]
    quadratic = lit(0.0)
    for i in range(len(names)):
        for j in range(len(names)):
            weight = float(inverse[i, j])
            if weight != 0.0:
                quadratic = quadratic + lit(weight) * centered[i] * centered[j]
    return ds.with_columns(**{output_column: quadratic.sqrt()})
