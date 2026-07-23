"""Numeric scalers — fit summary statistics, transform with an `Expr` projection.

Every scaler's `fit` is a single global aggregate (mean / min / max / quantiles) over
the existing mergeable runtime, and `transform` is an ordinary arithmetic `Expr` added
with `with_columns` — so the scaled column is computed in the engine, distributed and
spillable, never in per-row Python. Statistics are read back to the driver once and
become constants in the transform expression.
"""

from __future__ import annotations

import functools
import math
import operator
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg, fit_aggregate
from batcher.plan.expr_ir import col, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["MaxAbsScaler", "MinMaxScaler", "Normalizer", "RobustScaler", "StandardScaler"]


def _check_columns(columns: str | Sequence[str]) -> list[str]:
    """Normalize a scaler's `columns`, accepting a single name or a sequence."""
    return columns_arg(columns, what="a scaler")


class StandardScaler(Preprocessor):
    """Standardize columns to zero mean and unit variance: ``(x - mean) / std``.

    `std` is the **population** standard deviation (``ddof=0``, matching
    scikit-learn), derived from the engine's numerically stable (Welford) variance
    aggregate rather than the naive ``E[x^2] - E[x]^2``, which loses all precision
    to catastrophic cancellation on a large-magnitude column (e.g. values near
    ``1e8`` gave ``std=sqrt(2)`` instead of the true ``sqrt(1.25)``). A constant
    column (zero variance) scales by 1.0 (the column becomes its centered value),
    never dividing by zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import StandardScaler
            >>> ds = bt.from_pydict({"x": [1.0, 3.0]})
            >>> StandardScaler(["x"]).fit_transform(ds).to_pydict()
            {'x': [-1.0, 1.0]}

    Args:
        columns: the numeric columns to standardize (replaced in place).
        with_mean: subtract the mean (center) when True.
        with_std: divide by the standard deviation (scale) when True.
    """

    __slots__ = ("columns", "mean_", "scale_", "with_mean", "with_std")

    def __init__(
        self, columns: str | Sequence[str], *, with_mean: bool = True, with_std: bool = True
    ) -> None:
        self.columns = _check_columns(columns)
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}

    def fit(self, ds: Dataset) -> StandardScaler:
        """Learn each column's mean and population standard deviation from `ds`.

        Both come from one mergeable pass: `mean_[c]` is ``E[x]`` and `scale_[c]` is
        the population standard deviation from the engine's stable variance aggregate
        (1.0 for a zero-variance column).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import StandardScaler
                >>> pre = StandardScaler(["x"]).fit(bt.from_pydict({"x": [1.0, 3.0]}))
                >>> pre.mean_, pre.scale_
                ({'x': 2.0}, {'x': 1.0})

        Args:
            ds: The dataset to compute the per-column mean and std from.

        Returns:
            ``self``, fitted.
        """
        aggs = {}
        for c in self.columns:
            aggs[f"{c}__m"] = col(c).mean()
            # Sample variance (ddof=1) + count → population variance, using the engine's
            # numerically stable (Welford) aggregate. The old `E[x^2] - E[x]^2` form
            # cancelled catastrophically on large-magnitude columns (x^2 overflows f64's
            # 2^53 exact-integer range), yielding a badly wrong std.
            aggs[f"{c}__v"] = col(c).var()
            aggs[f"{c}__n"] = col(c).count()
        cell = fit_aggregate(ds, aggs)
        for c in self.columns:
            mean = cell[f"{c}__m"]
            svar = cell[f"{c}__v"]
            n = cell[f"{c}__n"]
            mean = 0.0 if mean is None else float(mean)
            # Convert sample variance (n-1 denominator) to population variance (n),
            # matching scikit-learn's ddof=0; a column with <2 values has zero variance.
            if svar is None or n is None or int(n) < 2:
                var = 0.0
            else:
                n = int(n)
                var = max(float(svar) * (n - 1) / n, 0.0)
            self.mean_[c] = mean
            self.scale_[c] = math.sqrt(var) if (self.with_std and var > 0.0) else 1.0
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each column with its standardized value ``(x - mean) / std``.

        Uses the statistics learned in `fit`; only the fitted columns are rewritten,
        all others pass through unchanged.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import StandardScaler
                >>> ds = bt.from_pydict({"x": [1.0, 3.0]})
                >>> pre = StandardScaler(["x"]).fit(ds)
                >>> pre.transform(bt.from_pydict({"x": [2.0, 4.0]})).to_pydict()
                {'x': [0.0, 2.0]}

        Args:
            ds: The dataset to standardize.

        Returns:
            A new lazy `Dataset` with the fitted columns standardized in place.
        """
        self._require_fitted()
        new = {}
        for c in self.columns:
            expr = col(c)
            if self.with_mean:
                expr = expr - self.mean_[c]
            if self.with_std and self.scale_[c] != 1.0:
                expr = expr / self.scale_[c]
            new[c] = expr
        return ds.with_columns(**new)


class MinMaxScaler(Preprocessor):
    """Scale columns into ``feature_range`` (default ``[0, 1]``) by min and max.

    ``x' = (x - min) / (max - min) * (hi - lo) + lo``. A constant column maps to
    `lo` (range collapses), never dividing by zero.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import MinMaxScaler
            >>> ds = bt.from_pydict({"x": [0.0, 5.0, 10.0]})
            >>> MinMaxScaler(["x"]).fit_transform(ds).to_pydict()
            {'x': [0.0, 0.5, 1.0]}

    Args:
        columns: the numeric columns to scale (replaced in place).
        feature_range: the ``(lo, hi)`` target range (``hi`` must exceed ``lo``).
    """

    __slots__ = ("columns", "data_max_", "data_min_", "feature_range")

    def __init__(
        self, columns: str | Sequence[str], *, feature_range: tuple[float, float] = (0.0, 1.0)
    ) -> None:
        self.columns = _check_columns(columns)
        lo, hi = feature_range
        if hi <= lo:
            raise PlanError(f"feature_range must be (lo, hi) with hi > lo, got {feature_range}")
        self.feature_range = (float(lo), float(hi))
        self.data_min_: dict[str, float] = {}
        self.data_max_: dict[str, float] = {}

    def fit(self, ds: Dataset) -> MinMaxScaler:
        """Learn each column's min and max from `ds` (one mergeable aggregate).

        Stored as `data_min_[c]` / `data_max_[c]`; `transform` reads them as constants.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MinMaxScaler
                >>> pre = MinMaxScaler(["x"]).fit(bt.from_pydict({"x": [0.0, 5.0, 10.0]}))
                >>> pre.data_min_, pre.data_max_
                ({'x': 0.0}, {'x': 10.0})

        Args:
            ds: The dataset to compute the per-column min and max from.

        Returns:
            ``self``, fitted.
        """
        aggs = {}
        for c in self.columns:
            aggs[f"{c}__min"] = col(c).min()
            aggs[f"{c}__max"] = col(c).max()
        cell = fit_aggregate(ds, aggs)
        for c in self.columns:
            lo = cell[f"{c}__min"]
            hi = cell[f"{c}__max"]
            self.data_min_[c] = 0.0 if lo is None else float(lo)
            self.data_max_[c] = 0.0 if hi is None else float(hi)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Rescale each fitted column into ``feature_range`` in place.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MinMaxScaler
                >>> ds = bt.from_pydict({"x": [0.0, 5.0, 10.0]})
                >>> MinMaxScaler(["x"]).fit(ds).transform(ds).to_pydict()
                {'x': [0.0, 0.5, 1.0]}

        Args:
            ds: The dataset to rescale.

        Returns:
            A new lazy `Dataset` with the fitted columns rescaled in place.
        """
        self._require_fitted()
        lo, hi = self.feature_range
        new = {}
        for c in self.columns:
            span = self.data_max_[c] - self.data_min_[c]
            if span == 0.0:
                new[c] = col(c) * 0.0 + lo
            else:
                scaled = (col(c) - self.data_min_[c]) / span
                new[c] = scaled * (hi - lo) + lo if (hi - lo) != 1.0 or lo != 0.0 else scaled
        return ds.with_columns(**new)


class MaxAbsScaler(Preprocessor):
    """Scale each column by its maximum absolute value into ``[-1, 1]``.

    ``x' = x / max(|x|)``; preserves sparsity (no centering). An all-zero column is
    left unchanged (scale 1.0).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import MaxAbsScaler
            >>> ds = bt.from_pydict({"x": [-2.0, 1.0, 4.0]})
            >>> MaxAbsScaler(["x"]).fit_transform(ds).to_pydict()
            {'x': [-0.5, 0.25, 1.0]}

    Args:
        columns: the numeric columns to scale (replaced in place).
    """

    __slots__ = ("columns", "max_abs_")

    def __init__(self, columns: str | Sequence[str]) -> None:
        self.columns = _check_columns(columns)
        self.max_abs_: dict[str, float] = {}

    def fit(self, ds: Dataset) -> MaxAbsScaler:
        """Learn each column's maximum absolute value from `ds`.

        Computed as ``max(|min|, |max|)`` (stored in `max_abs_`) so `fit` reuses the
        mergeable min/max aggregates.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MaxAbsScaler
                >>> pre = MaxAbsScaler(["x"]).fit(bt.from_pydict({"x": [-2.0, 1.0, 4.0]}))
                >>> pre.max_abs_
                {'x': 4.0}

        Args:
            ds: The dataset to compute each column's max absolute value from.

        Returns:
            ``self``, fitted.
        """
        aggs = {}
        for c in self.columns:
            aggs[f"{c}__min"] = col(c).min()
            aggs[f"{c}__max"] = col(c).max()
        cell = fit_aggregate(ds, aggs)
        for c in self.columns:
            lo = cell[f"{c}__min"]
            hi = cell[f"{c}__max"]
            lo = 0.0 if lo is None else float(lo)
            hi = 0.0 if hi is None else float(hi)
            self.max_abs_[c] = max(abs(lo), abs(hi))
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Divide each fitted column by its max absolute value in place.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MaxAbsScaler
                >>> ds = bt.from_pydict({"x": [-2.0, 1.0, 4.0]})
                >>> MaxAbsScaler(["x"]).fit(ds).transform(ds).to_pydict()
                {'x': [-0.5, 0.25, 1.0]}

        Args:
            ds: The dataset to scale.

        Returns:
            A new lazy `Dataset` with the fitted columns scaled into ``[-1, 1]``.
        """
        self._require_fitted()
        new = {}
        for c in self.columns:
            scale = self.max_abs_[c]
            new[c] = col(c) / scale if scale != 0.0 else col(c)
        return ds.with_columns(**new)


class RobustScaler(Preprocessor):
    """Scale columns by the median and interquartile range (outlier-robust).

    ``x' = (x - median) / (q75 - q25)``. A zero-IQR column scales by 1.0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import RobustScaler
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
            >>> RobustScaler(["x"]).fit_transform(ds).to_pydict()
            {'x': [-1.0, -0.5, 0.0, 0.5, 1.0]}

    Args:
        columns: the numeric columns to scale (replaced in place).
        quantile_range: the ``(lo, hi)`` percentiles bounding the IQR (default 25/75).
    """

    __slots__ = ("center_", "columns", "iqr_", "quantile_range")

    def __init__(
        self, columns: str | Sequence[str], *, quantile_range: tuple[float, float] = (25.0, 75.0)
    ) -> None:
        self.columns = _check_columns(columns)
        lo, hi = quantile_range
        if not (0.0 <= lo < hi <= 100.0):
            raise PlanError(f"quantile_range must be 0 <= lo < hi <= 100, got {quantile_range}")
        self.quantile_range = (lo / 100.0, hi / 100.0)
        self.center_: dict[str, float] = {}
        self.iqr_: dict[str, float] = {}

    def fit(self, ds: Dataset) -> RobustScaler:
        """Learn each column's median (`center_`) and interquartile range (`iqr_`).

        Both come from one mergeable quantile aggregate; a zero-IQR column keeps a
        scale of 1.0.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RobustScaler
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
                >>> pre = RobustScaler(["x"]).fit(ds)
                >>> pre.center_, pre.iqr_
                ({'x': 3.0}, {'x': 2.0})

        Args:
            ds: The dataset to compute each column's median and IQR from.

        Returns:
            ``self``, fitted.
        """
        q_lo, q_hi = self.quantile_range
        aggs = {}
        for c in self.columns:
            aggs[f"{c}__med"] = col(c).median()
            aggs[f"{c}__lo"] = col(c).quantile(q_lo)
            aggs[f"{c}__hi"] = col(c).quantile(q_hi)
        cell = fit_aggregate(ds, aggs)
        for c in self.columns:
            med = cell[f"{c}__med"]
            lo = cell[f"{c}__lo"]
            hi = cell[f"{c}__hi"]
            self.center_[c] = 0.0 if med is None else float(med)
            iqr = 0.0 if (lo is None or hi is None) else float(hi) - float(lo)
            self.iqr_[c] = iqr
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Center each fitted column by its median and scale by its IQR in place.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RobustScaler
                >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
                >>> RobustScaler(["x"]).fit(ds).transform(ds).to_pydict()
                {'x': [-1.0, -0.5, 0.0, 0.5, 1.0]}

        Args:
            ds: The dataset to scale.

        Returns:
            A new lazy `Dataset` with the fitted columns robustly scaled in place.
        """
        self._require_fitted()
        new = {}
        for c in self.columns:
            expr = col(c) - self.center_[c]
            if self.iqr_[c] != 0.0:
                expr = expr / self.iqr_[c]
            new[c] = expr
        return ds.with_columns(**new)


class Normalizer(Preprocessor):
    """Scale each **row** to unit norm across the given columns (sklearn ``Normalizer``).

    A per-row operation, so it is **stateless** (no `fit`). ``norm="l2"`` (default)
    divides each value by ``sqrt(Σ xᵢ²)`` over the row's columns; ``"l1"`` by
    ``Σ|xᵢ|``; ``"max"`` by ``max|xᵢ|``. A zero-norm row (all zeros) is left unchanged.
    The whole transform is one `Expr` per column — no per-row Python.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import Normalizer
            >>> ds = bt.from_pydict({"a": [3.0, 0.0], "b": [4.0, 0.0]})
            >>> Normalizer(["a", "b"]).fit_transform(ds).to_pydict()
            {'a': [0.6, 0.0], 'b': [0.8, 0.0]}

    Args:
        columns: the numeric columns that together form each row's vector.
        norm: the norm to divide by — ``"l1"``, ``"l2"``, or ``"max"``.
    """

    __slots__ = ("columns", "norm")

    def __init__(self, columns: str | Sequence[str], *, norm: str = "l2") -> None:
        self.columns = _check_columns(columns)
        if norm not in ("l1", "l2", "max"):
            raise PlanError(f"norm must be 'l1', 'l2', or 'max', got {norm!r}")
        self.norm = norm
        self._fitted = True  # stateless

    def transform(self, ds: Dataset) -> Dataset:
        """Divide each column by the row's norm across all the given columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Normalizer
                >>> ds = bt.from_pydict({"a": [3.0, 0.0], "b": [4.0, 0.0]})
                >>> Normalizer(["a", "b"]).transform(ds).to_pydict()
                {'a': [0.6, 0.0], 'b': [0.8, 0.0]}

        Args:
            ds: The dataset whose rows to normalize.

        Returns:
            A new lazy `Dataset` with each column divided by the per-row norm.
        """
        cols = [col(c) for c in self.columns]
        if self.norm == "l2":
            norm = functools.reduce(operator.add, (c * c for c in cols)).sqrt()
        elif self.norm == "l1":
            norm = functools.reduce(operator.add, (c.abs() for c in cols))
        else:  # max
            from batcher.plan.expr_ir.constructors import greatest

            norm = greatest(*(c.abs() for c in cols))
        # Guard a zero-norm row: divide by 1 so the (all-zero) values stay unchanged.
        divisor = when(norm == 0.0).then(1.0).otherwise(norm)
        return ds.with_columns(**{c: col(c) / divisor for c in self.columns})
