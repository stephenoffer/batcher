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
from batcher.ml.preprocessors.base import Preprocessor, fit_aggregate
from batcher.plan.expr_ir import col, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["MaxAbsScaler", "MinMaxScaler", "Normalizer", "RobustScaler", "StandardScaler"]


def _check_columns(columns: Sequence[str]) -> list[str]:
    cols = list(columns)
    if not cols:
        raise PlanError("a scaler requires at least one column")
    return cols


class StandardScaler(Preprocessor):
    """Standardize columns to zero mean and unit variance: ``(x - mean) / std``.

    `std` is the **population** standard deviation (``ddof=0``, matching
    scikit-learn), computed as ``sqrt(E[x^2] - E[x]^2)`` so `fit` reuses only the
    mergeable `mean` aggregate. A constant column (zero variance) scales by 1.0
    (the column becomes its centered value), never dividing by zero.

    Args:
        columns: the numeric columns to standardize (replaced in place).
        with_mean: subtract the mean (center) when True.
        with_std: divide by the standard deviation (scale) when True.
    """

    __slots__ = ("columns", "mean_", "scale_", "with_mean", "with_std")

    def __init__(
        self, columns: Sequence[str], *, with_mean: bool = True, with_std: bool = True
    ) -> None:
        self.columns = _check_columns(columns)
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}

    def fit(self, ds: Dataset) -> StandardScaler:
        aggs = {}
        for c in self.columns:
            aggs[f"{c}__m"] = col(c).mean()
            aggs[f"{c}__sq"] = (col(c) * col(c)).mean()
        cell = fit_aggregate(ds, aggs)
        for c in self.columns:
            mean = cell[f"{c}__m"]
            mean_sq = cell[f"{c}__sq"]
            mean = 0.0 if mean is None else float(mean)
            var = max(float(mean_sq) - mean * mean, 0.0) if mean_sq is not None else 0.0
            self.mean_[c] = mean
            self.scale_[c] = math.sqrt(var) if (self.with_std and var > 0.0) else 1.0
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
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
    """

    __slots__ = ("columns", "data_max_", "data_min_", "feature_range")

    def __init__(
        self, columns: Sequence[str], *, feature_range: tuple[float, float] = (0.0, 1.0)
    ) -> None:
        self.columns = _check_columns(columns)
        lo, hi = feature_range
        if hi <= lo:
            raise PlanError(f"feature_range must be (lo, hi) with hi > lo, got {feature_range}")
        self.feature_range = (float(lo), float(hi))
        self.data_min_: dict[str, float] = {}
        self.data_max_: dict[str, float] = {}

    def fit(self, ds: Dataset) -> MinMaxScaler:
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
    """

    __slots__ = ("columns", "max_abs_")

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = _check_columns(columns)
        self.max_abs_: dict[str, float] = {}

    def fit(self, ds: Dataset) -> MaxAbsScaler:
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
        self._require_fitted()
        new = {}
        for c in self.columns:
            scale = self.max_abs_[c]
            new[c] = col(c) / scale if scale != 0.0 else col(c)
        return ds.with_columns(**new)


class RobustScaler(Preprocessor):
    """Scale columns by the median and interquartile range (outlier-robust).

    ``x' = (x - median) / (q75 - q25)``. A zero-IQR column scales by 1.0.
    """

    __slots__ = ("center_", "columns", "iqr_", "quantile_range")

    def __init__(
        self, columns: Sequence[str], *, quantile_range: tuple[float, float] = (25.0, 75.0)
    ) -> None:
        self.columns = _check_columns(columns)
        lo, hi = quantile_range
        if not (0.0 <= lo < hi <= 100.0):
            raise PlanError(f"quantile_range must be 0 <= lo < hi <= 100, got {quantile_range}")
        self.quantile_range = (lo / 100.0, hi / 100.0)
        self.center_: dict[str, float] = {}
        self.iqr_: dict[str, float] = {}

    def fit(self, ds: Dataset) -> RobustScaler:
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
        self._require_fitted()
        new = {}
        for c in self.columns:
            expr = col(c) - self.center_[c]
            if self.iqr_[c] != 0.0:
                expr = expr / self.iqr_[c]
            new[c] = expr
        return ds.with_columns(**new)


class Normalizer(Preprocessor):
    """Scale each **row** to unit norm across the given columns (sklearn
    ``Normalizer``) — a per-row operation, so it is **stateless** (no `fit`).

    ``norm="l2"`` (default) divides each value by ``sqrt(Σ xᵢ²)`` over the row's
    columns; ``"l1"`` by ``Σ|xᵢ|``; ``"max"`` by ``max|xᵢ|``. A zero-norm row (all
    zeros) is left unchanged. The whole transform is one `Expr` per column — no
    per-row Python.
    """

    __slots__ = ("columns", "norm")

    def __init__(self, columns: Sequence[str], *, norm: str = "l2") -> None:
        self.columns = _check_columns(columns)
        if norm not in ("l1", "l2", "max"):
            raise PlanError(f"norm must be 'l1', 'l2', or 'max', got {norm!r}")
        self.norm = norm
        self._fitted = True  # stateless

    def transform(self, ds: Dataset) -> Dataset:
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
