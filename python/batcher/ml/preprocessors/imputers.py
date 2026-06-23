"""Missing-value imputation — fit a fill value per column, transform with COALESCE.

`fit` learns each column's fill value (mean / median / most-frequent / a constant),
each a single aggregate over the engine; `transform` replaces nulls with
``coalesce(col, fill)`` — an `Expr`, so the fill happens in the data plane. ``mean``
and ``median`` cast the column to float (the scikit-learn convention); ``most_frequent``
and ``constant`` keep the original type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, fit_aggregate
from batcher.plan.expr_ir import coalesce, col, count, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["SimpleImputer"]

_STRATEGIES = ("mean", "median", "most_frequent", "constant")


class SimpleImputer(Preprocessor):
    """Fill missing values in `columns` using a per-column statistic.

    Args:
        columns: the columns to impute in place.
        strategy: ``"mean"``, ``"median"``, ``"most_frequent"``, or ``"constant"``.
        fill_value: the constant to use when ``strategy="constant"`` (required then).
    """

    __slots__ = ("columns", "fill_value", "statistics_", "strategy")

    def __init__(
        self,
        columns: Sequence[str],
        *,
        strategy: str = "mean",
        fill_value: Any = None,
    ) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("SimpleImputer requires at least one column")
        if strategy not in _STRATEGIES:
            raise PlanError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")
        if strategy == "constant" and fill_value is None:
            raise PlanError("SimpleImputer(strategy='constant') requires fill_value")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: dict[str, Any] = {}

    def fit(self, ds: Dataset) -> SimpleImputer:
        if self.strategy == "constant":
            self.statistics_ = dict.fromkeys(self.columns, self.fill_value)
        elif self.strategy == "most_frequent":
            self.statistics_ = {c: self._mode(ds, c) for c in self.columns}
        elif self.strategy == "mean":
            self.statistics_ = dict(fit_aggregate(ds, {c: col(c).mean() for c in self.columns}))
        else:  # median
            self.statistics_ = dict(fit_aggregate(ds, {c: col(c).median() for c in self.columns}))
        for c in self.columns:
            if self.statistics_[c] is None:
                raise PlanError(
                    f"SimpleImputer cannot fit column {c!r}: no non-null values for "
                    f"strategy {self.strategy!r}"
                )
        self._fitted = True
        return self

    @staticmethod
    def _mode(ds: Dataset, column: str) -> Any:
        """The most frequent non-null value of `column` (ties: engine order)."""
        grouped = (
            ds.filter(col(column).is_not_null())
            .group_by(column)
            .agg(__n=count())
            .sort("__n", descending=True)
            .limit(1)
            .collect()
        )
        return grouped.column(column)[0].as_py() if grouped.num_rows else None

    def transform(self, ds: Dataset) -> Dataset:
        self._require_fitted()
        cast_float = self.strategy in ("mean", "median")
        new = {}
        for c in self.columns:
            base = col(c).cast("float64") if cast_float else col(c)
            new[c] = coalesce(base, lit(self.statistics_[c]))
        return ds.with_columns(**new)
