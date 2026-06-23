"""Categorical encoders — fit the category set, transform with a CASE projection.

`fit` learns each column's sorted distinct values (one `distinct` over the engine);
`transform` lowers to a `CASE`/`when` expression chain (`OrdinalEncoder`/`LabelEncoder`)
or a set of 0/1 indicator columns (`OneHotEncoder`). No per-row Python: the mapping is
an `Expr` the engine evaluates. Values not seen at fit time map to `unknown_value`
(ordinal) or all-zero indicators (one-hot); nulls are treated as unknown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, distinct_values
from batcher.plan.expr_ir import Expr, col, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["LabelEncoder", "OneHotEncoder", "OrdinalEncoder"]


def _ordinal_expr(column: str, categories: list[Any], unknown_value: int) -> Expr:
    """A CASE expression mapping each category to its index, else `unknown_value`."""
    builder = None
    for idx, cat in enumerate(categories):
        cond = col(column) == cat
        builder = when(cond).then(idx) if builder is None else builder.when(cond).then(idx)
    if builder is None:
        return col(column) * 0 + unknown_value
    return builder.otherwise(unknown_value)


class OrdinalEncoder(Preprocessor):
    """Map each categorical column to an integer code by sorted category order.

    Args:
        columns: the categorical columns to encode in place.
        unknown_value: the code for values unseen at fit time (and nulls).
    """

    __slots__ = ("categories_", "columns", "unknown_value")

    def __init__(self, columns: Sequence[str], *, unknown_value: int = -1) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("OrdinalEncoder requires at least one column")
        self.unknown_value = unknown_value
        self.categories_: dict[str, list[Any]] = {}

    def fit(self, ds: Dataset) -> OrdinalEncoder:
        for c in self.columns:
            self.categories_[c] = distinct_values(ds, c)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        self._require_fitted()
        new = {c: _ordinal_expr(c, self.categories_[c], self.unknown_value) for c in self.columns}
        return ds.with_columns(**new)


class LabelEncoder(Preprocessor):
    """Encode a single (target) column's labels as integers ``0..k-1``.

    The 1-D analogue of `OrdinalEncoder` for a label column `y`.
    """

    __slots__ = ("classes_", "column", "unknown_value")

    def __init__(self, column: str, *, unknown_value: int = -1) -> None:
        self.column = column
        self.unknown_value = unknown_value
        self.classes_: list[Any] = []

    def fit(self, ds: Dataset) -> LabelEncoder:
        self.classes_ = distinct_values(ds, self.column)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        self._require_fitted()
        expr = _ordinal_expr(self.column, self.classes_, self.unknown_value)
        return ds.with_columns(**{self.column: expr})


class OneHotEncoder(Preprocessor):
    """Expand each categorical column into 0/1 indicator columns, one per category.

    The encoded columns are dropped and replaced by ``{column}_{category}`` indicators
    (the scikit-learn convention). Unseen values produce all-zero indicators.

    Args:
        columns: the categorical columns to one-hot encode.
        drop_first: omit the first category's indicator (dummy encoding) when True.
    """

    __slots__ = ("categories_", "columns", "drop_first")

    def __init__(self, columns: Sequence[str], *, drop_first: bool = False) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("OneHotEncoder requires at least one column")
        self.drop_first = drop_first
        self.categories_: dict[str, list[Any]] = {}

    def fit(self, ds: Dataset) -> OneHotEncoder:
        for c in self.columns:
            self.categories_[c] = distinct_values(ds, c)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        self._require_fitted()
        encoded = set(self.columns)
        keep = [c for c in ds.columns if c not in encoded]
        indicators: dict[str, Expr] = {}
        for c in self.columns:
            cats = self.categories_[c][1:] if self.drop_first else self.categories_[c]
            for cat in cats:
                indicators[f"{c}_{cat}"] = when(col(c) == cat).then(1).otherwise(0)
        return ds.select(*keep, **indicators)
