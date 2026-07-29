"""Ordinal encoders — fit the category set, transform with a CASE projection.

`fit` learns each column's sorted distinct values (one bounded `distinct` over the
engine); `transform` lowers to a `CASE`/`when` expression chain. No per-row Python: the
mapping is an `Expr` the engine evaluates. Values not seen at fit time map to
`unknown_value`; nulls are treated as unknown.

The CASE chain carries one arm per category, so the size of the *plan* grows with the
fitted cardinality. `max_categories` bounds that: a fit over an unbounded column fails
with an actionable error rather than building a million-arm expression. Lowering a
high-cardinality mapping in constant plan size needs a native dictionary-lookup `Expr`
in `bc-expr`, which does not exist yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import (
    MAX_CATEGORIES,
    Preprocessor,
    column_arg,
    columns_arg,
    distinct_values,
)
from batcher.plan.expr_ir import Expr, col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["LabelEncoder", "OrdinalEncoder"]


def ordinal_expr(column: str, categories: list[Any], unknown_value: int) -> Expr:
    """A CASE expression mapping each category to its index, else `unknown_value`."""
    builder = None
    for idx, cat in enumerate(categories):
        cond = col(column) == cat
        builder = when(cond).then(idx) if builder is None else builder.when(cond).then(idx)
    if builder is None:
        # No categories were learned (an all-null column, or an empty fit set): every row
        # is "unseen", so the whole column is `unknown_value`. A broadcast literal is the
        # only correct constant here — `col(column) * 0` raises on a null/string column
        # (`Null * Int64` / `Utf8 * Int64`), crashing the documented all-unknown result.
        return lit(unknown_value)
    return builder.otherwise(unknown_value)


class OrdinalEncoder(Preprocessor):
    """Map each categorical column to an integer code by sorted category order.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import OrdinalEncoder
            >>> ds = bt.from_pydict({"c": ["b", "a", "c", "a"]})
            >>> OrdinalEncoder(["c"]).fit_transform(ds).to_pydict()
            {'c': [1, 0, 2, 0]}

    Args:
        columns: the categorical columns to encode in place.
        unknown_value: the code for values unseen at fit time (and nulls).
        max_categories: the ceiling on each column's fitted cardinality. Each category
            becomes one CASE arm, so this bounds both the plan size and the category set
            read back to the driver.
    """

    __slots__ = ("categories_", "columns", "max_categories", "unknown_value")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        unknown_value: int = -1,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        self.columns = columns_arg(columns, what="OrdinalEncoder")
        if not self.columns:
            raise PlanError("OrdinalEncoder requires at least one column")
        self.unknown_value = unknown_value
        self.max_categories = max_categories
        self.categories_: dict[str, list[Any]] = {}

    def fit(self, ds: Dataset) -> OrdinalEncoder:
        """Learn each column's sorted distinct categories from `ds`.

        Stored in `categories_[c]`; the code assigned to a value at transform time is
        its index into that sorted list.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import OrdinalEncoder
                >>> pre = OrdinalEncoder(["c"]).fit(bt.from_pydict({"c": ["b", "a", "c"]}))
                >>> pre.categories_
                {'c': ['a', 'b', 'c']}

        Args:
            ds: The dataset to learn each column's category set from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column has more than `max_categories` distinct values.
        """
        for c in self.columns:
            self.categories_[c] = distinct_values(
                ds, c, what="OrdinalEncoder", max_categories=self.max_categories
            )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its integer category code.

        Values unseen at fit time (and nulls) map to `unknown_value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import OrdinalEncoder
                >>> ds = bt.from_pydict({"c": ["b", "a", "c", "a"]})
                >>> OrdinalEncoder(["c"]).fit(ds).transform(ds).to_pydict()
                {'c': [1, 0, 2, 0]}

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with each fitted column replaced by its codes.
        """
        self._require_fitted()
        new = {c: ordinal_expr(c, self.categories_[c], self.unknown_value) for c in self.columns}
        return ds.with_columns(**new)


class LabelEncoder(Preprocessor):
    """Encode a single (target) column's labels as integers ``0..k-1``.

    The 1-D analogue of `OrdinalEncoder` for a label column `y`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import LabelEncoder
            >>> ds = bt.from_pydict({"y": ["cat", "dog", "cat"]})
            >>> LabelEncoder("y").fit_transform(ds).to_pydict()
            {'y': [0, 1, 0]}

    Args:
        column: the single label column to encode in place.
        unknown_value: the code for labels unseen at fit time (and nulls).
        max_categories: the ceiling on the fitted class count (one CASE arm each).
    """

    __slots__ = ("classes_", "column", "max_categories", "unknown_value")

    def __init__(
        self,
        column: str,
        *,
        unknown_value: int = -1,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        self.column = column_arg(column, what="LabelEncoder")
        self.unknown_value = unknown_value
        self.max_categories = max_categories
        self.classes_: list[Any] = []

    def fit(self, ds: Dataset) -> LabelEncoder:
        """Learn the sorted distinct labels of the column into `classes_`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LabelEncoder
                >>> LabelEncoder("y").fit(bt.from_pydict({"y": ["dog", "cat"]})).classes_
                ['cat', 'dog']

        Args:
            ds: The dataset to learn the label set from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the column has more than `max_categories` distinct labels.
        """
        self.classes_ = distinct_values(
            ds, self.column, what="LabelEncoder", max_categories=self.max_categories
        )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace the label column with each row's integer class index.

        Labels unseen at fit time (and nulls) map to `unknown_value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LabelEncoder
                >>> ds = bt.from_pydict({"y": ["cat", "dog", "cat"]})
                >>> LabelEncoder("y").fit(ds).transform(ds).to_pydict()
                {'y': [0, 1, 0]}

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with the label column replaced by its codes.
        """
        self._require_fitted()
        expr = ordinal_expr(self.column, self.classes_, self.unknown_value)
        return ds.with_columns(**{self.column: expr})
