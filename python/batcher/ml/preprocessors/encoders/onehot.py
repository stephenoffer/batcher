"""Indicator encoders — one 0/1 output column per learned category.

`OneHotEncoder` expands a scalar categorical column; `MultiHotEncoder` does the same for
a list column. Both emit one column per category, so the fitted cardinality is the width
of the output schema: `max_categories` bounds it, because an unguarded fit over a
high-cardinality column produces a plan with as many columns as the column has values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import (
    MAX_CATEGORIES,
    Preprocessor,
    check_cardinality,
    columns_arg,
    distinct_values,
    require_categories,
)
from batcher.plan.expr_ir import Expr, col, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["MultiHotEncoder", "OneHotEncoder"]


class OneHotEncoder(Preprocessor):
    """Expand each categorical column into 0/1 indicator columns, one per category.

    The encoded columns are dropped and replaced by ``{column}_{category}`` indicators
    (the scikit-learn convention). Unseen values produce all-zero indicators.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import OneHotEncoder
            >>> ds = bt.from_pydict({"c": ["a", "b", "a"]})
            >>> OneHotEncoder(["c"]).fit_transform(ds).to_pydict()
            {'c_a': [1, 0, 1], 'c_b': [0, 1, 0]}

    Args:
        columns: the categorical columns to one-hot encode.
        drop_first: omit the first category's indicator (dummy encoding) when True.
        max_categories: the ceiling on each column's fitted cardinality. One-hot emits a
            column per category, so this is the guard against an unbounded output schema.
    """

    __slots__ = ("categories_", "columns", "drop_first", "max_categories")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        drop_first: bool = False,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        self.columns = columns_arg(columns, what="OneHotEncoder")
        if not self.columns:
            raise PlanError("OneHotEncoder requires at least one column")
        self.drop_first = drop_first
        self.max_categories = max_categories
        self.categories_: dict[str, list[Any]] = {}

    def fit(self, ds: Dataset) -> OneHotEncoder:
        """Learn each column's sorted distinct categories (one indicator per category).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import OneHotEncoder
                >>> pre = OneHotEncoder(["c"]).fit(bt.from_pydict({"c": ["a", "b", "a"]}))
                >>> pre.categories_
                {'c': ['a', 'b']}

        Args:
            ds: The dataset to learn each column's category set from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column has more than `max_categories` distinct values.
        """
        for c in self.columns:
            self.categories_[c] = require_categories(
                distinct_values(ds, c, what="OneHotEncoder", max_categories=self.max_categories),
                what="OneHotEncoder",
                column=c,
            )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Drop each fitted column and emit its ``{column}_{category}`` 0/1 indicators.

        Non-encoded columns pass through; unseen values yield all-zero indicators.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import OneHotEncoder
                >>> ds = bt.from_pydict({"c": ["a", "b", "a"]})
                >>> OneHotEncoder(["c"]).fit(ds).transform(ds).to_pydict()
                {'c_a': [1, 0, 1], 'c_b': [0, 1, 0]}

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with indicator columns replacing the fitted ones.
        """
        self._require_fitted()
        encoded = set(self.columns)
        keep = [c for c in ds.columns if c not in encoded]
        indicators: dict[str, Expr] = {}
        for c in self.columns:
            cats = self.categories_[c][1:] if self.drop_first else self.categories_[c]
            for cat in cats:
                indicators[f"{c}_{cat}"] = when(col(c) == cat).then(1).otherwise(0)
        return ds.select(*keep, **indicators)


class MultiHotEncoder(Preprocessor):
    """Multi-hot encode a **list** column into one 0/1 indicator column per category.

    The multi-label counterpart of `OneHotEncoder`. `fit` learns the distinct elements
    across all the column's lists (explode + distinct); `transform` emits
    ``{column}_{category}`` columns, each 1 where that category appears in the row's
    list (via ``.list.contains``), 0 otherwise. Useful for tag/label sets. Pass
    `categories` to fix the vocabulary and skip `fit`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import MultiHotEncoder
            >>> ds = bt.from_pydict({"tags": [["x", "y"], ["y"]]})
            >>> MultiHotEncoder("tags").fit_transform(ds).to_pydict()
            {'tags': [['x', 'y'], ['y']], 'tags_x': [1, 0], 'tags_y': [1, 1]}

    Args:
        column: the list column to encode (kept alongside the indicators).
        categories: an explicit category vocabulary; learned from the data if None.
        max_categories: the ceiling on the vocabulary size, learned or explicit — one
            output column each.
    """

    __slots__ = ("categories_", "column", "max_categories")

    def __init__(
        self,
        column: str,
        *,
        categories: Sequence[Any] | None = None,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        self.column = column
        self.max_categories = max_categories
        self.categories_: list[Any] | None = list(categories) if categories is not None else None
        if self.categories_ is not None:
            check_cardinality(
                "MultiHotEncoder", column, len(self.categories_), max_categories, exact=True
            )
        self._fitted = categories is not None

    def fit(self, ds: Dataset) -> MultiHotEncoder:
        """Learn the distinct elements across all the column's lists into `categories_`.

        A no-op when an explicit `categories` vocabulary was given to the constructor.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MultiHotEncoder
                >>> ds = bt.from_pydict({"tags": [["x", "y"], ["y"]]})
                >>> MultiHotEncoder("tags").fit(ds).categories_
                ['x', 'y']

        Args:
            ds: The dataset whose list column supplies the category vocabulary.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the vocabulary is larger than `max_categories`.
        """
        if self.categories_ is None:
            exploded = ds.select(self.column).explode(self.column)
            self.categories_ = require_categories(
                distinct_values(
                    exploded,
                    self.column,
                    what="MultiHotEncoder",
                    max_categories=self.max_categories,
                ),
                what="MultiHotEncoder",
                column=self.column,
            )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Add one ``{column}_{category}`` 0/1 indicator per learned category.

        Each indicator is 1 where the category is in the row's list. The source list
        column is kept.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MultiHotEncoder
                >>> ds = bt.from_pydict({"tags": [["x", "y"], ["y"]]})
                >>> MultiHotEncoder("tags").fit(ds).transform(ds).to_pydict()
                {'tags': [['x', 'y'], ['y']], 'tags_x': [1, 0], 'tags_y': [1, 1]}

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with an indicator column per category added.
        """
        self._require_fitted()
        if self.categories_ is None:  # pragma: no cover - _require_fitted guarantees this
            raise PlanError("MultiHotEncoder has no category vocabulary")
        indicators = {
            f"{self.column}_{cat}": col(self.column).list.contains(cat).cast("int64")
            for cat in self.categories_
        }
        return ds.with_columns(**indicators)
