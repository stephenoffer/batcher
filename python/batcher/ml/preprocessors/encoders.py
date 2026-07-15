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
from batcher.plan.expr_ir import Expr, col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["LabelEncoder", "MultiHotEncoder", "OneHotEncoder", "OrdinalEncoder"]


def _ordinal_expr(column: str, categories: list[Any], unknown_value: int) -> Expr:
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
    """

    __slots__ = ("categories_", "columns", "unknown_value")

    def __init__(self, columns: Sequence[str], *, unknown_value: int = -1) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("OrdinalEncoder requires at least one column")
        self.unknown_value = unknown_value
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
        """
        for c in self.columns:
            self.categories_[c] = distinct_values(ds, c)
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
        new = {c: _ordinal_expr(c, self.categories_[c], self.unknown_value) for c in self.columns}
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
    """

    __slots__ = ("classes_", "column", "unknown_value")

    def __init__(self, column: str, *, unknown_value: int = -1) -> None:
        self.column = column
        self.unknown_value = unknown_value
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
        """
        self.classes_ = distinct_values(ds, self.column)
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
        expr = _ordinal_expr(self.column, self.classes_, self.unknown_value)
        return ds.with_columns(**{self.column: expr})


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
    """

    __slots__ = ("categories_", "columns", "drop_first")

    def __init__(self, columns: Sequence[str], *, drop_first: bool = False) -> None:
        self.columns = list(columns)
        if not self.columns:
            raise PlanError("OneHotEncoder requires at least one column")
        self.drop_first = drop_first
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
        """
        for c in self.columns:
            self.categories_[c] = distinct_values(ds, c)
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
    """

    __slots__ = ("categories_", "column")

    def __init__(self, column: str, *, categories: Sequence[Any] | None = None) -> None:
        self.column = column
        self.categories_: list[Any] | None = list(categories) if categories is not None else None
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
        """
        if self.categories_ is None:
            exploded = ds.select(self.column).explode(self.column)
            self.categories_ = distinct_values(exploded, self.column)
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
        assert self.categories_ is not None
        indicators = {
            f"{self.column}_{cat}": col(self.column).list.contains(cat).cast("int64")
            for cat in self.categories_
        }
        return ds.with_columns(**indicators)
