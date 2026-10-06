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
    output_columns_arg,
    output_pairs,
)
from batcher.plan.expr_ir import Expr, col, lit, when
from batcher.plan.functions.collection import element

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["LabelEncoder", "OrdinalEncoder"]


def ordinal_expr(subject: str | Expr, categories: list[Any], unknown_value: int) -> Expr:
    """A CASE expression mapping each category to its index, else `unknown_value`.

    `subject` is a column name, or an expression such as `element()` when the codes are
    computed per list element.
    """
    value = col(subject) if isinstance(subject, str) else subject
    builder = None
    for idx, cat in enumerate(categories):
        cond = value == cat
        builder = when(cond).then(idx) if builder is None else builder.when(cond).then(idx)
    if builder is None:
        # No categories were learned (an all-null column, or an empty fit set): every row
        # is "unseen", so the whole column is `unknown_value`. A broadcast literal is the
        # only correct constant here — `col(column) * 0` raises on a null/string column
        # (`Null * Int64` / `Utf8 * Int64`), crashing the documented all-unknown result.
        return lit(unknown_value)
    return builder.otherwise(unknown_value)


def category_expr(subject: str | Expr, categories: list[Any]) -> Expr:
    """The inverse of `ordinal_expr`: each code back to its category, anything else null.

    The unknown code has no category to return to, because unseen values and nulls share it,
    so it maps to null rather than to a guess.
    """
    value = col(subject) if isinstance(subject, str) else subject
    builder = None
    for idx, cat in enumerate(categories):
        cond = value == idx
        builder = when(cond).then(cat) if builder is None else builder.when(cond).then(cat)
    return lit(None) if builder is None else builder


def _is_list_column(ds: Dataset, column: str) -> bool:
    """Whether `column` is a list column, read from the schema without a scan."""
    import pyarrow as pa

    schema = ds.schema
    if column not in schema.names:
        return False
    dtype = schema.field(column).type
    return pa.types.is_list(dtype) or pa.types.is_large_list(dtype)


class OrdinalEncoder(Preprocessor):
    """Map each categorical column to an integer code by sorted category order.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import OrdinalEncoder
            >>> ds = bt.from_pydict({"c": ["b", "a", "c", "a"]})
            >>> OrdinalEncoder(["c"]).fit_transform(ds).to_pydict()
            {'c': [1, 0, 2, 0]}

            >>> tags = bt.from_pydict({"t": [["b", "a"], ["c"], []]})
            >>> OrdinalEncoder("t", output_columns="t_codes").fit_transform(tags).to_pydict()
            {'t': [['b', 'a'], ['c'], []], 't_codes': [[1, 0], [2], []]}

    Args:
        columns: the categorical columns to encode in place.
        unknown_value: the code for values unseen at fit time (and nulls).
        max_categories: the ceiling on each column's fitted cardinality. Each category
            becomes one CASE arm, so this bounds both the plan size and the category set
            read back to the driver.
        encode_lists: encode each element of a list column, learning the categories from
            the elements (Ray Data's default). ``False`` would treat each whole list as one
            category, which needs list literals the expression layer does not have, so it
            raises on a list column. A scalar column ignores this.
        output_columns: write each code column to this name instead of over its input,
            one name per column in order, keeping the inputs (Ray Data's
            ``output_columns``). ``None`` (the default) encodes in place.
    """

    __slots__ = (
        "categories_",
        "columns",
        "encode_lists",
        "max_categories",
        "output_columns",
        "unknown_value",
    )

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        unknown_value: int = -1,
        max_categories: int = MAX_CATEGORIES,
        encode_lists: bool = True,
        output_columns: str | Sequence[str] | None = None,
    ) -> None:
        self.columns = columns_arg(columns, what="OrdinalEncoder")
        if not self.columns:
            raise PlanError("OrdinalEncoder requires at least one column")
        self.unknown_value = unknown_value
        self.max_categories = max_categories
        self.encode_lists = encode_lists
        self.output_columns = output_columns_arg(
            self.columns, output_columns, what="OrdinalEncoder"
        )
        self.categories_: dict[str, list[Any]] = {}

    def fit(self, ds: Dataset) -> OrdinalEncoder:
        """Learn each column's sorted distinct categories from `ds`.

        Unlike `transform`, which only builds a lazy plan, `fit` executes: it runs a
        query over `ds` now and reads the learned state back to the driver.

        Stored in `categories_[c]`; the code assigned to a value at transform time is
        its index into that sorted list. A list column learns from its elements.

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
            source = ds
            if _is_list_column(ds, c):
                self._require_element_encoding(c)
                source = ds.select(c).explode(c)
            self.categories_[c] = distinct_values(
                source, c, what="OrdinalEncoder", max_categories=self.max_categories
            )
        self._fitted = True
        return self

    def _require_element_encoding(self, column: str) -> None:
        """Refuse ``encode_lists=False`` on a list column, naming what it would need."""
        if not self.encode_lists:
            raise PlanError(
                f"OrdinalEncoder(encode_lists=False): column {column!r} is a list column, and "
                "encoding each whole list as one category needs list literals, which the "
                "expression layer does not support. Use encode_lists=True to encode each "
                "element, or turn the list into a string first (col(...).list.join(','))."
            )

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its integer category code.

        Values unseen at fit time (and nulls) map to `unknown_value`. A list column is
        encoded element by element, and a null list stays null.

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
        new = {}
        for c, out in output_pairs(self.columns, self.output_columns):
            if _is_list_column(ds, c):
                self._require_element_encoding(c)
                codes = ordinal_expr(element(), self.categories_[c], self.unknown_value)
                new[out] = col(c).list.transform(codes)
            else:
                new[out] = ordinal_expr(c, self.categories_[c], self.unknown_value)
        return ds.with_columns(**new)

    def inverse_transform(self, ds: Dataset) -> Dataset:
        """Map each code back to its category, written to the source column.

        Exact for every code `fit` assigned. The `unknown_value` code becomes null, since
        the unseen values and nulls that share it cannot be told apart. A list column is
        decoded element by element.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import OrdinalEncoder
                >>> ds = bt.from_pydict({"c": ["b", "a", "c"]})
                >>> pre = OrdinalEncoder("c").fit(ds)
                >>> pre.inverse_transform(pre.transform(ds)).to_pydict()
                {'c': ['b', 'a', 'c']}

        Args:
            ds: A dataset holding the code (output) columns.

        Returns:
            A new lazy `Dataset` with each source column restored.
        """
        self._require_fitted()
        new = {}
        for c, out in output_pairs(self.columns, self.output_columns):
            if _is_list_column(ds, out):
                new[c] = col(out).list.transform(category_expr(element(), self.categories_[c]))
            else:
                new[c] = category_expr(out, self.categories_[c])
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

            >>> LabelEncoder("y", output_column="y_code").fit_transform(ds).to_pydict()
            {'y': ['cat', 'dog', 'cat'], 'y_code': [0, 1, 0]}

    Args:
        column: the single label column to encode in place.
        unknown_value: the code for labels unseen at fit time (and nulls).
        max_categories: the ceiling on the fitted class count (one CASE arm each).
        output_column: write the codes to this column and keep the labels (Ray Data's
            ``output_column``). ``None`` (the default) encodes in place.
    """

    __slots__ = ("classes_", "column", "max_categories", "output_column", "unknown_value")

    def __init__(
        self,
        column: str,
        *,
        unknown_value: int = -1,
        max_categories: int = MAX_CATEGORIES,
        output_column: str | None = None,
    ) -> None:
        self.column = column_arg(column, what="LabelEncoder")
        if output_column is not None and not isinstance(output_column, str):
            raise PlanError(
                f"LabelEncoder: output_column must be a single column name, got {output_column!r}"
            )
        self.output_column = output_column
        self.unknown_value = unknown_value
        self.max_categories = max_categories
        self.classes_: list[Any] = []

    def fit(self, ds: Dataset) -> LabelEncoder:
        """Learn the sorted distinct labels of the column into `classes_`.

        Unlike `transform`, which only builds a lazy plan, `fit` executes: it runs a
        query over `ds` now and reads the learned state back to the driver.

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
        return ds.with_columns(**{self.output_column or self.column: expr})

    def inverse_transform(self, ds: Dataset) -> Dataset:
        """Map each class index back to its label, written to the label column.

        The `unknown_value` code becomes null. The usual use is decoding a model's
        predicted class indices: ``enc.inverse_transform(scored)`` with the predictions in
        the encoder's output column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LabelEncoder
                >>> enc = LabelEncoder("y").fit(bt.from_pydict({"y": ["cat", "dog"]}))
                >>> enc.inverse_transform(bt.from_pydict({"y": [1, 0, -1]})).to_pydict()
                {'y': ['dog', 'cat', None]}

        Args:
            ds: A dataset holding the codes, in `output_column` (or the label column).

        Returns:
            A new lazy `Dataset` with the label column restored.
        """
        self._require_fitted()
        source = self.output_column or self.column
        return ds.with_columns(**{self.column: category_expr(source, self.classes_)})
