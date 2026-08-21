"""Rank and label transforms — order-based rescaling and one-vs-rest label expansion.

Three transforms that reshape a column by its *structure* rather than its statistics:

`RankTransformer`
    Replace a value with its rank, as a percentile. Like `QuantileTransformer` it destroys
    the scale and keeps only the order, so it is immune to an outlier and needs no
    distribution assumption — but it is exact (every distinct value gets its own rank) rather
    than binned, which matters on a small column. Reached as a rank because a model that
    splits on order (a tree) or measures monotone association (a rank correlation) wants the
    rank, not the raw value.

`LabelBinarizer`
    Turn a multi-class label into one 0/1 column per class — the one-vs-rest expansion a
    binary-only model or a per-class metric needs. This is `OneHotEncoder` pointed at the
    *target* rather than a feature, and named separately because that is a different step in
    a pipeline with a different meaning.

`MultiLabelBinarizer`
    The same, for a column that holds a *list* of labels per row: each row can belong to many
    classes at once (tags, genres, diagnoses), and each class becomes a column that is 1 when
    the list contains it. The standard input shaping for a multi-label classifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import (
    MAX_CATEGORIES,
    Preprocessor,
    columns_arg,
    distinct_values,
    require_categories,
)
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["LabelBinarizer", "MultiLabelBinarizer", "RankTransformer"]


class RankTransformer(Preprocessor):
    """Replace each column with its percentile rank in ``[0, 1]`` — exact, outlier-proof.

    Every value is mapped to the fraction of rows below it, so the result is uniform on
    ``[0, 1]`` whatever the input distribution and a single extreme value becomes simply "the
    largest" at rank 1. Unlike `QuantileTransformer` the mapping is exact rather than binned —
    each distinct value gets its own rank — which is the better choice on a column small enough
    that binning would collapse distinct values together.

    Stateless in the sense that there are no fitted parameters, but the rank is computed
    *within the frame being transformed* (a window), so it is a relative measure — a serving
    row is ranked against the serving batch, not the training set. Use it as a feature
    transform on a whole dataset, not as a fitted train/serve transform.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import RankTransformer
            >>> ds = bt.from_pydict({"x": [10.0, 40.0, 20.0, 1000.0]})
            >>> RankTransformer("x").fit_transform(ds).to_pydict()["x"]
            [0.0, 0.6666666666666666, 0.3333333333333333, 1.0]

    Args:
        columns: The numeric columns to rank (replaced in place).
    """

    numeric_only = True

    __slots__ = ("columns",)

    def __init__(self, columns: str | Sequence[str]) -> None:
        self.columns = columns_arg(columns, what="RankTransformer")

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each column with its percentile rank.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RankTransformer
                >>> ds = bt.from_pydict({"x": [3.0, 1.0, 2.0]})
                >>> RankTransformer("x").transform(ds).to_pydict()["x"]
                [1.0, 0.0, 0.5]

        Args:
            ds: The dataset to rank.

        Returns:
            A new lazy `Dataset` with the fitted columns replaced by their percentile rank.
        """
        # rank_pct() is a window, which cannot write back to its own input column in a single
        # with_columns; compute each rank into a temp, then swap it in for the original.
        #
        # A null row ranks too, and it ranked *last* — so every missing value came out at
        # percentile 1.0, the top of the feature's range, which is the worst place to put one
        # and looks like an ordinary extreme value downstream. Null it back out afterwards:
        # a rank is meaningless for a value that is not there.
        temps = {f"__bt_rank_{name}": col(name).rank_pct() for name in self.columns}
        out = ds.with_columns(**temps)
        for name in self.columns:
            temp = f"__bt_rank_{name}"
            out = out.with_columns(
                **{
                    temp: when(col(name).is_null())
                    .then(col(name).cast("float64"))
                    .otherwise(col(temp))
                }
            )
            out = out.drop(name).rename({temp: name})
        return out


class LabelBinarizer(Preprocessor):
    """One-vs-rest expand a categorical *label* into a 0/1 column per class.

    The target-side counterpart of `OneHotEncoder`: it learns the class set and appends one
    ``{column}_{class}`` indicator per class, 1 where the label is that class. This is the
    shaping a per-class metric or a set of one-vs-rest binary models expects, and it is a
    different pipeline step from encoding a feature, which is why it is its own class.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import LabelBinarizer
            >>> ds = bt.from_pydict({"label": ["cat", "dog", "cat"]})
            >>> out = LabelBinarizer("label").fit_transform(ds).to_pydict()
            >>> out["label_cat"], out["label_dog"]
            ([1, 0, 1], [0, 1, 0])

    Args:
        column: The label column to expand.
        drop_original: Remove the source label column after expanding it.
        max_categories: The ceiling on the learned class set — one output column each.
    """

    __slots__ = ("classes_", "column", "drop_original", "max_categories")

    def __init__(
        self, column: str, *, drop_original: bool = False, max_categories: int = MAX_CATEGORIES
    ) -> None:
        if not isinstance(column, str):
            raise PlanError(f"LabelBinarizer takes one column name, got {column!r}")
        self.column = column
        self.drop_original = drop_original
        self.max_categories = max_categories
        self.classes_: list[object] = []

    def fit(self, ds: Dataset) -> LabelBinarizer:
        """Learn the sorted class set with one bounded `distinct`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LabelBinarizer
                >>> LabelBinarizer("y").fit(bt.from_pydict({"y": ["b", "a", "b"]})).classes_
                ['a', 'b']

        Args:
            ds: The dataset to learn the class set from.

        Returns:
            ``self``, fitted.
        """
        self.classes_ = require_categories(
            distinct_values(
                ds, self.column, what="LabelBinarizer", max_categories=self.max_categories
            ),
            what="LabelBinarizer",
            column=self.column,
        )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Append one 0/1 indicator column per learned class.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LabelBinarizer
                >>> ds = bt.from_pydict({"y": ["a", "b"]})
                >>> LabelBinarizer("y").fit(ds).transform(ds).columns
                ['y', 'y_a', 'y_b']

        Args:
            ds: The dataset to expand.

        Returns:
            A new lazy `Dataset` with one indicator column per class.
        """
        self._require_fitted()
        projections = {
            f"{self.column}_{cls}": when(col(self.column) == lit(cls))
            .then(lit(1))
            .otherwise(lit(0))
            for cls in self.classes_
        }
        out = ds.with_columns(**projections)
        return out.drop(self.column) if self.drop_original else out


class MultiLabelBinarizer(Preprocessor):
    """Expand a *list* column into a 0/1 column per label — the multi-label input shaping.

    Where `LabelBinarizer` handles one label per row, this handles a list of labels per row: a
    row can carry many tags, genres, or diagnoses at once, and each learned label becomes a
    column that is 1 when the row's list contains it. This is the standard way to turn a
    ``List<String>`` target into the indicator matrix a multi-label classifier trains on.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import MultiLabelBinarizer
            >>> ds = bt.from_pydict({"tags": [["a", "b"], ["b"]]})
            >>> out = MultiLabelBinarizer("tags").fit_transform(ds).to_pydict()
            >>> out["tags_a"], out["tags_b"]
            ([1, 0], [1, 1])

    Args:
        column: The list column to expand.
        labels: The label set to expand into; learned from the data when omitted.
        drop_original: Remove the source list column after expanding it.
        max_categories: The ceiling on the learned label set.
    """

    __slots__ = ("column", "drop_original", "labels", "labels_", "max_categories")

    def __init__(
        self,
        column: str,
        *,
        labels: Sequence[object] | None = None,
        drop_original: bool = False,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        if not isinstance(column, str):
            raise PlanError(f"MultiLabelBinarizer takes one column name, got {column!r}")
        self.column = column
        self.labels = list(labels) if labels is not None else None
        self.drop_original = drop_original
        self.max_categories = max_categories
        self.labels_: list[object] = []

    def fit(self, ds: Dataset) -> MultiLabelBinarizer:
        """Learn the label set by exploding the list column, unless it was given.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MultiLabelBinarizer
                >>> ds = bt.from_pydict({"t": [["x", "y"], ["y"]]})
                >>> MultiLabelBinarizer("t").fit(ds).labels_
                ['x', 'y']

        Args:
            ds: The dataset to learn the label set from.

        Returns:
            ``self``, fitted.
        """
        if self.labels is not None:
            self.labels_ = list(self.labels)
        else:
            exploded = ds.select(self.column).explode(self.column)
            self.labels_ = require_categories(
                distinct_values(
                    exploded,
                    self.column,
                    what="MultiLabelBinarizer",
                    max_categories=self.max_categories,
                ),
                what="MultiLabelBinarizer",
                column=self.column,
            )
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Append one 0/1 membership column per learned label.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import MultiLabelBinarizer
                >>> ds = bt.from_pydict({"t": [["x"], ["x", "y"]]})
                >>> MultiLabelBinarizer("t").fit(ds).transform(ds).columns
                ['t', 't_x', 't_y']

        Args:
            ds: The dataset to expand.

        Returns:
            A new lazy `Dataset` with one membership column per label.
        """
        self._require_fitted()
        projections = {
            f"{self.column}_{label}": col(self.column).list.contains(label).cast("int64")
            for label in self.labels_
        }
        out = ds.with_columns(**projections)
        return out.drop(self.column) if self.drop_original else out
