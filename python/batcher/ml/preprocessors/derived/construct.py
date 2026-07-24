"""Feature construction — the columns a model needs that the source table doesn't have.

Scaling and encoding change columns that already exist; these *make* columns. A model
learns interactions only if you hand them to it, ratios only if you compute them, and a
threshold flag only if you binarize. Each of these is the transform that turns a raw table
into a feature matrix, and each is a lazy `Expr` projection, so a hundred derived columns
add no pass over the data.

Most are stateless — a product or a ratio is a function of the row, learned from nothing —
which is what lets the same expression apply to training and serving data with no state to
persist and no way for the two to skew. `VarianceThreshold` is the exception: it *learns*
which columns are worth keeping, and so follows the fit/transform contract.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg, fit_aggregate
from batcher.plan.expr_ir.constructors import col, lit, nullif, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = [
    "Binarizer",
    "ColumnDropper",
    "ColumnSelector",
    "InteractionFeatures",
    "RatioFeatures",
    "VarianceThreshold",
]


class Binarizer(Preprocessor):
    """Map each column to 0/1 by a threshold — the simplest possible feature.

    Turns a continuous column into "is it above the line", which is exactly the feature a
    linear model cannot build for itself and a business rule usually wants: is the balance
    over the limit, is the latency over the SLA, is the score over the cutoff. Stateless, so
    the same threshold applies everywhere.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import Binarizer
            >>> ds = bt.from_pydict({"x": [0.2, 0.6, 0.9]})
            >>> Binarizer("x", threshold=0.5).fit_transform(ds).to_pydict()
            {'x': [0, 1, 1]}

    Args:
        columns: The columns to binarize (replaced in place).
        threshold: A value strictly above this becomes 1, at or below becomes 0.
    """

    __slots__ = ("columns", "threshold")

    def __init__(self, columns: str | Sequence[str], *, threshold: float = 0.0) -> None:
        self.columns = columns_arg(columns, what="Binarizer")
        self.threshold = threshold

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each column with ``1`` where it exceeds the threshold, ``0`` otherwise.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import Binarizer
                >>> ds = bt.from_pydict({"x": [-1.0, 1.0]})
                >>> Binarizer("x").fit_transform(ds).to_pydict()
                {'x': [0, 1]}

        Args:
            ds: The dataset to binarize.

        Returns:
            A new lazy `Dataset` with the fitted columns replaced by 0/1 integers.
        """
        return ds.with_columns(
            **{
                name: when(col(name) > lit(self.threshold)).then(lit(1)).otherwise(lit(0))
                for name in self.columns
            }
        )


class ColumnSelector(Preprocessor):
    """Keep only the named columns — the projection step, as a pipeline stage.

    A preprocessor rather than a bare `ds.select` so it composes inside a `Chain`, and so the
    exact feature set a model was trained on travels with the fitted pipeline rather than
    living in a separate line of code that drifts out of sync.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import ColumnSelector
            >>> ds = bt.from_pydict({"keep": [1], "drop": [2]})
            >>> ColumnSelector(["keep"]).fit_transform(ds).columns
            ['keep']

    Args:
        columns: The columns to keep, in the order given.
    """

    __slots__ = ("columns",)

    def __init__(self, columns: str | Sequence[str]) -> None:
        self.columns = columns_arg(columns, what="ColumnSelector")

    def transform(self, ds: Dataset) -> Dataset:
        """Return the dataset projected to the selected columns.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import ColumnSelector
                >>> ds = bt.from_pydict({"a": [1], "b": [2]})
                >>> ColumnSelector(["a"]).transform(ds).columns
                ['a']

        Args:
            ds: The dataset to project.

        Returns:
            A new lazy `Dataset` with only the selected columns.
        """
        return ds.select(*self.columns)


class ColumnDropper(Preprocessor):
    """Remove the named columns — the mirror of `ColumnSelector`.

    Use it for the columns a model must not see: an identifier, a leaked target, a
    post-outcome field. As a pipeline stage the exclusion is part of the fitted object, so a
    column that should never reach the model cannot slip in because someone forgot the
    `drop` at the call site.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import ColumnDropper
            >>> ds = bt.from_pydict({"feature": [1], "user_id": [42]})
            >>> ColumnDropper(["user_id"]).fit_transform(ds).columns
            ['feature']

    Args:
        columns: The columns to remove.
    """

    __slots__ = ("columns",)

    def __init__(self, columns: str | Sequence[str]) -> None:
        self.columns = columns_arg(columns, what="ColumnDropper")

    def transform(self, ds: Dataset) -> Dataset:
        """Return the dataset with the named columns removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import ColumnDropper
                >>> ds = bt.from_pydict({"a": [1], "b": [2]})
                >>> ColumnDropper(["b"]).transform(ds).columns
                ['a']

        Args:
            ds: The dataset to reduce.

        Returns:
            A new lazy `Dataset` without the dropped columns.
        """
        return ds.drop(*self.columns)


class InteractionFeatures(Preprocessor):
    """Append the pairwise products of the columns — interactions a linear model can't learn.

    A linear model sees each feature in isolation: it can learn that price matters and that
    region matters, but not that price matters *more* in one region, which is an
    interaction. Handing it the product ``price * region_flag`` is how it learns that, and
    it is exactly the term a tree finds for free and a linear model never does.

    Only products of *distinct* pairs are added (a column times itself is a squared term —
    that is `PolynomialFeatures`), and the source columns are kept.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import InteractionFeatures
            >>> ds = bt.from_pydict({"a": [2.0], "b": [3.0], "c": [4.0]})
            >>> out = InteractionFeatures(["a", "b", "c"]).fit_transform(ds).to_pydict()
            >>> out["a_x_b"], out["a_x_c"], out["b_x_c"]
            ([6.0], [8.0], [12.0])

    Args:
        columns: The columns to cross; every distinct pair becomes a product column.
    """

    __slots__ = ("columns",)

    def __init__(self, columns: Sequence[str]) -> None:
        cols = columns_arg(columns, what="InteractionFeatures")
        if len(cols) < 2:
            raise PlanError("InteractionFeatures needs at least two columns to cross")
        self.columns = cols

    def transform(self, ds: Dataset) -> Dataset:
        """Append one ``{a}_x_{b}`` product column per distinct pair.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import InteractionFeatures
                >>> ds = bt.from_pydict({"a": [2.0], "b": [3.0]})
                >>> InteractionFeatures(["a", "b"]).transform(ds).columns
                ['a', 'b', 'a_x_b']

        Args:
            ds: The dataset to extend.

        Returns:
            A new lazy `Dataset` with the interaction columns appended.
        """
        projections = {
            f"{left}_x_{right}": col(left) * col(right)
            for left, right in itertools.combinations(self.columns, 2)
        }
        return ds.with_columns(**projections)


class RatioFeatures(Preprocessor):
    """Append the ratio of column pairs — the normalized quantity a raw pair hides.

    A count is rarely the feature; a *rate* is. Errors per request, spend per visit, price
    per square foot — each is a ratio of two columns the source table stores separately, and
    each is what actually generalizes across rows of different scale. A division by zero
    becomes null rather than infinity, so one bad denominator does not poison the column.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import RatioFeatures
            >>> ds = bt.from_pydict({"errors": [4.0, 0.0], "requests": [100.0, 0.0]})
            >>> out = RatioFeatures([("errors", "requests")]).fit_transform(ds).to_pydict()
            >>> out["errors_per_requests"]
            [0.04, None]

    Args:
        pairs: The ``(numerator, denominator)`` column pairs to divide.
    """

    __slots__ = ("pairs",)

    def __init__(self, pairs: Sequence[tuple[str, str]]) -> None:
        if not pairs:
            raise PlanError("RatioFeatures needs at least one (numerator, denominator) pair")
        cleaned = []
        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise PlanError(f"each ratio must be a (numerator, denominator) pair, got {pair!r}")
            cleaned.append((str(pair[0]), str(pair[1])))
        self.pairs = cleaned

    def transform(self, ds: Dataset) -> Dataset:
        """Append one ``{num}_per_{den}`` column per pair, null on a zero denominator.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import RatioFeatures
                >>> ds = bt.from_pydict({"a": [10.0], "b": [2.0]})
                >>> RatioFeatures([("a", "b")]).transform(ds).to_pydict()["a_per_b"]
                [5.0]

        Args:
            ds: The dataset to extend.

        Returns:
            A new lazy `Dataset` with the ratio columns appended.
        """
        # `nullif(den, 0)` turns a zero denominator into null, and division by null
        # propagates to null — so a bad denominator costs its own row, not the column.
        projections = {
            f"{numerator}_per_{denominator}": col(numerator) / nullif(col(denominator), lit(0.0))
            for numerator, denominator in self.pairs
        }
        return ds.with_columns(**projections)


class VarianceThreshold(Preprocessor):
    """Drop columns whose variance is at or below a threshold — the unsupervised feature filter.

    A near-constant column carries almost no information, costs scan time and plan width, and
    dilutes a model's regularization budget. This learns which columns clear the threshold in
    one aggregate and drops the rest, with no reference to a target — so it is the screen to
    run before any label is even available.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import VarianceThreshold
            >>> ds = bt.from_pydict({"varies": [1.0, 2.0, 3.0], "flat": [7.0, 7.0, 7.0]})
            >>> VarianceThreshold(["varies", "flat"]).fit_transform(ds).columns
            ['varies']

    Args:
        columns: The columns to consider; only these can be dropped, others pass through.
        threshold: A column with variance at or below this is dropped (0.0 drops only
            genuinely constant columns).
    """

    __slots__ = ("columns", "kept_", "threshold")

    def __init__(self, columns: str | Sequence[str], *, threshold: float = 0.0) -> None:
        self.columns = columns_arg(columns, what="VarianceThreshold")
        if threshold < 0.0:
            raise PlanError(f"threshold must be non-negative, got {threshold}")
        self.threshold = threshold
        self.kept_: list[str] = []

    def fit(self, ds: Dataset) -> VarianceThreshold:
        """Learn which columns have variance above the threshold, in one aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import VarianceThreshold
                >>> pre = VarianceThreshold(["a", "b"]).fit(
                ...     bt.from_pydict({"a": [1.0, 2.0], "b": [5.0, 5.0]})
                ... )
                >>> pre.kept_
                ['a']

        Args:
            ds: The dataset to learn the per-column variance from.

        Returns:
            ``self``, fitted.
        """
        cell = fit_aggregate(ds, {f"{c}__v": col(c).var() for c in self.columns})
        self.kept_ = [
            c
            for c in self.columns
            if (v := cell[f"{c}__v"]) is not None and float(v) > self.threshold
        ]
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Drop the fitted columns whose variance did not clear the threshold.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import VarianceThreshold
                >>> ds = bt.from_pydict({"a": [1.0, 2.0], "b": [5.0, 5.0]})
                >>> VarianceThreshold(["a", "b"]).fit_transform(ds).columns
                ['a']

        Args:
            ds: The dataset to reduce.

        Returns:
            A new lazy `Dataset` with the low-variance columns removed.
        """
        self._require_fitted()
        dropped = [c for c in self.columns if c not in self.kept_]
        return ds.drop(*dropped) if dropped else ds
