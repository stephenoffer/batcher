"""Target encoders that shrink differently — leave-one-out, and James-Stein.

`TargetEncoder` shrinks every category toward the global mean by the same fixed weight
`smoothing`, which is the m-estimate. That is one answer to the same question all three
encoders here ask: *how much should I believe this category's own mean?* The others answer
it differently, and the difference matters on a long-tailed column.

`LeaveOneOutEncoder`
    Doesn't shrink at all; it removes the row's own contribution instead. A row's encoding
    is the mean of the *other* rows in its category, so a category's own target cannot leak
    into its own feature even when fitting and encoding the same split.
`JamesSteinEncoder`
    Derives the shrinkage from the data rather than taking it as a hyperparameter: a
    category whose target variance is high relative to the between-category variance is
    trusted less. That removes the one number `TargetEncoder` asks you to guess.

Both keep the `Preprocessor` shape — `fit` is one mergeable ``group_by`` per column,
`transform` is a lazy CASE expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher.ml.preprocessors.base import (
    MAX_CATEGORIES,
    Preprocessor,
    check_cardinality,
    columns_arg,
    fit_aggregate,
)
from batcher.ml.preprocessors.encoders.target import target_expr
from batcher.plan.expr_ir import Expr, col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["JamesSteinEncoder", "LeaveOneOutEncoder"]


def _group_totals(ds: Dataset, column: str, target: str, limit: int, what: str) -> dict[Any, Any]:
    """Each category's ``(count, sum, sum_of_squares)`` from one mergeable aggregate."""
    grouped = ds.group_by(column).agg(
        __bt_n=col(target).count(),
        __bt_s=col(target).sum(),
        __bt_ss=(col(target) * col(target)).sum(),
    )
    rows = grouped.limit(limit + 1).to_pydict()
    check_cardinality(what, column, len(rows[column]), limit, exact=False)
    totals: dict[Any, Any] = {}
    for category, n, s, ss in zip(
        rows[column], rows["__bt_n"], rows["__bt_s"], rows["__bt_ss"], strict=False
    ):
        if category is None or not n:
            continue
        totals[category] = (int(n), float(s or 0.0), float(ss or 0.0))
    return totals


class LeaveOneOutEncoder(Preprocessor):
    """Encode a category by the mean target of every *other* row in that category.

    The leakage-free target encoding that needs no folds. `TargetEncoder` without `cv` puts
    a row's own target inside its own feature; with `cv` it removes that by holding out
    folds. This removes it exactly, per row, by subtracting the row's own contribution:
    ``(sum(category) - y) / (count(category) - 1)``.

    That exactness is also its weakness, and worth knowing before choosing it. On a binary
    target in a category of size two, the encoding is the *other* row's label, so the
    feature is nearly a copy of the answer inverted — a model can learn to read it backwards.
    Prefer it on categories with a reasonable number of rows each, and prefer `TargetEncoder`
    with `cv` when the tail is thin.

    `fit_transform` applies the leave-one-out form, because those rows are the training rows.
    `transform` applies the plain category mean, because a held-out row contributed nothing
    to subtract.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import LeaveOneOutEncoder
            >>> ds = bt.from_pydict({"c": ["a", "a", "a"], "y": [0.0, 3.0, 6.0]})
            >>> LeaveOneOutEncoder(["c"], "y").fit_transform(ds).to_pydict()["c"]
            [4.5, 3.0, 1.5]

    Args:
        columns: The categorical columns to replace in place with their encoding.
        target: The numeric or 0/1 target column supplying the means.
        max_categories: The ceiling on each column's fitted cardinality.
    """

    __slots__ = ("columns", "counts_", "max_categories", "prior_", "sums_", "target")

    def __init__(
        self,
        columns: str | Sequence[str],
        target: str,
        *,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        self.columns = columns_arg(columns, what="LeaveOneOutEncoder")
        self.target = target
        self.max_categories = max_categories
        self.prior_: float = 0.0
        self.counts_: dict[str, dict[Any, float]] = {}
        self.sums_: dict[str, dict[Any, float]] = {}

    def fit(self, ds: Dataset) -> LeaveOneOutEncoder:
        """Learn each category's target count and sum, plus the global mean.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LeaveOneOutEncoder
                >>> ds = bt.from_pydict({"c": ["a", "a", "b"], "y": [1.0, 3.0, 5.0]})
                >>> LeaveOneOutEncoder(["c"], "y").fit(ds).counts_["c"]["a"]
                2.0

        Args:
            ds: The training dataset supplying the categories and the target.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column has more than `max_categories` distinct values.
        """
        prior = fit_aggregate(ds, {"__bt_p": col(self.target).mean()})["__bt_p"]
        self.prior_ = float(prior) if prior is not None else 0.0
        for column in self.columns:
            totals = _group_totals(
                ds, column, self.target, self.max_categories, "LeaveOneOutEncoder"
            )
            self.counts_[column] = {k: float(v[0]) for k, v in totals.items()}
            self.sums_[column] = {k: v[1] for k, v in totals.items()}
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with the plain category mean.

        This is the held-out form. A row that was not in the fit set contributed nothing to
        the category's sum, so there is nothing to leave out.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LeaveOneOutEncoder
                >>> ds = bt.from_pydict({"c": ["a", "a", "b"], "y": [1.0, 3.0, 5.0]})
                >>> enc = LeaveOneOutEncoder(["c"], "y").fit(ds)
                >>> enc.transform(bt.from_pydict({"c": ["a", "z"]})).to_pydict()["c"]
                [2.0, 3.0]

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with each fitted column replaced.
        """
        self._require_fitted()
        projections: dict[str, Expr] = {}
        for column in self.columns:
            means = {
                category: self.sums_[column][category] / count
                for category, count in self.counts_[column].items()
                if count
            }
            projections[column] = target_expr(column, means, self.prior_)
        return ds.with_columns(**projections)

    def fit_transform(self, ds: Dataset) -> Dataset:
        """`fit(ds)` then encode `ds` with the row's own target removed.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import LeaveOneOutEncoder
                >>> ds = bt.from_pydict({"c": ["a", "a"], "y": [2.0, 4.0]})
                >>> LeaveOneOutEncoder(["c"], "y").fit_transform(ds).to_pydict()["c"]
                [4.0, 2.0]

        Args:
            ds: The training dataset to fit on and encode.

        Returns:
            A new lazy `Dataset` with each fitted column replaced by its leave-one-out
            encoding.
        """
        self.fit(ds)
        target = col(self.target).cast("float64")
        projections: dict[str, Expr] = {}
        for column in self.columns:
            sums = target_expr(column, self.sums_[column], 0.0)
            counts = target_expr(column, self.counts_[column], 0.0)
            remaining = counts - lit(1.0)
            # A category with a single row has nothing left once its own row is removed, so
            # it falls back to the global prior rather than dividing by zero.
            projections[column] = (
                when(remaining > lit(0.0))
                .then((sums - target) / remaining)
                .otherwise(lit(self.prior_))
            )
        return ds.with_columns(**projections)


class JamesSteinEncoder(Preprocessor):
    """Encode a category by its target mean, shrunk toward the global mean by its own noise.

    The same shape as `TargetEncoder`, with the shrinkage *derived* rather than configured.
    A category's weight is ``1 - (within variance / total variance)``, so a category whose
    own target scatters widely relative to the spread between categories is trusted less,
    and one that is both large and consistent is trusted almost fully.

    That removes the one number `TargetEncoder` asks you to guess, which is worth having
    when the categories differ wildly in size — a single `smoothing` that suits a category
    with ten rows over-shrinks one with ten thousand.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import JamesSteinEncoder
            >>> ds = bt.from_pydict(
            ...     {"c": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]}
            ... )
            >>> encoded = JamesSteinEncoder(["c"], "y").fit_transform(ds).to_pydict()["c"]
            >>> encoded[0] > encoded[2]
            True

    Args:
        columns: The categorical columns to replace in place with their encoding.
        target: The numeric or 0/1 target column supplying the means.
        max_categories: The ceiling on each column's fitted cardinality.
    """

    __slots__ = ("columns", "mapping_", "max_categories", "prior_", "target")

    def __init__(
        self,
        columns: str | Sequence[str],
        target: str,
        *,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        self.columns = columns_arg(columns, what="JamesSteinEncoder")
        self.target = target
        self.max_categories = max_categories
        self.prior_: float = 0.0
        self.mapping_: dict[str, dict[Any, float]] = {}

    def fit(self, ds: Dataset) -> JamesSteinEncoder:
        """Learn each category's shrunk mean from one grouped aggregate per column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import JamesSteinEncoder
                >>> ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]})
                >>> round(JamesSteinEncoder(["c"], "y").fit(ds).prior_, 4)
                0.5

        Args:
            ds: The training dataset supplying the categories and the target.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column has more than `max_categories` distinct values.
        """
        prior = fit_aggregate(ds, {"__bt_p": col(self.target).mean()})["__bt_p"]
        self.prior_ = float(prior) if prior is not None else 0.0
        for column in self.columns:
            totals = _group_totals(
                ds, column, self.target, self.max_categories, "JamesSteinEncoder"
            )
            self.mapping_[column] = _shrunk_means(totals, self.prior_)
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its shrunk target encoding.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import JamesSteinEncoder
                >>> ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]})
                >>> enc = JamesSteinEncoder(["c"], "y").fit(ds)
                >>> enc.transform(bt.from_pydict({"c": ["z"]})).to_pydict()["c"]
                [0.5]

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with each fitted column replaced.
        """
        self._require_fitted()
        return ds.with_columns(
            **{c: target_expr(c, self.mapping_[c], self.prior_) for c in self.columns}
        )


def _shrunk_means(totals: dict[Any, Any], prior: float) -> dict[Any, float]:
    """Each category's mean pulled toward `prior` by its own noise-to-signal ratio.

    The weight is ``1 - within / (within + between)``: `within` is the variance of the
    category's own mean, which falls as the category grows, and `between` is the spread of
    the category means around the prior. A category that is small or internally noisy gets a
    weight near zero and encodes as the prior; a large, consistent one keeps its own mean.
    """
    means = {category: total / count for category, (count, total, _) in totals.items() if count}
    if not means:
        return {}
    # Between-category variance, measured on the category means around the global prior.
    between = sum((value - prior) ** 2 for value in means.values()) / len(means)
    out: dict[str, float] = {}
    for category, (count, total, sum_squares) in totals.items():
        if not count:
            continue
        mean = total / count
        if count > 1:
            variance = max(sum_squares / count - mean * mean, 0.0)
            within = variance / count  # the variance of this category's *mean*
        else:
            # One observation says nothing about its own scatter, so the only honest answer
            # is to trust it as little as the between-category spread allows.
            within = between if between > 0 else 1.0
        weight = 0.0 if within + between <= 0 else between / (within + between)
        out[category] = weight * mean + (1.0 - weight) * prior
    return out
