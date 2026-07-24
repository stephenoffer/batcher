"""Weight-of-evidence encoding — the credit-scorecard categorical transform.

Target encoding replaces a category with the mean of the target; weight-of-evidence replaces
it with the *log odds* of the target relative to the overall odds. That difference is not
cosmetic. WOE is additive in log-odds space, which is exactly the space a logistic regression
works in, so a WOE-encoded feature enters a linear scorecard as a straight, interpretable
coefficient — the reason regulated credit models are built on it rather than on target means.

`WOEEncoder` learns each category's WOE on the training data and applies it as a lazy CASE
expression. An unseen category, or one with no positives or no negatives, maps to 0 (neutral
evidence) rather than to an infinite log-odds, which is the convention a scorecard uses and
the only one that does not blow up on a rare category.

It is supervised, so the leakage caveat that applies to `TargetEncoder` applies here too: fit
on the training split only, or a row's own outcome leaks into its own feature.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import MAX_CATEGORIES, Preprocessor, columns_arg
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.functions.aggregate import count_if

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["WOEEncoder"]

# The floor on a category's positive or negative share, so a category with none of one class
# gets a large-but-finite WOE rather than infinity. 0.5 of a row is the "adjusted" smoothing
# the credit-scoring literature uses, applied as a share of the class total.
_SMOOTHING = 0.5


class WOEEncoder(Preprocessor):
    """Replace a categorical column with the weight of evidence of a binary target.

    Each category maps to ``ln( (positives_in_cat / total_positives) / (negatives_in_cat /
    total_negatives) )`` — how much more (or less) likely the positive class is within that
    category than overall, in log-odds. A positive WOE means the category leans toward the
    positive class; 0 means it carries no evidence either way.

    `fit` learns every category's WOE in one grouped aggregate. `transform` is a CASE
    expression, so it stays lazy and distributes. An unseen or single-class category encodes
    as 0.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import WOEEncoder
            >>> ds = bt.from_pydict(
            ...     {"grade": ["a", "a", "b", "b"], "default": [0, 0, 1, 1]}
            ... )
            >>> out = WOEEncoder(["grade"], "default").fit_transform(ds).to_pydict()["grade"]
            >>> out[0] < 0 < out[2]
            True

    Args:
        columns: The categorical columns to encode (replaced in place).
        target: The binary (0/1 or two-value) target column supplying the evidence.
        positive: The target value that counts as the positive class.
        max_categories: The ceiling on each column's fitted cardinality — one CASE arm each.
    """

    __slots__ = ("columns", "max_categories", "positive", "target", "woe_")

    def __init__(
        self,
        columns: str | Sequence[str],
        target: str,
        *,
        positive: object = 1,
        max_categories: int = MAX_CATEGORIES,
    ) -> None:
        self.columns = columns_arg(columns, what="WOEEncoder")
        if not isinstance(target, str):
            raise PlanError(f"target must be a column name, got {target!r}")
        self.target = target
        self.positive = positive
        self.max_categories = max_categories
        self.woe_: dict[str, dict[object, float]] = {}

    def fit(self, ds: Dataset) -> WOEEncoder:
        """Learn each category's weight of evidence with one grouped aggregate per column.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import WOEEncoder
                >>> ds = bt.from_pydict({"g": ["a", "b", "b"], "y": [0, 1, 1]})
                >>> pre = WOEEncoder(["g"], "y").fit(ds)
                >>> pre.woe_["g"]["b"] > 0
                True

        Args:
            ds: The dataset to learn the category weights from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column exceeds `max_categories`.
        """
        is_positive = col(self.target) == lit(self.positive)
        totals = ds.agg(
            __bt_pos=count_if(is_positive),
            __bt_neg=count_if(~is_positive),
        ).collect()
        total_positive = float(totals.column("__bt_pos")[0].as_py() or 0)
        total_negative = float(totals.column("__bt_neg")[0].as_py() or 0)
        if total_positive == 0 or total_negative == 0:
            raise PlanError(
                f"the target {self.target!r} has only one class, so there is no weight of "
                "evidence to learn. WOE needs both a positive and a negative outcome."
            )
        for column in self.columns:
            grouped = (
                ds.filter(col(column).is_not_null())
                .group_by(column)
                .agg(__bt_pos=count_if(is_positive), __bt_neg=count_if(~is_positive))
                .limit(self.max_categories + 1)
                .collect()
            )
            if grouped.num_rows > self.max_categories:
                from batcher.ml.preprocessors.base import check_cardinality

                check_cardinality(
                    "WOEEncoder", column, grouped.num_rows, self.max_categories, exact=False
                )
            weights: dict[object, float] = {}
            categories = grouped.column(column).to_pylist()
            positives = grouped.column("__bt_pos").to_pylist()
            negatives = grouped.column("__bt_neg").to_pylist()
            for category, pos, neg in zip(categories, positives, negatives, strict=True):
                weights[category] = _woe(float(pos), float(neg), total_positive, total_negative)
            self.woe_[column] = weights
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replace each fitted column with its per-category weight of evidence.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import WOEEncoder
                >>> ds = bt.from_pydict({"g": ["a", "b", "b"], "y": [0, 1, 1]})
                >>> pre = WOEEncoder(["g"], "y").fit(ds)
                >>> pre.transform(bt.from_pydict({"g": ["unseen"]})).to_pydict()["g"]
                [0.0]

        Args:
            ds: The dataset to encode.

        Returns:
            A new lazy `Dataset` with the fitted columns replaced by their WOE.
        """
        self._require_fitted()
        return ds.with_columns(**{name: self._expr(name) for name in self.columns})

    def _expr(self, name: str) -> Expr:
        """A CASE ladder mapping each learned category to its WOE, unseen to 0."""
        builder = None
        for category, weight in self.woe_[name].items():
            branch = col(name) == lit(category)
            builder = (
                when(branch).then(lit(weight))
                if builder is None
                else builder.when(branch).then(lit(weight))
            )
        if builder is None:
            return lit(0.0)
        return builder.otherwise(lit(0.0))


def _woe(positives: float, negatives: float, total_pos: float, total_neg: float) -> float:
    """One category's weight of evidence, smoothed so a single-class category stays finite."""
    pos_share = (positives + _SMOOTHING) / (total_pos + _SMOOTHING)
    neg_share = (negatives + _SMOOTHING) / (total_neg + _SMOOTHING)
    return math.log(pos_share / neg_share)
