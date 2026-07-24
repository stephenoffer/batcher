"""Naive Bayes — a probabilistic classifier whose whole fit is a grouped aggregate.

Gaussian naive Bayes assumes each feature is normally distributed within a class and independent
of the others. That assumption is exactly what makes it the cheapest classifier to fit in a
columnar engine: the only parameters are a per-class mean and variance and a class prior, and all
three fall out of a single ``group_by(target)`` aggregate. Prediction is a closed-form
log-likelihood per class and an argmax, both expressions, so scoring is one streaming pass.

It is naive by name and often right by luck — the independence assumption is usually false, yet
the argmax it produces is a strong, instant baseline, especially in high dimensions where a
model with more parameters would overfit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml._estimator import argmax_prediction, require_fitted
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["BernoulliNB", "GaussianNB", "MultinomialNB"]


class GaussianNB:
    """Gaussian naive Bayes, fitted from one grouped aggregate over the training data.

    Learns each class's prior, per-feature mean, and per-feature variance in a single
    ``group_by(target)`` pass, then classifies by the maximum-a-posteriori log-likelihood under
    the Gaussian-independence assumption. Reproduces scikit-learn's ``GaussianNB`` predictions,
    including its ``var_smoothing`` floor on the variances. `predict` appends the predicted class.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.naive_bayes import GaussianNB
            >>> ds = bt.from_pydict(
            ...     {"x": [0.0, 0.5, 5.0, 5.5], "y": ["a", "a", "b", "b"]}
            ... )
            >>> model = GaussianNB(["x"], "y").fit(ds)
            >>> model.predict(bt.from_pydict({"x": [0.2, 5.2]})).to_pydict()["prediction"]
            ['a', 'b']

    Args:
        features: The numeric feature columns.
        target: The class label column.
        var_smoothing: A fraction of the largest feature variance added to every variance for
            numerical stability, matching scikit-learn's default.
        output_column: The name of the predicted-class column `predict` appends.
    """

    __slots__ = (
        "classes_",
        "features",
        "output_column",
        "priors_",
        "target",
        "theta_",
        "var_",
        "var_smoothing",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        var_smoothing: float = 1e-9,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("GaussianNB needs at least one feature column.")
        self.target = target
        self.var_smoothing = var_smoothing
        self.output_column = output_column
        self.classes_: list[object] = []
        self.priors_: dict[object, float] = {}
        self.theta_: dict[object, list[float]] = {}
        self.var_: dict[object, list[float]] = {}

    def fit(self, ds: Dataset) -> GaussianNB:
        """Learn the class priors, means, and variances in one grouped aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.naive_bayes import GaussianNB
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 10.0, 11.0], "y": [0, 0, 1, 1]})
                >>> model = GaussianNB(["x"], "y").fit(ds)
                >>> sorted(model.classes_)
                [0, 1]

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        from batcher.plan.functions.aggregate import mean as mean_
        from batcher.plan.functions.statistics import var_pop

        for name in (*self.features, self.target):
            if name not in ds.columns:
                from batcher._internal.errors import ColumnNotFoundError, unknown_message

                raise ColumnNotFoundError(
                    unknown_message("column", name, ds.columns, hint="Pass an existing column.")
                )
        total = ds.count()
        global_variance = ds.agg(
            **{f"v{i}": var_pop(col(name)) for i, name in enumerate(self.features)}
        ).collect()
        smoothing = self.var_smoothing * max(
            float(global_variance.column(f"v{i}")[0].as_py() or 0.0)
            for i in range(len(self.features))
        )
        aggregates: dict[str, object] = {"__bt_n": col(self.target).count()}
        for i, name in enumerate(self.features):
            aggregates[f"m{i}"] = mean_(col(name))
            aggregates[f"v{i}"] = var_pop(col(name))
        grouped = ds.group_by(self.target).agg(**aggregates).collect()
        self.classes_ = []
        self.priors_, self.theta_, self.var_ = {}, {}, {}
        for row in range(grouped.num_rows):
            label = grouped.column(self.target)[row].as_py()
            self.classes_.append(label)
            self.priors_[label] = float(grouped.column("__bt_n")[row].as_py()) / total
            self.theta_[label] = [
                float(grouped.column(f"m{i}")[row].as_py()) for i in range(len(self.features))
            ]
            self.var_[label] = [
                float(grouped.column(f"v{i}")[row].as_py() or 0.0) + smoothing
                for i in range(len(self.features))
            ]
        return self

    def _log_likelihood(self, label: object):
        """The (row-varying part of the) Gaussian log-likelihood expression for one class."""
        means = self.theta_[label]
        variances = self.var_[label]
        constant = math.log(self.priors_[label]) - 0.5 * sum(
            math.log(2.0 * math.pi * v) for v in variances
        )
        expression = lit(constant)
        for name, mean, variance in zip(self.features, means, variances, strict=True):
            delta = col(name) - lit(mean)
            expression = expression - lit(0.5 / variance) * delta * delta
        return expression

    def predict(self, ds: Dataset) -> Dataset:
        """Append the maximum-a-posteriori class label for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.naive_bayes import GaussianNB
                >>> ds = bt.from_pydict({"x": [0.0, 1.0, 10.0, 11.0], "y": [0, 0, 1, 1]})
                >>> model = GaussianNB(["x"], "y").fit(ds)
                >>> model.predict(bt.from_pydict({"x": [0.5, 10.5]})).to_pydict()["prediction"]
                [0, 1]

        Args:
            ds: The dataset to classify.

        Returns:
            A new lazy `Dataset` with the predicted-class column appended.
        """
        require_fitted(self, self.classes_)
        prediction = argmax_prediction(self.classes_, self._log_likelihood)
        return ds.with_columns(**{self.output_column: prediction})


class MultinomialNB:
    """Multinomial naive Bayes for count features, fitted from grouped feature sums.

    The text-classification workhorse: it models each class as a multinomial over the feature
    counts (word counts, event tallies), learning a smoothed probability per feature per class
    from one grouped aggregate. Prediction is the maximum-a-posteriori log-likelihood, an
    expression linear in the feature columns. Reproduces scikit-learn's ``MultinomialNB``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.naive_bayes import MultinomialNB
            >>> ds = bt.from_pydict(
            ...     {"a": [3, 4, 0, 0], "b": [0, 0, 3, 4], "y": [0, 0, 1, 1]}
            ... )
            >>> model = MultinomialNB(["a", "b"], "y").fit(ds)
            >>> model.predict(bt.from_pydict({"a": [5, 0], "b": [0, 5]})).to_pydict()["prediction"]
            [0, 1]

    Args:
        features: The non-negative count feature columns.
        target: The class label column.
        alpha: The additive (Laplace) smoothing strength.
        output_column: The name of the predicted-class column `predict` appends.
    """

    __slots__ = (
        "alpha",
        "classes_",
        "features",
        "log_prior_",
        "log_prob_",
        "output_column",
        "target",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alpha: float = 1.0,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("MultinomialNB needs at least one feature column.")
        self.target = target
        self.alpha = alpha
        self.output_column = output_column
        self.classes_: list[object] = []
        self.log_prior_: dict[object, float] = {}
        self.log_prob_: dict[object, list[float]] = {}

    def fit(self, ds: Dataset) -> MultinomialNB:
        """Learn each class's prior and smoothed per-feature log-probabilities in one aggregate.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.naive_bayes import MultinomialNB
                >>> ds = bt.from_pydict({"a": [2, 3], "b": [0, 1], "y": [0, 1]})
                >>> sorted(MultinomialNB(["a", "b"], "y").fit(ds).classes_)
                [0, 1]

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        from batcher.plan.functions.aggregate import sum as sum_

        _check_columns(ds, self.features, self.target)
        total = ds.count()
        aggregates: dict[str, object] = {"__bt_n": col(self.target).count()}
        for i, name in enumerate(self.features):
            aggregates[f"s{i}"] = sum_(col(name))
        grouped = ds.group_by(self.target).agg(**aggregates).collect()
        d = len(self.features)
        self.classes_, self.log_prior_, self.log_prob_ = [], {}, {}
        for row in range(grouped.num_rows):
            label = grouped.column(self.target)[row].as_py()
            self.classes_.append(label)
            self.log_prior_[label] = math.log(float(grouped.column("__bt_n")[row].as_py()) / total)
            feature_sums = [float(grouped.column(f"s{i}")[row].as_py() or 0.0) for i in range(d)]
            denominator = sum(feature_sums) + self.alpha * d
            self.log_prob_[label] = [
                math.log((value + self.alpha) / denominator) for value in feature_sums
            ]
        return self

    def _score(self, label: object):
        """The log-posterior expression for one class: log-prior plus counts times log-probs."""
        expression = lit(self.log_prior_[label])
        for name, log_probability in zip(self.features, self.log_prob_[label], strict=True):
            expression = expression + lit(log_probability) * col(name)
        return expression

    def predict(self, ds: Dataset) -> Dataset:
        """Append the maximum-a-posteriori class label for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.naive_bayes import MultinomialNB
                >>> ds = bt.from_pydict({"a": [3, 0], "b": [0, 3], "y": [0, 1]})
                >>> model = MultinomialNB(["a", "b"], "y").fit(ds)
                >>> model.predict(ds).to_pydict()["prediction"]
                [0, 1]

        Args:
            ds: The dataset to classify.

        Returns:
            A new lazy `Dataset` with the predicted-class column appended.
        """
        require_fitted(self, self.classes_)
        return ds.with_columns(
            **{self.output_column: argmax_prediction(self.classes_, self._score)}
        )


class BernoulliNB:
    """Bernoulli naive Bayes for binary features, fitted from grouped presence counts.

    Models each feature as present or absent within a class, learning the smoothed presence
    probability per feature per class. Unlike `MultinomialNB` it explicitly penalizes a feature's
    *absence*, which is what makes it the right choice for binary indicators (a word occurs or
    not) rather than counts. Features are binarized at `threshold`. Reproduces scikit-learn's
    ``BernoulliNB``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.naive_bayes import BernoulliNB
            >>> ds = bt.from_pydict(
            ...     {"a": [1, 1, 0, 0], "b": [0, 0, 1, 1], "y": [0, 0, 1, 1]}
            ... )
            >>> model = BernoulliNB(["a", "b"], "y").fit(ds)
            >>> model.predict(bt.from_pydict({"a": [1, 0], "b": [0, 1]})).to_pydict()["prediction"]
            [0, 1]

    Args:
        features: The feature columns, binarized at `threshold`.
        target: The class label column.
        alpha: The additive (Laplace) smoothing strength.
        threshold: Values above this count as "present"; at or below as "absent".
        output_column: The name of the predicted-class column `predict` appends.
    """

    __slots__ = (
        "alpha",
        "classes_",
        "features",
        "log_neg_prob_",
        "log_prior_",
        "log_prob_",
        "output_column",
        "target",
        "threshold",
    )

    def __init__(
        self,
        features: Sequence[str],
        target: str,
        *,
        alpha: float = 1.0,
        threshold: float = 0.0,
        output_column: str = "prediction",
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise PlanError("BernoulliNB needs at least one feature column.")
        self.target = target
        self.alpha = alpha
        self.threshold = threshold
        self.output_column = output_column
        self.classes_: list[object] = []
        self.log_prior_: dict[object, float] = {}
        self.log_prob_: dict[object, list[float]] = {}
        self.log_neg_prob_: dict[object, list[float]] = {}

    def _present(self, name: str):
        """The 0/1 indicator that feature `name` is present (above the threshold)."""
        return when(col(name) > lit(self.threshold)).then(lit(1.0)).otherwise(lit(0.0))

    def fit(self, ds: Dataset) -> BernoulliNB:
        """Learn each class's prior and smoothed per-feature presence log-probabilities.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.naive_bayes import BernoulliNB
                >>> ds = bt.from_pydict({"a": [1, 0], "b": [0, 1], "y": [0, 1]})
                >>> sorted(BernoulliNB(["a", "b"], "y").fit(ds).classes_)
                [0, 1]

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            ColumnNotFoundError: If a named column is missing.
        """
        from batcher.plan.functions.aggregate import sum as sum_

        _check_columns(ds, self.features, self.target)
        total = ds.count()
        aggregates: dict[str, object] = {"__bt_n": col(self.target).count()}
        for i, name in enumerate(self.features):
            aggregates[f"p{i}"] = sum_(self._present(name))
        grouped = ds.group_by(self.target).agg(**aggregates).collect()
        d = len(self.features)
        self.classes_, self.log_prior_, self.log_prob_, self.log_neg_prob_ = [], {}, {}, {}
        for row in range(grouped.num_rows):
            label = grouped.column(self.target)[row].as_py()
            class_count = float(grouped.column("__bt_n")[row].as_py())
            self.classes_.append(label)
            self.log_prior_[label] = math.log(class_count / total)
            probabilities = [
                (float(grouped.column(f"p{i}")[row].as_py() or 0.0) + self.alpha)
                / (class_count + 2 * self.alpha)
                for i in range(d)
            ]
            self.log_prob_[label] = [math.log(p) for p in probabilities]
            self.log_neg_prob_[label] = [math.log(1.0 - p) for p in probabilities]
        return self

    def _score(self, label: object):
        """The log-posterior expression, crediting present features and penalizing absent ones."""
        expression = lit(self.log_prior_[label])
        for name, log_p, log_neg in zip(
            self.features, self.log_prob_[label], self.log_neg_prob_[label], strict=True
        ):
            present = self._present(name)
            expression = expression + present * lit(log_p) + (lit(1.0) - present) * lit(log_neg)
        return expression

    def predict(self, ds: Dataset) -> Dataset:
        """Append the maximum-a-posteriori class label for each row.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.naive_bayes import BernoulliNB
                >>> ds = bt.from_pydict({"a": [1, 0], "b": [0, 1], "y": [0, 1]})
                >>> model = BernoulliNB(["a", "b"], "y").fit(ds)
                >>> model.predict(ds).to_pydict()["prediction"]
                [0, 1]

        Args:
            ds: The dataset to classify.

        Returns:
            A new lazy `Dataset` with the predicted-class column appended.
        """
        require_fitted(self, self.classes_)
        return ds.with_columns(
            **{self.output_column: argmax_prediction(self.classes_, self._score)}
        )


def _check_columns(ds: Dataset, features: Sequence[str], target: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest match for any missing column."""
    for name in (*features, target):
        if name not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, ds.columns, hint="Pass an existing column.")
            )
