"""Missing-value imputation — fit a fill value per column, transform with COALESCE.

`fit` learns each column's fill value (mean / median / most-frequent / a constant),
each a single aggregate over the engine; `transform` replaces nulls with
``coalesce(col, fill)`` — an `Expr`, so the fill happens in the data plane. ``mean``
and ``median`` cast the column to float (the scikit-learn convention); ``most_frequent``
and ``constant`` keep the original type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, columns_arg, fit_aggregate
from batcher.plan.expr_ir import coalesce, col, count, lit, when

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["IterativeImputer", "SimpleImputer"]

_STRATEGIES = ("mean", "median", "most_frequent", "constant")


class SimpleImputer(Preprocessor):
    """Fill missing values in `columns` using a per-column statistic.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import SimpleImputer
            >>> ds = bt.from_pydict({"a": [1.0, None, 3.0]})
            >>> SimpleImputer(["a"]).fit_transform(ds).to_pydict()
            {'a': [1.0, 2.0, 3.0]}

    Args:
        columns: the columns to impute in place.
        strategy: ``"mean"``, ``"median"``, ``"most_frequent"``, or ``"constant"``.
        fill_value: the constant to use when ``strategy="constant"`` (required then).
    """

    __slots__ = ("columns", "fill_value", "statistics_", "strategy")

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        strategy: str = "mean",
        fill_value: Any = None,
    ) -> None:
        self.columns = columns_arg(columns, what="SimpleImputer")
        if not self.columns:
            raise PlanError("SimpleImputer requires at least one column")
        if strategy not in _STRATEGIES:
            raise PlanError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")
        if strategy == "constant" and fill_value is None:
            raise PlanError("SimpleImputer(strategy='constant') requires fill_value")
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: dict[str, Any] = {}

    def _check_numeric(self, ds: Dataset) -> None:
        """Require numeric columns only for the statistics that are arithmetic.

        `numeric_only` is a property of the class everywhere else, but here it is a property of
        the *strategy*: filling with the most frequent value, or with a constant, is exactly how
        a categorical column is imputed and must keep working on strings. Only the mean and the
        median need a number, and those were the two that failed inside the engine.

        Args:
            ds: The dataset whose schema to read.

        Raises:
            PlanError: If the strategy is arithmetic and a named column is not a number.
        """
        if self.strategy not in ("mean", "median"):
            return
        from batcher.ml._estimator import require_numeric

        require_numeric(self, ds, self.columns, role="column")

    def fit(self, ds: Dataset) -> SimpleImputer:
        """Learn each column's fill value into `statistics_` per the chosen strategy.

        ``"mean"`` / ``"median"`` run one mergeable aggregate; ``"most_frequent"`` a
        grouped count; ``"constant"`` just reuses `fill_value`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import SimpleImputer
                >>> SimpleImputer(["a"]).fit(bt.from_pydict({"a": [1.0, None, 3.0]})).statistics_
                {'a': 2.0}

        Args:
            ds: The dataset to compute each column's fill value from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column has no non-null values to learn a statistic from.
        """
        self._check_numeric(ds)
        if self.strategy == "constant":
            self.statistics_ = dict.fromkeys(self.columns, self.fill_value)
        elif self.strategy == "most_frequent":
            self.statistics_ = {c: self._mode(ds, c) for c in self.columns}
        elif self.strategy == "mean":
            self.statistics_ = dict(fit_aggregate(ds, {c: col(c).mean() for c in self.columns}))
        else:  # median
            self.statistics_ = dict(fit_aggregate(ds, {c: col(c).median() for c in self.columns}))
        for c in self.columns:
            if self.statistics_[c] is None:
                raise PlanError(
                    f"SimpleImputer cannot fit column {c!r}: no non-null values for "
                    f"strategy {self.strategy!r}"
                )
        self._fitted = True
        return self

    @staticmethod
    def _mode(ds: Dataset, column: str) -> Any:
        """The most frequent non-null value of `column` (ties: engine order)."""
        grouped = (
            ds.filter(col(column).is_not_null())
            .group_by(column)
            .agg(__n=count())
            .sort("__n", descending=True)
            .limit(1)
            .collect()
        )
        return grouped.column(column)[0].as_py() if grouped.num_rows else None

    def transform(self, ds: Dataset) -> Dataset:
        """Replace nulls in each fitted column with its learned fill value.

        Lowered to ``coalesce(col, fill)``; ``"mean"``/``"median"`` first cast the
        column to float (the scikit-learn convention).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import SimpleImputer
                >>> ds = bt.from_pydict({"a": [1.0, None, 3.0]})
                >>> SimpleImputer(["a"]).fit(ds).transform(ds).to_pydict()
                {'a': [1.0, 2.0, 3.0]}

        Args:
            ds: The dataset to fill.

        Returns:
            A new lazy `Dataset` with nulls in the fitted columns filled.
        """
        self._require_fitted()
        cast_float = self.strategy in ("mean", "median")
        new = {}
        for c in self.columns:
            base = col(c).cast("float64") if cast_float else col(c)
            new[c] = coalesce(base, lit(self.statistics_[c]))
        return ds.with_columns(**new)


class IterativeImputer(Preprocessor):
    """Impute each column by regressing it on the others, repeatedly.

    `SimpleImputer` fills a column with one number, which throws away everything the row's
    *other* columns say about it: a missing income in a row with a known job title and
    postcode is not well described by the global mean. This is scikit-learn's
    ``IterativeImputer``, the MICE-style alternative — model each incomplete column from the
    remaining ones, fill the gaps with the model's prediction, and repeat so that later
    rounds see better fills than earlier ones did.

    `fit` records the whole schedule: the initial per-column fill, then one linear model per
    incomplete column per round, in order. `transform` replays exactly that schedule, so a
    serving row is imputed by the same models in the same sequence as a training row — which
    is the part a hand-rolled loop usually gets wrong.

    The cost is real: a fit is up to ``max_iter * len(incomplete columns)`` model fits, each
    a pass over the data. Reach for `SimpleImputer` when the columns are unrelated, and for
    this when they are not.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import IterativeImputer
            >>> ds = bt.from_pydict(
            ...     {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, None, 8.0]}
            ... )
            >>> out = IterativeImputer(["a", "b"], max_iter=3).fit_transform(ds)
            >>> round(out.to_pydict()["b"][2], 6)
            6.0

    Args:
        columns: The numeric columns to impute and to regress on each other.
        max_iter: The maximum number of rounds over the incomplete columns.
        initial_strategy: How to fill before the first round — ``"mean"`` or ``"median"``.
        tol: Stop early once a round moves no imputed value by more than this.
        ridge: The ridge penalty on each round's regression. A small positive value is the
            default because the columns are correlated by assumption, which is exactly when
            an unpenalized least-squares fit is unstable.
    """

    numeric_only = True

    __slots__ = (
        "columns",
        "imputations_",
        "initial_",
        "initial_strategy",
        "max_iter",
        "n_iter_",
        "ridge",
        "tol",
    )

    def __init__(
        self,
        columns: str | Sequence[str],
        *,
        max_iter: int = 10,
        initial_strategy: str = "mean",
        tol: float = 1e-3,
        ridge: float = 1e-6,
    ) -> None:
        self.columns = columns_arg(columns, what="IterativeImputer")
        if len(self.columns) < 2:
            raise PlanError(
                "IterativeImputer needs at least two columns: it imputes each one *from the "
                "others*, so a single column has nothing to regress on. Use SimpleImputer."
            )
        if max_iter < 1:
            raise PlanError(f"IterativeImputer: max_iter must be at least 1, got {max_iter}")
        if initial_strategy not in ("mean", "median"):
            raise PlanError(
                f"IterativeImputer: initial_strategy must be 'mean' or 'median', "
                f"got {initial_strategy!r}"
            )
        if tol < 0:
            raise PlanError(f"IterativeImputer: tol must be non-negative, got {tol!r}")
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.tol = tol
        self.ridge = ridge
        self.initial_: dict[str, float] = {}
        self.imputations_: list[dict[str, Any]] = []
        self.n_iter_: int = 0

    def _missing_flag(self, column: str) -> str:
        """The helper column name carrying `column`'s original missingness."""
        return f"__bt_missing_{column}"

    def _staged(self, ds: Dataset) -> Dataset:
        """`ds` with the missingness flags captured and the initial fills applied.

        The flags have to be taken before any filling, and carried alongside: once a column
        has been filled it no longer knows which of its values were imputed, and every later
        round must only overwrite the ones that were.
        """
        flags = {self._missing_flag(c): col(c).is_null() for c in self.columns}
        filled = {c: coalesce(col(c).cast("float64"), lit(self.initial_[c])) for c in self.columns}
        return ds.with_columns(**flags).with_columns(**filled)

    def _apply(self, ds: Dataset, step: dict[str, Any]) -> Dataset:
        """Overwrite one column's originally-missing entries with a fitted model's output."""
        from batcher.ml._estimator import linear_score

        target = step["column"]
        predicted = linear_score(step["features"], step["coef"], step["intercept"])
        return ds.with_columns(
            **{target: when(col(self._missing_flag(target))).then(predicted).otherwise(col(target))}
        )

    def fit(self, ds: Dataset) -> IterativeImputer:
        """Learn the initial fills and the per-round regression schedule.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import IterativeImputer
                >>> ds = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, None, 6.0]})
                >>> fitted = IterativeImputer(["a", "b"], max_iter=2).fit(ds)
                >>> fitted.imputations_[0]["column"]
                'b'

        Args:
            ds: The training dataset.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If a column is entirely null, leaving nothing to learn a fill from.
            ColumnNotFoundError: If a named column is missing.
        """
        self._check_numeric(ds)
        from batcher.ml.linear import Ridge

        aggregates = {}
        for name in self.columns:
            expression = col(name).cast("float64")
            aggregates[name] = (
                expression.mean() if self.initial_strategy == "mean" else expression.median()
            )
        learned = fit_aggregate(ds, aggregates)
        empty = [name for name, value in learned.items() if value is None]
        if empty:
            raise PlanError(
                f"IterativeImputer: column {empty[0]!r} has no non-null values, so there is "
                "nothing to learn an initial fill from. Drop it, or supply a constant with "
                "SimpleImputer first."
            )
        self.initial_ = {name: float(value) for name, value in learned.items()}
        incomplete = [
            name for name in self.columns if ds.filter(col(name).is_null()).limit(1).count() > 0
        ]
        self.imputations_ = []
        self.n_iter_ = 0
        if not incomplete:
            self._fitted = True
            return self

        working = self._staged(ds)
        for _round in range(self.max_iter):
            self.n_iter_ += 1
            largest_move = 0.0
            for target in incomplete:
                features = [c for c in self.columns if c != target]
                observed = working.filter(~col(self._missing_flag(target)))
                model = Ridge(features, target, alpha=self.ridge).fit(observed)
                step = {
                    "column": target,
                    "features": features,
                    "coef": [float(v) for v in model.coef_],
                    "intercept": float(model.intercept_),
                }
                # Keep the pre-update value beside the column so the round's movement is a
                # per-row difference inside one frame, not a comparison across two datasets
                # whose rows nothing lines up.
                previous = f"__bt_previous_{target}"
                staged = working.with_columns(**{previous: col(target)})
                updated = self._apply(staged, step)
                moved = updated.agg(__bt_move=(col(target) - col(previous)).abs().max()).collect()
                movement = moved.column("__bt_move")[0].as_py()
                largest_move = max(largest_move, abs(float(movement)) if movement else 0.0)
                working = updated.drop(previous)
                self.imputations_.append(step)
            if largest_move <= self.tol:
                break
        self._fitted = True
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Replay the fitted schedule: initial fills, then each round's model in order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import IterativeImputer
                >>> train = bt.from_pydict({"a": [1.0, 2.0, 3.0], "b": [2.0, None, 6.0]})
                >>> fitted = IterativeImputer(["a", "b"], max_iter=2).fit(train)
                >>> out = fitted.transform(bt.from_pydict({"a": [4.0], "b": [None]}))
                >>> out.to_pydict()["b"][0] is not None
                True

        Args:
            ds: The dataset to impute.

        Returns:
            A new lazy `Dataset` with the imputed columns and no helper columns left behind.
        """
        self._require_fitted()
        working = self._staged(ds)
        for step in self.imputations_:
            working = self._apply(working, step)
        return working.drop(*(self._missing_flag(c) for c in self.columns))
