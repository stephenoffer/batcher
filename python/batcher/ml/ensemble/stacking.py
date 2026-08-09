"""`StackingEnsemble` — let a second model learn how to combine the first ones.

Blending averages the base models with weights you choose. Stacking learns the weights, and
learns them *conditionally*: a meta-model over the base predictions can discover that one
model is the one to trust on short documents and another on long ones, which no fixed
average can express.

The thing that makes stacking work, and the thing everyone gets wrong, is what the
meta-model trains on. Train it on predictions the base models made about rows they were
fitted on and it learns to trust whichever model memorized hardest, which is the one that
will do worst in production. The meta-model has to see *out-of-fold* predictions: each row
scored by a base model that never saw it.

`batcher.ml.model_selection.cross_val_predict` already produces exactly that for one model.
What this adds is producing it for several models **in the same fold loop**, so every base
model's out-of-fold column describes the same row without a join and without needing a key
column to join on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from batcher.api.dataset import Dataset

__all__ = ["StackingEnsemble", "out_of_fold_features"]


def _validate_bases(bases: Mapping[str, tuple[Any, Any]], prediction: str) -> None:
    """Reject a base-model spec that cannot produce usable feature columns."""
    if not bases:
        raise PlanError("StackingEnsemble needs at least one base model")
    for name, pair in bases.items():
        if name == prediction:
            raise PlanError(
                f"StackingEnsemble: base model {name!r} collides with the prediction column "
                f"name {prediction!r}. Rename the base model, or pass a different "
                "prediction= name."
            )
        if not isinstance(pair, tuple) or len(pair) != 2 or not all(callable(f) for f in pair):
            raise PlanError(
                f"StackingEnsemble: base model {name!r} must be a (fit, predict) pair of "
                "callables, as cross_val_score takes."
            )


def out_of_fold_features(
    ds: Dataset,
    bases: Mapping[str, tuple[Callable[[Dataset], Any], Callable[[Any, Dataset], Dataset]]],
    *,
    k: int = 5,
    seed: int = 0,
    key: str | list[str] | None = None,
    stratify: str | None = None,
    prediction: str = "prediction",
) -> Dataset:
    """Every row's out-of-fold prediction from each base model, as one column each.

    All the base models are fitted and scored inside a single fold loop, so each row carries
    one column per base model describing *that row*, with no join and no row key. Running
    `cross_val_predict` once per model instead would give datasets whose row orders do not
    correspond, which is why this exists.

    Args:
        ds: The training split.
        bases: ``{name: (fit, predict)}`` — the same callable pair `cross_val_score` takes.
            Each name becomes a feature column.
        k: How many folds.
        seed: Seed for the fold assignment.
        key: The column(s) identifying a row, for a stable content-hash split.
        stratify: A label column to stratify the folds on.
        prediction: The column name each base model's `predict` writes into.

    Returns:
        A `Dataset` of every row with one out-of-fold prediction column per base model. Row
        order is not preserved.

    Raises:
        PlanError: If `k` is less than 2, or a base model is malformed.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import Ridge
            >>> from batcher.ml.ensemble import out_of_fold_features
            >>> ds = bt.from_pydict(
            ...     {"x": [float(i) for i in range(20)], "y": [2.0 * i for i in range(20)]}
            ... )
            >>> bases = {"ridge": (lambda d: Ridge(["x"], "y").fit(d), lambda m, d: m.predict(d))}
            >>> sorted(out_of_fold_features(ds, bases, k=4, key="x").columns)
            ['ridge', 'x', 'y']
    """
    from batcher.ml.model_selection import _folds

    if k < 2:
        raise PlanError(f"out_of_fold_features needs at least 2 folds, got {k}")
    _validate_bases(bases, prediction)
    parts: list[Dataset] = []
    for train, validate in _folds(ds, k, seed, key, stratify):
        part = validate
        for name, (fit, predict) in bases.items():
            scored = predict(fit(train), part)
            if prediction not in scored.columns:
                raise PlanError(
                    f"StackingEnsemble: base model {name!r} produced no {prediction!r} column. "
                    "Its predict must append that column, or pass prediction= naming the one "
                    "it does append."
                )
            part = scored.with_columns(**{name: col(prediction)}).drop(prediction)
        parts.append(part)
    combined = parts[0]
    for part in parts[1:]:
        combined = combined.union(part)
    return combined


class StackingEnsemble:
    """Combine several base models with a meta-model fitted on out-of-fold predictions.

    `fit` does two things: it builds the out-of-fold feature table the meta-model learns
    from, and it refits every base model on the *whole* training split so `predict` has
    something to score new rows with. Both are necessary and they are not the same models —
    the out-of-fold columns come from `k` fold-fitted copies, the prediction path from one
    full-data fit.

    Base models are ``(fit, predict)`` callable pairs, the same shape `cross_val_score`
    takes, so a Batcher estimator, a scikit-learn one, or a whole preprocessing pipeline all
    compose without an adapter.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression, Ridge
            >>> from batcher.ml.ensemble import StackingEnsemble
            >>> ds = bt.from_pydict(
            ...     {"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]}
            ... )
            >>> bases = {
            ...     "ridge": (lambda d: Ridge(["x"], "y", alpha=1.0).fit(d),
            ...               lambda m, d: m.predict(d)),
            ...     "ols": (lambda d: LinearRegression(["x"], "y").fit(d),
            ...             lambda m, d: m.predict(d)),
            ... }
            >>> meta = (lambda d: LinearRegression(["ridge", "ols"], "y").fit(d),
            ...         lambda m, d: m.predict(d))
            >>> stack = StackingEnsemble(bases, meta, k=4, key="x").fit(ds)
            >>> "prediction" in stack.predict(ds).columns
            True

    Args:
        bases: ``{name: (fit, predict)}`` — the base models. Each name becomes a feature
            column the meta-model sees.
        meta: The ``(fit, predict)`` pair for the meta-model, fitted on those columns.
        k: How many folds the out-of-fold features are built from.
        seed: Seed for the fold assignment.
        key: The column(s) identifying a row, for a stable content-hash split.
        stratify: A label column to stratify the folds on.
        prediction: The column name every model's `predict` writes into.
    """

    __slots__ = (
        "bases",
        "bases_",
        "k",
        "key",
        "meta",
        "meta_",
        "prediction",
        "seed",
        "stratify",
    )

    def __init__(
        self,
        bases: Mapping[str, tuple[Callable[[Dataset], Any], Callable[[Any, Dataset], Dataset]]],
        meta: tuple[Callable[[Dataset], Any], Callable[[Any, Dataset], Dataset]],
        *,
        k: int = 5,
        seed: int = 0,
        key: str | list[str] | None = None,
        stratify: str | None = None,
        prediction: str = "prediction",
    ) -> None:
        _validate_bases(bases, prediction)
        if not isinstance(meta, tuple) or len(meta) != 2 or not all(callable(f) for f in meta):
            raise PlanError("StackingEnsemble: meta must be a (fit, predict) pair of callables")
        if k < 2:
            raise PlanError(f"StackingEnsemble needs at least 2 folds, got {k}")
        self.bases = dict(bases)
        self.meta = meta
        self.k = k
        self.seed = seed
        self.key = key
        self.stratify = stratify
        self.prediction = prediction
        self.bases_: dict[str, Any] = {}
        self.meta_: Any = None

    def fit(self, ds: Dataset) -> StackingEnsemble:
        """Build the out-of-fold features, fit the meta-model, and refit the bases in full.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression
                >>> from batcher.ml.ensemble import StackingEnsemble
                >>> ds = bt.from_pydict(
                ...     {"x": [float(i) for i in range(20)], "y": [2.0 * i for i in range(20)]}
                ... )
                >>> base = (lambda d: LinearRegression(["x"], "y").fit(d),
                ...         lambda m, d: m.predict(d))
                >>> meta = (lambda d: LinearRegression(["ols"], "y").fit(d),
                ...         lambda m, d: m.predict(d))
                >>> stack = StackingEnsemble({"ols": base}, meta, k=4, key="x").fit(ds)
                >>> sorted(stack.bases_)
                ['ols']

        Args:
            ds: The training split.

        Returns:
            ``self``, fitted.
        """
        features = out_of_fold_features(
            ds,
            self.bases,
            k=self.k,
            seed=self.seed,
            key=self.key,
            stratify=self.stratify,
            prediction=self.prediction,
        )
        self.meta_ = self.meta[0](features)
        # Refit on everything: the fold-fitted copies exist only to produce honest features,
        # and each of them saw a fraction of the data. Scoring a new row with one of them
        # would throw away the rest of the training split for no reason.
        self.bases_ = {name: fit(ds) for name, (fit, _) in self.bases.items()}
        return self

    def predict(self, ds: Dataset) -> Dataset:
        """Score `ds` with every base model, then combine them with the meta-model.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression
                >>> from batcher.ml.ensemble import StackingEnsemble
                >>> ds = bt.from_pydict(
                ...     {"x": [float(i) for i in range(20)], "y": [2.0 * i for i in range(20)]}
                ... )
                >>> base = (lambda d: LinearRegression(["x"], "y").fit(d),
                ...         lambda m, d: m.predict(d))
                >>> meta = (lambda d: LinearRegression(["ols"], "y").fit(d),
                ...         lambda m, d: m.predict(d))
                >>> stack = StackingEnsemble({"ols": base}, meta, k=4, key="x").fit(ds)
                >>> stack.predict(ds).count()
                20

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the base columns and the ensemble's prediction.

        Raises:
            PlanError: If `fit` has not run yet.
        """
        if self.meta_ is None:
            raise PlanError("StackingEnsemble must be fitted before predict().")
        scored = ds
        for name, (_, predict) in self.bases.items():
            produced = predict(self.bases_[name], scored)
            scored = produced.with_columns(**{name: col(self.prediction)}).drop(self.prediction)
        return self.meta[1](self.meta_, scored)
