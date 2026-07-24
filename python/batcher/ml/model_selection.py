"""Cross-validated scoring and learning curves — the model-selection loop, tied together.

Choosing a model means fitting it on several train/test splits and reading how it does across
them. The splits, the model, and the metric are three separate things everywhere else; here
they compose into one call. The splits come from `batcher.ml.splitting` (content-hash folds,
identical however the data is partitioned), the model is any object with `fit`/`predict`, and
the metric is any callable — so a cross-validated score is one line and runs each fold's data
through the engine rather than a driver-held array.

The pieces:

`cross_val_score`
    Fit and score the model on each fold, returning the per-fold scores. The mean is the
    headline; the spread across folds is the honesty — a model with a great mean and a huge
    variance is one split away from looking bad.
`cross_val_predict`
    The out-of-fold prediction for every row: each row scored by a model that never saw it in
    training. The right input for a stacked ensemble, and the only unbiased way to plot
    predicted-vs-actual on the training data.
`learning_curve`
    The score as a function of training-set size, which answers "would more data help". A gap
    that is still closing means collect more; a gap that has flattened means the ceiling is
    the model, not the data.

None of these fits the model for you beyond the folds — they call the `fit` you pass, so a
custom preprocessing-plus-model pipeline works as long as it exposes `fit`/`predict`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["cross_val_predict", "cross_val_score", "learning_curve", "validation_curve"]

# A fit callable takes a training Dataset and returns something predict can score; a predict
# callable takes (fitted, dataset) and returns a Dataset with a prediction column. Kept as
# plain callables so any framework composes without an adapter.
Fit = "Callable[[Dataset], Any]"
Predict = "Callable[[Any, Dataset], Dataset]"
Metric = "Callable[[Dataset, str, str], float]"


def _folds(
    ds: Dataset,
    k: int,
    seed: int,
    key: str | list[str] | None,
    stratify: str | None,
) -> list[tuple[Dataset, Dataset]]:
    """The k train/validation splits, stratified when a label is given."""
    from batcher.ml.splitting import kfold, stratified_kfold

    if stratify is not None:
        return stratified_kfold(ds, stratify, k, seed=seed, key=key)
    return kfold(ds, k, seed=seed, key=key)


def cross_val_score(
    ds: Dataset,
    fit: Fit,
    predict: Predict,
    *,
    y_true: str,
    metric: Metric,
    k: int = 5,
    prediction: str = "prediction",
    seed: int = 0,
    key: str | list[str] | None = None,
    stratify: str | None = None,
) -> list[float]:
    """Fit and score the model on each of `k` folds, returning the per-fold scores.

    The standard cross-validation loop: for each fold, `fit` on the other ``k-1`` folds,
    `predict` the held-out fold, and score it with `metric`. The list of `k` scores is what
    you actually want — its mean estimates generalization and its spread estimates how much to
    trust that mean.

    Args:
        ds: The full dataset.
        fit: ``fit(train_ds) -> model`` — trains on a fold's training split.
        predict: ``predict(model, ds) -> ds_with_prediction`` — scores a split.
        y_true: The label column.
        metric: ``metric(ds, y_true, prediction) -> float`` — the score per fold.
        k: How many folds.
        prediction: The name `predict` gives its output column.
        seed: Seed for the fold assignment.
        key: The column(s) identifying a row, for a stable content-hash split.
        stratify: A label column to stratify the folds on (for an imbalanced target).

    Returns:
        The `k` per-fold scores, in fold order.

    Raises:
        PlanError: If `k` is less than 2.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import evaluate
            >>> from batcher.ml.model_selection import cross_val_score
            >>> from sklearn.linear_model import LinearRegression
            >>> rows = {"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]}
            >>> ds = bt.from_pydict(rows)
            >>> def fit(train):
            ...     frame = train.to_pydict()
            ...     return LinearRegression().fit([[v] for v in frame["x"]], frame["y"])
            >>> def predict(model, d):
            ...     return d.ml.predict(model, features=["x"])
            >>> scores = cross_val_score(
            ...     ds, fit, predict, y_true="y", metric=lambda d, t, p: evaluate(
            ...         d, t, y_pred=p, task="regression", metrics=["r2"]
            ...     )["r2"], k=4, key="x"
            ... )
            >>> all(s > 0.99 for s in scores)
            True
    """
    if k < 2:
        raise PlanError(f"cross_val_score needs at least 2 folds, got {k}")
    scores = []
    for train, validate in _folds(ds, k, seed, key, stratify):
        model = fit(train)
        scored = predict(model, validate)
        scores.append(float(metric(scored, y_true, prediction)))
    return scores


def cross_val_predict(
    ds: Dataset,
    fit: Fit,
    predict: Predict,
    *,
    k: int = 5,
    seed: int = 0,
    key: str | list[str] | None = None,
    stratify: str | None = None,
) -> Dataset:
    """Return every row's out-of-fold prediction — scored by a model that never trained on it.

    For each fold, the model is trained on the other folds and used to predict the held-out
    one; the held-out predictions are concatenated so every row is covered exactly once. This
    is the unbiased prediction on the training data — the input a stacking ensemble needs from
    its base models, and the only honest way to draw a predicted-vs-actual plot without the
    optimism of scoring rows the model memorized.

    Args:
        ds: The full dataset.
        fit: ``fit(train_ds) -> model``.
        predict: ``predict(model, ds) -> ds_with_prediction``.
        k: How many folds.
        seed: Seed for the fold assignment.
        key: The column(s) identifying a row.
        stratify: A label column to stratify the folds on.

    Returns:
        A `Dataset` of every row with its out-of-fold prediction. Row order is not preserved.

    Raises:
        PlanError: If `k` is less than 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.model_selection import cross_val_predict
            >>> from sklearn.linear_model import LinearRegression
            >>> ds = bt.from_pydict(
            ...     {"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]}
            ... )
            >>> def fit(train):
            ...     f = train.to_pydict()
            ...     return LinearRegression().fit([[v] for v in f["x"]], f["y"])
            >>> oof = cross_val_predict(
            ...     ds, fit, lambda m, d: d.ml.predict(m, features=["x"]), k=4, key="x"
            ... )
            >>> oof.count()
            40
    """
    if k < 2:
        raise PlanError(f"cross_val_predict needs at least 2 folds, got {k}")
    parts = []
    for train, validate in _folds(ds, k, seed, key, stratify):
        model = fit(train)
        parts.append(predict(model, validate))
    result = parts[0]
    for part in parts[1:]:
        result = result.union(part)
    return result


def learning_curve(
    ds: Dataset,
    fit: Fit,
    predict: Predict,
    *,
    y_true: str,
    metric: Metric,
    fractions: list[float] | None = None,
    holdout: float = 0.25,
    prediction: str = "prediction",
    seed: int = 0,
    key: str | list[str] | None = None,
) -> Dataset:
    """The validation score as a function of training-set size — does more data help.

    Holds out a fixed validation split, then trains on growing fractions of the rest and
    scores each on that same split. The resulting curve answers the question a cross-validated
    score cannot: whether the model is data-limited (the score is still rising, so collect
    more) or model-limited (the score has plateaued, so a bigger dataset will not help and a
    better model might).

    Args:
        ds: The full dataset.
        fit: ``fit(train_ds) -> model``.
        predict: ``predict(model, ds) -> ds_with_prediction``.
        y_true: The label column.
        metric: ``metric(ds, y_true, prediction) -> float``.
        fractions: The training-set fractions to evaluate; ``[0.2, 0.4, 0.6, 0.8, 1.0]`` by
            default. Each is a fraction of the *training* portion, not the whole dataset.
        holdout: The fraction reserved as the fixed validation split.
        prediction: The name `predict` gives its output column.
        seed: Seed for the splits.
        key: The column(s) identifying a row.

    Returns:
        A `Dataset` of ``train_fraction``, ``train_rows``, ``score``, ordered by fraction.

    Raises:
        PlanError: If `holdout` is not in ``(0, 1)`` or a fraction is out of range.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import evaluate
            >>> from batcher.ml.model_selection import learning_curve
            >>> from sklearn.linear_model import LinearRegression
            >>> ds = bt.from_pydict(
            ...     {"x": [float(i) for i in range(200)], "y": [2.0 * i for i in range(200)]}
            ... )
            >>> def fit(train):
            ...     f = train.to_pydict()
            ...     return LinearRegression().fit([[v] for v in f["x"]], f["y"])
            >>> curve = learning_curve(
            ...     ds, fit, lambda m, d: d.ml.predict(m, features=["x"]),
            ...     y_true="y",
            ...     metric=lambda d, t, p: evaluate(d, t, y_pred=p, task="regression",
            ...         metrics=["r2"])["r2"],
            ...     fractions=[0.5, 1.0], key="x",
            ... )
            >>> curve.columns
            ['train_fraction', 'train_rows', 'score']
    """
    import batcher as bt

    if not 0.0 < holdout < 1.0:
        raise PlanError(f"holdout must be in (0, 1), got {holdout}")
    grid = fractions if fractions is not None else [0.2, 0.4, 0.6, 0.8, 1.0]
    for fraction in grid:
        if not 0.0 < fraction <= 1.0:
            raise PlanError(f"each training fraction must be in (0, 1], got {fraction}")
    train_pool, validation = ds.ml.train_test_split(holdout, seed=seed, key=_as_key(key))
    validation = validation.cache()
    rows: dict[str, list[Any]] = {"train_fraction": [], "train_rows": [], "score": []}
    for fraction in sorted(grid):
        subset = (
            train_pool
            if fraction >= 1.0
            else train_pool.ml.train_test_split(1.0 - fraction, seed=seed, key=_as_key(key))[0]
        )
        model = fit(subset)
        scored = predict(model, validation)
        rows["train_fraction"].append(float(fraction))
        rows["train_rows"].append(subset.count())
        rows["score"].append(float(metric(scored, y_true, prediction)))
    return bt.from_pydict(rows).sort("train_fraction")


def validation_curve(
    ds: Dataset,
    fit: Fit,
    predict: Predict,
    *,
    y_true: str,
    metric: Metric,
    param_values: list[Any],
    param_name: str = "param",
    k: int = 5,
    prediction: str = "prediction",
    seed: int = 0,
    key: str | list[str] | None = None,
    stratify: str | None = None,
) -> Dataset:
    """The cross-validated score as a function of one hyperparameter — the tuning curve.

    Where `learning_curve` varies the training-set size, this varies a *hyperparameter* and
    cross-validates at each value, so the resulting curve shows the bias-variance trade-off
    directly: the score rises as the parameter adds capacity, peaks, then falls as the model
    starts to overfit. The peak is the value to pick, and the shape says whether the search
    range was wide enough. The `fit` callable takes the parameter value as its second argument,
    ``fit(train_ds, value) -> model``.

    Args:
        ds: The full dataset.
        fit: ``fit(train_ds, value) -> model`` — trains at one parameter value.
        predict: ``predict(model, ds) -> ds_with_prediction``.
        y_true: The label column.
        metric: ``metric(ds, y_true, prediction) -> float``.
        param_values: The hyperparameter values to sweep.
        param_name: The name of the parameter column in the result.
        k: How many cross-validation folds per value.
        prediction: The name `predict` gives its output column.
        seed: Seed for the fold assignment.
        key: The column(s) identifying a row.
        stratify: A label column to stratify the folds on.

    Returns:
        A `Dataset` of ``<param_name>`` and ``score`` (the mean cross-validated score), one row
        per parameter value, ordered by the value.

    Raises:
        PlanError: If `k` is less than 2.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import evaluate
            >>> from batcher.ml.model_selection import validation_curve
            >>> from sklearn.tree import DecisionTreeRegressor
            >>> ds = bt.from_pydict(
            ...     {"x": [float(i % 20) for i in range(200)],
            ...      "y": [float((i % 20) ** 2) for i in range(200)]}
            ... )
            >>> def fit(train, depth):
            ...     f = train.to_pydict()
            ...     return DecisionTreeRegressor(max_depth=depth).fit([[v] for v in f["x"]], f["y"])
            >>> curve = validation_curve(
            ...     ds, fit, lambda m, d: d.ml.predict(m, features=["x"]),
            ...     y_true="y",
            ...     metric=lambda d, t, p: evaluate(d, t, y_pred=p, task="regression",
            ...         metrics=["r2"])["r2"],
            ...     param_values=[1, 5], param_name="max_depth", k=2, key="x",
            ... )
            >>> curve.columns
            ['max_depth', 'score']
    """
    import batcher as bt

    if k < 2:
        raise PlanError(f"validation_curve needs at least 2 folds, got {k}")
    rows: dict[str, list[Any]] = {param_name: [], "score": []}
    for value in param_values:
        fold_scores = []
        for train, validate in _folds(ds, k, seed, key, stratify):
            model = fit(train, value)
            scored = predict(model, validate)
            fold_scores.append(float(metric(scored, y_true, prediction)))
        rows[param_name].append(value)
        rows["score"].append(sum(fold_scores) / len(fold_scores))
    return bt.from_pydict(rows).sort(param_name)


def _as_key(key: str | list[str] | None) -> str | list[str] | None:
    """Pass a hash key through unchanged (train_test_split accepts the same shapes)."""
    return key
