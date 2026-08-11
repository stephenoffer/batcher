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

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from batcher.api.dataset import Dataset

    # `Fit`/`Predict` are the unbound pair (this module refits per fold) and `Scorer` grades
    # the result; all three are defined once in `ml._estimator`. They stay plain callables
    # rather than requiring the `Estimator` protocol, so a framework model composes without an
    # adapter — which is the reason this module took callables in the first place.
    from batcher.ml._estimator import Fit, Predict, Scorer

__all__ = [
    "SearchResult",
    "cross_val_predict",
    "cross_val_score",
    "grid_search",
    "learning_curve",
    "param_grid",
    "param_samples",
    "random_search",
    "validation_curve",
]


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
    metric: Scorer,
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
    metric: Scorer,
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
    metric: Scorer,
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


@dataclass(frozen=True)
class SearchResult:
    """What a hyperparameter search found: the winner, its score, and every trial.

    `trials` is the part worth reading. A best score reported alone hides whether the search
    found a peak or a plateau — and a plateau means the parameter did not matter, which is
    more useful to know than which arbitrary point on it won.

    Examples:
        .. doctest::

            >>> from batcher.ml.model_selection import SearchResult
            >>> found = SearchResult(
            ...     best_params={"alpha": 0.1}, best_score=0.9,
            ...     trials=[{"params": {"alpha": 0.1}, "mean": 0.9, "std": 0.01,
            ...              "scores": [0.9, 0.9]}],
            ... )
            >>> found.best_params
            {'alpha': 0.1}

    Args:
        best_params: The parameter combination that scored best.
        best_score: Its mean cross-validated score.
        trials: Every combination tried, best first, each with its ``params``, ``mean``,
            ``std``, and per-fold ``scores``.
    """

    best_params: dict[str, Any]
    best_score: float
    trials: list[dict[str, Any]]

    def to_dataset(self) -> Dataset:
        """The trials as a `Dataset`, one row per combination, best first.

        Examples:
            .. doctest::

                >>> from batcher.ml.model_selection import SearchResult
                >>> found = SearchResult(
                ...     best_params={"alpha": 0.1}, best_score=0.9,
                ...     trials=[{"params": {"alpha": 0.1}, "mean": 0.9, "std": 0.0,
                ...              "scores": [0.9]}],
                ... )
                >>> found.to_dataset().to_pydict()["alpha"]
                [0.1]

        Returns:
            A `Dataset` with one column per parameter plus ``mean_score`` and ``std_score``.
        """
        import batcher as bt

        names = sorted({name for trial in self.trials for name in trial["params"]})
        rows: dict[str, list[Any]] = {name: [] for name in names}
        rows["mean_score"] = []
        rows["std_score"] = []
        for trial in self.trials:
            for name in names:
                rows[name].append(trial["params"].get(name))
            rows["mean_score"].append(trial["mean"])
            rows["std_score"].append(trial["std"])
        return bt.from_pydict(rows)


def param_grid(**options: Sequence[Any]) -> list[dict[str, Any]]:
    """Every combination of the given parameter values, as a list of dicts.

    scikit-learn's ``ParameterGrid``, spelled as keyword arguments. Combinations come out in
    a stable order — the cartesian product with the last keyword varying fastest — so two
    runs of the same search try the same things in the same sequence.

    Args:
        **options: One sequence of candidate values per parameter name.

    Returns:
        Every combination, as ``{name: value}`` dicts.

    Raises:
        PlanError: If a parameter is given no candidate values.

    Examples:
        .. doctest::

            >>> from batcher.ml.model_selection import param_grid
            >>> param_grid(alpha=[0.1, 1.0], fit_intercept=[True])
            [{'alpha': 0.1, 'fit_intercept': True}, {'alpha': 1.0, 'fit_intercept': True}]
    """
    import itertools

    empty = [name for name, values in options.items() if not list(values)]
    if empty:
        raise PlanError(
            f"param_grid: {empty[0]!r} was given no candidate values, so the grid is empty. "
            "Every parameter needs at least one value."
        )
    names = list(options)
    return [
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*(list(options[n]) for n in names))
    ]


def param_samples(
    n: int, *, seed: int = 0, **distributions: Sequence[Any] | Callable[[Random], Any]
) -> list[dict[str, Any]]:
    """`n` random parameter combinations drawn from the given candidates.

    The sampling half of scikit-learn's ``RandomizedSearchCV``. A parameter's candidates are
    either a sequence, drawn from uniformly, or a callable taking a `random.Random` — which
    is how a continuous range is expressed without this module growing a dependency on a
    distribution library.

    Random search beats grid search once more than two or three parameters are in play:
    a grid spends its budget re-testing the parameters that do not matter, while random
    search gives every parameter `n` distinct values.

    Args:
        n: How many combinations to draw.
        seed: Seed for the draw, so a search is reproducible.
        **distributions: Per parameter, a sequence of candidates or a
            ``(random.Random) -> value`` callable.

    Returns:
        `n` ``{name: value}`` dicts.

    Raises:
        PlanError: If `n` is not positive, or a parameter has no candidates.

    Examples:
        .. doctest::

            >>> from batcher.ml.model_selection import param_samples
            >>> draws = param_samples(3, seed=0, alpha=[0.1, 1.0],
            ...                       depth=lambda rng: rng.randint(1, 5))
            >>> len(draws), sorted(draws[0])
            (3, ['alpha', 'depth'])
    """
    if n < 1:
        raise PlanError(f"param_samples: n must be at least 1, got {n}")
    for name, candidates in distributions.items():
        if not callable(candidates) and not list(candidates):
            raise PlanError(f"param_samples: {name!r} was given no candidate values")
    rng = Random(seed)
    draws: list[dict[str, Any]] = []
    for _ in range(n):
        draw: dict[str, Any] = {}
        for name, candidates in distributions.items():
            draw[name] = candidates(rng) if callable(candidates) else rng.choice(list(candidates))
        draws.append(draw)
    return draws


def grid_search(
    ds: Dataset,
    fit: Callable[[Dataset, dict[str, Any]], Any],
    predict: Predict,
    *,
    y_true: str,
    metric: Scorer,
    grid: Sequence[dict[str, Any]],
    greater_is_better: bool = True,
    k: int = 5,
    prediction: str = "prediction",
    seed: int = 0,
    key: str | list[str] | None = None,
    stratify: str | None = None,
) -> SearchResult:
    """Cross-validate every parameter combination in `grid` and return the best.

    scikit-learn's ``GridSearchCV``, over `cross_val_score`'s folds. Each combination is
    scored on the same `k` folds, so the comparison between combinations is paired — the
    same rows train and validate every candidate, which removes fold-assignment luck from
    the difference between two scores.

    Nothing is refitted on the full dataset afterwards. Returning the winning *parameters*
    rather than a fitted model keeps this independent of what `fit` builds, and refitting is
    one call the caller was going to make anyway.

    Args:
        ds: The dataset to cross-validate over.
        fit: ``fit(train_ds, params) -> model`` — trains one candidate on a fold.
        predict: ``predict(model, ds) -> ds_with_prediction``.
        y_true: The label column.
        metric: ``metric(ds, y_true, prediction) -> float`` — the score per fold.
        grid: The parameter combinations to try, as from `param_grid`.
        greater_is_better: Whether a larger `metric` is better. Set it to ``False`` for a
            loss such as RMSE, or the search returns the worst combination.
        k: How many folds.
        prediction: The name `predict` gives its output column.
        seed: Seed for the fold assignment.
        key: The column(s) identifying a row, for a stable content-hash split.
        stratify: A label column to stratify the folds on.

    Returns:
        A `SearchResult` with the winning parameters, its mean score, and every trial.

    Raises:
        PlanError: If `grid` is empty or `k` is less than 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import Ridge
            >>> from batcher.ml.metrics import evaluate
            >>> from batcher.ml.model_selection import grid_search, param_grid
            >>> ds = bt.from_pydict(
            ...     {"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]}
            ... )
            >>> found = grid_search(
            ...     ds,
            ...     lambda train, p: Ridge(["x"], "y", alpha=p["alpha"]).fit(train),
            ...     lambda model, d: model.predict(d),
            ...     y_true="y",
            ...     metric=lambda d, t, p: evaluate(
            ...         d, t, y_pred=p, task="regression", metrics=["r2"]
            ...     )["r2"],
            ...     grid=param_grid(alpha=[0.01, 100.0]),
            ...     k=4,
            ...     key="x",
            ... )
            >>> found.best_params
            {'alpha': 0.01}
    """
    combinations = list(grid)
    if not combinations:
        raise PlanError("grid_search needs at least one parameter combination to try")
    if k < 2:
        raise PlanError(f"grid_search needs at least 2 folds, got {k}")
    folds = _folds(ds, k, seed, key, stratify)
    trials: list[dict[str, Any]] = []
    for params in combinations:
        scores = []
        for train, validate in folds:
            scored = predict(fit(train, params), validate)
            scores.append(float(metric(scored, y_true, prediction)))
        trials.append({"params": dict(params), **_summary(scores)})
    trials.sort(key=lambda t: t["mean"], reverse=greater_is_better)
    return SearchResult(
        best_params=trials[0]["params"], best_score=trials[0]["mean"], trials=trials
    )


def random_search(
    ds: Dataset,
    fit: Callable[[Dataset, dict[str, Any]], Any],
    predict: Predict,
    *,
    y_true: str,
    metric: Scorer,
    distributions: dict[str, Sequence[Any] | Callable[[Random], Any]],
    n_iter: int = 10,
    greater_is_better: bool = True,
    k: int = 5,
    prediction: str = "prediction",
    seed: int = 0,
    key: str | list[str] | None = None,
    stratify: str | None = None,
) -> SearchResult:
    """Cross-validate `n_iter` random parameter draws and return the best.

    scikit-learn's ``RandomizedSearchCV``. Prefer it to `grid_search` once there are more
    than two or three parameters: a grid spends most of its budget re-testing the ones that
    do not matter, while a random draw gives every parameter `n_iter` distinct values for
    the same number of fits.

    Args:
        ds: The dataset to cross-validate over.
        fit: ``fit(train_ds, params) -> model``.
        predict: ``predict(model, ds) -> ds_with_prediction``.
        y_true: The label column.
        metric: ``metric(ds, y_true, prediction) -> float``.
        distributions: Per parameter, a sequence of candidates or a
            ``(random.Random) -> value`` callable.
        n_iter: How many combinations to draw and score.
        greater_is_better: Whether a larger `metric` is better.
        k: How many folds.
        prediction: The name `predict` gives its output column.
        seed: Seed for both the draw and the fold assignment.
        key: The column(s) identifying a row.
        stratify: A label column to stratify the folds on.

    Returns:
        A `SearchResult` with the winning parameters, its mean score, and every trial.

    Raises:
        PlanError: If `n_iter` is not positive or `k` is less than 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import Ridge
            >>> from batcher.ml.metrics import evaluate
            >>> from batcher.ml.model_selection import random_search
            >>> ds = bt.from_pydict(
            ...     {"x": [float(i) for i in range(40)], "y": [2.0 * i for i in range(40)]}
            ... )
            >>> found = random_search(
            ...     ds,
            ...     lambda train, p: Ridge(["x"], "y", alpha=p["alpha"]).fit(train),
            ...     lambda model, d: model.predict(d),
            ...     y_true="y",
            ...     metric=lambda d, t, p: evaluate(
            ...         d, t, y_pred=p, task="regression", metrics=["r2"]
            ...     )["r2"],
            ...     distributions={"alpha": [0.01, 0.1, 1.0]},
            ...     n_iter=3,
            ...     k=4,
            ...     key="x",
            ... )
            >>> found.best_score > 0.99
            True
    """
    draws = param_samples(n_iter, seed=seed, **distributions)
    return grid_search(
        ds,
        fit,
        predict,
        y_true=y_true,
        metric=metric,
        grid=draws,
        greater_is_better=greater_is_better,
        k=k,
        prediction=prediction,
        seed=seed,
        key=key,
        stratify=stratify,
    )


def _summary(scores: list[float]) -> dict[str, Any]:
    """The mean, population standard deviation, and raw per-fold scores of one trial."""
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return {"mean": mean, "std": variance**0.5, "scores": scores}
