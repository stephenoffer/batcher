"""Model interpretation at scale — why the model predicts what it does, over the whole set.

Explaining a model is normally done on a sample small enough to fit a driver, because the
standard tools re-score the data many times. Both techniques here re-score too, but they do
it *through the engine*, so the explanation runs over the same data the model scores rather
than a hopeful subsample of it.

`permutation_importance`
    How much the model relies on each feature, measured by how far a metric falls when that
    feature's values are shuffled. Model-agnostic (it treats the model as a black box), and
    honest in a way a tree's built-in importance is not: a tree can report a feature as
    important because it *split* on it, even if permuting it changes nothing.
`partial_dependence`
    What the model does as one feature varies, holding the rest as they are. The curve that
    says "the prediction rises with tenure up to two years, then flattens" — the shape a
    stakeholder can read, averaged over the real joint distribution of the other features
    rather than a synthetic grid.

Both take a `predict` callable — usually `ds.ml.predict` bound to a model — so they compose
with the tabular inference path rather than duplicating it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns as _require
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.ml._estimator import Predictor, Scorer

__all__ = ["partial_dependence", "permutation_importance"]


def permutation_importance(
    ds: Dataset,
    predict: Predictor,
    features: list[str],
    *,
    y_true: str,
    prediction: str = "prediction",
    metric: Scorer | None = None,
    n_repeats: int = 3,
    seed: int = 0,
) -> Dataset:
    """Rank features by how much the model's error rises when each is shuffled.

    For each feature, its column is permuted (destroying its relationship with the target
    while keeping its marginal distribution), the model is re-scored, and the increase in the
    error metric is the feature's importance. A feature the model does not actually use shows
    an importance near zero however prominently it appears in the model's own internals.

    The permutation is a full-column shuffle, so this runs over the whole dataset, not a
    sample — the explanation describes the model's behaviour on the real data.

    Args:
        ds: The scored dataset, holding the features and the true label.
        predict: A callable turning a `Dataset` into one with a `prediction` column —
            typically a model bound through `ds.ml.predict`.
        features: The feature columns to measure.
        y_true: The true-label column.
        prediction: The name `predict` gives its output column.
        metric: The error metric ``(ds, y_true, y_pred) -> float``, lower better. Defaults to
            RMSE for a numeric target.
        n_repeats: How many independent shuffles to average per feature; more is steadier.
        seed: Seed for the shuffles; the same seed reproduces the ranking.

    Returns:
        A `Dataset` of ``feature``, ``importance`` (mean metric rise), ``std``, ordered by
        descending importance.

    Raises:
        PlanError: If `features` is empty, or `n_repeats` is less than 1.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.interpret import permutation_importance
            >>> ds = bt.from_pydict(
            ...     {"drives": [1.0, 2.0, 3.0, 4.0], "noise": [1.0, 1.0, 1.0, 1.0],
            ...      "y": [1.0, 2.0, 3.0, 4.0]}
            ... )
            >>> scored = lambda d: d.with_columns(prediction=bt.col("drives"))
            >>> imp = permutation_importance(ds, scored, ["drives", "noise"], y_true="y")
            >>> imp.to_pydict()["feature"][0]
            'drives'
    """
    import batcher as bt

    _require(ds, y_true, *features)
    if not features:
        raise PlanError("permutation_importance needs at least one feature")
    if n_repeats < 1:
        raise PlanError(f"n_repeats must be at least 1, got {n_repeats}")
    scorer = metric or _rmse
    baseline = scorer(predict(ds), y_true, prediction)

    rows: dict[str, list[Any]] = {"feature": [], "importance": [], "std": []}
    for feature in features:
        rises = []
        for repeat in range(n_repeats):
            shuffled = _permute_column(ds, feature, seed=seed + repeat)
            rises.append(scorer(predict(shuffled), y_true, prediction) - baseline)
        mean = sum(rises) / len(rises)
        variance = sum((r - mean) ** 2 for r in rises) / len(rises)
        rows["feature"].append(feature)
        rows["importance"].append(mean)
        rows["std"].append(variance**0.5)
    return bt.from_pydict(rows).sort("importance", descending=True)


def _permute_column(ds: Dataset, column: str, *, seed: int) -> Dataset:
    """`ds` with `column` replaced by a shuffled copy of itself, other columns intact.

    A permutation must move the column's values *relative to the other columns* while keeping
    its own marginal distribution. Shuffling the whole dataset and lifting only this column
    off the shuffled copy does exactly that, and it stays a relational op — a shuffle plus a
    positional zip — rather than a driver-side array shuffle.
    """
    others = [c for c in ds.columns if c != column]
    shuffled = ds.select(column).shuffle(seed=seed).rename({column: "__bt_perm"})
    # Positional recombination: attach the shuffled column back by row index, so the
    # permuted values pair with the *original* rows' other features.
    left = ds.select(*others).with_row_index("__bt_i")
    right = shuffled.with_row_index("__bt_i")
    return (
        left.join(right, on="__bt_i", how="inner")
        .with_columns(**{column: col("__bt_perm")})
        .drop("__bt_i", "__bt_perm")
        .select(*ds.columns)
    )


def partial_dependence(
    ds: Dataset,
    predict: Predictor,
    feature: str,
    *,
    grid: list[float] | None = None,
    grid_points: int = 10,
    prediction: str = "prediction",
) -> Dataset:
    """The average prediction as one feature is swept across a grid, others held as they are.

    For each grid value, the feature is set to that value for *every* row, the model is
    re-scored, and the mean prediction is recorded. The resulting curve is what the model
    does as that feature moves, averaged over the real joint distribution of the other
    features — the shape a stakeholder reads as "the risk rises with balance, then plateaus".

    Averaging over the actual rows rather than a synthetic background is the honest version:
    it weights each combination of the other features by how often it really occurs.

    Args:
        ds: The dataset whose rows supply the held-fixed features.
        predict: A callable turning a `Dataset` into one with a `prediction` column.
        feature: The feature to sweep.
        grid: The values to set the feature to. Defaults to `grid_points` evenly spaced
            between the feature's own min and max.
        grid_points: How many evenly spaced values to use when `grid` is not given.
        prediction: The name `predict` gives its output column.

    Returns:
        A `Dataset` of ``value`` (the grid point) and ``mean_prediction``, ordered by value.

    Raises:
        PlanError: If `grid_points` is less than 2 when a grid must be derived.
        ColumnNotFoundError: If `feature` is not a column of `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.interpret import partial_dependence
            >>> ds = bt.from_pydict({"x": [1.0, 2.0, 3.0], "other": [5.0, 1.0, 9.0]})
            >>> scored = lambda d: d.with_columns(prediction=bt.col("x") * bt.lit(2.0))
            >>> pd = partial_dependence(ds, scored, "x", grid=[0.0, 1.0])
            >>> pd.to_pydict()["mean_prediction"]
            [0.0, 2.0]
    """
    import batcher as bt

    _require(ds, feature)
    values = grid if grid is not None else _feature_grid(ds, feature, grid_points)
    base = ds.cache()
    curve: dict[str, list[Any]] = {"value": [], "mean_prediction": []}
    for value in values:
        fixed = base.with_columns(**{feature: lit(float(value))})
        scored = predict(fixed)
        mean = scored.agg(__bt_m=bt.mean(col(prediction))).collect().column("__bt_m")[0].as_py()
        curve["value"].append(float(value))
        curve["mean_prediction"].append(None if mean is None else float(mean))
    return bt.from_pydict(curve).sort("value")


def _feature_grid(ds: Dataset, feature: str, grid_points: int) -> list[float]:
    """`grid_points` evenly spaced values between the feature's min and max."""
    import batcher as bt

    if grid_points < 2:
        raise PlanError(f"grid_points must be at least 2, got {grid_points}")
    bounds = ds.agg(lo=bt.min(col(feature)), hi=bt.max(col(feature))).collect()
    low = bounds.column("lo")[0].as_py()
    high = bounds.column("hi")[0].as_py()
    if low is None or high is None:
        raise PlanError(f"cannot build a grid for {feature!r}: it is empty or entirely null")
    low, high = float(low), float(high)
    if low == high:
        return [low]
    step = (high - low) / (grid_points - 1)
    return [low + step * i for i in range(grid_points)]


def _rmse(ds: Dataset, y_true: str, y_pred: str) -> float:
    """Root mean squared error — the default permutation-importance metric."""
    import batcher as bt

    row = ds.agg(__bt_e=bt.rmse(y_true, y_pred)).collect()
    value = row.column("__bt_e")[0].as_py() if row.num_rows else None
    return float("inf") if value is None else float(value)
