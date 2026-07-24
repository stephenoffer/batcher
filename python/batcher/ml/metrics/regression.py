"""Regression-specific diagnostics — the residual, the interval, and the top-k label.

The scalar regression metrics (`rmse`, `mae`, `r2`, …) are aggregate expressions and live in
`plan.functions.metrics`. What is here needs more than one aggregate or a per-row table:

`residual_summary`
    Where the error lives, as a `Dataset` of one row per segment. A single RMSE hides that
    the model is unbiased on average while badly over-predicting one cohort; grouping the
    residual statistics surfaces exactly that.
`prediction_interval_coverage`
    Whether a quantile model's intervals mean what they claim. A "90% interval" is only
    useful if roughly 90% of actuals fall inside it, and that is a property of the *data*,
    not of the model's confidence — the number nobody checks and every calibrated forecast
    depends on.
`top_k_accuracy`
    The multi-class metric for a ranked prediction: was the true label among the model's top
    `k` guesses. The honest number for a recommender or a retrieval-style classifier, where
    the top-1 label being wrong does not mean the model was useless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.functions.aggregate import count_if

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "d2_absolute_error_score",
    "d2_pinball_score",
    "d2_tweedie_score",
    "prediction_interval_coverage",
    "residual_summary",
    "top_k_accuracy",
]


def _require(ds: Dataset, *names: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column for any missing name."""
    for name in names:
        if name not in ds.columns:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, ds.columns, hint="Pass an existing column.")
            )


def residual_summary(
    ds: Dataset,
    y_true: str,
    y_pred: str,
    *,
    by: str | list[str] | None = None,
) -> Dataset | dict[str, float]:
    """Summarize the prediction residuals — mean, std, and the error quantiles.

    The residual (``y_pred - y_true``) is where a regression model's problems live, and its
    *shape* says which problem. A non-zero mean is systematic bias; a large spread is
    imprecision; a skew means the model is worse in one direction than the other. Grouped by
    `by`, it shows which segment carries the error a global RMSE averages away.

    Args:
        ds: The scored dataset.
        y_true: The observed-value column.
        y_pred: The predicted-value column.
        by: Column(s) to summarize the residuals separately for.

    Returns:
        A ``{statistic: value}`` dict when ungrouped, or a `Dataset` of one row per group
        with ``mean_residual``, ``std_residual``, ``mae``, ``p05``, ``p50``, ``p95``.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import residual_summary
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 4.0]})
            >>> round(residual_summary(ds, "y", "p")["mean_residual"], 4)
            0.3333
    """
    import batcher as bt

    _require(ds, y_true, y_pred)
    residual = col(y_pred) - col(y_true)
    with_residual = ds.with_columns(__bt_res=residual)
    statistics = {
        "mean_residual": bt.mean(col("__bt_res")),
        "std_residual": bt.std(col("__bt_res")),
        "mae": bt.mean(col("__bt_res").abs()),
        "p05": col("__bt_res").quantile(0.05),
        "p50": col("__bt_res").quantile(0.5),
        "p95": col("__bt_res").quantile(0.95),
    }
    groups = [] if by is None else ([by] if isinstance(by, str) else list(by))
    if groups:
        return with_residual.group_by(*groups).agg(**statistics)
    row = with_residual.agg(**statistics).collect()
    return {name: row.column(name)[0].as_py() for name in row.column_names}


def prediction_interval_coverage(
    ds: Dataset,
    y_true: str,
    lower: str,
    upper: str,
    *,
    by: str | list[str] | None = None,
) -> float | Dataset:
    """The fraction of actuals that fall inside their predicted ``[lower, upper]`` interval.

    A quantile model's headline promise — a "90% interval" — is a claim about *coverage*,
    and coverage is the one thing the model's own output cannot verify. This checks it: feed
    the predicted lower and upper bounds and it returns the empirical fraction that landed
    inside. A well-calibrated 90% interval covers ≈0.9; a lower number means the intervals
    are too narrow and the model is overconfident, which is the usual direction.

    Args:
        ds: The scored dataset.
        y_true: The observed-value column.
        lower: The predicted lower-bound column.
        upper: The predicted upper-bound column.
        by: Column(s) to measure coverage separately for.

    Returns:
        The coverage fraction, or a `Dataset` of one row per group when `by` is given.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import prediction_interval_coverage
            >>> ds = bt.from_pydict(
            ...     {"y": [1.0, 5.0, 9.0], "lo": [0.0, 4.0, 20.0], "hi": [2.0, 6.0, 30.0]}
            ... )
            >>> round(prediction_interval_coverage(ds, "y", "lo", "hi"), 4)
            0.6667
    """
    _require(ds, y_true, lower, upper)
    inside = (col(y_true) >= col(lower)) & (col(y_true) <= col(upper))
    covered = count_if(inside)
    total = count_if(col(y_true).is_not_null())
    groups = [] if by is None else ([by] if isinstance(by, str) else list(by))
    if groups:
        return ds.group_by(*groups).agg(coverage=covered / total)
    row = ds.agg(__bt_cov=covered / total).collect()
    value = row.column("__bt_cov")[0].as_py() if row.num_rows else None
    return float("nan") if value is None else float(value)


def top_k_accuracy(
    ds: Dataset,
    y_true: str,
    score_columns: list[str],
    *,
    k: int = 1,
    labels: list[Any] | None = None,
) -> float:
    """The fraction of rows whose true label is among the `k` highest-scored classes.

    Multi-class accuracy assumes the model gets exactly one guess; a recommender, a
    retrieval classifier, or any UI that shows several suggestions gets `k`. Top-k accuracy
    scores that: it is right if the true label is anywhere in the top `k` predicted classes,
    which is the number that actually predicts user satisfaction on those surfaces.

    Each `score_columns[i]` holds the probability of class `labels[i]` (or of class ``i`` when
    `labels` is omitted). The comparison is done with rank arithmetic over the score columns,
    so no per-row Python is involved.

    Args:
        ds: The scored dataset, one probability column per class.
        y_true: The true-label column.
        score_columns: The per-class score columns, in class order.
        k: How many top classes count as a hit.
        labels: The class label each score column corresponds to; ``0..n-1`` when omitted.

    Returns:
        The top-`k` accuracy in ``[0, 1]``.

    Raises:
        PlanError: If `k` exceeds the number of classes, or is less than 1.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import top_k_accuracy
            >>> ds = bt.from_pydict(
            ...     {"y": [0, 1, 2], "c0": [0.6, 0.1, 0.1], "c1": [0.3, 0.5, 0.2],
            ...      "c2": [0.1, 0.4, 0.7]}
            ... )
            >>> top_k_accuracy(ds, "y", ["c0", "c1", "c2"], k=2)
            1.0
    """
    _require(ds, y_true, *score_columns)
    if k < 1:
        raise PlanError(f"k must be at least 1, got {k}")
    if k > len(score_columns):
        raise PlanError(
            f"k={k} exceeds the {len(score_columns)} score column(s); a hit cannot be in a "
            "top-k wider than the class set."
        )
    class_labels = labels if labels is not None else list(range(len(score_columns)))
    if len(class_labels) != len(score_columns):
        raise PlanError(
            f"labels has {len(class_labels)} entries but there are {len(score_columns)} "
            "score columns; give one label per column."
        )
    # The true class's score, picked out by matching the label: exactly one arm is non-zero,
    # so their sum is that score. Then count how many class scores strictly beat it — a hit
    # is when fewer than k classes outrank the truth.
    true_score = lit(0.0)
    for label, column in zip(class_labels, score_columns, strict=True):
        true_score = true_score + when(col(y_true) == lit(label)).then(col(column)).otherwise(
            lit(0.0)
        )
    outranking = lit(0)
    for column in score_columns:
        outranking = outranking + (col(column) > true_score).cast("int64")
    hit = outranking < lit(k)
    row = (
        ds.with_columns(__bt_hit=hit)
        .agg(__bt_acc=count_if(col("__bt_hit")) / count_if(col(y_true).is_not_null()))
        .collect()
    )
    value = row.column("__bt_acc")[0].as_py() if row.num_rows else None
    return float("nan") if value is None else float(value)


def d2_tweedie_score(ds: Dataset, y_true: str, y_pred: str, *, power: float = 1.5) -> float:
    """The fraction of Tweedie deviance a model explains — R² for a count or rate model.

    ``1 - deviance(y, yhat) / deviance(y, mean(y))``. 1 is a perfect fit, 0 matches predicting
    the target's mean, negative is worse. It gives a Poisson, gamma, or Tweedie model the same
    self-contained 0-to-1 score that R² gives a least-squares one, on the model's own loss
    rather than on squared error — which for a skewed count target is the only honest scale.

    It is a Dataset function rather than an expression because the reference deviance needs the
    target's own mean as a constant baseline; both deviances are still computed in a single
    pass.

    Args:
        ds: The scored dataset.
        y_true: The observed values.
        y_pred: The predicted values.
        power: The Tweedie power — 1 (Poisson), 2 (gamma), ``(1, 2)`` (compound). See
            `tweedie_deviance`.

    Returns:
        The D² score; at most 1.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import d2_tweedie_score
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 3.0]})
            >>> d2_tweedie_score(ds, "y", "p", power=1.0)
            1.0
    """
    import batcher as bt

    _require(ds, y_true, y_pred)
    baselined = ds.with_columns(__bt_base=bt.mean(col(y_true)).over())
    row = baselined.agg(
        model=bt.tweedie_deviance(y_true, y_pred, power=power),
        null=bt.tweedie_deviance(y_true, "__bt_base", power=power),
    ).collect()
    model = row.column("model")[0].as_py()
    null = row.column("null")[0].as_py()
    if model is None or null is None or null == 0.0:
        return float("nan")
    return 1.0 - float(model) / float(null)


def d2_absolute_error_score(ds: Dataset, y_true: str, y_pred: str) -> float:
    """The fraction of absolute error a model explains — R² on the L1 scale.

    ``1 - sum|y - yhat| / sum|y - median(y)|``. It is to the mean absolute error what R² is to
    the squared error: 1 is a perfect fit, 0 matches always predicting the target's median, and
    negative is worse than that. Because the baseline is the median rather than the mean, it is
    the right self-contained score for a model trained on absolute error or a heavy-tailed
    target where the mean is not the thing being predicted.

    Args:
        ds: The scored dataset.
        y_true: The observed values.
        y_pred: The predicted values.

    Returns:
        The D² score on the absolute-error scale; at most 1.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import d2_absolute_error_score
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0], "p": [1.0, 2.0, 3.0, 4.0]})
            >>> d2_absolute_error_score(ds, "y", "p")
            1.0
    """
    import batcher as bt

    _require(ds, y_true, y_pred)
    median = float(ds.agg(m=col(y_true).median()).collect().column("m")[0].as_py())
    row = ds.agg(
        model=bt.sum((col(y_true) - col(y_pred)).abs()),
        null=bt.sum((col(y_true) - bt.lit(median)).abs()),
    ).collect()
    model = row.column("model")[0].as_py()
    null = row.column("null")[0].as_py()
    if model is None or null is None or null == 0.0:
        return float("nan")
    return 1.0 - float(model) / float(null)


def d2_pinball_score(ds: Dataset, y_true: str, y_pred: str, *, alpha: float = 0.5) -> float:
    """The fraction of pinball loss a quantile model explains — R² for a quantile forecast.

    ``1 - pinball(y, yhat) / pinball(y, quantile_alpha(y))`` at the same quantile `alpha`. It
    scores a quantile regression the way R² scores a mean regression: the baseline is the
    optimal constant for the pinball loss, which is the target's own `alpha`-quantile. At
    ``alpha=0.5`` it reduces to `d2_absolute_error_score`.

    Args:
        ds: The scored dataset.
        y_true: The observed values.
        y_pred: The predicted quantile.
        alpha: The quantile the model targets, in ``(0, 1)``.

    Returns:
        The D² score on the pinball scale; at most 1.

    Raises:
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import d2_pinball_score
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0, 4.0], "p": [1.0, 2.0, 3.0, 4.0]})
            >>> d2_pinball_score(ds, "y", "p", alpha=0.5)
            1.0
    """
    import batcher as bt

    _require(ds, y_true, y_pred)
    baseline = float(ds.agg(q=col(y_true).quantile(alpha)).collect().column("q")[0].as_py())
    baselined = ds.with_columns(__bt_base=bt.lit(baseline))
    row = baselined.agg(
        model=bt.pinball_loss(y_true, y_pred, quantile=alpha),
        null=bt.pinball_loss(y_true, "__bt_base", quantile=alpha),
    ).collect()
    model = row.column("model")[0].as_py()
    null = row.column("null")[0].as_py()
    if model is None or null is None or null == 0.0:
        return float("nan")
    return 1.0 - float(model) / float(null)
