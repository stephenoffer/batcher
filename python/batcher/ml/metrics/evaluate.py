"""`evaluate` — every metric for a task in as few passes as the metrics allow.

Model evaluation is normally a dozen separate calls, each re-reading the predictions. Here
the aggregate metrics for a task are one `agg`, so asking for ten of them costs exactly
what asking for one does; only the rank-based metrics (which need a sort) add a pass, and
only when you ask for them.

The metric sets are named per task rather than assembled by the caller, because the choice
of metric *is* the hard part and getting a sensible default set is most of the value. A
caller who wants something else passes `metrics=[...]`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher._internal.logging import note_suppressed
from batcher.ml.metrics import ranked
from batcher.plan.functions import metrics as agg_metrics

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["METRIC_SETS", "evaluate", "multiclass_averages"]

# The aggregate metrics of each task, in report order. Every entry is a callable taking
# (label_column, value_column) and returning an `Expr`, so the whole set lowers into one
# aggregate no matter how many are selected.
_REGRESSION: dict[str, Callable[..., Any]] = {
    "rmse": agg_metrics.rmse,
    "mae": agg_metrics.mae,
    "mse": agg_metrics.mse,
    "r2": agg_metrics.r2,
    "mape": agg_metrics.mape,
    "smape": agg_metrics.smape,
    "wape": agg_metrics.wape,
    "medae": agg_metrics.medae,
    "max_error": agg_metrics.max_error,
    "mean_bias": agg_metrics.mean_bias,
    "explained_variance": agg_metrics.explained_variance,
}

_LABEL_METRICS: dict[str, Callable[..., Any]] = {
    "accuracy": agg_metrics.accuracy,
    "precision": agg_metrics.precision,
    "recall": agg_metrics.recall,
    "f1": agg_metrics.f1_score,
    "balanced_accuracy": agg_metrics.balanced_accuracy,
    "specificity": agg_metrics.specificity,
    "mcc": agg_metrics.matthews_corrcoef,
    "cohen_kappa": agg_metrics.cohen_kappa,
    "true_positives": agg_metrics.true_positives,
    "false_positives": agg_metrics.false_positives,
    "false_negatives": agg_metrics.false_negatives,
    "true_negatives": agg_metrics.true_negatives,
}

_SCORE_METRICS: dict[str, Callable[..., Any]] = {
    "log_loss": agg_metrics.log_loss,
    "brier_score": agg_metrics.brier_score,
}

# The rank-based metrics, which each cost a sort and so are listed apart.
_RANK_METRICS: dict[str, Callable[..., Any]] = {
    "roc_auc": ranked.roc_auc,
    "average_precision": ranked.average_precision,
    "ks": ranked.ks_statistic,
    "gini": ranked.gini_coefficient,
}


def _needs_score(name: str) -> bool:
    """Whether `name` can only be computed from a probability, not a hard prediction."""
    return name in _SCORE_METRICS or name in _RANK_METRICS


#: The metrics reported for each task when none are named, in report order.
METRIC_SETS: dict[str, tuple[str, ...]] = {
    "regression": tuple(_REGRESSION),
    "binary": (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
    ),
    "multiclass": (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ),
}

# The multi-class averages, which are not aggregates: each one is a mean over the per-class
# report, so they are computed from `classification_report` rather than inside `agg`.
_MULTICLASS_AVERAGES = (
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)


def _resolve_task(
    task: str,
    y_pred: str | None,
    y_score: str | None,
    ds: Dataset | None = None,
    y_true: str | None = None,
    max_classes: int = 100,
) -> str:
    """Pick the task from the labels, or validate the one the caller named.

    `"auto"` used to answer from the *argument list* alone: a `y_score` meant binary and a
    `y_pred` meant regression, whatever the labels held. So the most ordinary call there is
    — ``evaluate("label", y_pred="prediction")`` over a 0/1 classification — reported RMSE,
    MAE and R². Those are real numbers, computed correctly, answering a question nobody
    asked, and nothing in the result says so.

    The labels decide it now: a float label column is a regression, and an integer, string
    or boolean one is a classification with as many classes as it has distinct values. That
    is one cheap pass over a single column, and only on the `auto` path.
    """
    if task != "auto":
        if task not in METRIC_SETS:
            from batcher._internal.errors import suggestion

            hint = suggestion(task, sorted(METRIC_SETS))
            tail = f" {hint}" if hint else ""
            raise PlanError(f"task must be one of {sorted(METRIC_SETS)}, got {task!r}.{tail}")
        return task
    if y_pred is None and y_score is None:
        raise PlanError(
            "evaluate() needs y_pred= (a prediction column) or y_score= (a probability)"
        )
    if y_pred is None:
        return "binary"  # a score with no hard prediction is the binary shape by construction
    inferred = _infer_task_from_labels(ds, y_true, max_classes)
    return inferred if inferred is not None else "regression"


def _infer_task_from_labels(ds: Dataset | None, y_true: str | None, max_classes: int) -> str | None:
    """The task `y_true`'s own values imply, or `None` when they cannot be read.

    Returning `None` rather than guessing keeps the previous behaviour for a caller whose
    dataset cannot answer (an un-inferable schema, an unreadable source): the task falls
    back to regression exactly as before, so this can only ever add information.
    """
    if ds is None or y_true is None:
        return None
    try:
        import pyarrow as pa

        from batcher.plan.expr_ir.constructors import col

        dtype = ds.schema.field(y_true).type
        if pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
            return "regression"  # a continuous label is a regression, however few values it has
        if not (
            pa.types.is_integer(dtype) or pa.types.is_boolean(dtype) or pa.types.is_string(dtype)
        ):
            return None
        classes = ds.select(**{"__bt_label": col(y_true)}).distinct().count()
    except Exception as exc:  # pragma: no cover - inference must never break the report
        note_suppressed("ml", "infer the evaluation task from the labels", exc)
        return None
    if classes <= 1:
        return None  # degenerate; leave the caller's own default rather than inventing one
    if classes == 2:
        return "binary"
    return "multiclass" if classes <= max_classes else "regression"


def evaluate(
    ds: Dataset,
    y_true: str,
    *,
    y_pred: str | None = None,
    y_score: str | None = None,
    task: str = "auto",
    metrics: list[str] | None = None,
    positive: Any = 1,
    threshold: float = 0.5,
    by: str | list[str] | None = None,
    max_classes: int = 100,
) -> dict[str, float] | Dataset:
    """Score a set of predictions, returning every metric for the task in one call.

    The aggregate metrics are evaluated together as a single `agg`, so a ten-metric report
    is one pass over the predictions. The rank-based metrics (`roc_auc`,
    `average_precision`, `ks`, `gini`) each add a sort and are computed only when they are
    in the requested set.

    For a binary task, giving `y_score` alone is enough: the hard predictions are derived
    at `threshold`, so precision, recall, and AUC all come from one scored column.

    **Nulls are excluded, not counted as wrong.** Every metric is an Arrow aggregate, and
    those skip nulls, so a row whose label or prediction is missing contributes to nothing:
    the denominator is the rows where both are present. That is the usual convention, but it
    is worth knowing which way it cuts — a model that predicts null on the half of the data
    it finds hard scores on the easy half alone, and reports a clean number for it. Nothing
    here says how many rows survived, so check that yourself when nulls are possible:
    ``ds.filter(bt.col(y_pred).is_null()).count()``.

    A metric that is undefined for the data returns ``nan`` rather than raising — `roc_auc`
    and `balanced_accuracy` when only one class is present, `precision`/`recall`/`f1` when
    there are no positives. A `nan` in the report means "not defined here", not "zero".

    Args:
        ds: The dataset holding labels and predictions.
        y_true: The label column.
        y_pred: The hard-prediction column (a label, or a value for regression).
        y_score: The predicted probability of the positive class, for a binary task.
        task: ``"binary"``, ``"multiclass"``, ``"regression"``, or ``"auto"``. ``"auto"``
            reads it off `y_true`: a float label is a regression, and an integer, string or
            boolean one is a classification with as many classes as it has distinct values
            (above `max_classes`, a regression again).
        metrics: The metric names to compute. Omitted, the task's default set is used,
            minus any metric that needs a probability when no `y_score` was given -- so a
            binary report over hard predictions is the six label-only metrics rather than
            an error. A metric named here explicitly still raises if it cannot be computed.
        positive: The label value that counts as the positive class.
        threshold: The cutoff turning `y_score` into a hard prediction.
        by: Column(s) to report a separate row of metrics for.
        max_classes: The ceiling on the discovered class set, for a multi-class task.

    Returns:
        A ``{metric: value}`` dict, or a `Dataset` of one row per group when `by` is given.

    Raises:
        PlanError: On an unknown task or metric name, or when neither `y_pred` nor
            `y_score` is given.
        ColumnNotFoundError: If a named column is not in `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import evaluate
            >>> ds = bt.from_pydict({"y": [1.0, 2.0, 3.0], "p": [1.0, 2.0, 4.0]})
            >>> round(evaluate(ds, "y", y_pred="p", task="regression")["mae"], 6)
            0.333333
    """
    from batcher.plan.expr_ir.constructors import col, lit, when

    resolved = _resolve_task(task, y_pred, y_score, ds, y_true, max_classes)
    if metrics is not None:
        requested = list(metrics)
    else:
        # A metric the caller *named* and cannot have still raises, naming it — that error is
        # actionable. But the task's **default** set is not a request, and four of the ten
        # binary defaults need a probability: with hard predictions alone, the canonical
        # `evaluate("y", y_pred="p", task="binary")` raised instead of reporting the six
        # metrics it had everything for.
        requested = [
            name for name in METRIC_SETS[resolved] if y_score is not None or not _needs_score(name)
        ]
    _validate_metrics(requested)
    groups = ranked._group_keys(by)

    frame = ds
    prediction = y_pred
    if resolved in ("binary", "multiclass") and prediction is None:
        if y_score is None:
            raise PlanError("a classification task needs y_pred= or y_score=")
        # Deriving the hard prediction here (rather than asking the caller to) is what lets
        # one scored column serve both the threshold metrics and the rank metrics.
        prediction = "__bt_hard_pred"
        frame = ds.with_columns(
            **{
                prediction: when(col(y_score) >= lit(threshold))
                .then(lit(positive))
                .otherwise(lit(_negative_of(positive)))
            }
        )

    aggregates = _aggregate_exprs(requested, y_true, prediction, y_score, positive)
    results: dict[str, Any] = {}
    if aggregates:
        reduced = frame.group_by(*groups).agg(**aggregates) if groups else frame.agg(**aggregates)
        results["__aggregates"] = reduced

    rank_requested = [name for name in requested if name in _RANK_METRICS]
    if rank_requested and y_score is None:
        raise PlanError(
            f"{rank_requested[0]!r} needs y_score= (a continuous score), not just a hard "
            "prediction. Pass the model's probability column."
        )
    rank_frames = [
        _RANK_METRICS[name](ds, y_true, y_score, positive=positive, by=by, metric=name)
        for name in rank_requested
    ]

    averages_requested = [name for name in requested if name in _MULTICLASS_AVERAGES]
    if averages_requested and groups:
        raise PlanError(
            "the multi-class averages are computed from a per-class report, which by= cannot "
            "partition. Ask for them without by=, or group the dataset and call evaluate() "
            "per group."
        )
    averages = (
        multiclass_averages(frame, y_true, prediction, max_classes=max_classes)
        if averages_requested and prediction is not None
        else {}
    )

    if groups:
        return _join_group_results(results.get("__aggregates"), rank_frames, groups, requested)
    scalars = _scalar_results(results.get("__aggregates"), rank_requested, rank_frames, requested)
    scalars.update({k: v for k, v in averages.items() if k in requested})
    return {name: scalars[name] for name in requested if name in scalars}


def _rank_metric_names() -> frozenset[str]:
    """The metrics that need a global sort, so a caller can refuse to batch them."""
    return frozenset(_RANK_METRICS)


def _negative_of(positive: Any) -> Any:
    """A value distinct from `positive` to use as the derived negative label."""
    if isinstance(positive, bool):
        return not positive
    if isinstance(positive, (int, float)):
        return 0 if positive != 0 else 1
    return f"not_{positive}"


def multiclass_averages(
    ds: Dataset, y_true: str, y_pred: str, *, max_classes: int
) -> dict[str, float]:
    """Macro and support-weighted averages of the per-class precision, recall, and F1.

    Two ways to average a per-class metric, and they answer different questions. The
    **macro** average weights every class equally, so a rare class the model ignores drags
    it down — which is usually what you want to know. The **weighted** average weights by
    support, so it tracks overall accuracy and a rare class barely moves it. Reporting only
    one of them is how a model that never predicts the minority class passes review.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_pred: The predicted-label column.
        max_classes: The ceiling on the discovered class set.

    Returns:
        A ``{name: value}`` dict of the six averages.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics.evaluate import multiclass_averages
            >>> ds = bt.from_pydict({"y": ["a", "a", "b"], "p": ["a", "a", "b"]})
            >>> multiclass_averages(ds, "y", "p", max_classes=10)["macro_f1"]
            1.0
    """
    from batcher.ml.metrics.tables import classification_report

    report = classification_report(ds, y_true, y_pred, max_classes=max_classes).to_pydict()
    supports = [float(s) for s in report["support"]]
    total = sum(supports)
    averages: dict[str, float] = {}
    for metric in ("precision", "recall", "f1"):
        values = [float(v) for v in report[metric]]
        averages[f"macro_{metric}"] = sum(values) / len(values) if values else float("nan")
        averages[f"weighted_{metric}"] = (
            sum(v * w for v, w in zip(values, supports, strict=True)) / total
            if total
            else float("nan")
        )
    return averages


def _validate_metrics(names: list[str]) -> None:
    """Raise on any unknown metric name, naming the closest match."""
    known = {**_REGRESSION, **_LABEL_METRICS, **_SCORE_METRICS, **_RANK_METRICS}
    known.update(dict.fromkeys(_MULTICLASS_AVERAGES))
    for name in names:
        if name not in known:
            from batcher._internal.errors import suggestion

            hint = suggestion(name, sorted(known))
            tail = f" {hint}" if hint else ""
            raise PlanError(f"unknown metric {name!r}.{tail}")


def _aggregate_exprs(
    names: list[str], y_true: str, y_pred: str | None, y_score: str | None, positive: Any
) -> dict[str, Any]:
    """The `agg` keyword mapping for every requested metric that is a single-pass aggregate."""
    out: dict[str, Any] = {}
    for name in names:
        if name in _REGRESSION and y_pred is not None:
            out[name] = _REGRESSION[name](y_true, y_pred)
        elif name in _LABEL_METRICS and y_pred is not None:
            builder = _LABEL_METRICS[name]
            out[name] = (
                builder(y_true, y_pred)
                if name == "accuracy"
                else builder(y_true, y_pred, positive=positive)
            )
        elif name in _SCORE_METRICS:
            if y_score is None:
                raise PlanError(f"{name!r} needs y_score= (a predicted probability)")
            out[name] = _SCORE_METRICS[name](y_true, y_score, positive=positive)
    return out


def _scalar_results(
    aggregated: Dataset | None,
    rank_names: list[str],
    rank_values: list[Any],
    order: list[str],
) -> dict[str, float]:
    """Merge the aggregate row and the rank scalars into one ordered dict."""
    values: dict[str, float] = {}
    if aggregated is not None:
        row = aggregated.collect()
        for name in row.column_names:
            values[name] = row.column(name)[0].as_py()
    values.update(dict(zip(rank_names, rank_values, strict=True)))
    return {name: values[name] for name in order if name in values}


def _join_group_results(
    aggregated: Dataset | None, rank_frames: list[Dataset], groups: list[str], order: list[str]
) -> Dataset:
    """Join the per-group aggregate frame with each per-group rank frame on the group keys."""
    frames = ([aggregated] if aggregated is not None else []) + rank_frames
    if not frames:
        raise PlanError("evaluate() computed no metrics; check the metrics= list")
    joined = frames[0]
    for other in frames[1:]:
        joined = joined.join(other, on=groups, how="inner")
    return joined.select(*groups, *[name for name in order if name in joined.columns])
