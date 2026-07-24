"""Comparing several models on the same data, in one pass rather than N.

Model selection means scoring a handful of candidates and reading the numbers side by side.
Done the obvious way that is one full evaluation per candidate, each re-reading the
predictions — and because the metrics here are aggregate expressions, it does not have to
be: every model's metrics go into the *same* `agg`, so comparing six candidates costs what
comparing one does.

The result is a `Dataset` with one row per model rather than a printed table, so it sorts,
joins to a cost or latency column, and appends to an experiment log — which is what turns a
comparison into a record of why a model was chosen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.metrics.evaluate import (
    METRIC_SETS,
    _aggregate_exprs,
    _negative_of,
    _rank_metric_names,
    _resolve_task,
    _validate_metrics,
)
from batcher.plan.expr_ir.constructors import col, lit, when

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = ["compare_models"]


def compare_models(
    ds: Dataset,
    y_true: str,
    predictions: dict[str, str],
    *,
    task: str = "auto",
    metrics: list[str] | None = None,
    positive: Any = 1,
    threshold: float = 0.5,
    scores: bool = True,
) -> Dataset:
    """Score several models' prediction columns side by side, in one pass over the data.

    `predictions` maps a model name to the column holding its output. With ``scores=True``
    (the default) those columns hold probabilities and the hard predictions are derived at
    `threshold`; with ``scores=False`` they hold hard labels or regression values already.

    Every model's aggregate metrics are evaluated in the *same* `agg`, so the comparison
    costs one scan regardless of how many candidates there are. Rank-based metrics need a
    sort each and are therefore excluded — ask for them per model with `roc_auc(..., by=)`
    when you want them.

    Args:
        ds: The dataset holding the labels and every model's predictions.
        y_true: The label column.
        predictions: A ``{model_name: column}`` mapping.
        task: ``"binary"``, ``"multiclass"``, ``"regression"``, or ``"auto"``.
        metrics: The metric names to compute; the task's default aggregate set when omitted.
        positive: The label value that counts as the positive class.
        threshold: The cutoff turning a score into a hard prediction.
        scores: Whether the columns hold probabilities (the default) or final predictions.

    Returns:
        A `Dataset` with a ``model`` column and one column per metric, one row per model.

    Raises:
        PlanError: If `predictions` is empty, names an unknown metric, or asks for a
            rank-based metric.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import compare_models
            >>> ds = bt.from_pydict(
            ...     {"y": [1, 0, 1, 0], "a": [0.9, 0.1, 0.8, 0.2], "b": [0.4, 0.6, 0.4, 0.6]}
            ... )
            >>> got = compare_models(ds, "y", {"good": "a", "bad": "b"}, metrics=["accuracy"])
            >>> got.sort("accuracy", descending=True).to_pydict()["model"]
            ['good', 'bad']
    """
    import batcher as bt

    if not predictions:
        raise PlanError("compare_models needs at least one model in predictions=")
    missing = [c for c in predictions.values() if c not in ds.columns]
    if missing:
        from batcher._internal.errors import ColumnNotFoundError, unknown_message

        raise ColumnNotFoundError(
            unknown_message("column", missing[0], ds.columns, hint="Pass a prediction column.")
        )
    first = next(iter(predictions.values()))
    resolved = _resolve_task(task, None if scores else first, first if scores else None)
    requested = list(metrics) if metrics is not None else list(METRIC_SETS[resolved])
    _validate_metrics(requested)
    ranked = [name for name in requested if name in _rank_metric_names()]
    if ranked:
        raise PlanError(
            f"{ranked[0]!r} needs a sort per model, so it cannot share the comparison's single "
            "pass. Drop it from metrics= and call the rank metric per model instead."
        )

    frame = ds
    columns: dict[str, str] = {}
    for name, column in predictions.items():
        if scores and resolved != "regression":
            derived = f"__bt_hard_{name}"
            frame = frame.with_columns(
                **{
                    derived: when(col(column) >= lit(threshold))
                    .then(lit(positive))
                    .otherwise(lit(_negative_of(positive)))
                }
            )
            columns[name] = derived
        else:
            columns[name] = column

    aggregates: dict[str, Any] = {}
    for name, column in columns.items():
        score_column = predictions[name] if scores else None
        for metric, builder in _aggregate_exprs(
            requested, y_true, column, score_column, positive
        ).items():
            aggregates[f"{name}__{metric}"] = builder
    if not aggregates:
        raise PlanError("compare_models computed no metrics; check the metrics= list")
    row = frame.agg(**aggregates).collect()

    table: dict[str, list[Any]] = {"model": list(columns)}
    for metric in requested:
        values = []
        for name in columns:
            key = f"{name}__{metric}"
            values.append(row.column(key)[0].as_py() if key in row.column_names else None)
        if any(v is not None for v in values):
            table[metric] = values
    return bt.from_pydict(table)
