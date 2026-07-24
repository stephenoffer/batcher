"""Diagnostic tables — confusion matrix, threshold sweep, lift, calibration.

A single metric tells you how good a model is; these tell you *where* it is wrong, which
is what changes what you do next. Each returns a lazy `Dataset` rather than a printed
table, so the result composes: filter it, join a cost column onto it, write it to the
monitoring table.

Everything here is a `group_by` or a bucketed aggregate over the scored dataset, so a
diagnostic over a billion scored rows costs one pass and never materializes on the driver
— which is the difference from every ``sklearn.metrics`` equivalent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.expr_ir.nodes import ntile
from batcher.plan.functions.aggregate import count_if
from batcher.plan.functions.aggregate import sum as sum_
from batcher.plan.functions.metrics.model.classification import positive_mask

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "calibration_curve",
    "classification_report",
    "confusion_matrix",
    "lift_table",
    "threshold_sweep",
]

_BUCKET = "__bt_bucket"
_LABEL = "__bt_label"


def confusion_matrix(
    ds: Dataset, y_true: str, y_pred: str, *, count_column: str = "count"
) -> Dataset:
    """The full confusion matrix as one row per ``(actual, predicted)`` pair.

    Long form rather than a square array, because that is what stays correct when the label
    set is large, sparse, or discovered from the data — and because a long table joins,
    filters, and writes, which a NumPy array does not. Pairs that never occur are absent
    rather than zero.

    Multi-class by construction: it is a `group_by` on the two label columns, so nothing
    about it assumes two classes.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_pred: The predicted-label column.
        count_column: Name of the appended count column.

    Returns:
        A lazy `Dataset` of ``y_true``, ``y_pred``, ``count``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import confusion_matrix
            >>> ds = bt.from_pydict({"y": [1, 1, 0], "p": [1, 0, 0]})
            >>> confusion_matrix(ds, "y", "p").sort("y", "p").to_pydict()
            {'y': [0, 1, 1], 'p': [0, 0, 1], 'count': [1, 1, 1]}
    """
    _require_columns(ds, y_true, y_pred)
    return ds.group_by(y_true, y_pred).agg(**{count_column: col(y_true).count()})


def classification_report(
    ds: Dataset,
    y_true: str,
    y_pred: str,
    *,
    max_classes: int = 100,
) -> Dataset:
    """Per-class precision, recall, F1, and support — the multi-class scorecard.

    The table to read instead of a single accuracy the moment there are more than two
    classes. An overall accuracy of 0.91 across ten classes routinely hides one class the
    model never predicts at all, and that class is usually the one anybody cared about.

    Every class's four counts are computed in **one aggregate**: each class contributes a
    handful of `count_if` terms to the same pass, so a twenty-class report costs what a
    two-class one does rather than twenty scans.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_pred: The predicted-label column.
        max_classes: The ceiling on the discovered class set; each class adds terms to the
            aggregate, so an accidental report over an identifier column fails fast.

    Returns:
        A lazy `Dataset` of ``class``, ``precision``, ``recall``, ``f1``, ``support``,
        ``predicted``, ordered by descending support.

    Raises:
        PlanError: If the label column has more than `max_classes` distinct values.
        ColumnNotFoundError: If a named column is missing.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import classification_report
            >>> ds = bt.from_pydict({"y": ["a", "a", "b"], "p": ["a", "b", "b"]})
            >>> classification_report(ds, "y", "p").to_pydict()["class"]
            ['a', 'b']
    """
    import batcher as bt

    _require_columns(ds, y_true, y_pred)
    classes = _label_set(ds, y_true, y_pred, max_classes)
    counts: dict[str, Any] = {}
    for index, label in enumerate(classes):
        actual = col(y_true) == lit(label)
        predicted = col(y_pred) == lit(label)
        counts[f"__bt_tp_{index}"] = count_if(actual & predicted)
        counts[f"__bt_fp_{index}"] = count_if(~actual & predicted)
        counts[f"__bt_fn_{index}"] = count_if(actual & ~predicted)
    row = ds.agg(**counts).collect()
    rows: dict[str, list[Any]] = {
        "class": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "support": [],
        "predicted": [],
    }
    for index, label in enumerate(classes):
        true_positive = row.column(f"__bt_tp_{index}")[0].as_py() or 0
        false_positive = row.column(f"__bt_fp_{index}")[0].as_py() or 0
        false_negative = row.column(f"__bt_fn_{index}")[0].as_py() or 0
        precision_value = _ratio(true_positive, true_positive + false_positive)
        recall_value = _ratio(true_positive, true_positive + false_negative)
        rows["class"].append(label)
        rows["precision"].append(precision_value)
        rows["recall"].append(recall_value)
        rows["f1"].append(
            _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
        )
        rows["support"].append(true_positive + false_negative)
        rows["predicted"].append(true_positive + false_positive)
    return bt.from_pydict(rows).sort("support", descending=True)


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 when the denominator vanishes.

    scikit-learn's convention: a class the model never predicts has an undefined precision,
    and reporting 0 is both what it does and the honest reading — nothing it predicted for
    that class was right, because it predicted nothing.
    """
    return float(numerator) / float(denominator) if denominator else 0.0


def _label_set(ds: Dataset, y_true: str, y_pred: str, max_classes: int) -> list[Any]:
    """Every label appearing in either column, sorted, bounded by `max_classes`."""
    labels = (
        ds.select(y_true)
        .rename({y_true: "__bt_label"})
        .union(ds.select(y_pred).rename({y_pred: "__bt_label"}))
        .distinct()
        .limit(max_classes + 1)
        .collect()
        .column("__bt_label")
        .to_pylist()
    )
    present = sorted(v for v in labels if v is not None)
    if len(present) > max_classes:
        raise PlanError(
            f"{y_true!r} has more than {max_classes} distinct labels. Each class adds terms "
            "to the aggregate, so this is almost certainly an identifier column rather than "
            "a label. Raise max_classes to accept the cost."
        )
    return present


def threshold_sweep(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    thresholds: int = 100,
    positive: Any = 1,
) -> Dataset:
    """Precision, recall, and the confusion counts at every one of `thresholds` cutoffs.

    Picking an operating point is the step between "the model scores 0.87 AUC" and
    "deploy it", and it needs the whole curve, not one number. The score range is divided
    into `thresholds` equal buckets and the counts are accumulated from the top down, so
    each row answers "if I alerted on everything scoring at or above this, what would
    happen".

    One pass and one running aggregate regardless of `thresholds`, unlike the usual loop
    that re-scans the predictions once per cutoff.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted-score column, assumed to be in ``[0, 1]``.
        thresholds: How many cutoffs to evaluate.
        positive: The label value that counts as the positive class.

    Returns:
        A lazy `Dataset` of ``threshold``, ``tp``, ``fp``, ``fn``, ``tn``, ``precision``,
        ``recall``, ``f1``, ``predicted_positive_rate``, ordered by descending threshold.

    Raises:
        PlanError: If `thresholds` is less than 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import threshold_sweep
            >>> ds = bt.from_pydict({"y": [0, 1], "s": [0.2, 0.9]})
            >>> threshold_sweep(ds, "y", "s", thresholds=2).columns[:5]
            ['threshold', 'tp', 'fp', 'fn', 'tn']
    """
    _require_columns(ds, y_true, y_score)
    if thresholds < 2:
        raise PlanError(f"threshold_sweep needs at least 2 thresholds, got {thresholds}")
    step = 1.0 / thresholds
    is_positive = positive_mask(col(y_true), positive)
    # Bucket each row by the cutoff it is the *last* to survive, then accumulate downward:
    # a row scoring 0.83 is predicted positive at every threshold at or below 0.83.
    bucketed = ds.with_columns(
        **{
            _BUCKET: (col(y_score) / lit(step)).floor().clip(lit(0), lit(thresholds - 1)),
            _LABEL: when(is_positive).then(lit(1.0)).otherwise(lit(0.0)),
        }
    )
    per_bucket = bucketed.group_by(_BUCKET).agg(
        bucket_positives=sum_(col(_LABEL)),
        bucket_rows=col(_LABEL).count(),
    )
    # Descending cumulative sums: everything at or above this bucket is predicted positive.
    running = per_bucket.with_columns(
        tp=sum_(col("bucket_positives")).over(order_by=[(_BUCKET, True)]),
        predicted_positive=sum_(col("bucket_rows")).over(order_by=[(_BUCKET, True)]),
        __bt_total_positive=sum_(col("bucket_positives")).over(),
        __bt_total=sum_(col("bucket_rows")).over(),
    )
    precision = col("tp") / col("predicted_positive")
    recall = col("tp") / col("__bt_total_positive")
    return running.select(
        threshold=col(_BUCKET).cast("float64") * lit(step),
        tp=col("tp"),
        fp=col("predicted_positive") - col("tp"),
        fn=col("__bt_total_positive") - col("tp"),
        tn=col("__bt_total") - col("predicted_positive") - (col("__bt_total_positive") - col("tp")),
        precision=precision,
        recall=recall,
        f1=lit(2.0) * precision * recall / (precision + recall),
        predicted_positive_rate=col("predicted_positive") / col("__bt_total"),
    ).sort("threshold", descending=True)


def lift_table(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    buckets: int = 10,
    positive: Any = 1,
) -> Dataset:
    """The decile (or `buckets`-ile) lift table — how much better than random the top scores are.

    The table a marketing or risk team actually reads: sort by score, cut into equal-sized
    groups, and report each group's positive rate against the overall base rate. A lift of
    3.0 in the top decile means those customers convert three times as often as average,
    which is the sentence that justifies a campaign.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted-score column.
        buckets: How many equal-sized groups to cut the ranking into (10 = deciles).
        positive: The label value that counts as the positive class.

    Returns:
        A lazy `Dataset` of ``bucket`` (1 = highest scores), ``rows``, ``positives``,
        ``positive_rate``, ``lift``, ``cumulative_positive_rate``, ``cumulative_lift``,
        ``capture_rate``.

    Raises:
        PlanError: If `buckets` is less than 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import lift_table
            >>> ds = bt.from_pydict({"y": [1, 1, 0, 0], "s": [0.9, 0.8, 0.2, 0.1]})
            >>> lift_table(ds, "y", "s", buckets=2).to_pydict()["lift"]
            [2.0, 0.0]
    """
    _require_columns(ds, y_true, y_score)
    if buckets < 2:
        raise PlanError(f"lift_table needs at least 2 buckets, got {buckets}")
    labelled = ds.with_columns(
        **{_LABEL: when(positive_mask(col(y_true), positive)).then(lit(1.0)).otherwise(lit(0.0))}
    )
    ranked = labelled.with_columns(**{_BUCKET: ntile(buckets).over(order_by=[(y_score, True)])})
    per_bucket = ranked.group_by(_BUCKET).agg(rows=col(_LABEL).count(), positives=sum_(col(_LABEL)))
    running = per_bucket.with_columns(
        __bt_cum_pos=sum_(col("positives")).over(order_by=[_BUCKET]),
        __bt_cum_rows=sum_(col("rows")).over(order_by=[_BUCKET]),
        __bt_tot_pos=sum_(col("positives")).over(),
        __bt_tot=sum_(col("rows")).over(),
    )
    base_rate = col("__bt_tot_pos") / col("__bt_tot")
    rate = col("positives") / col("rows")
    cumulative_rate = col("__bt_cum_pos") / col("__bt_cum_rows")
    return running.select(
        bucket=col(_BUCKET),
        rows=col("rows"),
        positives=col("positives"),
        positive_rate=rate,
        lift=rate / base_rate,
        cumulative_positive_rate=cumulative_rate,
        cumulative_lift=cumulative_rate / base_rate,
        capture_rate=col("__bt_cum_pos") / col("__bt_tot_pos"),
    ).sort("bucket")


def calibration_curve(
    ds: Dataset,
    y_true: str,
    y_score: str,
    *,
    bins: int = 10,
    positive: Any = 1,
) -> Dataset:
    """Predicted probability against observed frequency, in `bins` equal-width buckets.

    The diagnostic no single number gives you: a model can rank perfectly (AUC 0.95) while
    every predicted probability is twice the true one, which matters the moment a
    prediction is multiplied by a dollar amount. A well-calibrated model has
    ``mean_predicted`` ≈ ``observed_rate`` in every row of this table.

    Args:
        ds: The scored dataset.
        y_true: The label column.
        y_score: The predicted-probability column, in ``[0, 1]``.
        bins: How many equal-width probability buckets to use.
        positive: The label value that counts as the positive class.

    Returns:
        A lazy `Dataset` of ``bin``, ``bin_lower``, ``bin_upper``, ``rows``,
        ``mean_predicted``, ``observed_rate``, ``calibration_error``.

    Raises:
        PlanError: If `bins` is less than 2.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.metrics import calibration_curve
            >>> ds = bt.from_pydict({"y": [0, 1], "s": [0.05, 0.95]})
            >>> calibration_curve(ds, "y", "s", bins=2).to_pydict()["observed_rate"]
            [0.0, 1.0]
    """
    _require_columns(ds, y_true, y_score)
    if bins < 2:
        raise PlanError(f"calibration_curve needs at least 2 bins, got {bins}")
    width = 1.0 / bins
    labelled = ds.with_columns(
        **{
            _BUCKET: (col(y_score) / lit(width)).floor().clip(lit(0), lit(bins - 1)),
            _LABEL: when(positive_mask(col(y_true), positive)).then(lit(1.0)).otherwise(lit(0.0)),
        }
    )
    grouped = labelled.group_by(_BUCKET).agg(
        rows=col(_LABEL).count(),
        mean_predicted=col(y_score).mean(),
        observed_rate=col(_LABEL).mean(),
    )
    return grouped.select(
        bin=col(_BUCKET).cast("int64"),
        bin_lower=col(_BUCKET).cast("float64") * lit(width),
        bin_upper=(col(_BUCKET).cast("float64") + lit(1.0)) * lit(width),
        rows=col("rows"),
        mean_predicted=col("mean_predicted"),
        observed_rate=col("observed_rate"),
        calibration_error=(col("mean_predicted") - col("observed_rate")).abs(),
    ).sort("bin")


def _require_columns(ds: Dataset, *names: str) -> None:
    """Raise a `ColumnNotFoundError` naming the closest real column for any missing name."""
    available = ds.columns
    for name in names:
        if name not in available:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message("column", name, available, hint="Pass an existing column.")
            )
