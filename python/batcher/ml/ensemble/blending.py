"""`blend_predictions` — combine several models' prediction columns into one.

The cheapest ensemble there is, and often most of the benefit: average the predictions of
models that make different mistakes. It needs no meta-model, no extra fit, and no held-out
split, because there is nothing to learn — which is exactly why it is the thing to try
before `StackingEnsemble`.

It is a single `Expr` over columns already in the frame, so it costs one projection and
distributes like any other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.plan.expr_ir import Expr

__all__ = ["blend_predictions", "majority_vote"]


def blend_predictions(
    ds: Dataset,
    columns: Sequence[str],
    *,
    weights: Sequence[float] | None = None,
    output_column: str = "prediction",
) -> Dataset:
    """Append the weighted mean of several prediction columns.

    Weights are normalized to sum to one, so ``[2, 1]`` and ``[0.667, 0.333]`` mean the same
    thing and the blend stays on the same scale as its inputs whatever numbers you pass.

    Averaging *probabilities* is the useful case. Averaging hard class labels is not
    meaningful — the mean of labels 0 and 2 is 1, which may be a class nobody predicted — so
    blend the probability columns and threshold afterwards.

    Args:
        ds: The dataset carrying the prediction columns.
        columns: The prediction columns to combine.
        weights: One weight per column; equal weights when omitted.
        output_column: Where to write the blended prediction.

    Returns:
        A new lazy `Dataset` with the blended column appended.

    Raises:
        PlanError: If no columns are given, a weight is missing, or the weights sum to zero.
        ColumnNotFoundError: If a named column is not in `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.ensemble import blend_predictions
            >>> ds = bt.from_pydict({"a": [0.0, 1.0], "b": [1.0, 1.0]})
            >>> blend_predictions(ds, ["a", "b"]).to_pydict()["prediction"]
            [0.5, 1.0]
            >>> blend_predictions(ds, ["a", "b"], weights=[3, 1]).to_pydict()["prediction"]
            [0.25, 1.0]
    """
    names = list(columns)
    if not names:
        raise PlanError("blend_predictions needs at least one prediction column")
    require_columns(ds, *names, hint="Pass the models' prediction columns.")
    if weights is None:
        share = [1.0 / len(names)] * len(names)
    else:
        values = [float(w) for w in weights]
        if len(values) != len(names):
            raise PlanError(
                f"blend_predictions: {len(values)} weight(s) for {len(names)} column(s). "
                "Pass one weight per column, or none for an equal blend."
            )
        if any(w < 0 for w in values):
            raise PlanError(f"blend_predictions: weights must be non-negative, got {values}")
        total = sum(values)
        if total <= 0:
            raise PlanError("blend_predictions: the weights sum to zero, so nothing is blended")
        share = [w / total for w in values]
    blended = lit(0.0)
    for name, weight in zip(names, share, strict=True):
        blended = blended + lit(weight) * col(name).cast("float64")
    return ds.with_columns(**{output_column: blended})


def majority_vote(
    ds: Dataset,
    columns: Sequence[str],
    *,
    labels: Sequence[object] | None = None,
    weights: Sequence[float] | None = None,
    output_column: str = "prediction",
) -> Dataset:
    """Append the label most of the given classifiers predicted for each row.

    Hard voting, the classification counterpart of `blend_predictions`. Averaging class
    *labels* is meaningless — the mean of labels 0 and 2 is 1, which may be a class nobody
    predicted — so a classifier ensemble counts votes instead. Where the models expose
    probabilities, prefer `blend_predictions` on those and threshold afterwards; soft voting
    uses more of what each model knows.

    Ties go to whichever label appears earliest in `labels`, which makes the result
    reproducible rather than dependent on evaluation order.

    Args:
        ds: The dataset carrying the prediction columns.
        columns: The label columns to combine, one per model.
        labels: The candidate labels. Learned with one `distinct` scan when omitted, which
            makes this call eager; pass them to keep it lazy.
        weights: One weight per column, letting a better model count for more. Equal
            weights when omitted.
        output_column: Where to write the winning label.

    Returns:
        A new lazy `Dataset` with the voted label appended.

    Raises:
        PlanError: If no columns are given, the weights do not match, or no candidate
            labels could be found.
        ColumnNotFoundError: If a named column is not in `ds`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.ensemble import majority_vote
            >>> ds = bt.from_pydict(
            ...     {"m1": ["a", "b"], "m2": ["a", "b"], "m3": ["b", "a"]}
            ... )
            >>> majority_vote(ds, ["m1", "m2", "m3"]).to_pydict()["prediction"]
            ['a', 'b']
    """
    from batcher.ml._estimator import argmax_prediction

    names = list(columns)
    if not names:
        raise PlanError("majority_vote needs at least one prediction column")
    require_columns(ds, *names, hint="Pass the models' label columns.")
    if weights is None:
        share = [1.0] * len(names)
    else:
        share = [float(w) for w in weights]
        if len(share) != len(names):
            raise PlanError(
                f"majority_vote: {len(share)} weight(s) for {len(names)} column(s). "
                "Pass one weight per column, or none for an equal vote."
            )
        if any(w < 0 for w in share):
            raise PlanError(f"majority_vote: weights must be non-negative, got {share}")

    candidates = list(labels) if labels is not None else _observed_labels(ds, names)
    if not candidates:
        raise PlanError(
            "majority_vote: the prediction columns hold no non-null labels, so there is "
            "nothing to vote on."
        )

    def votes_for(label: object) -> Expr:
        total = lit(0.0)
        for name, weight in zip(names, share, strict=True):
            total = total + lit(weight) * (col(name) == lit(label)).cast("float64")
        return total

    return ds.with_columns(**{output_column: argmax_prediction(candidates, votes_for)})


def _observed_labels(ds: Dataset, columns: list[str]) -> list[object]:
    """The sorted distinct non-null labels appearing across `columns`.

    One `distinct` per column rather than a union of the whole frame: the columns are the
    same label space by construction, and a union would shuffle every row of every column
    to discover what the schema already implies.
    """
    seen: set[object] = set()
    for name in columns:
        values = ds.select(name).distinct().collect().column(name).to_pylist()
        seen.update(v for v in values if v is not None)
    return sorted(seen, key=repr)
