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
from batcher.plan.expr_ir.constructors import col, lit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["blend_predictions"]


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
    available = ds.columns
    present = set(available)
    for name in names:
        if name not in present:
            from batcher._internal.errors import ColumnNotFoundError, unknown_message

            raise ColumnNotFoundError(
                unknown_message(
                    "column", name, available, hint="Pass the models' prediction columns."
                )
            )
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
