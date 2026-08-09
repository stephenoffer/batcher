"""Feature selection as fit/transform preprocessors.

`batcher.ml.feature_scores` and `batcher.ml.selection` answer "which columns look useful";
these turn that answer into a step of a pipeline. The difference matters more than it
sounds: a selection is *fitted state*, so it has to be learned on the training split and
then applied unchanged to the validation split. Recomputing it per split silently lets the
held-out data choose its own features, and the resulting score is optimistic by an amount
nothing reports.

Which to reach for, cheapest first:

`SelectKBest` / `SelectPercentile`
    Univariate filters. One mergeable aggregate per feature, blind to interaction. Run
    first, on a wide table, to remove obvious dead weight.
`DropCorrelated`
    Removes columns that duplicate each other. Cheap, and the fix for the unstable
    coefficients two near-identical features give a linear model.
`SelectFromModel`
    Reads one fitted model's coefficients. The standard partner for an L1 penalty.
`RFE`
    Refits after every elimination. The most faithful and the most expensive.
"""

from __future__ import annotations

from batcher.ml.preprocessors.selection.model_based import (
    RFE,
    SelectFromModel,
    feature_importances,
)
from batcher.ml.preprocessors.selection.redundancy import DropCorrelated
from batcher.ml.preprocessors.selection.univariate import (
    SCORE_FUNCTIONS,
    SelectKBest,
    SelectPercentile,
)

__all__ = [
    "RFE",
    "SCORE_FUNCTIONS",
    "DropCorrelated",
    "SelectFromModel",
    "SelectKBest",
    "SelectPercentile",
    "feature_importances",
]
