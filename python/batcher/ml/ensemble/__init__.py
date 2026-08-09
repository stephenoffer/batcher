"""Ensembling — combining several models into one prediction.

Two ways, in increasing order of what they cost and what they can express:

`blend_predictions`
    A weighted average of prediction columns already in the frame. No fit, no held-out
    split, one projection. Try this first: most of the benefit of ensembling comes from
    models that make *different* mistakes, and averaging is enough to collect it.
`StackingEnsemble`
    A meta-model fitted on the base models' out-of-fold predictions, so it can learn that
    one model is the one to trust in one region of the input and another elsewhere — which
    no fixed average can express.

The out-of-fold part is the whole difficulty of stacking. A meta-model trained on
predictions the base models made about rows they were fitted on learns to trust whichever
model memorized hardest, which is the one that will do worst in production. `StackingEnsemble`
builds every base model's out-of-fold column inside a single fold loop, so the columns
describe the same row without a join and without needing a row key.
"""

from __future__ import annotations

from batcher.ml.ensemble.blending import blend_predictions, majority_vote
from batcher.ml.ensemble.stacking import StackingEnsemble, out_of_fold_features

__all__ = [
    "StackingEnsemble",
    "blend_predictions",
    "majority_vote",
    "out_of_fold_features",
]
