"""Composing preprocessing and a model into one fitted object.

`Chain` composes preprocessors and stops there, which leaves the caller responsible for
remembering which transforms a model was trained behind and applying exactly those at
serving time. That is where train/serve skew comes from, and it fails silently — a model
scored behind one fewer transform returns numbers, not an error.

`Pipeline` owns both halves, so the sequence `predict` replays is by construction the one
`fit` used, and the whole recipe saves as a single file.

`TransformedTargetRegressor` does the same for the other end of the model: it fits on a
reshaped target and inverts the prediction back, so the inverse cannot be forgotten at
serving time — where forgetting it produces predictions wrong by a factor of *e* with
nothing to indicate it.
"""

from __future__ import annotations

from batcher.ml.compose.pipeline import Pipeline
from batcher.ml.compose.target import TransformedTargetRegressor

__all__ = ["Pipeline", "TransformedTargetRegressor"]
