"""Features derived from time — calendar parts, and history as columns.

Two modules, split by where the feature comes from. `calendar` turns one timestamp into
what a model can learn from: its parts, and the circular coordinates a periodic part needs
so midnight is adjacent to hour 23 rather than 23 units away. `history` turns *other rows*
into columns: the value `n` steps back, and aggregates over the window before the current
row.

The distinction matters because only the second can leak. A calendar part is a function of
the row itself; a rolling mean is a function of its neighbours, and one that includes the
current row puts the target's own value inside its own feature. Everything in `history` is
built to make that impossible.
"""

from __future__ import annotations

from batcher.ml.preprocessors.timeseries.calendar import CyclicalEncoder, DateTimeFeaturizer
from batcher.ml.preprocessors.timeseries.history import (
    ROLLING_AGGREGATES,
    LagFeaturizer,
    RollingFeaturizer,
)

__all__ = [
    "ROLLING_AGGREGATES",
    "CyclicalEncoder",
    "DateTimeFeaturizer",
    "LagFeaturizer",
    "RollingFeaturizer",
]
