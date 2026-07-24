"""Categorical encoders — ordinal codes, 0/1 indicators, and target encoding.

Every encoder learns its category set in `fit` (one bounded `distinct` or `group_by`
over the engine) and applies a lazy `Expr` projection in `transform`, with no per-row
Python. Each learned category costs either a CASE arm or an output column, so every
encoder takes a `max_categories` ceiling that fails a runaway fit fast instead of
building an unbounded plan.

The `frequency` module is the answer for a column whose cardinality makes that ceiling
the wrong tool: frequency encoding, rare-category bucketing, and the stateless hashing
trick each tolerate a category set no plan could enumerate.
"""

from __future__ import annotations

from batcher.ml.preprocessors.encoders.binary import BinaryEncoder
from batcher.ml.preprocessors.encoders.frequency import (
    FrequencyEncoder,
    HashingEncoder,
    RareCategoryEncoder,
)
from batcher.ml.preprocessors.encoders.onehot import MultiHotEncoder, OneHotEncoder
from batcher.ml.preprocessors.encoders.ordinal import LabelEncoder, OrdinalEncoder
from batcher.ml.preprocessors.encoders.target import TargetEncoder
from batcher.ml.preprocessors.encoders.woe import WOEEncoder

__all__ = [
    "BinaryEncoder",
    "FrequencyEncoder",
    "HashingEncoder",
    "LabelEncoder",
    "MultiHotEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "RareCategoryEncoder",
    "TargetEncoder",
    "WOEEncoder",
]
