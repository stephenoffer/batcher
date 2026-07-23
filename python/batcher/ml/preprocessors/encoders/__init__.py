"""Categorical encoders — ordinal codes, 0/1 indicators, and target encoding.

Every encoder learns its category set in `fit` (one bounded `distinct` or `group_by`
over the engine) and applies a lazy `Expr` projection in `transform`, with no per-row
Python. Each learned category costs either a CASE arm or an output column, so every
encoder takes a `max_categories` ceiling that fails a runaway fit fast instead of
building an unbounded plan.
"""

from __future__ import annotations

from batcher.ml.preprocessors.encoders.onehot import MultiHotEncoder, OneHotEncoder
from batcher.ml.preprocessors.encoders.ordinal import LabelEncoder, OrdinalEncoder
from batcher.ml.preprocessors.encoders.target import TargetEncoder

__all__ = ["LabelEncoder", "MultiHotEncoder", "OneHotEncoder", "OrdinalEncoder", "TargetEncoder"]
