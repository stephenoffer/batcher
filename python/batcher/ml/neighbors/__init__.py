"""Nearest-neighbour models — prediction and imputation by local similarity.

The models that assume nothing about the shape of the relationship: to answer a question
about a row, find the training rows most like it and see what was true of them. That makes
them the natural check on whether a problem has local structure, and the right tool when a
boundary is irregular or a missing value is better described by similar rows than by a
column-wide statistic.

All three fold a **bounded** reference set into the prediction as literals, the way a fitted
linear model folds in its coefficients, so what reaches the engine is one arithmetic
expression over the feature columns — no join, no shuffle, and the same plan on one core or
a hundred. The bound is the trade: a k-NN over a genuinely large corpus wants an approximate
index (`batcher.ml.build_vector_index`), not a broadcast reference set.

Scale the features first. Distance treats every column alike, so a column measured in
millions decides every neighbour and one measured in fractions is ignored.
"""

from __future__ import annotations

from batcher.ml.neighbors.estimators import KNeighborsClassifier, KNeighborsRegressor
from batcher.ml.neighbors.imputer import KNNImputer
from batcher.ml.neighbors.reference import MAX_REFERENCE_ROWS

__all__ = [
    "MAX_REFERENCE_ROWS",
    "KNNImputer",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
]
