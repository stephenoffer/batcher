"""Derived-feature preprocessors — the columns a model needs that the table doesn't have.

Three families, all about *constructing* features rather than rescaling existing ones.
`construct` makes columns from a row's own values — thresholds, interactions, ratios,
projections, and the variance filter. `grouped` makes columns from a row's *group* — the
per-group statistics and group-aware imputation that encode behaviour an individual row
cannot carry. `decomposition` makes a few uncorrelated columns from many correlated ones (PCA).
"""

from __future__ import annotations

from batcher.ml.preprocessors.derived.construct import (
    Binarizer,
    ColumnDropper,
    ColumnSelector,
    InteractionFeatures,
    RatioFeatures,
    VarianceThreshold,
)
from batcher.ml.preprocessors.derived.decomposition import PCA, TruncatedSVD
from batcher.ml.preprocessors.derived.encode import (
    LabelBinarizer,
    MultiLabelBinarizer,
    RankTransformer,
)
from batcher.ml.preprocessors.derived.grouped import (
    GROUP_STATISTICS,
    GroupImputer,
    GroupStatEncoder,
)

__all__ = [
    "GROUP_STATISTICS",
    "PCA",
    "Binarizer",
    "ColumnDropper",
    "ColumnSelector",
    "GroupImputer",
    "GroupStatEncoder",
    "InteractionFeatures",
    "LabelBinarizer",
    "MultiLabelBinarizer",
    "RankTransformer",
    "RatioFeatures",
    "TruncatedSVD",
    "VarianceThreshold",
]
