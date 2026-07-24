"""Tabular batch inference — XGBoost, LightGBM, CatBoost, scikit-learn, ONNX.

The classical-ML counterpart of `batcher.ml.inference`: score a fitted tabular model over
a `Dataset` at any scale, with the model loaded once per worker and the feature matrix
assembled from Arrow columns rather than row dicts. Reached as `ds.ml.predict(...)`.

Each framework is a `TabularAdapter` in the `FRAMEWORKS` registry, so a new one is a new
adapter rather than a new call path.
"""

from __future__ import annotations

from batcher.ml.tabular.features import feature_matrix, prediction_columns, resolve_features
from batcher.ml.tabular.predictor import predicted_column_names, tabular_predictor
from batcher.ml.tabular.registry import (
    FRAMEWORKS,
    TabularAdapter,
    detect_framework,
    get_adapter,
)

__all__ = [
    "FRAMEWORKS",
    "TabularAdapter",
    "detect_framework",
    "feature_matrix",
    "get_adapter",
    "predicted_column_names",
    "prediction_columns",
    "resolve_features",
    "tabular_predictor",
]
