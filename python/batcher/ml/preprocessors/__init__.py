"""Preprocessors — sklearn-style fit/transform that reuses Batcher's relational algebra.

`fit` learns state with one mergeable aggregate/distinct over the engine; `transform`
is a lazy `Expr` projection. Fit on train, `transform` train and test with the same
state. Compose by sequencing — each step's fitted object transforms both splits.

    from batcher.ml.preprocessors import SimpleImputer, StandardScaler

    imputer = SimpleImputer(["age"])
    scaler = StandardScaler(["age", "income"])
    train2 = scaler.fit_transform(imputer.fit_transform(train))
    test2 = scaler.transform(imputer.transform(test))
"""

from __future__ import annotations

from batcher.ml.preprocessors.base import Preprocessor
from batcher.ml.preprocessors.binning import KBinsDiscretizer
from batcher.ml.preprocessors.chain import Chain
from batcher.ml.preprocessors.encoders import (
    LabelEncoder,
    MultiHotEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
from batcher.ml.preprocessors.imputers import SimpleImputer
from batcher.ml.preprocessors.polynomial import PolynomialFeatures
from batcher.ml.preprocessors.scalers import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)
from batcher.ml.preprocessors.text import Concatenator, Tokenizer

__all__ = [
    "Chain",
    "Concatenator",
    "KBinsDiscretizer",
    "LabelEncoder",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MultiHotEncoder",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PolynomialFeatures",
    "Preprocessor",
    "RobustScaler",
    "SimpleImputer",
    "StandardScaler",
    "TargetEncoder",
    "Tokenizer",
]
