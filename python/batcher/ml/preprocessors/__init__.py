"""Preprocessors — sklearn-style fit/transform that reuses Batcher's relational algebra.

`fit` learns state with one mergeable aggregate/distinct over the engine; `transform`
is a lazy `Expr` projection. Fit on train, `transform` train and test with the same
state. Compose with `Chain`.

    from batcher.ml.preprocessors import StandardScaler, Chain, SimpleImputer

    pipe = Chain([SimpleImputer(["age"]), StandardScaler(["age", "income"])])
    train2 = pipe.fit_transform(train)
    test2 = pipe.transform(test)
"""

from __future__ import annotations

from batcher.ml.preprocessors.base import Chain, Preprocessor
from batcher.ml.preprocessors.binning import KBinsDiscretizer
from batcher.ml.preprocessors.encoders import (
    LabelEncoder,
    MultiHotEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)
from batcher.ml.preprocessors.imputers import SimpleImputer
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
    "Preprocessor",
    "RobustScaler",
    "SimpleImputer",
    "StandardScaler",
    "Tokenizer",
]
