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
from batcher.ml.preprocessors.derived import (
    PCA,
    Binarizer,
    ColumnDropper,
    ColumnSelector,
    GroupImputer,
    GroupStatEncoder,
    InteractionFeatures,
    LabelBinarizer,
    MultiLabelBinarizer,
    RankTransformer,
    RatioFeatures,
    TruncatedSVD,
    VarianceThreshold,
)
from batcher.ml.preprocessors.encoders import (
    BinaryEncoder,
    FrequencyEncoder,
    HashingEncoder,
    LabelEncoder,
    MultiHotEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareCategoryEncoder,
    TargetEncoder,
    WOEEncoder,
)
from batcher.ml.preprocessors.imputers import SimpleImputer
from batcher.ml.preprocessors.persistence import from_dict, load, save, to_dict
from batcher.ml.preprocessors.polynomial import PolynomialFeatures, SplineTransformer
from batcher.ml.preprocessors.power import BoxCoxTransformer, PowerTransformer
from batcher.ml.preprocessors.scalers import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)
from batcher.ml.preprocessors.selection import (
    RFE,
    DropCorrelated,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    feature_importances,
)
from batcher.ml.preprocessors.text import Concatenator, Tokenizer
from batcher.ml.preprocessors.text_features import TextStatFeaturizer
from batcher.ml.preprocessors.timeseries import (
    CyclicalEncoder,
    DateTimeFeaturizer,
    LagFeaturizer,
    RollingFeaturizer,
)
from batcher.ml.preprocessors.transforms import (
    Clipper,
    FunctionTransformer,
    LogTransformer,
    MissingIndicator,
    QuantileTransformer,
)
from batcher.ml.preprocessors.vectorizers import CountVectorizer, HashingVectorizer, TfidfVectorizer

__all__ = [
    "PCA",
    "RFE",
    "Binarizer",
    "BinaryEncoder",
    "BoxCoxTransformer",
    "Chain",
    "Clipper",
    "ColumnDropper",
    "ColumnSelector",
    "Concatenator",
    "CountVectorizer",
    "CyclicalEncoder",
    "DateTimeFeaturizer",
    "DropCorrelated",
    "FrequencyEncoder",
    "FunctionTransformer",
    "GroupImputer",
    "GroupStatEncoder",
    "HashingEncoder",
    "HashingVectorizer",
    "InteractionFeatures",
    "KBinsDiscretizer",
    "LabelBinarizer",
    "LabelEncoder",
    "LagFeaturizer",
    "LogTransformer",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MissingIndicator",
    "MultiHotEncoder",
    "MultiLabelBinarizer",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PolynomialFeatures",
    "PowerTransformer",
    "Preprocessor",
    "QuantileTransformer",
    "RankTransformer",
    "RareCategoryEncoder",
    "RatioFeatures",
    "RobustScaler",
    "RollingFeaturizer",
    "SelectFromModel",
    "SelectKBest",
    "SelectPercentile",
    "SimpleImputer",
    "SplineTransformer",
    "StandardScaler",
    "TargetEncoder",
    "TextStatFeaturizer",
    "TfidfVectorizer",
    "Tokenizer",
    "TruncatedSVD",
    "VarianceThreshold",
    "WOEEncoder",
    "feature_importances",
    "from_dict",
    "load",
    "save",
    "to_dict",
]
