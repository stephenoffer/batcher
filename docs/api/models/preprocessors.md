# Preprocessors API

`batcher.ml.preprocessors` holds the fit/transform feature-engineering estimators. Each
one `fit`s over a {py:class}`Dataset <batcher.Dataset>` to learn its statistics, then `transform`s any {py:class}`Dataset <batcher.Dataset>` with
them, and {py:class}`Chain <batcher.ml.preprocessors.Chain>` composes several into one pipeline.

This page is the reference. For how they fit into a training workflow, read
{doc}`/ml/preparing/preprocessors/index`.

Every estimator implements the same `fit` / `transform` / `fit_transform` protocol, and
`Chain` is itself one of them:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   Preprocessor
   Chain
```

## Scalers and normalizers

These rescale numeric columns:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   StandardScaler
   MinMaxScaler
   MaxAbsScaler
   RobustScaler
   Normalizer
```

## Distribution shaping

Reshaping a column's *distribution* rather than only its scale. Reach for these when a
feature is heavily skewed or long-tailed and a linear rescale would leave it that way:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   PowerTransformer
   BoxCoxTransformer
   QuantileTransformer
   RankTransformer
   LogTransformer
   PCA
   TruncatedSVD
```

## Encoders

These turn categorical columns into numeric ones:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   OneHotEncoder
   MultiHotEncoder
   LabelBinarizer
   MultiLabelBinarizer
   LabelEncoder
   OrdinalEncoder
   BinaryEncoder
   TargetEncoder
   FrequencyEncoder
   HashingEncoder
   RareCategoryEncoder
   LeaveOneOutEncoder
   JamesSteinEncoder
   WOEEncoder
```

## Binning, imputation, text, and assembly

The rest of the estimators cover discretization, missing values, text splitting, and feature assembly:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   KBinsDiscretizer
   SimpleImputer
   IterativeImputer
   Tokenizer
   Concatenator
   PolynomialFeatures
   SplineTransformer
   FunctionTransformer
   Clipper
   MissingIndicator
   Binarizer
   VarianceThreshold
   ColumnSelector
   ColumnDropper
```

## Feature selection

These prune columns rather than transform them. The choice is held as fitted state, so the
validation split is pruned by the training split's decision:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   SelectKBest
   SelectPercentile
   DropCorrelated
   SelectFromModel
   RFE
   feature_importances
```

## Random projection and kernel approximation

Dimensionality reduction and kernel feature maps that need no covariance matrix. All four
lower to plain arithmetic over the source columns, so the transform runs column-wise:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   GaussianRandomProjection
   SparseRandomProjection
   RBFSampler
   Nystroem
   johnson_lindenstrauss_min_dim
```

## Probability calibration

`batcher.ml.metrics` measures calibration. These two correct it. Both are fitted on a split
the model did not train on, and both are monotone in the score, so neither changes the
model's ranking:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   PlattCalibrator
   IsotonicCalibrator
```

## Text vectorizers

These turn a text column into the bag-of-words features a classical text model trains on.
`CountVectorizer` and `TfidfVectorizer` learn a vocabulary. `HashingVectorizer` decides a
term's feature index arithmetically and so needs no fit pass at all:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   CountVectorizer
   TfidfVectorizer
   HashingVectorizer
```

## Derived and grouped features

New columns built out of existing ones: products and ratios that a linear model cannot learn
on its own, and group-relative statistics that let a row see its cohort:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   InteractionFeatures
   RatioFeatures
   GroupStatEncoder
   GroupImputer
```

## Timestamp features

A raw timestamp is the least useful column in a feature table. These turn it into parts a
model can learn from: integer parts for a tree, and circular coordinates for anything that
measures distance:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   DateTimeFeaturizer
   CyclicalEncoder
```

## Lag and rolling features

History as columns, for a forecasting model. Both exclude the current row by construction,
because a rolling window that includes it puts the target's own value inside its own
feature. That is the most common leak in a forecasting pipeline, and one that raises nothing:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   LagFeaturizer
   RollingFeaturizer
```

## Text surface features

Length, word count, and character mix. These need no model at all, and they often carry most
of the signal a gradient-boosted model splits on:

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   TextStatFeaturizer
```

## Persistence

A fitted preprocessor's state has to outlive the process that fitted it, or a serving
request is standardized with its own mean instead of the training set's. These read and write
that state as plain JSON: reviewable, diffable, portable, and safe to load from a store you
do not fully control. A pickle is none of those.

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autosummary::
   :toctree: generated
   :nosignatures:

   save
   load
   to_dict
   from_dict
```

## See also

- {doc}`/ml/preparing/preprocessors/index`: the guide, with the fit-on-train contract.
- {doc}`/api/models/ml`: the `.ml` accessor these estimators sit beside.
- {doc}`/api/models/ml-models`: the estimators that consume the features they produce.
- {doc}`/cookbook/ml/preprocessing/preprocessing_chain`: chaining preprocessors into one fitted pipeline, as a script.
