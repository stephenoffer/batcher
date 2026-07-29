# Preprocessors API

`batcher.ml.preprocessors` holds the fit/transform feature-engineering estimators. Each
one `fit`s over a `Dataset` to learn its statistics, then `transform`s any `Dataset` with
them, and `Chain` composes several into one pipeline.

This page is the reference. For how they fit into a training workflow, read
{doc}`../ml/preprocessors/index`.

`batcher.ml.preprocessors` holds the fit/transform feature-engineering estimators.
Each one `fit`s over a `Dataset` to learn its statistics, then `transform`s any `Dataset` with them. `Chain` composes several into one pipeline. See the
[preprocessors guide](../ml/preprocessors/index.md) for how they fit into a training
workflow.

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autoclass:: Preprocessor
   :members:

.. autoclass:: Chain
   :members:
```

## Scalers and normalizers

These rescale numeric columns:

```{eval-rst}
.. autoclass:: StandardScaler
   :members:

.. autoclass:: MinMaxScaler
   :members:

.. autoclass:: MaxAbsScaler
   :members:

.. autoclass:: RobustScaler
   :members:

.. autoclass:: Normalizer
   :members:
```

## Distribution shaping

These reshape a column's *distribution* rather than only its scale. Reach for them when a
feature is heavily skewed or long-tailed and a linear rescale would leave it that way:

```{eval-rst}
.. autoclass:: PowerTransformer
   :members:

.. autoclass:: BoxCoxTransformer
   :members:

.. autoclass:: QuantileTransformer
   :members:

.. autoclass:: RankTransformer
   :members:

.. autoclass:: LogTransformer
   :members:

.. autoclass:: PCA
   :members:

.. autoclass:: TruncatedSVD
   :members:
```

## Encoders

These turn categorical columns into numeric ones:

```{eval-rst}
.. autoclass:: OneHotEncoder
   :members:

.. autoclass:: MultiHotEncoder
   :members:

.. autoclass:: LabelBinarizer
   :members:

.. autoclass:: MultiLabelBinarizer
   :members:

.. autoclass:: LabelEncoder
   :members:

.. autoclass:: OrdinalEncoder
   :members:

.. autoclass:: BinaryEncoder
   :members:

.. autoclass:: TargetEncoder
   :members:

.. autoclass:: FrequencyEncoder
   :members:

.. autoclass:: HashingEncoder
   :members:

.. autoclass:: RareCategoryEncoder
   :members:

.. autoclass:: WOEEncoder
   :members:
```

## Binning, imputation, text, and assembly

The rest of the estimators cover discretization, missing values, text splitting, and feature assembly:

```{eval-rst}
.. autoclass:: KBinsDiscretizer
   :members:

.. autoclass:: SimpleImputer
   :members:

.. autoclass:: Tokenizer
   :members:

.. autoclass:: Concatenator
   :members:

.. autoclass:: PolynomialFeatures
   :members:

.. autoclass:: Clipper
   :members:

.. autoclass:: MissingIndicator
   :members:

.. autoclass:: Binarizer
   :members:

.. autoclass:: VarianceThreshold
   :members:

.. autoclass:: ColumnSelector
   :members:

.. autoclass:: ColumnDropper
   :members:
```

## Derived and grouped features

These build new columns out of existing ones: products and ratios that a linear model
cannot learn on its own, and group-relative statistics that let a row see its cohort:

```{eval-rst}
.. autoclass:: InteractionFeatures
   :members:

.. autoclass:: RatioFeatures
   :members:

.. autoclass:: GroupStatEncoder
   :members:

.. autoclass:: GroupImputer
   :members:
```

## Timestamp features

A raw timestamp is the least useful column in a feature table. These turn it into parts a
model can learn from — integer parts for a tree, circular coordinates for anything that
measures distance:

```{eval-rst}
.. autoclass:: DateTimeFeaturizer
   :members:

.. autoclass:: CyclicalEncoder
   :members:
```

## Lag and rolling features

History as columns, for a forecasting model. Both exclude the current row by construction,
because a rolling window that includes it puts the target's own value inside its own
feature — the most common leak in a forecasting pipeline, and one that raises nothing:

```{eval-rst}
.. autoclass:: LagFeaturizer
   :members:

.. autoclass:: RollingFeaturizer
   :members:
```

## Text surface features

Cheap, interpretable text signals — length, word count, character mix — that need no model
and often carry most of the signal a gradient-boosted model splits on:

```{eval-rst}
.. autoclass:: TextStatFeaturizer
   :members:
```

## Persistence

A fitted preprocessor's state has to outlive the process that fitted it, or a serving
request is standardized with its own mean instead of the training set's. These read and
write that state as plain JSON — reviewable, diffable, portable, and safe to load from a
store you do not fully control, which a pickle is none of.

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autofunction:: save
.. autofunction:: load
.. autofunction:: to_dict
.. autofunction:: from_dict
```

## See also

:::{seealso}
- {doc}`../ml/preprocessors/index`: the guide, with the fit-on-train contract.
- {doc}`ml`: the `.ml` accessor these estimators sit beside.
- {doc}`ml-models`: the estimators that consume the features they produce.
:::
