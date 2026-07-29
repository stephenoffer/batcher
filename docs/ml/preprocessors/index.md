# Preprocessors

Preprocessors are scikit-learn-style `fit` and `transform` feature transformers that run
on the engine. `fit` learns its state with one mergeable aggregate over the data, so it
is distributed and spillable for free. `transform` is a lazy column rewrite. Fit
on the training set, then `transform` the training **and** validation sets with the
same learned state.

Every preprocessor is importable from `batcher.ml.preprocessors`. Most are also
re-exported from `batcher.ml`, with the exceptions of `TargetEncoder` and
`PolynomialFeatures`, which you import from `batcher.ml.preprocessors`.

## Splitting first

`fit` must see the training rows only. Fit on the whole frame and held-out statistics
leak into your features. `ds.ml.train_test_split` gives disjoint parts that
together cover every row, assigned by a reproducible hash of each row's own content.
Each part is a plain row-wise filter, so the split streams, distributes, and is
*partition-independent*. A row lands in the same part however the data is laid out.

```python
import batcher as bt

ds = bt.range(0, 1000)
train, test = ds.ml.train_test_split(0.2, seed=42)
print(train.count() + test.count())
# 1000
```

`ds.ml.random_split([0.7, 0.15, 0.15], seed=42)` generalizes it to a
train/validation/test split.

Pass `key=` to hash only the columns that identify a row:

```python
users = bt.range(0, 1000).select(id=bt.col("value"), score=bt.col("value") * 2)
train, test = users.ml.train_test_split(0.2, seed=42, key="id")
print(train.count() + test.count())
# 1000
```

Then re-deriving a feature column does not move rows between train and test. The split
follows `id` alone, so recomputing `score` leaves every row where it was:

```python
rescored = users.with_columns(score=bt.col("id") * 3)
again, _ = rescored.ml.train_test_split(0.2, seed=42, key="id")
print(sorted(again.to_pydict()["id"]) == sorted(train.to_pydict()["id"]))
# True
```

Without `key=` every column is hashed, which is correct but re-splits whenever any value
changes.

## The three-call contract

Every preprocessor exposes the same `Preprocessor` API. `fit(ds)` runs a small aggregate,
stores the learned state on the object, and returns `self`. Even a stateless transform
such as `Normalizer`, `Concatenator`, or `Tokenizer` needs a `fit` or `fit_transform`
before `transform`. `transform(ds)` returns a new lazy `Dataset` with the learned rewrite
applied, and runs no work until a terminal op such as `collect` or `write.parquet`.
`fit_transform(ds)` is `fit(ds).transform(ds)`, the common single-split path.

`fit` is the one place a preprocessor *executes* and touches data. `transform` stays
lazy, so it composes with the rest of the pipeline and runs inside the engine. Calling
`transform` before `fit` raises `PlanError`.

## Available preprocessors

The table lists every preprocessor, what its `fit` learns, and what its `transform`
does. Stateless entries learn nothing and only need a `fit` call to satisfy the contract.

| Class | `fit` learns | `transform` |
| --- | --- | --- |
| `StandardScaler` | mean, population std | `(x - mean) / std` |
| `MinMaxScaler` | min, max | scale into `feature_range`, default `[0, 1]` |
| `MaxAbsScaler` | max absolute value | `x / max(\|x\|)` into `[-1, 1]` |
| `RobustScaler` | median, IQR | `(x - median) / IQR`, outlier-robust |
| `OrdinalEncoder` | sorted categories | integer code per category |
| `LabelEncoder` | sorted classes | integer code for one target column |
| `OneHotEncoder` | categories | one 0/1 indicator column per category |
| `BinaryEncoder` | categories | the category's integer code in base 2, one column per bit |
| `MultiHotEncoder` | distinct list elements | one 0/1 indicator column per category, for a list column |
| `TargetEncoder` | per-category target mean, global prior | smoothed mean-target code per high-cardinality category |
| `KBinsDiscretizer` | bin edges, quantile or uniform | integer bin index `0..n_bins-1` |
| `Normalizer` | nothing, stateless | scale each row to unit L1, L2, or max norm across columns |
| `SimpleImputer` | mean, median, mode, or constant | fill nulls |
| `Concatenator` | nothing, stateless | stack columns into one tensor column |
| `PolynomialFeatures` | nothing, stateless | add interaction and power terms such as `a*b` and `a^2` up to a degree |
| `Tokenizer` | nothing, stateless | tokenize text with a user tokenizer |
| `QuantileTransformer` | `n_quantiles` cut points | map onto a uniform or normal distribution by rank |
| `PowerTransformer` | the Yeo-Johnson lambda, by maximum likelihood | make a skewed column more Gaussian |
| `BoxCoxTransformer` | the Box-Cox lambda, by maximum likelihood | the same, for a strictly positive column |
| `LogTransformer` | nothing, stateless | `log(x + offset)`, the explainable shape fix |
| `Clipper` | lower and upper quantiles | clamp values into the learned range |
| `MissingIndicator` | nothing, stateless | append a boolean flag per column, before imputation |
| `FrequencyEncoder` | per-category frequency | replace a category with how often it occurs |
| `RareCategoryEncoder` | the categories clearing `min_frequency` | collapse the tail into one bucket |
| `HashingEncoder` | nothing, stateless | hash a category into one of `n_buckets` |

All preprocessors share the `Preprocessor` base contract of `fit`, `transform`, and
`fit_transform`.

Each scaler matches scikit-learn's definitions, and `StandardScaler` uses population
variance. `fit` lowers to the existing `group_by().agg(...)` and `distinct()`
operators, so it is partition-independent. A fit on a distributed dataset learns the
same statistics as a single-node fit.

## What happens to a missing value

A preprocessor never invents a value for a missing one. A null goes in and a null comes
out, so a gap in the data stays a gap the model can see rather than becoming a real value
at one end of the feature's range:

```python
import batcher as bt
from batcher.ml.preprocessors import KBinsDiscretizer

gaps = bt.from_pydict({"v": [0.0, 2.0, 8.0, 10.0, None]})
print(KBinsDiscretizer("v", n_bins=2, strategy="uniform").fit_transform(gaps).to_pydict())
# {'v': [0, 0, 1, 1, None]}
```

Three groups depart from that, each deliberately:

| Preprocessor | A null becomes | Why |
|---|---|---|
| `SimpleImputer`, `GroupImputer` | the fitted fill value | Filling nulls is the whole point of them. |
| `LabelEncoder`, `OrdinalEncoder`, `FrequencyEncoder` | `unknown_value` | A missing value is not a category, so it joins the unknown bucket. |
| `OneHotEncoder`, `LabelBinarizer`, `RareCategoryEncoder` | all-zero indicators, or the rare bucket | It belongs to no class, which the indicators already express. |

Reach for `MissingIndicator` before an encoder when the difference between "absent" and
"rare" carries signal for your model. It adds a 0/1 column recording where the nulls were,
so the information survives whatever the next step does with them.

## Where they run

`transform` is a lazy `Dataset`, so it composes with the rest of a pipeline and the
result is computed by a terminal op such as `collect()` or `write.parquet(...)`, on one
node or across a cluster. Use preprocessors before a training loop, covered in
[PyTorch integration](../pytorch.md), or before batch [inference](../inference.md).

## The rest of this section

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Scaling and distributions
:link: scaling
:link-type: doc
Scalers, the row normalizer, and the distribution reshapers.
:::

:::{grid-item-card} {octicon}`hash;1.1em` Encoding and imputation
:link: encoding
:link-type: doc
Categorical encoders, missing-value imputation, and binning.
:::

:::{grid-item-card} {octicon}`plus-circle;1.1em` Generating features
:link: feature-generation
:link-type: doc
Timestamps, text statistics, history, PCA, and assembly.
:::

:::{grid-item-card} {octicon}`link;1.1em` Chaining and persisting
:link: pipelines
:link-type: doc
`Chain`, fitting a whole pipeline, and saving the fitted state.
:::

:::{grid-item-card} {octicon}`duplicate;1.1em` Deduplication and matching
:link: deduplication
:link-type: doc
Fuzzy dedup with MinHash, and `similarity_join` on embeddings.
:::
::::

## Next steps

- [Feature engineering tutorial](../../tutorials/feature-engineering.md): the full workflow
  from raw table to model-ready matrix, end to end, with `Chain`.
- [PyTorch integration](../pytorch.md): hand the assembled features to a training loop.
- [ML API reference](../../api/ml.md): the complete `Preprocessor` surface.

```{toctree}
:hidden:

scaling
encoding
feature-generation
pipelines
deduplication
```
