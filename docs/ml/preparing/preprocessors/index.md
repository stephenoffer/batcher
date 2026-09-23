# Preprocessors

Preprocessors are scikit-learn-style `fit` and `transform` feature transformers that run on the engine. If you know `sklearn.preprocessing`, you already know the API. What changes is where the work happens. `fit` learns its state with one mergeable aggregate over the data, so fitting a scaler on a billion rows is one distributed, spillable pass rather than a sample pulled into memory. `transform` is a lazy column rewrite that runs inside the plan. Fit on the training set, then `transform` the training **and** validation sets with the same learned state.

Every preprocessor is importable from both `batcher.ml.preprocessors` and `batcher.ml`.
[`tests/unit/test_ml_preprocessing_workflow.py`](https://github.com/stephenoffer/batcher/blob/main/tests/unit/test_ml_preprocessing_workflow.py) pins that, so the two paths cannot drift.

## Splitting first

`fit` must see the training rows only. Fit on the whole frame and held-out statistics
leak into your features. {py:meth}`ds.ml.train_test_split <batcher.api.dataset.ml.DatasetML.train_test_split>` gives disjoint parts that
together cover every row, assigned by a reproducible hash of each row's own content.
Each part is a plain row-wise filter, so the split streams, distributes, and is
*partition-independent*. A row lands in the same part however the data is laid out.

The two pipelines below differ by a single arrow into `fit`, and that arrow is the whole difference between an honest offline score and a leaking one:

![Fitting on the training split and fitting on everything differ by one arrow, and nothing in the engine tells the two apart. On the correct path, scaler.fit(train) learns mean_ and scale_ from the training rows only, transform applies that one learned scale to both splits, and the held-out statistics never existed. On the leaking path, scaler.fit(ds) learns from every row, so the test rows help set the scale that is then applied to them, and every offline score afterwards is optimistic. Underneath, fit executes one aggregate and reads a few scalars back to the driver as the trailing-underscore state, while transform bakes those scalars into an expression and stays lazy. Nothing detects the leak: fit runs the same aggregate over whatever rows it is handed, with no error, no warning, and no flag.](/_static/diagrams/fit_transform_leakage.svg)

```python
import batcher as bt

ds = bt.range(0, 1000)
train, test = ds.ml.train_test_split(0.2, seed=42)
print(train.count() + test.count())
# 1000
```

{py:meth}`ds.ml.random_split([0.7, 0.15, 0.15], seed=42) <batcher.api.dataset.ml.DatasetML.random_split>` generalizes it to a
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

Every preprocessor exposes the same {py:class}`Preprocessor <batcher.ml.preprocessors.Preprocessor>` API. `fit(ds)` runs a small aggregate,
stores the learned state on the object, and returns `self`. `transform(ds)` returns a new
lazy {py:class}`Dataset <batcher.Dataset>` with the learned rewrite applied, and runs no work until a terminal op
such as `collect` or `write.parquet`. `fit_transform(ds)` is `fit(ds).transform(ds)`, the
common single-split path.

`fit` is the one place a preprocessor *executes* and touches data. `transform` stays
lazy, so it composes with the rest of the pipeline and runs inside the engine.

Call `fit` even on a transform that learns nothing. Several of them enforce it:
{py:class}`Concatenator <batcher.ml.preprocessors.Concatenator>` and {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` raise {py:exc}`PlanError <batcher.PlanError>` on a `transform` that no `fit`
preceded. Others don't check, and {py:class}`Normalizer <batcher.ml.preprocessors.Normalizer>` is one, so a bare `transform` works
there. Relying on that buys you nothing and breaks the moment the step moves into a
{py:class}`Chain <batcher.ml.preprocessors.Chain>` beside something stateful.

## Available preprocessors

These are the ones you reach for most, with what each `fit` learns and what its
`transform` does. {doc}`/api/models/preprocessors` has the rest.

| Class | `fit` learns | `transform` |
| --- | --- | --- |
| {py:class}`StandardScaler <batcher.ml.preprocessors.StandardScaler>` | mean, population std | `(x - mean) / std` |
| {py:class}`MinMaxScaler <batcher.ml.preprocessors.MinMaxScaler>` | min, max | scale into `feature_range`, default `[0, 1]` |
| {py:class}`MaxAbsScaler <batcher.ml.preprocessors.MaxAbsScaler>` | max absolute value | `x / max(\|x\|)` into `[-1, 1]` |
| {py:class}`RobustScaler <batcher.ml.preprocessors.RobustScaler>` | median, IQR | `(x - median) / IQR`, outlier-robust |
| {py:class}`OrdinalEncoder <batcher.ml.preprocessors.OrdinalEncoder>` | sorted categories | integer code per category |
| {py:class}`LabelEncoder <batcher.ml.preprocessors.LabelEncoder>` | sorted classes | integer code for one target column |
| {py:class}`OneHotEncoder <batcher.ml.preprocessors.OneHotEncoder>` | categories | one 0/1 indicator column per category |
| {py:class}`BinaryEncoder <batcher.ml.preprocessors.BinaryEncoder>` | categories | the category's integer code in base 2, one column per bit |
| {py:class}`MultiHotEncoder <batcher.ml.preprocessors.MultiHotEncoder>` | distinct list elements | one 0/1 indicator column per category, for a list column |
| {py:class}`TargetEncoder <batcher.ml.preprocessors.TargetEncoder>` | per-category target mean, global prior | smoothed mean-target code per high-cardinality category |
| {py:class}`KBinsDiscretizer <batcher.ml.preprocessors.KBinsDiscretizer>` | bin edges, quantile or uniform | integer bin index `0..n_bins-1` |
| {py:class}`Normalizer <batcher.ml.preprocessors.Normalizer>` | nothing, stateless | scale each row to unit L1, L2, or max norm across columns |
| {py:class}`SimpleImputer <batcher.ml.preprocessors.SimpleImputer>` | mean, median, mode, or constant | fill nulls |
| {py:class}`Concatenator <batcher.ml.preprocessors.Concatenator>` | nothing, stateless | stack columns into one tensor column |
| {py:class}`PolynomialFeatures <batcher.ml.preprocessors.PolynomialFeatures>` | nothing, stateless | add interaction and power terms such as `a*b` and `a^2` up to a degree |
| {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` | nothing, stateless | tokenize text with a user tokenizer |
| {py:class}`QuantileTransformer <batcher.ml.preprocessors.QuantileTransformer>` | `n_quantiles` cut points | map onto a uniform or normal distribution by rank |
| {py:class}`PowerTransformer <batcher.ml.preprocessors.PowerTransformer>` | the Yeo-Johnson lambda, by maximum likelihood | make a skewed column more Gaussian |
| {py:class}`BoxCoxTransformer <batcher.ml.preprocessors.BoxCoxTransformer>` | the Box-Cox lambda, by maximum likelihood | the same, for a strictly positive column |
| {py:class}`LogTransformer <batcher.ml.preprocessors.LogTransformer>` | nothing, stateless | `log(x + offset)`, the explainable shape fix |
| {py:class}`Clipper <batcher.ml.preprocessors.Clipper>` | lower and upper quantiles | clamp values into the learned range |
| {py:class}`MissingIndicator <batcher.ml.preprocessors.MissingIndicator>` | nothing, stateless | append a boolean flag per column, before imputation |
| {py:class}`FrequencyEncoder <batcher.ml.preprocessors.FrequencyEncoder>` | per-category frequency | replace a category with how often it occurs |
| {py:class}`RareCategoryEncoder <batcher.ml.preprocessors.RareCategoryEncoder>` | the categories clearing `min_frequency` | collapse the tail into one bucket |
| {py:class}`HashingEncoder <batcher.ml.preprocessors.HashingEncoder>` | nothing, stateless | hash a category into one of `n_buckets` |

Each scaler matches scikit-learn's definitions, and `StandardScaler` uses population
variance. `fit` lowers to the existing {py:meth}`group_by().agg(...) <batcher.Dataset.group_by>` and {py:meth}`distinct() <batcher.Dataset.distinct>`
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
| {py:class}`SimpleImputer <batcher.ml.preprocessors.SimpleImputer>`, {py:class}`GroupImputer <batcher.ml.preprocessors.GroupImputer>` | the fitted fill value | Filling nulls is the whole point of them. |
| `LabelEncoder`, `OrdinalEncoder`, `FrequencyEncoder` | `unknown_value` | A missing value is not a category, so it joins the unknown bucket. |
| {py:class}`OneHotEncoder <batcher.ml.preprocessors.OneHotEncoder>`, {py:class}`LabelBinarizer <batcher.ml.preprocessors.LabelBinarizer>`, {py:class}`RareCategoryEncoder <batcher.ml.preprocessors.RareCategoryEncoder>` | all-zero indicators, or the rare bucket | It belongs to no class, which the indicators already express. |

Reach for `MissingIndicator` before an encoder when the difference between "absent" and
"rare" carries signal for your model. It adds a 0/1 column recording where the nulls were,
so the information survives whatever the next step does with them.

### NaN counts as missing

In a floating-point column, IEEE NaN is treated exactly like null. That is the rule scikit-learn applies with `missing_values=np.nan`, and it matters because NaN is how pandas, NumPy, and Parquet files written from either spell a missing value. The rule has three parts:

- Every `fit` statistic skips NaN. The scalers, `SimpleImputer`, `IterativeImputer`, `GroupImputer`, `GroupStatEncoder`, the power transforms, `QuantileTransformer`, `KBinsDiscretizer`, `Clipper`, and `SplineTransformer` rewrite NaN to null before their aggregate, so one NaN no longer turns a learned mean or maximum into NaN.
- `SimpleImputer`, `IterativeImputer`, and `GroupImputer` fill NaN as well as null.
- `MissingIndicator` flags NaN as well as null.

The rewrite is a lazy `nullif(x, NaN)` projection in the engine, with no per-row Python, and it leaves integer and string columns alone because they can't hold a NaN. A transform that doesn't fill values, such as a scaler, passes a NaN through as NaN, the same way it passes a null through as null.

```python
import batcher as bt
from batcher.ml.preprocessors import MissingIndicator, SimpleImputer, StandardScaler

raw = bt.from_pydict({"income": [40.0, float("nan"), 60.0, None]})
print(StandardScaler("income").fit(raw).mean_)
# {'income': 50.0}
flagged = MissingIndicator("income").fit_transform(raw)
print(SimpleImputer("income").fit_transform(flagged).to_pydict())
# {'income': [40.0, 50.0, 60.0, 50.0], 'income_missing': [False, True, False, True]}
```

## What a preprocessor accepts

The arithmetic preprocessors, meaning the scalers, the binners, the power and rank transforms, the projections and the kernel approximations, need a number in every column they name. Given a string one they say so, naming the column and the preprocessor:

```python
import batcher as bt
from batcher.ml.preprocessors import StandardScaler

ds = bt.from_pydict({"city": ["a", "b", "c"], "amount": [1.0, 4.0, 2.0]})
try:
    StandardScaler(["city"]).fit(ds)
except Exception as error:
    print(str(error)[:60])
# StandardScaler: column 'city' has type string, and a fit
```

The encoders are the opposite case and take strings by design, which is why the requirement is opt-in per preprocessor rather than a rule for all of them. {py:class}`SimpleImputer <batcher.ml.preprocessors.SimpleImputer>` sits between the two: `strategy="mean"` and `strategy="median"` are arithmetic and need numbers, while `most_frequent` and `constant` are how a categorical column gets imputed and keep working on strings.

The check reads the schema rather than the data, so it costs no pass over the dataset and raises before any work is scheduled. An empty dataset is not rejected, because an untyped column is what both an all-null column and an empty partition look like.

## Where they run

`transform` is a lazy `Dataset`, so it composes with the rest of a pipeline and the
result is computed by a terminal op such as {py:meth}`collect() <batcher.Dataset.collect>` or `write.parquet(...)`, on one
node or across a cluster. Use preprocessors before a training loop, covered in
{doc}`PyTorch integration </ml/inference/pytorch>`, or before batch {doc}`inference </ml/inference/inference>`.

`fit` takes no `distributed=` argument. Its aggregate runs through a bare `collect()`, so it follows the session's `distributed="auto"` routing, which distributes only when Ray is already connected to a multi-node cluster and the input is estimated to be large enough to pay for the fan-out. To pin every fit to the cluster, or keep every fit off it, set the session-wide `distributed.mode` option to `"always"` or `"never"`:

```python
# docs: skip
import batcher.config

with batcher.config.option_context("distributed.mode", "always"):
    scaler = StandardScaler(["amount"]).fit(train)
```

The learned state is the same either way, up to floating-point reassociation in the last bits of a mean or variance, because every fit is a mergeable aggregate. [`tests/integration/test_ml_preprocessors_distributed.py`](https://github.com/stephenoffer/batcher/blob/main/tests/integration/test_ml_preprocessors_distributed.py) checks that on four workers for a representative set of preprocessors.

{py:class}`Chain <batcher.ml.preprocessors.Chain>` needs one more decision. With the default `cache=True`, it collects the whole training set to the driver once and fits every step against that in-memory copy, which saves a source scan per step but holds the data in driver memory and runs the later fits single-node. For a training set that is large or lives on a cluster, pass `cache=False`, so each step's fit is its own aggregate over the source.

## Where results differ from scikit-learn

Most preprocessors match scikit-learn to floating-point precision on the same data. The ones below differ on purpose, usually because the scikit-learn behaviour needs a per-row search or a second pass that the expression language can't express cheaply:

| Preprocessor | Difference |
| --- | --- |
| `QuantileTransformer` | A step function that reports each step's midpoint, so outputs run from `0.5 / n_quantiles` to `1 - 0.5 / n_quantiles`. scikit-learn interpolates between quantiles and maps the training extremes to exactly 0 and 1. |
| `KBinsDiscretizer(strategy="quantile")` | Edges come from a mergeable quantile sketch, so a value near an edge can land one bin from scikit-learn's answer. `strategy="uniform"` is exact. |
| `HashingVectorizer` | The index is a 64-bit FNV-1a hash of the term. scikit-learn uses signed 32-bit MurmurHash3 with alternating signs, so indices and signs differ. |
| `PowerTransformer`, `BoxCoxTransformer` | Lambda is found by a coarse grid over `[-2, 2]` plus two zoomed grids, within about 1e-4 of scikit-learn's Brent optimizer. An optimum far outside that bracket gets a boundary value. |
| `KNeighborsClassifier`, `KNeighborsRegressor`, `KNNImputer` | Every reference row tied with the k-th nearest distance is a neighbour, so a row can use more than `k`. A `k` larger than the reference set uses every row rather than raising. |

No preprocessor has an `inverse_transform`. To recover original units, keep the source column beside the transformed one, or apply the inverse yourself from the learned state, such as `mean_` and `scale_` on a scaler.

Saving is exact for every preprocessor that holds only data. `FunctionTransformer`, `Tokenizer`, and `RFE` take a Python callable, which has no JSON form, so `save` and `to_dict` raise `PlanError` for them rather than write a file that can't be loaded. `pickle` works for them when the callable is importable.

## The rest of this section

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Scaling and distributions
:link: /ml/preparing/preprocessors/scaling
:link-type: doc
Scalers, the row normalizer, and the distribution reshapers.
:::

:::{grid-item-card} {octicon}`hash;1.1em` Encoding and imputation
:link: /ml/preparing/preprocessors/encoding
:link-type: doc
Categorical encoders, missing-value imputation, and binning.
:::

:::{grid-item-card} {octicon}`typography;1.1em` Text vectorization
:link: /ml/preparing/preprocessors/text-vectorization
:link-type: doc
Bag of words, TF-IDF, n-grams, and hashing vectorizers.
:::

:::{grid-item-card} {octicon}`plus-circle;1.1em` Generating features
:link: /ml/preparing/preprocessors/feature-generation
:link-type: doc
Timestamps, text statistics, history, PCA, and assembly.
:::

:::{grid-item-card} {octicon}`checklist;1.1em` Feature selection
:link: /ml/preparing/preprocessors/feature-selection
:link-type: doc
Univariate filters, correlated-column removal, and recursive elimination.
:::

:::{grid-item-card} {octicon}`link;1.1em` Chaining and persisting
:link: /ml/preparing/preprocessors/pipelines
:link-type: doc
{py:class}`Chain <batcher.ml.preprocessors.Chain>`, fitting a whole pipeline, and saving the fitted state.
:::

:::{grid-item-card} {octicon}`duplicate;1.1em` Deduplication and matching
:link: /ml/preparing/preprocessors/deduplication
:link-type: doc
Fuzzy dedup with MinHash, and {py:meth}`similarity_join <batcher.api.dataset.ml.DatasetML.similarity_join>` on embeddings.
:::
::::

## See also

- {doc}`Feature engineering tutorial </getting-started/tutorials/ml/feature-engineering>`: the full workflow
  from raw table to model-ready matrix, end to end, with `Chain`.
- {doc}`PyTorch integration </ml/inference/pytorch>`: hand the assembled features to a training loop.
- {doc}`ML API reference </api/models/ml>`: the complete `Preprocessor` surface.

```{toctree}
:hidden:

scaling
encoding
text-vectorization
feature-generation
feature-selection
pipelines
deduplication
```
