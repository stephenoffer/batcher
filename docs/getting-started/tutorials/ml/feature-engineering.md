# Feature engineering with preprocessors

Build a model-ready feature matrix from a raw table with Batcher's scikit-learn-style {doc}`preprocessors </ml/preparing/preprocessors/index>`. You impute, scale, encode, and bin, then compose the lot with `Chain`, fitting on the training split and replaying the *same* learned statistics on the test split.

A preprocessor is an object because `fit` learns state (a mean, a category set, bin edges) that is reused on held-out data. `fit` runs one mergeable aggregate in the engine, and `transform` is a lazy column rewrite. Every block runs on `pip install batcher-engine` except the closing training loop, which needs `torch`.

:::{dropdown} The same workflow as plain expressions
For the expression form (broadcast aggregates, `when/then` bucketing, one-hot via boolean casts), see [`examples/feature_engineering.py`](https://github.com/stephenoffer/batcher/blob/main/examples/feature_engineering.py). This page as a runnable script is [`examples/preprocessors.py`](https://github.com/stephenoffer/batcher/blob/main/examples/preprocessors.py).
:::

## The raw data

Start with a small customer table split into train and test. Both have a missing `age`, and the test set carries a `plan` value, `"student"`, that never appears in training.

```python
import batcher as bt

train = bt.from_pydict(
    {
        "user_id": [1, 2, 3, 4, 5, 6],
        "age": [25.0, 40.0, None, 33.0, 52.0, 19.0],  # a null to impute
        "tenure": [2.0, 8.0, 5.0, 12.0, 20.0, 1.0],
        "plan": ["free", "pro", "free", "enterprise", "pro", "free"],
        "spend": [10.0, 55.0, 12.0, 90.0, 70.0, 5.0],
        "churned": ["yes", "no", "yes", "no", "no", "yes"],  # the target label
    }
)

test = bt.from_pydict(
    {
        "user_id": [7, 8],
        "age": [None, 45.0],
        "tenure": [3.0, 15.0],
        "plan": ["pro", "student"],  # "student" is unseen at fit time
        "spend": [20.0, 80.0],
        "churned": ["no", "yes"],
    }
)

print(train.columns)
# ['user_id', 'age', 'tenure', 'plan', 'spend', 'churned']
```

In practice, {py:obj}`ds.ml.train_test_split <batcher.api.dataset.ml.DatasetML.train_test_split>` produces the two splits by a reproducible hash of each row:

```python
everyone = bt.from_pydict({"user_id": list(range(64))})
tr, te = everyone.ml.train_test_split(test_size=0.25, seed=7, key="user_id")
print(tr.count(), te.count())
# 44 20
```

## Why you fit on train, never on test

`fit` reads the data to learn a statistic, so each split learns a different one:

```python
from batcher.ml.preprocessors import StandardScaler

on_train = StandardScaler(["tenure"]).fit(train)
on_test = StandardScaler(["tenure"]).fit(test)
print(round(on_train.mean_["tenure"], 3), round(on_test.mean_["tenure"], 3))
# 8.0 9.0
```

:::{important}
Call `fit` (or `fit_transform`) on `train` only, and put the held-out split through `transform`, so it inherits the training statistics. Every step below follows this rule.
:::

## Impute missing values

{py:obj}`SimpleImputer <batcher.ml.preprocessors.SimpleImputer>` learns a per-column fill value (here the median of `age`) and fills nulls with it, on both splits:

```python
from batcher.ml.preprocessors import SimpleImputer

imputer = SimpleImputer(["age"], strategy="median").fit(train)
print(imputer.statistics_)
# {'age': 33.0}

print(imputer.transform(train).to_pydict()["age"])
# [25.0, 40.0, 33.0, 33.0, 52.0, 19.0]
print(imputer.transform(test).to_pydict()["age"])
# [33.0, 45.0]
```


## Scale the numeric columns

{py:obj}`StandardScaler <batcher.ml.preprocessors.StandardScaler>` maps each column to `(x - mean) / std`. Fit it on the *imputed* training data:

```python
imputed_train = imputer.transform(train)

scaler = StandardScaler(["age", "tenure"]).fit(imputed_train)
scaled_train = scaler.transform(imputed_train)
print([round(v, 3) for v in scaled_train.to_pydict()["age"]])
# [-0.822, 0.601, -0.063, -0.063, 1.738, -1.391]
```

{py:class}`MinMaxScaler <batcher.ml.preprocessors.MinMaxScaler>`, {py:class}`MaxAbsScaler <batcher.ml.preprocessors.MaxAbsScaler>`, and {py:class}`RobustScaler <batcher.ml.preprocessors.RobustScaler>` are drop-in alternatives:

```python
from batcher.ml.preprocessors import MinMaxScaler

minmax = MinMaxScaler(["tenure"]).fit(train)
print([round(v, 2) for v in minmax.transform(train).to_pydict()["tenure"]])
# [0.05, 0.37, 0.21, 0.58, 1.0, 0.0]
```

## Encode the categorical column

{py:obj}`OneHotEncoder <batcher.ml.preprocessors.OneHotEncoder>` learns the category set and emits one `{column}_{category}` 0/1 indicator per category:

```python
from batcher.ml.preprocessors import OneHotEncoder

encoder = OneHotEncoder(["plan"]).fit(train)
print(encoder.categories_)
# {'plan': ['enterprise', 'free', 'pro']}

encoded_test = encoder.transform(test).to_pydict()
print(encoded_test["plan_enterprise"], encoded_test["plan_free"], encoded_test["plan_pro"])
# [0, 0] [0, 0] [1, 0]
```

The second test row's `"student"` plan, unseen at fit, encodes as all zeros. For an ordinal column use
{py:obj}`OrdinalEncoder <batcher.ml.preprocessors.OrdinalEncoder>`; for a list-valued
tag column use {py:obj}`MultiHotEncoder <batcher.ml.preprocessors.MultiHotEncoder>`.

## Bin a continuous column

{py:obj}`KBinsDiscretizer <batcher.ml.preprocessors.KBinsDiscretizer>` turns a continuous column into a bin index `0..n_bins-1`, with `"uniform"` (equal-width) or `"quantile"` (equal-count) edges:

```python
from batcher.ml.preprocessors import KBinsDiscretizer

binner = KBinsDiscretizer(["spend"], n_bins=3, strategy="uniform").fit(train)
print([round(e, 2) for e in binner.edges_["spend"]])
# [33.33, 61.67]

print(binner.transform(train).to_pydict()["spend"])
# [0, 1, 0, 2, 2, 0]
print(binner.transform(test).to_pydict()["spend"])
# [0, 2]
```

## Encode the target label

{py:obj}`LabelEncoder <batcher.ml.preprocessors.LabelEncoder>` maps a target's sorted classes to `0..k-1`:

```python
from batcher.ml.preprocessors import LabelEncoder

target = LabelEncoder("churned").fit(train)
print(target.classes_)
# ['no', 'yes']
print(target.transform(train).to_pydict()["churned"])
# [1, 0, 1, 0, 0, 1]
```

## What each step learns

Each preprocessor learns one kind of state at `fit` and replays it at `transform`:

| Preprocessor | Learns at `fit` | Does at `transform` |
|---|---|---|
| `SimpleImputer` | A per-column fill value (here the median) | Replaces nulls with it |
| `StandardScaler` | The mean and standard deviation | `(x - mean) / std` |
| `OneHotEncoder` | The category set | One 0/1 indicator per learned category |
| `KBinsDiscretizer` | The bin edges | An integer bin index |
| `LabelEncoder` | The sorted classes | Maps the target to `0..k-1` |

## Compose the whole pipeline with `Chain`

{py:obj}`Chain <batcher.ml.preprocessors.Chain>` writes the fit-then-replay loop once. `fit` threads each step's output into the next, and `transform` replays the fitted steps in order. A `Chain` is itself a {py:class}`Preprocessor <batcher.ml.preprocessors.Preprocessor>`, so it nests.

```python
from batcher.ml.preprocessors import Chain

pipeline = Chain(
    SimpleImputer(["age"], strategy="median"),
    StandardScaler(["age", "tenure"]),
    KBinsDiscretizer(["spend"], n_bins=3, strategy="uniform"),
    OneHotEncoder(["plan"]),
    LabelEncoder("churned"),
).fit(train)

print(pipeline)
# Chain(SimpleImputer, StandardScaler, KBinsDiscretizer, OneHotEncoder, LabelEncoder)

train_features = pipeline.transform(train)
test_features = pipeline.transform(test)
print(train_features.collect().column_names)
# ['user_id', 'age', 'tenure', 'spend', 'churned', 'plan_enterprise', 'plan_free', 'plan_pro']
```

`test_features` carries the training statistics. The steps stay introspectable:

```python
print(round(pipeline[1].mean_["age"], 3))
# 33.667
print([round(v, 3) for v in test_features.to_pydict()["age"]])
# [-0.063, 1.075]
```

## Assemble the feature vector

{py:obj}`Concatenator <batcher.ml.preprocessors.Concatenator>` stacks the feature columns into one list column for a training loop (`drop=True` removes the sources):

```python
from batcher.ml.preprocessors import Concatenator

feature_cols = ["age", "tenure", "spend", "plan_enterprise", "plan_free", "plan_pro"]
assembler = Concatenator(feature_cols, output_column="features", drop=True)
model_ready = assembler.fit_transform(train_features)

print(model_ready.collect().column_names)
# ['user_id', 'churned', 'features']
print([round(v, 3) for v in model_ready.to_pydict()["features"][0]])
# [-0.822, -0.922, 0.0, 0.0, 1.0, 0.0]
```

Apply the same assembler to `test_features` for the held-out matrix.

## Hand the matrix to a training loop

{py:obj}`ds.ml.iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` streams the matrix to PyTorch in bounded memory, one `{column: tensor}` batch at a time:

```python
# docs: skip
import torch

model = torch.nn.Linear(6, 1)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.BCEWithLogitsLoss()

for batch in model_ready.ml.iter_torch_batches(batch_size=256, columns=["features", "churned"]):
    logits = model(batch["features"].float())
    loss = loss_fn(logits.squeeze(1), batch["churned"].float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The training and evaluation matrices share every learned statistic, because one fitted `Chain` produced both.

## Where to go next

More preprocessors, the ranks that consume this matrix, or the training-loop side of it:

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.1em` Preprocessors guide
:link: /ml/preparing/preprocessors/index
:link-type: doc
Every preprocessor there is, plus splitting, fuzzy dedup, and where the transforms run.
:::

:::{grid-item-card} {octicon}`git-branch;1.1em` Distributed training pipeline
:link: /getting-started/tutorials/ml/distributed-training-pipeline
:link-type: doc
Feed these features to DDP ranks, balanced and resumable.
:::

:::{grid-item-card} {octicon}`plug;1.1em` PyTorch integration
:link: /ml/inference/pytorch
:link-type: doc
The training-loop side: device transfer, prefetch, zero-copy.
:::
::::

## See also

- {doc}`Expressions </user-guide/transform/columns/expressions>`: the same feature work said as column math,
  which is what these objects lower to.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: the mergeable pass every `fit` runs.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why a `fit` gives the same answer
  on one core and on a cluster.
- {doc}`ML API reference </api/models/ml>`: the full `Preprocessor` surface.
- {doc}`Feature pipeline recipe </cookbook/ml/pipelines/features/feature-pipeline>` and
  {doc}`train/test split recipe </cookbook/ml/pipelines/features/train-test-split>`: the short versions.
- [`examples/preprocessors.py`](https://github.com/stephenoffer/batcher/blob/main/examples/preprocessors.py): this workflow as a runnable, asserted script.
