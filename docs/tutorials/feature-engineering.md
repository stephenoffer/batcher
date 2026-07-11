# Feature engineering with preprocessors

This tutorial builds a **model-ready feature matrix** from a raw table, end to end,
using Batcher's scikit-learn-style [preprocessors](../ml/preprocessors.md).
You will fit on a training split, transform the training **and** test splits with the
*same* learned statistics (the discipline that keeps a model honest), impute missing
values, scale numerics, encode categoricals, bin a continuous column, and compose the
whole pipeline with `Chain`. Every block runs as written; the final training-loop
block needs PyTorch and so is shown but not executed.

A preprocessor is an object, not a `Dataset` method, for one reason: `fit` learns
state (a mean, a category set, bin edges) that has to be *reused* on held-out data.
`fit` runs one mergeable aggregate in the engine; `transform` is a lazy column rewrite.
Nothing here touches a row in Python.

If you want the same workflow as expressions instead of objects — broadcast
aggregates, `when/then` bucketing, one-hot via boolean casts — see
`examples/feature_engineering.py`. This tutorial is the preprocessor-object counterpart;
the runnable script version is `examples/preprocessors.py`.

## The raw data

Start with a small customer table split into a training set and a held-out test set.
The training set has a missing `age`; the test set has a missing `age` **and** a `plan`
value (`"student"`) that never appears in training — both are situations a real feature
pipeline has to survive.

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

In practice you would produce `train` / `test` with
{py:obj}`ds.ml.train_test_split <batcher.api.dataset.ml.DatasetML.train_test_split>`,
which assigns each row by a reproducible hash of its own content. Here we write the two
splits out explicitly so the numbers below are deterministic.

## Why you fit on train, never on test

`fit` *executes* — it reads the data to learn a statistic. If it reads the test rows,
their distribution leaks into your features and every offline metric becomes optimistic.
The learned state is what differs: fit the same scaler on each split and the mean and
scale it learns are not the same number.

```python
from batcher.ml.preprocessors import StandardScaler

on_train = StandardScaler(["tenure"]).fit(train)
on_test = StandardScaler(["tenure"]).fit(test)
print(round(on_train.mean_["tenure"], 3), round(on_test.mean_["tenure"], 3))
# 8.0 9.0
```

The rule that follows from this: call `fit` (or `fit_transform`) on `train` **only**,
then `transform` — never `fit_transform` — the held-out split, so it inherits the
training statistics. Every step below does exactly that.

## Impute missing values

{py:obj}`SimpleImputer <batcher.ml.preprocessors.SimpleImputer>` learns a per-column
fill value in `fit` (here the median of `age`) and replaces nulls with it via a
`coalesce` in `transform`. The fitted value lives on the object, so the test set's null
is filled with the **training** median, not its own.

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

The test row with a missing age became `33.0` — the training median — even though the
test set never saw it.

## Scale the numeric columns

{py:obj}`StandardScaler <batcher.ml.preprocessors.StandardScaler>` standardizes each
column to zero mean and unit variance: `(x - mean) / std`. Fit it on the **imputed**
training data (so the mean it learns already reflects the fill), then transform both
splits.

```python
imputed_train = imputer.transform(train)

scaler = StandardScaler(["age", "tenure"]).fit(imputed_train)
scaled_train = scaler.transform(imputed_train)
print([round(v, 3) for v in scaled_train.to_pydict()["age"]])
# [-0.822, 0.601, -0.063, -0.063, 1.738, -1.391]
```

The two imputed ages (`33.0`) both land on the same standardized value, just below the
mean. `MinMaxScaler`, `MaxAbsScaler`, and `RobustScaler` are drop-in alternatives with
the same `fit`/`transform` contract.

## Encode the categorical column

{py:obj}`OneHotEncoder <batcher.ml.preprocessors.OneHotEncoder>` learns the category set
in `fit` and, in `transform`, drops the source column and emits one `{column}_{category}`
0/1 indicator per learned category. A value unseen at fit time produces **all-zero**
indicators — which is exactly why the encoder is fitted once, on train.

```python
from batcher.ml.preprocessors import OneHotEncoder

encoder = OneHotEncoder(["plan"]).fit(train)
print(encoder.categories_)
# {'plan': ['enterprise', 'free', 'pro']}

encoded_test = encoder.transform(test).to_pydict()
print(encoded_test["plan_enterprise"], encoded_test["plan_free"], encoded_test["plan_pro"])
# [0, 0] [0, 0] [1, 0]
```

The second test row (`plan="student"`, unseen at fit) is all zeros across the three
indicators — it encodes deterministically instead of shifting every column. For an
ordinal-target column use
{py:obj}`OrdinalEncoder <batcher.ml.preprocessors.OrdinalEncoder>`; for a list-valued
tag column use {py:obj}`MultiHotEncoder <batcher.ml.preprocessors.MultiHotEncoder>`.

## Bin a continuous column

{py:obj}`KBinsDiscretizer <batcher.ml.preprocessors.KBinsDiscretizer>` turns a
continuous column into an integer bin index `0..n_bins-1`. `strategy="uniform"` learns
equal-width edges from the min and max; `"quantile"` learns edges that give each bin
roughly equal counts.

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

{py:obj}`LabelEncoder <batcher.ml.preprocessors.LabelEncoder>` is the one-column
encoder for a target: it maps the sorted classes to `0..k-1`.

```python
from batcher.ml.preprocessors import LabelEncoder

target = LabelEncoder("churned").fit(train)
print(target.classes_)
# ['no', 'yes']
print(target.transform(train).to_pydict()["churned"])
# [1, 0, 1, 0, 0, 1]
```

## Compose the whole pipeline with `Chain`

Running those five steps by hand means fitting each on the previous step's output and
replaying them, in order, over every split — four or five chances to fit on the wrong
frame and leak test statistics without ever failing.
{py:obj}`Chain <batcher.ml.preprocessors.Chain>` is that loop written once: `fit` threads
each step's output into the next; `transform` replays the fitted steps in order. A
`Chain` is itself a `Preprocessor`, so it nests.

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

`fit(train)` learns every step's state on the training rows; `transform` is called on
both splits, so `test_features` carries the training median, mean, edges, and category
set — not its own. The steps stay introspectable (`pipeline[1].mean_`, `len(pipeline)`)
if you need to read a fitted statistic back.

```python
print(round(pipeline[1].mean_["age"], 3))
# 33.667
print([round(v, 3) for v in test_features.to_pydict()["age"]])
# [-0.063, 1.075]
```

## Assemble the feature vector

A training loop wants one tensor column, not many scalar columns.
{py:obj}`Concatenator <batcher.ml.preprocessors.Concatenator>` stacks the numeric
feature columns into a single list column (`drop=True` removes the sources), leaving the
id and the label alongside it.

```python
from batcher.ml.preprocessors import Concatenator

feature_cols = ["age", "tenure", "spend", "plan_enterprise", "plan_free", "plan_pro"]
assembler = Concatenator(feature_cols, output_column="features", drop=True)
model_ready = assembler.fit_transform(train_features)

print(model_ready.collect().column_names)
# ['user_id', 'churned', 'features']
print(model_ready.to_pydict()["features"][0])
# [-0.8217814036133171, -0.9221679352414079, 0.0, 0.0, 1.0, 0.0]
```

`Concatenator` is stateless, so `fit_transform` (or `fit` then `transform`) is all it
needs; apply the *same* assembler to `test_features` for the held-out matrix.

## Hand the matrix to a training loop

The assembled dataset is a lazy plan — the feature matrix is materialized only when a
terminal op runs.
{py:obj}`ds.ml.iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`
streams it to PyTorch in bounded memory, one `{column: tensor}` batch at a time, so it
scales past memory. This block needs `torch`, so it is shown but not executed.

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

Because `fit` ran on `train` only and `transform` replayed the *same* fitted `Chain` on
`test`, the training and evaluation matrices share every learned statistic — the whole
point of doing feature engineering with fitted objects rather than ad-hoc column math.

## Next steps

- [Preprocessors guide](../ml/preprocessors.md): every preprocessor, splitting, fuzzy
  dedup, and where they run.
- [ML API reference](../api/ml.md): the full `Preprocessor` surface.
- [PyTorch integration](../ml/pytorch.md): the training-loop side in detail.
- `examples/preprocessors.py`: this same workflow as a runnable, asserted script.
