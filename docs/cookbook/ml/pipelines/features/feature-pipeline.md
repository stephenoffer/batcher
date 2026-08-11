# Feature pipeline

A raw table becomes a model matrix through a handful of learned transforms: fill the
nulls, standardize the numbers, encode the categories, stack it into a vector. Every one
of them learns something from the data (a mean, a category set), and every one of them is a
leak waiting to happen if it learns from rows the model is supposed to be evaluated on.

The rule is one line long: **fit on train, transform everything.** Batcher's preprocessors
enforce the shape of that rule; they cannot enforce that you split first.

## Split before you look at anything

```python
import batcher as bt

raw = bt.from_pydict(
    {
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "age": [22.0, 35.0, None, 51.0, 44.0, 29.0, None, 63.0],
        "spend": [10.0, 120.0, 45.0, 300.0, 80.0, 25.0, 60.0, 900.0],
        "plan": ["free", "pro", "free", "team", "pro", "free", "pro", "team"],
        "churned": [0, 1, 0, 1, 0, 0, 1, 1],
    }
)

train, test = raw.ml.train_test_split(0.25, seed=5, key="user_id")
print(train.count(), test.count())
# 6 2
```

:::{tip}
`key="user_id"` hashes the identifier rather than the whole row. That is what keeps a row
in the same split after you recompute a feature. Hash every column (the default) and changing
one derived value re-throws every row into a different part, quietly invalidating the
comparison you were about to make. See {doc}`train/test split </cookbook/ml/pipelines/features/train-test-split>` for
what else that decision buys.
:::

## Chain the transforms

{py:class}`Chain <batcher.ml.preprocessors.Chain>` is the sklearn `Pipeline` equivalent. It fits each step on the **previous step's
output** and replays the fitted steps, in order, over any split. Sequencing this by hand
is where leaks appear: fit step *i* on data that steps *0..i-1* have not transformed and
the statistics no longer match what the model will see at serving time. Nothing fails. The
metric is wrong.

The classic order is impute → scale → encode.

```python
from batcher.ml import Chain, OneHotEncoder, SimpleImputer, StandardScaler

pipeline = Chain(
    SimpleImputer(["age"], strategy="median"),
    StandardScaler(["age", "spend"]),
    OneHotEncoder(["plan"]),
).fit(train)  # train only: the test rows are never seen

train_x = pipeline.transform(train)
test_x = pipeline.transform(test)
print(train_x.collect().column_names)
# ['user_id', 'age', 'spend', 'churned', 'plan_free', 'plan_pro', 'plan_team']
```

`fit` is the one place a preprocessor touches data: it runs a single mergeable aggregate
over the engine, so it distributes and spills like any other aggregation. `transform` is a
lazy {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` projection, with no Python per row and nothing materialized until a terminal op.

The fitted state is on the object, so you can read what it learned:

```python
imputer = pipeline[0]
print(imputer.statistics_)
# {'age': 44.0}
```

:::{warning}
That 44.0 is the *training* median, and it is the value that fills the test set's nulls
too. Refit on test and the two splits stop sharing a scale, which is the leak in its most
common form. Nothing raises. The metric is wrong, and it is wrong in the flattering
direction.
:::

## Assemble the vector

{py:class}`Concatenator <batcher.ml.preprocessors.Concatenator>` stacks the numeric columns into one list column: the "make a feature vector"
step before a training loop. A tensor column travels with its shape, so
{py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` hands the loop an `(n, d)` tensor with no reshape at the edge.

```python
from batcher.ml import Concatenator

features = ["age", "spend", "plan_free", "plan_pro", "plan_team"]
assembler = Concatenator(features, output_column="features", drop=True)

train_v = assembler.fit_transform(train_x).select("features", "churned")
test_v = assembler.transform(test_x).select("features", "churned")
print(train_v.collect().column("features").to_pylist()[0])
# [-1.4901975067523408, -0.6956089924572142, 1.0, 0.0, 0.0]
```

`Concatenator` is stateless, but it still follows the three-call contract (`fit` /
`transform` / `fit_transform`), so the pipeline reads the same whether a step learns
something or not. Calling `transform` before `fit` raises {py:exc}`PlanError <batcher.PlanError>` rather than silently
using an unfitted object.

## Hand it to the training loop

```python
# docs: skip
for batch in train_v.ml.iter_torch_batches(
    batch_size=256,
    device="auto",                  # CUDA / ROCm / XPU / MPS / CPU
    pin_memory=True,
    local_shuffle_buffer_size=8192,
):
    loss = loss_fn(model(batch["features"]), batch["churned"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The transforms are still lazy at this point. Nothing was materialized: the loader pulls
Arrow batches through the imputer, the scaler, the encoder, and the assembler as it goes,
in bounded memory, which is what lets the same code run over a corpus that does not fit in
RAM.

`iter_torch_batches` is one of several loaders, and which one you want is decided by how the
training runs, not by the features:

| Loader | Hands you | Reach for it when |
| --- | --- | --- |
| {py:meth}`ds.ml.iter_torch_batches(...) <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` | `{column: tensor}` dicts, placed on the device | training in a single process |
| {py:meth}`ds.ml.stream_loader(...) <batcher.api.dataset.ml.DatasetML.stream_loader>` | a torch `IterableDataset`, one shard per rank | DDP or FSDP over a bounded corpus |
| {py:func}`batcher.ml.streaming_split(...) <batcher.ml.streaming_split>` | one rank iterator per rank, from a single read | the source is unbounded, so there is no length to shard on |
| {py:func}`batcher.ml.to_torch_iterable(...) <batcher.ml.to_torch_iterable>` | a torch iterable over any batch iterator | you already have a batch iterator and want to keep it |

The full map, including the two ways to shard a stream wrongly, is in
{doc}`data loaders </ml/training/data-loaders>`.

## Derived features belong in expressions

Preprocessors handle the *learned* transforms. Everything else (ratios, log scaling, buckets,
flags) is an expression, and expressions run in the data plane where they cost nothing. Do this before the `Chain`, so the scaler sees the feature you actually train on.

```python
from batcher import col, when

enriched = raw.with_columns(
    spend_per_year=col("spend") / (col("age") + 1.0),
    tier=when(col("spend") > 100.0).then("high").otherwise("low"),
)
print(enriched.select("tier").to_pydict()["tier"])
# ['low', 'high', 'low', 'high', 'low', 'low', 'low', 'high']
```

Reach for a `map_batches` UDF only when an expression cannot say it. A Python function is
opaque to the optimizer: it cannot be pushed down, the engine cannot see which columns it
reads, and its cardinality has to be measured rather than estimated.

## Bucketing a skewed column

:::{dropdown} Quantile bins, learned on the training split

{py:class}`KBinsDiscretizer <batcher.ml.preprocessors.KBinsDiscretizer>` with `strategy="quantile"` learns the quantile edges so each bin holds
roughly equal counts. That is the right move for a heavy-tailed column such as spend, where
equal *width* bins put 95% of your rows in bin 0.

```python
from batcher.ml import KBinsDiscretizer

binner = KBinsDiscretizer(["spend"], n_bins=4, strategy="quantile").fit(train)
print(binner.transform(raw).select("spend").to_pydict()["spend"])
# [0, 3, 1, 3, 2, 1, 2, 3]
```

The edges come from the training split; test rows fall into the bins train defined. A row
above the largest training value lands in the top bin instead of inventing a fifth.
:::

## See also

- {doc}`Preprocessors </ml/preparing/preprocessors/index>`: every estimator, and what each `fit` learns.
- {doc}`Data loaders </ml/training/data-loaders>`: the loader table above, in full.
- {doc}`PyTorch </ml/inference/pytorch>` and the
  {doc}`PyTorch integration </integrations/compute/pytorch>`: the training loop on the other end.
- {doc}`Train/test split </cookbook/ml/pipelines/features/train-test-split>`: hash splits, group leakage, and time-based cuts.
- {doc}`Recommender features </cookbook/ml/pipelines/features/recommender-features>`: aggregate and window features over an
  event log.
- {doc}`ML API reference </api/models/ml>`: `Chain`, the estimators, and the loaders.
- {doc}`Training-ingest benchmarks </benchmarks/results/ai-and-gpu>`: what `iter_torch_batches`
  sustains, and against what.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: why a feature vector travels with its
  shape.
