# Synthetic data generation

Build test datasets in memory with {py:func}`bt.range <batcher.range>`, {py:obj}`bt.from_pydict <batcher.from_pydict>`, and plain Python, at whatever size and shape a pipeline needs. Everything here runs on `pip install batcher-engine`, plus `numpy` for the numpy section.

:::{tip}
Seed every generator (`random.seed(0)`, `np.random.default_rng(0)`) so a failing test fails the same way every time.
:::

## Generate in the engine

{py:func}`bt.range <batcher.range>` builds an integer column in the engine, and expressions derive the rest, with no Python lists at all:

```python
import batcher as bt

gen = bt.range(6).with_columns(bucket=bt.col("value") % 3, amount=bt.col("value") * 1.5)
print(gen.to_pydict())
# {'value': [0, 1, 2, 3, 4, 5], 'bucket': [0, 1, 2, 0, 1, 2], 'amount': [0.0, 1.5, 3.0, 4.5, 6.0, 7.5]}
```

It takes `start`, `stop`, `step`, and a column `name`:

```python
print(bt.range(0, 10, 3, name="id").to_pydict())
# {'id': [0, 3, 6, 9]}
```

## A small fixed dataset

{py:obj}`bt.from_pydict <batcher.from_pydict>` takes a column-oriented dict:

```python
ds = bt.from_pydict(
    {
        "id": list(range(1, 6)),
        "category": ["a", "b", "a", "b", "a"],
        "value": [10, 20, 30, 40, 50],
    }
)
print(ds.to_pydict())
# {'id': [1, 2, 3, 4, 5], 'category': ['a', 'b', 'a', 'b', 'a'], 'value': [10, 20, 30, 40, 50]}
```

## Random columns

The standard library `random` module builds columns of any size:

```python
import random

random.seed(0)
n = 1000
categories = ["north", "south", "east", "west"]

events = bt.from_pydict(
    {
        "id": list(range(n)),
        "region": [random.choice(categories) for _ in range(n)],
        "amount": [round(random.uniform(1.0, 100.0), 2) for _ in range(n)],
    }
)
print(events.count())
# 1000
```

Query it like any other dataset:

```python
by_region = events.group_by("region").agg(total=bt.col("amount").sum(), n=bt.count()).sort("region")
print(by_region.to_pydict()["region"])
# ['east', 'north', 'south', 'west']
```

## numpy columns

With numpy, generation is vectorized. Convert arrays to lists for {py:func}`from_pydict <batcher.from_pydict>`:

```python
import numpy as np

rng = np.random.default_rng(0)
n = 1000

numeric = bt.from_pydict(
    {
        "id": np.arange(n).tolist(),
        "x": rng.normal(0.0, 1.0, n).tolist(),
        "y": rng.integers(0, 10, n).tolist(),
    }
)
print(numeric.columns)
# ['id', 'x', 'y']
```

## Joinable tables

To exercise joins, generate a fact table and a dimension table that share a key:

```python
random.seed(1)
regions = ["west", "east"]

facts = bt.from_pydict(
    {
        "id": list(range(20)),
        "region": [random.choice(regions) for _ in range(20)],
    }
)
dim = bt.from_pydict({"region": ["west", "east"], "label": ["W", "E"]})

joined = facts.join(dim, on="region", how="inner")
print(sorted(set(joined.to_pydict()["label"])))
# ['E', 'W']
```

## Which generator to reach for

Match your case to a row:

| You want | Use |
|---|---|
| Any size, generated in the engine | {py:func}`bt.range <batcher.range>` plus expressions |
| A handful of rows with exact values | A literal dict |
| Arbitrary size, no dependency beyond the standard library | `random`, seeded |
| Arbitrary size, fast, and numeric | `numpy`, with `default_rng(seed)` |
| To exercise a join | Two tables sharing a key, as above |
| A file on disk instead of memory | Generate, then {py:meth}`ds.write.parquet(path) <batcher.api.io_namespace.writer.Writer.parquet>` |

## Skewed keys

Production keys are rarely uniform. Build the skew in on purpose to test a pipeline against a hot key:

```python
skewed = bt.range(1000).with_columns(
    key=bt.when(bt.col("value") % 100 == 0).then(bt.lit("rare")).otherwise(bt.lit("hot"))
)
print(skewed.group_by("key").agg(n=bt.count()).sort("key").to_pydict())
# {'key': ['hot', 'rare'], 'n': [990, 10]}
```

## Where to go next

Now put the generated data through something:

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Your first pipeline
:link: /getting-started/tutorials/foundations/first-pipeline
:link-type: doc
The full transform, aggregate, sort flow over what you just built.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Batch inference
:link: /getting-started/tutorials/ml/batch-inference
:link-type: doc
Run a model over the data you generate.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Optimizing a slow query
:link: /getting-started/tutorials/foundations/optimizing-a-slow-query
:link-type: doc
Now make a 200,000-row query tell you why it is slow.
:::
::::

## See also

- {doc}`Joins </user-guide/analyze/joins>`: the operator the last section sets up.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: turning a generated dataset into files.
- {doc}`Data quality </user-guide/trust/data-quality>`: validating a corpus, synthetic or not.
- {doc}`Dataset API </api/relational/dataset>`: `from_pydict`, `join`, and the rest.
