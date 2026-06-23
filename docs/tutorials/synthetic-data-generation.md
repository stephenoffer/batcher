# Synthetic data generation

Build test datasets in memory with plain Python and `bt.from_pydict`. This is the
simplest way to produce inputs for trying out a pipeline at a chosen size and shape.
Everything here runs as written.

## A small fixed dataset

`bt.from_pydict` takes a column-oriented dict, so generate each column as a list.

```python
import batcher as bt

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

Use the standard library `random` module to build columns of arbitrary size. Seed
it for reproducible data.

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

Run a real query against the generated data to confirm it is well formed:

```python
by_region = (
    events.group_by("region")
    .agg(total=bt.col("amount").sum(), n=bt.count())
    .sort("region")
)
print(by_region.to_pydict()["region"])
# ['east', 'north', 'south', 'west']
```

## numpy columns

When numpy is available, vectorized column generation is faster and reads cleanly.
Convert arrays to lists for `from_pydict`.

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

Generate a fact table and a small dimension table that share a key, to exercise
joins.

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

## Next steps

- [Your first pipeline](first-pipeline.md): the full transform-aggregate-sort flow.
- [Batch inference](batch-inference.md): run a model over the data you generate.
