# Synthetic data generation

Build test datasets in memory with plain Python and {py:obj}`bt.from_pydict <batcher.from_pydict>`. This is the
simplest way to produce inputs for trying out a pipeline at a chosen size and shape.
Everything here runs as written.

:::{note}
**What you'll build.** Four generated datasets: a fixed one, a seeded random one, a numpy
one, and a pair of joinable tables. `pip install batcher-engine` covers all but the numpy
section, which needs `numpy`.
:::

:::{tip}
Seed the generator. `random.seed(0)` or `np.random.default_rng(0)` is the difference between
a test that fails reproducibly and a test that fails on Tuesdays. Every example below is
seeded for exactly that reason.
:::

## A small fixed dataset

{py:obj}`bt.from_pydict <batcher.from_pydict>` takes a column-oriented dict, so generate each column as a list.

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

## Which generator to reach for

| You want | Use |
|---|---|
| A handful of rows with exact values | A literal dict, as in the first section |
| Arbitrary size, no dependency beyond the standard library | `random`, seeded |
| Arbitrary size, fast, and numeric | `numpy`, with `default_rng(seed)` |
| To exercise a join | Two tables sharing a key, as above |
| A file on disk instead of memory | Generate, then `ds.write.parquet(path)` |

:::{warning}
Generated data is uniform, and real data is not. A pipeline that is fast on
`random.choice(["north", "south", "east", "west"])` may be slow on a production key with one
value in ten million rows and a million values with one row each. Skew is the thing your
synthetic corpus will not reproduce unless you build it in on purpose.
:::

## What you learned

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Your first pipeline
:link: first-pipeline
:link-type: doc
The full transform, aggregate, sort flow over what you just built.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Batch inference
:link: batch-inference
:link-type: doc
Run a model over the data you generate.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Optimizing a slow query
:link: optimizing-a-slow-query
:link-type: doc
Now make a 200,000-row query tell you why it is slow.
:::
::::

## See also

- [Joins](../user-guide/joins.md): the operator the last section sets up.
- [Writing data](../user-guide/writing-data.md): turning a generated dataset into files.
- [Data quality](../user-guide/data-quality.md): validating a corpus, synthetic or not.
- [Dataset API](../api/dataset.md): `from_pydict`, `join`, and the rest.
