# Sampling and splitting

Most sampling implementations draw from a random number generator per partition. That
is reproducible only if the partitioning is, which it is not: change the file layout,
the worker count, or run distributed instead of single-node, and you get a different
sample from the same data. Batcher assigns rows by a seeded hash of their *values*
instead, so a sample is a function of the data and the seed, and of nothing else.

## Setup

```python
import batcher as bt

ds = bt.range(0, 1000).with_columns(bucket=bt.col("value") % 4)
print(ds.columns)
# ['value', 'bucket']
```

## sample: a fraction or an exact count

The two forms are not interchangeable, and the difference is whether the operator has to
see the whole relation before it can emit anything.

::::{tab-set}
:::{tab-item} A fraction (streams)

`sample(fraction)` keeps each row whose hash falls under the fraction. No breaker, no
materialization, so it works on an unbounded source.

```python
half = ds.sample(0.5, seed=7)
print(half.count())
# 520
```

520, not 500. A hash-keyed fraction is binomial around `fraction * n`, not exact, which
is the same trade Spark's `sample` makes.

:::

:::{tab-item} An exact count (breaks)

`sample(n=...)` keeps the `n` smallest-hash rows, so it has to rank them all first.

```python
print(ds.sample(n=10, seed=7).count())
# 10
```

:::
::::

The same seed on the same data gives the same rows, every time, on any number of cores.

```python
a = ds.sample(0.1, seed=1).to_pydict()["value"]
b = ds.sample(0.1, seed=1).to_pydict()["value"]
print(a == b, len(a))
# True 107
```

With `seed=None` (the default) a fresh seed is baked in when the plan is *built*, not
when it runs, so the two `collect()` calls on one sampled dataset still agree with each
other. Pass a seed explicitly if the sample has to reproduce across processes.

## Sampling is not a shuffle

`sample(n=10)` gives you ten rows chosen by hash, which means the choice is stable but
the *order* is arbitrary. It is not "ten random rows re-drawn each call", and it is not
a permutation. If what you want is a random ordering, add a random column and sort by it.
`with_random(name, seed=)` is a deterministic per-row uniform draw.

```python
shuffled = ds.with_random("r", seed=3).sort("r").head(3)
print(shuffled.select("value").to_pydict())
# {'value': [91, 40, 731]}
```

That sort is a full breaker over the whole relation, so reach for it on the small side
of a pipeline, not before a 10 TB scan.

## Train/test splits

`train_test_split` is the split you want for modelling: the two parts are disjoint,
they cover every row, and neither materializes. Each is a row-wise filter, so both stay
lazy.

```python
train, test = ds.ml.train_test_split(0.2, seed=42, key="value")
print(train.count(), test.count(), train.count() + test.count())
# 821 179 1000
```

:::{warning}
Pass `key`. Without it the assignment hashes *every column*, so recomputing an unrelated
feature re-draws the split and rows silently migrate from train to test. That is the
classic leak, and nothing in the metrics will report it: the model simply scores better
than it should. Hashing a stable identifier instead keeps a row on the side it started on
however the other columns change.
:::

`random_split` is the n-way generalization.

```python
tr, val, te = ds.ml.random_split([0.7, 0.15, 0.15], seed=42, key="value")
print(tr.count(), val.count(), te.count())
# 728 149 123
```

Sizes are binomial around the requested fractions, for the same reason `sample(0.5)`
was not exactly 500. Disjointness and coverage are exact; the sizes are not.

## Stratified sampling

There is no `stratified=True` flag. Sample per stratum and union, which is explicit
about what the strata are and what fraction each one gets.

```python
parts = [
    ds.filter(bt.col("bucket") == b).sample(0.1, seed=11)
    for b in range(4)
]
stratified = parts[0].union(*parts[1:])
print(stratified.group_by("bucket").agg(n=bt.count()).sort("bucket").to_pydict())
# {'bucket': [0, 1, 2, 3], 'n': [24, 23, 16, 25]}
```

Use the same seed across strata and the sample stays reproducible as a whole.

## Sampling for a cheap estimate

:::{tip}
Sample when you want *rows*. Sketch when you want a *number*. The decision table:

| You want | Reach for | Why |
| --- | --- | --- |
| Rows to eyeball, or a dev fixture | `sample(fraction)` | streams, no breaker |
| An exact row count out | `sample(n=...)` | ranks by hash, so it breaks |
| Disjoint modelling splits | `ml.train_test_split` / `ml.random_split` | row-wise filters, both stay lazy |
| A random *ordering* | `with_random(...)` then `sort` | a full breaker; use it on the small side |
| A distinct count or a quantile | `approx_n_unique` / `approx_quantile` | one pass, mergeable, no sampling error to reason about |
:::

Sampling to *estimate* an aggregate is usually the wrong tool. The engine already has
exact and sketch-based answers that read the same data in one pass: `approx_n_unique`
runs a HyperLogLog (about 2% error) and `approx_quantile` a DDSketch. Both are mergeable,
so they give the same number single-node and distributed.

```python
print(ds.approx_n_unique("value"), ds.n_unique("value"))
# 993 1000

print(ds.approx_median("value"))
# 499.5
```

Sample when you want *rows* (to eyeball data, to build a dev fixture, to train on a
subset). Sketch when you want a *number*.

## See also

- [Aggregations](aggregations.md): the exact and approximate aggregate families.
- [Filtering](filtering.md): predicates, which is how a stratum is defined.
- [Preprocessors](../ml/preprocessors.md): fitting feature statistics on the train split
  only.
- [Cardinality estimation](../deep-dives/cardinality-estimation.md): the sketches behind
  `approx_n_unique`, and the error bounds they hold to.
- [Train/test split recipe](../examples/ml/train-test-split.md): the leak-free split on a
  real feature table.
- [A/B testing](../examples/analytics/ab-testing.md): hash-bucketed assignment, the same
  machinery pointed at an experiment.
- [Dataset API](../api/dataset.md): the `sample` and `with_random` reference.
