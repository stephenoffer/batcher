# Migrating from Ray Data

This page maps Ray Data's `Dataset` onto Batcher's and explains the one architectural difference that changes how you tune a job. Most of the vocabulary carries over, because both libraries are lazy Python APIs over Arrow batches that scale from a laptop to a Ray cluster.

## Bulk data leaves the object store

In Ray Data, a `Dataset` is a collection of blocks held as Ray objects, and every shuffle, split, and repartition moves those blocks through the Ray object store. That is why object store memory is the number you end up tuning, and why spilling shows up as the dominant cost on a large job.

Batcher uses Ray for *scheduling only*. Ray tasks carry control-plane metadata, and the bulk Arrow batches move directly between workers over Arrow Flight with credit-based flow control. Nothing large is written to the object store, so there is no object store size to tune and no spill storm to diagnose.

Two practical consequences follow. You don't call `ray.init()` or size an object store before running a job. And distribution is a property of a single terminal call, not of the dataset, so the same pipeline runs single-node or distributed without a rewrite.

```python
import batcher as bt
from batcher import col

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC", "SF"], "amount": [10, 20, 30, 40]})
out = ds.filter(col("amount") > 10).group_by("city").agg(total=col("amount").sum())
print(out.sort("city").to_pydict())
# {'city': ['LA', 'NYC', 'SF'], 'total': [20, 30, 40]}
```

The same plan runs across a cluster by passing `distributed=True` to the terminal call:

```python
# docs: skip
out.collect(distributed=True, num_workers=8)
```

## Relational verbs

Most Ray Data verbs have a Batcher spelling, and the column verbs take SQL names: `select_columns` is `select`, `drop_columns` is `drop`, and `groupby` is `group_by`. {doc}`ray-data/dataset` lists every `Dataset` method with its Batcher spelling and status. Read the status before renaming a call, because several verbs share a name and differ in behavior. `write_parquet` appends by default in Ray Data and overwrites by default here, and `random_shuffle` draws a new permutation on every execution where `ds.shuffle(seed=0)` returns the same one every run.

Prefer an expression over a callback wherever Ray Data accepts either. `ds.filter(col("amount") > 10)` lowers into the engine and is pushed down to the scan. `ds.filter(lambda r: r["amount"] > 10)` can't be. This is the single largest performance difference in a ported script.

## Splitting

Ray Data's positional splits carry the same names and the same semantics here, including how they treat an index past the end and a repeated index.

```python
first, middle, last = bt.range(0, 10).split_at_indices([2, 5])
print([first.to_pydict()["value"], middle.to_pydict()["value"], last.to_pydict()["value"]])
# [[0, 1], [2, 3, 4], [5, 6, 7, 8, 9]]
```

```python
a, b, c = bt.range(0, 10).split_proportionately([0.2, 0.5])
print([a.count(), b.count(), c.count()])
# [2, 5, 3]
```

They differ in one way, and it works in your favor. Ray Data materializes the dataset to split it. Batcher doesn't. Each part is a lazy plan, so a pipeline that consumes one part never computes the others. The trade is that collecting every part reads the input once per part, so call `ds.cache()` first when the source is expensive and you want all of them.

`ds.split(n)` keeps its name and its `equal=`, and takes the order a position is counted in as `order_by`. Ray Data counts position in block order, and a Batcher dataset has none, so number the rows where you read them with `with_row_index("i")` and pass `order_by="i"`. `ds.zip(other)` works the same way, pairing rows by position under `order_by` and refusing inputs whose row counts differ. `streaming_split` becomes `batcher.ml.streaming_split(dataset, world_size, rank=)`, which yields per-rank torch batches rather than n iterators. {doc}`ray-data/dataset` lists both differences.

For train and test sets, prefer {py:meth}`ds.ml.train_test_split(...) <batcher.api.dataset.ml.DatasetML.train_test_split>`. It assigns each row by a hash of its own values, not by position, so the split stays identical however the data is partitioned.

## Batch inference and UDFs

`map_batches` is spelled the same and keeps the same contract: your function receives a whole batch, never a row.

```python
import pyarrow as pa


def double(batch: pa.RecordBatch) -> pa.RecordBatch:
    doubled = pa.array([v * 2 for v in batch.column("amount").to_pylist()])
    return batch.set_column(batch.schema.get_field_index("amount"), "amount", doubled)


print(ds.map_batches(double).to_pydict()["amount"])
# [20, 40, 60, 80]
```

A Ray Data class-based UDF, which exists so the model loads once per actor rather than once per batch, ports to passing the class itself:

```python
# docs: skip
ds.map_batches(Classifier, concurrency=4, num_gpus=1, batch_size=64)
```

For a model rather than arbitrary code, {py:meth}`ds.ml.infer(...) <batcher.api.dataset.ml.DatasetML.infer>` is the shorter path. On a cluster, the actor pool stays warm for the session, so the same model loads once rather than once per execution.

## Reading and writing

Readers live on `bt.read` and writers on `ds.write`, not as module-level and method-level functions. `ray.data.read_parquet(path)` becomes `bt.read.parquet(path)`, and `ds.write_parquet(path)` becomes `ds.write.parquet(path)`. The defaults aren't all the same. {doc}`ray-data/io` lists every reader with what differs, such as `read_images` decoding by default in Ray Data and not here, and {doc}`ray-data/dataset` lists the writers.

## Consuming results

Ray Data distinguishes eager `take*` methods from lazy ones. Batcher stays lazy until a terminal call, so an eager name becomes a `limit` plus a terminal, such as `ds.take(n)` becoming `ds.limit(n).to_pylist()`. The iterators port too, with different defaults: Ray Data's `iter_batches` yields dicts of NumPy arrays, 256 rows at a time, where Batcher's yields Arrow record batches of the engine's size. {doc}`ray-data/dataset` gives the Batcher spelling and the difference for each of `take`, `take_all`, `take_batch`, `iter_batches`, `iter_rows`, `iter_torch_batches`, and `materialize`.

```python
print(ds.select("city", "amount").limit(2).to_pylist())
# [{'city': 'NYC', 'amount': 10}, {'city': 'LA', 'amount': 20}]
```

## What has no equivalent, on purpose

The block and object-ref surface is absent because the data plane doesn't use it. `get_internal_block_refs`, `to_arrow_refs`, `to_pandas_refs`, and `num_blocks` have nothing to return, because bulk Arrow never becomes a Ray object. Stream with `ds.iter_batches()` instead, and read what execution actually did from `ds.stats()`.

Datasets also carry no name, no id, and no per-dataset context. Configuration is process-wide through `bt.config` and `bt.set_config(...)`, and `ds.explain()` labels the plan.

Type one of these names and the error tells you what to use instead. It isn't a bare `AttributeError`:

```python
try:
    ds.random_shuffle()
except AttributeError as e:
    print(e)
# Dataset has no attribute 'random_shuffle'. Spelled ds.shuffle(seed=0) here (a full, seeded shuffle).
```

## Verifying the port

Run both versions and compare results rather than plans. {py:meth}`equals <batcher.Dataset.equals>` ignores row order by default, which is what you want unless the query has an `ORDER BY`.

```python
ported = ds.filter(col("amount") > 10).select("city", "amount")
expected = bt.from_pydict({"city": ["LA", "NYC", "SF"], "amount": [20, 30, 40]})
print(ported.equals(expected))
# True
```

Then check that the distributed result matches the single-node one, which is the property Batcher holds itself to.

```python
# docs: skip
assert ported.collect(distributed=True).equals(ported.collect())
```

## See also

- {doc}`ray-data/index`: every Ray Data name, with its Batcher spelling, its status, and what differs.
- {doc}`Running on Ray </integrations/compute/ray>`: cluster setup, worker counts, and the Flight shuffle.
- {doc}`Sampling and splitting </user-guide/transform/rows/sampling>`: the positional and hash-based splits side by side.
- {doc}`Batch inference and ML </getting-started/migration/ml-pipelines>`: models over batches, GPU pools, and the training feed.
- {doc}`Differences and verification </getting-started/migration/differences>`: what Batcher deliberately does not have.
- {doc}`Dataset API </api/relational/dataset>`: the reference for every verb named here.
