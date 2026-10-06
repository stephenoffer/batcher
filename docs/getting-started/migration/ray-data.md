# Porting a Ray Data pipeline

This page maps Ray Data's `Dataset` onto Batcher's. Most of the vocabulary carries over, because both are lazy Python APIs over Arrow batches that scale from a laptop to a Ray cluster. For a name-by-name lookup across all 530 of Ray Data's public names, read {doc}`ray-data/index`.

## Bulk data leaves the object store

Batcher uses Ray for *scheduling only*. Bulk Arrow batches move directly between workers over Arrow Flight with credit-based flow control, never through the object store. You don't call `ray.init()` or size an object store, and distribution is an argument to the terminal call, so the same pipeline runs single-node or distributed without a rewrite.

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

The column verbs take SQL names: `select_columns` is `select`, `drop_columns` is `drop`, and `groupby` is `group_by`:

```python
print(ds.select("city").columns, ds.drop("city").columns)
# ['city'] ['amount']
```

{doc}`ray-data/dataset` lists every method with its status. Two defaults differ: `write_parquet` appends in Ray Data and overwrites here, and `ds.shuffle(seed=0)` returns the same permutation every run.

Prefer an expression over a lambda. `ds.filter(col("amount") > 10)` is pushed down to the scan, and a string predicate works too:

```python
print(ds.filter("amount > 25").to_pydict())
# {'city': ['NYC', 'SF'], 'amount': [30, 40]}
```

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

Each part is a lazy plan, so a pipeline that consumes one part never computes the others. Call `ds.cache()` first when the source is expensive and you want all of them.

`ds.split(n)` and `ds.zip(other)` take the row order as `order_by`, such as a `with_row_index("i")` column. `streaming_split` becomes {py:obj}`batcher.ml.streaming_split(dataset, world_size, rank=) <batcher.ml.streaming_split>`. For train and test sets, {py:meth}`ds.ml.train_test_split(...) <batcher.api.dataset.ml.DatasetML.train_test_split>` assigns rows by a hash of their values, so the split is the same however the data is partitioned.

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

A class-based UDF ports by passing the class itself, so the model loads once per actor:

```python
# docs: skip
ds.map_batches(Classifier, concurrency=4, num_gpus=1, batch_size=64)
```

For a model, {py:meth}`ds.ml.infer(...) <batcher.api.dataset.ml.DatasetML.infer>` is the shorter path, and its actor pool stays warm for the session.

## Reading and writing

`ray.data.read_parquet(path)` becomes `bt.read.parquet(path)`, and `ds.write_parquet(path)` becomes `ds.write.parquet(path)`. {doc}`ray-data/io` lists every reader with what differs, such as `read_images` not decoding by default here.

## Consuming results

`ds.take(n)` becomes `ds.limit(n).to_pylist()`. Batcher's `iter_batches` yields Arrow record batches; {doc}`ray-data/dataset` covers `take_all`, `take_batch`, `iter_rows`, `iter_torch_batches`, and `materialize`.

```python
print(ds.select("city", "amount").limit(2).to_pylist())
# [{'city': 'NYC', 'amount': 10}, {'city': 'LA', 'amount': 20}]
```

## Block and object-ref APIs

Bulk Arrow never becomes a Ray object, so `get_internal_block_refs`, `to_arrow_refs`, and `num_blocks` have no equivalent: stream with `ds.iter_batches()` and read execution details from `ds.stats()`. Configuration is set for the current thread through {py:obj}`bt.set_config(...) <batcher.set_config>`, or for every process through `BATCHER_*` environment variables. Typing a Ray Data name tells you what to use instead:

```python
try:
    ds.random_shuffle()
except AttributeError as e:
    print(e)
# Dataset has no attribute 'random_shuffle'. Spelled ds.shuffle(seed=0) here (a full, seeded shuffle).
```

## Verifying the port

{py:meth}`equals <batcher.Dataset.equals>` compares results and ignores row order by default:

```python
ported = ds.filter(col("amount") > 10).select("city", "amount")
expected = bt.from_pydict({"city": ["LA", "NYC", "SF"], "amount": [20, 30, 40]})
print(ported.equals(expected))
# True
```

Then check the distributed result against the single-node one:

```python
# docs: skip
distributed = bt.from_arrow(ported.collect(distributed=True, num_workers=2))
assert distributed.equals(ported)
```

## See also

- {doc}`ray-data/index`: every Ray Data name, with its Batcher spelling, its status, and what differs.
- {doc}`Running on Ray </integrations/compute/ray>`: cluster setup, worker counts, and the Flight shuffle.
- {doc}`Sampling and splitting </user-guide/transform/rows/sampling>`: the positional and hash-based splits side by side.
- {doc}`Batch inference and ML </getting-started/migration/ml-pipelines>`: models over batches, GPU pools, and the training feed.
- {doc}`Differences and verification </getting-started/migration/differences>`: what Batcher deliberately does not have.
- {doc}`Dataset API </api/relational/dataset>`: the reference for every verb named here.
