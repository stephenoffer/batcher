# Streaming for training

This page describes how {py:meth}`iter_batches() <batcher.Dataset.iter_batches>` streams a
dataset into a training loop: which plans stream in bounded memory, which materialize first,
and where the work that shapes each batch should run.

A training loop wants a stream of batches, not one materialized result. `iter_batches` yields
Arrow `RecordBatch`es as the engine produces them, so memory stays bounded and the loop starts
consuming before the full dataset is read. Every training loader in `batcher.ml` is built on it.

## Streaming consumption

The simplest consumer counts rows as they arrive:

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4, 5, 6], "label": [0, 1, 0, 1, 0, 1]})

seen = 0
for batch in ds.iter_batches():
    seen += batch.num_rows
print(seen)
# 6
```

`batch_size` rebatches the output to a fixed row count. Leave it unset to keep the engine's
batches.

```python
for batch in ds.iter_batches(batch_size=2):
    print(batch.num_rows)
# 2
# 2
# 2
```

The yielded objects are ordinary `pyarrow.RecordBatch`es. PyArrow's compute kernels, its NumPy
and pandas conversions, and its tensor extraction all work on them without a detour through
Python lists.

## Which plans stream

`iter_batches` picks the execution mode from the plan, and there's no flag to set. The plan
shapes fall into three groups:

- A breaker-free pipeline over one source, such as filter, project, and `map_batches`, is
  consumed one source batch at a time. So is a top-level aggregate, distinct, or top-N over such
  a pipeline. A larger-than-memory or unbounded source streams incrementally.
- A top-level sort, join, or window over bounded sources streams from the out-of-core bucket
  pipeline. The input is consumed to disk first, and then the result is yielded one bounded
  bucket at a time. Memory stays bounded, but the first batch waits for the input.
- Anything else materializes first. Over an unbounded source, a plan that can't stream raises
  `PlanError` rather than hanging.

With `distributed=True`, a top-level breaker fans out across Ray workers and its result streams
back one reducer bucket at a time, so the driver never holds the whole distributed result.

On a dataset marked with `cache()`, an already-cached result streams straight from the cache.
Streaming doesn't *fill* the cache, because filling it means materializing the whole result.
Warm it once with a materializing terminal such as `collect()`, and later streams read from it.

## Shaping batches before the stream

Do feature engineering with expressions and `map_batches`, so the work runs in the engine and
the loop receives batches ready to convert:

```python
import pyarrow.compute as pc


def normalize(batch):
    scaled = pc.divide(pc.cast(batch.column("x"), "float64"), 6.0)
    return batch.set_column(0, "x", scaled)


prepared = ds.map_batches(normalize)
first = next(prepared.iter_batches())
print(first.column("x").to_pylist())
# [0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0]
```

For learned feature statistics such as standardization, encoding, or imputation, fit a
{doc}`preprocessor </ml/preparing/preprocessors/index>` on the training split and `transform`
the stream. The fit is one mergeable pass and the transform stays inside the engine, so neither
touches the training hot path.

## From batches to tensors

Converting each Arrow batch to tensors by hand is boilerplate Batcher already ships.
{py:meth}`ds.ml.iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` folds
the conversion into the stream and keeps the bounded memory of `iter_batches`, adding device
transfer, pinned memory, prefetch, and a local shuffle. {doc}`PyTorch </ml/inference/pytorch>`
covers its options, and `ds.ml.to_tf` is the TensorFlow counterpart.

Across ranks, the stream needs an owner for the shard, and the loader depends on the source:

- {py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` for a bounded
  corpus. It reads the corpus as a stream and keeps only its own rank's rows.
- {py:func}`batcher.ml.shard_stream_loader <batcher.ml.shard_stream_loader>` for a corpus larger
  than one rank's memory, written once with `ds.ml.write_shards`.
- {py:func}`batcher.ml.streaming_split <batcher.ml.streaming_split>` for an unbounded source with
  no global length, which fans one read out to every rank.

{doc}`Data loaders </ml/training/data-loaders>` compares all of them.
{doc}`Distributed training </ml/training/distributed-training>` states the guarantees the first
two share: balanced ranks, a global order that doesn't depend on the cluster size, and resume with
no sample repeated or skipped.

## See also

- {doc}`Data loaders </ml/training/data-loaders>`: which loader fits which training setup.
- {doc}`PyTorch </ml/inference/pytorch>`: tensor batches, device transfer, and DDP and FSDP wiring.
- {doc}`Distributed training </ml/training/distributed-training>`: the ordering, balance, and resume contract.
- {doc}`Preprocessors </ml/preparing/preprocessors/index>`: fit feature transforms before the stream.
- {doc}`Streaming </user-guide/moving-data/streaming/index>`: continuous queries over unbounded sources.
- {doc}`Inference </ml/inference/inference>`: batch prediction and embeddings.
