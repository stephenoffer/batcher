# Streaming for training

A training loop wants a stream of batches, not one materialized result. Batcher
produces that with `iter_batches()`: the engine yields Arrow
`RecordBatch`es as they are produced, so memory stays bounded and the loop starts
consuming before the full dataset is read.

Transforms applied before the stream (`map_batches`, `select`, `filter`) run
inside the engine, so the batches arrive already shaped for training.

## Streaming consumption

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4, 5, 6], "label": [0, 1, 0, 1, 0, 1]})

seen = 0
for batch in ds.iter_batches():
    seen += batch.num_rows
print(seen)
# 6
```

`iter_batches` picks the execution mode automatically: a breaker-free pipeline (and
a top-level aggregate / distinct / top-N over one) is delivered incrementally in
bounded memory, so large or unbounded inputs stream without materializing; other
plans materialize first. Set `batch_size` to control rows per batch.

```python
for batch in ds.iter_batches(batch_size=2):
    print(batch.num_rows)
# 2
# 2
# 2
```

The yielded objects are ordinary `pyarrow.RecordBatch`es, so anything in the PyArrow
ecosystem (compute kernels, NumPy/pandas conversion, tensor extraction) works on them
without copying through Python lists.

## Shaping batches before the stream

Do feature engineering with expressions and `map_batches` so the work runs in the
engine, not the training loop. The loop then receives ready-to-use Arrow batches.

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

For learned feature statistics (standardization, encoding, imputation), fit a
[preprocessor](preprocessors.md) on the training split and `transform` the stream — the
fit is one mergeable pass and the transform stays inside the engine, so neither touches
the training hot path.

## Building a training-data pipeline

The pattern is: shape the data with the DataFrame and `map_batches` API, then
stream batches into the framework. Each Arrow batch converts to tensors with zero
or one copy. The framework-specific part (a PyTorch `IterableDataset`, a training
actor) is outside the engine, so it is shown but not run here.

```python
# docs: skip
import torch


def to_tensors(batch):
    x = torch.tensor(batch.column("x").to_pylist())
    y = torch.tensor(batch.column("label").to_pylist())
    return x, y


for batch in prepared.iter_batches(batch_size=256):
    x, y = to_tensors(batch)
    # forward, loss, backward, step ...
```

## Tensor batches without the boilerplate

`ds.ml.iter_torch_batches` folds the convert-to-tensor step into the stream: it
consumes `iter_batches()` incrementally and yields `{column: tensor}` dicts (numeric
columns; others dropped), moving each batch to `device` and overlapping that move with
the next batch's host work. It is the single-process training-iteration path, in
bounded memory, so it scales to larger-than-memory and streaming sources.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/train/*.parquet")
for batch in ds.ml.iter_torch_batches(
    batch_size=256,
    device="auto",                   # CUDA / ROCm / XPU / MPS / CPU
    pin_memory=True,                 # fast async host→device copies
    local_shuffle_buffer_size=8192,  # streaming approximation of a shuffle
):
    train_step(batch["features"], batch["label"])
```

See [PyTorch integration](pytorch.md) for the device-transfer, prefetch, collate, and
zero-copy options in full (and a runnable in-memory example).

## Distributed and resumable training

For data-parallel training across ranks, `ds.ml.stream_loader` gives each rank a
`torch.utils.data.IterableDataset` over its slice of a single, seed-reproducible global
order. It is the streaming-ingest path for PyTorch DDP/FSDP/DeepSpeed, and it holds the
guarantees a distributed loop needs:

- **balanced** — with `drop_last=True` every rank yields the same number of batches, so
  no rank finishes early and stalls the others at the all-reduce barrier;
- **deterministic / elastic** — the same `(seed, epoch)` produces the same global order
  *regardless of `world_size`*, so a job can resume on a differently-sized cluster;
- **resumable** — pass `global_consumed` (the sample count already processed this epoch,
  read from a checkpoint) to resume mid-epoch with no repeated or skipped samples;
- **independent ranks** — each rank reads its own index slice with no central
  coordinator, so a slow rank never blocks the others.

```python
# docs: skip
import batcher as bt
from torch.utils.data import DataLoader

ds = bt.read.parquet("s3://bucket/train/*.parquet")
iterable = ds.ml.stream_loader(
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    seed=42,
    columns=["features", "label"],
    global_consumed=resume_offset,  # 0 for a fresh epoch
)
# stream_loader is the only shard authority — do not add a DistributedSampler.
for batch in DataLoader(iterable, batch_size=None):  # batches are already sized
    train_step(batch["features"].cuda(), batch["label"].cuda())
```

`stream_loader` materializes the dataset once (fine up to RAM). For a larger-than-RAM
corpus, write it with `batcher.io.formats.ml.write_shards` and stream from disk with
`batcher.ml.shard_stream_loader`, which keeps a bounded shard cache and the identical
sample-order contract. For an unbounded or streaming source with no global length, use
`batcher.ml.streaming_split` instead, which fans one read of the stream out to
`world_size` rank iterators (consumed concurrently, with backpressure).

At the top of each epoch, bump `epoch` so the shuffle reseeds; on restart, pass the
checkpointed `global_consumed` so the rank picks up exactly where it stopped.

## Next steps

- [PyTorch integration](pytorch.md): feed Arrow batches to a `DataLoader`, device
  transfer, and DDP/FSDP.
- [Preprocessors](preprocessors.md): fit feature transforms before the stream.
- [Inference](inference.md): batch prediction and embeddings.
- [GPU scheduling](gpu.md): run transforms on GPU workers.
