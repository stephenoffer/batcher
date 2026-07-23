# Data loaders

This page maps the training data loaders: which one to reach for in which situation, and
what each one guarantees.

A training loop that waits on data is a loop with an expensive GPU sitting idle, and the
usual cause is a loader doing per-row Python work that should have been a columnar
operator. Shape the data in the engine, and let the loader do nothing but hand tensors to
the step function.

## Which loader

| Situation | Reach for |
| --- | --- |
| Anything, in Arrow, no framework | `ds.iter_batches()` |
| Single-process PyTorch training | `ds.ml.iter_torch_batches(...)` |
| Multi-rank DDP/FSDP over a bounded corpus | `ds.ml.stream_loader(...)` |
| Corpus larger than RAM | `batcher.ml.shard_stream_loader(...)` |
| Unbounded / streaming source, no global length | `batcher.ml.streaming_split(...)` |
| A batch iterator you already have, and torch tensors | `batcher.ml.to_torch_iterable(...)` |
| NumPy, no torch | `batcher.ml.to_numpy_batches(...)` |
| TensorFlow | `batcher.ml.to_tf_dataset(...)` |
| The whole result in memory as NumPy / JAX arrays | `ds.to_numpy()` / `ds.to_jax()` |

`ds.to_numpy()` and `ds.to_jax()` materialize the *entire* result as a
`{column: array}` dict, where a tensor or embedding column comes back shaped
`(n, *shape)`. Use them when the result fits in memory and you want arrays rather than a
streaming loader.

The three that get confused with each other are worth stating plainly.
`iter_torch_batches` is the single-process loop, and it owns the read, the device
transfer, and the batching. `stream_loader` is the multi-rank one, and it owns the
*shard*, which is why nothing else may. `to_torch_iterable` owns nothing at all. It is a
converter you hand an existing batch iterator, for a pipeline you assembled yourself.

## iter_batches: the base

Everything else is built on this. It yields ordinary `pyarrow.RecordBatch`es as they are
produced, so memory stays bounded and the loop starts before the read finishes.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "f0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "label": [0, 1, 0, 1, 0, 1],
    }
)

for batch in ds.iter_batches(batch_size=2):
    print(batch.num_rows, batch.schema.names)
# 2 ['f0', 'f1', 'label']
# 2 ['f0', 'f1', 'label']
# 2 ['f0', 'f1', 'label']
```

A breaker-free pipeline is delivered incrementally. A plan that must materialize, such as
one containing a sort or a global aggregate, does that first. So `iter_batches` over a
filter-and-project chain streams a 10 TB source in bounded memory, and over a sort it does
not. That is a property of the plan, not of the loader.

## iter_torch_batches: tensors, single process

It folds the tensor conversion into the stream and yields `{column: tensor}` dicts over
the numeric columns. `device="auto"` picks CUDA, ROCm, XPU, MPS, or CPU. Here it is CPU
so the example runs with no GPU.

```python
batches = list(ds.ml.iter_torch_batches(batch_size=2, device="cpu"))
print(len(batches), sorted(batches[0]))
# 3 ['f0', 'f1', 'label']
print(batches[0]["f0"].shape, batches[0]["label"].dtype)
# torch.Size([2]) torch.int64
```

Non-numeric columns are dropped, since a string column has no tensor. `columns=[...]`
selects explicitly, which is what you want anyway, because it lets projection pushdown
prune the scan.

Three options matter in a real loop. `pin_memory=True` page-locks the host tensors so the
copy to the device can be asynchronous. `prefetch_batches`, which is 1 by default,
overlaps that copy with the next batch's host work, so the GPU is not waiting on the PCIe
bus. `local_shuffle_buffer_size` is a streaming approximation of a shuffle. It keeps a
reservoir and draws from it, which is not a global permutation but costs nothing extra to
read.

```python
# docs: skip
for batch in ds.ml.iter_torch_batches(
    batch_size=256,
    columns=["features", "label"],
    device="auto",
    pin_memory=True,
    prefetch_batches=2,
    local_shuffle_buffer_size=8192,
):
    loss = loss_fn(model(batch["features"]), batch["label"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## stream_loader: one shard per rank

For DDP, each rank needs a disjoint slice of a single global order. `ds.ml.stream_loader`
returns a `torch.utils.data.IterableDataset` for one rank, already batched.

```python
# docs: skip
from torch.utils.data import DataLoader

iterable = ds.ml.stream_loader(
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    seed=42,
    columns=["features", "label"],
    global_consumed=resume_offset,  # 0 for a fresh epoch
)
# stream_loader is the only shard authority: do NOT also add a DistributedSampler.
for batch in DataLoader(iterable, batch_size=None):  # already sized
    train_step(batch["features"].cuda(), batch["label"].cuda())
```

:::{warning}
Wrapping this in a `DistributedSampler` shards an already-sharded stream, so each rank
sees a fraction of its fraction and most of your corpus is never read. The job runs, the
loss falls, and you are training on a quarter of the data. `stream_loader` is the only
shard authority. While you are there, note that passing `batch_size` to the `DataLoader`
re-batches batches that are already the right size.
:::

See [distributed training](distributed-training.md) for the balance, determinism, and
resume guarantees.

## streaming_split: an unbounded source

A stream has no length, so there is no index to shard on. `streaming_split` fans a
*single* read of the source out to `world_size` rank iterators, round-robin, with
bounded queues. The data is read once total, not once per rank.

```python
from batcher.ml import streaming_split

train = bt.from_pydict({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "label": [0, 1, 0, 1, 0, 1]})

rank0 = streaming_split(train, world_size=2, rank=0, batch_size=1, device="cpu")
print([b["f"].tolist() for b in rank0])
# [[1.0], [3.0], [5.0]]
```

Called without `rank` it returns a list of `world_size` iterators instead, and those must
be consumed **concurrently**, because one reader is feeding all of them through bounded
queues. Consume them serially and the first one blocks forever waiting for a queue the
others are not draining.

Only complete rounds of `world_size` batches are emitted; a trailing partial round is
dropped, so every rank yields the same count and none of them stalls the all-reduce.

## Framework converters

The converters take a batch iterator and yield framework-native batches. `to_numpy_batches`
is the NumPy one:

```python
from batcher.ml import to_numpy_batches

for arrays in to_numpy_batches(ds.iter_batches(batch_size=3), columns=["f0", "label"]):
    print(sorted(arrays), arrays["f0"].shape)
# ['f0', 'label'] (3,)
# ['f0', 'label'] (3,)
```

`to_torch_iterable` and `to_tf_dataset` are the same idea for the other two frameworks.
All three take a batch iterator, so they compose with anything upstream that yields
batches, including a hand-built pipeline.

## Do the work upstream

:::{tip}
Anything you can express as an operator should be an operator. Filters, projections,
feature arithmetic, casts, and a fitted preprocessor's transform all run inside the
engine, vectorized and parallel, and out of the training process entirely.
:::

```python
from batcher import col

prepared = (
    ds.filter(col("f1") > 1.0)
    .with_columns(f_scaled=col("f1") / 6.0)
    .select("f0", "f_scaled", "label")
)
first = next(prepared.iter_batches())
print(first.schema.names, first.num_rows)
# ['f0', 'f_scaled', 'label'] 5
```

For learned statistics such as standardization, one-hot encoding, or imputation, fit a
[preprocessor](preprocessors.md) on the train split and `transform` the stream. The fit
is one mergeable pass over the data, and the transform is an engine stage. Neither one
runs in the training loop.

## Diagnosing an idle GPU

If the GPU is starved, the loader is rarely the culprit. The stage feeding it usually is.
In order of frequency:

| What is actually happening | What to do about it |
| --- | --- |
| Per-row Python in a UDF | make it a vectorized batch function |
| A tokenizer or decoder running every epoch | run it once, write the column out |
| `batch_size` too small to amortize the kernel launch | size the batch to the model, not to the file |
| `prefetch_batches=0`, so the device copy is serialized against compute | prefetch, and pin the host memory |
| A global shuffle where a local buffer would have done | `local_shuffle_buffer_size` |

## See also

- [PyTorch](pytorch.md): device transfer, collate, zero-copy, and DDP in full.
- [Distributed training](distributed-training.md): the multi-rank sharding contract.
- [Streaming for training](streaming.md): the bounded-memory ingest path.
- [Preprocessors](preprocessors.md): the fit-on-train, transform-the-stream contract.
- [Tensor columns](../deep-dives/tensor-columns.md): how a fixed-shape tensor reaches the
  loop with its shape intact.
- [GPU execution](../deep-dives/gpu-execution.md): what the device is waiting on when it
  is waiting.
- [Distributed training pipeline](../tutorials/distributed-training-pipeline.md): the
  whole path, from files to a loop.
- [ML API](../api/ml.md): the loader and converter reference.
