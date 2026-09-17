# PyTorch

This page covers feeding PyTorch from Batcher: the tensor loader and its options, the
converters for a batch stream you built yourself, and the wiring for DDP and FSDP.

Batcher feeds PyTorch's training loop rather than replacing it. The engine reads, filters,
joins, and computes features, then hands the loop batches that are already shaped. The torch
side does nothing but convert and step. {doc}`Data loaders </ml/training/data-loaders>` is the
map of every loader, including the TensorFlow and larger-than-RAM ones. This page is the PyTorch
detail behind it.

## Shape the data before the stream

Feature work belongs in expressions and `map_batches`, where the engine vectorizes it and runs
it in parallel, not in a `__getitem__`. This step needs no torch:

```python
import batcher as bt
import pyarrow.compute as pc

ds = bt.from_pydict(
    {
        "f0": [0.1, 0.2, 0.3, 0.4],
        "f1": [1.0, 2.0, 3.0, 4.0],
        "label": [0, 1, 0, 1],
    }
)


def scale(batch):
    f1 = pc.divide(batch.column("f1"), 4.0)
    return batch.set_column(1, "f1", f1)


prepared = ds.map_batches(scale)
print(prepared.to_pydict())
# {'f0': [0.1, 0.2, 0.3, 0.4], 'f1': [0.25, 0.5, 0.75, 1.0], 'label': [0, 1, 0, 1]}
```

## Tensors straight from the engine

{py:meth}`ds.ml.iter_torch_batches(...) <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`
is the single-process training path. It consumes
{py:meth}`iter_batches() <batcher.Dataset.iter_batches>` incrementally, in bounded memory, and
yields one `{column: tensor}` dict per batch over the numeric columns. Other columns are dropped
with a warning. The conversion is the only torch dependency, so this runs on CPU with no GPU and
no model:

```python
import batcher as bt

ds = bt.from_pydict({"f0": [0.1, 0.2, 0.3, 0.4], "f1": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1]})

batches = list(ds.ml.iter_torch_batches(batch_size=2, device="cpu"))
print(len(batches))
# 2
first = batches[0]
print(sorted(first), first["label"].shape[0])
# ['f0', 'f1', 'label'] 2
```

In real training, leave `device="auto"`, the default. It picks CUDA, ROCm, Intel XPU, or Apple
MPS when one is available, falls back to CPU, and moves each batch there. MPS has no 64-bit
tensors, so on a Mac the loader downcasts 64-bit columns rather than crashing.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/train/*.parquet")
loader = ds.ml.iter_torch_batches(
    batch_size=256,
    device="auto",  # CUDA / ROCm / XPU / MPS / CPU
    pin_memory=True,  # faster async host-to-device copies
    prefetch_batches=2,  # overlap the device move with compute
    local_shuffle_buffer_size=8192,  # streaming approximation of a shuffle
)
for batch in loader:
    logits = model(batch["features"])
    loss = loss_fn(logits, batch["label"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### The options that matter

`pin_memory=True` page-locks the host tensors so the copy to the device can be asynchronous. On
CUDA the copy runs on its own stream. `prefetch_batches`, 2 by default, prepares batches on a
background thread so the device move overlaps the next batch's host work. Two to four is the
useful band: one batch of look-ahead can't cover a device copy plus the next read, and past four
the extra batches sit in memory no budget accounts for.

`local_shuffle_buffer_size` is a streaming approximation of a shuffle, and it's a block
permutation rather than a reservoir. The loader fills a block to that row count or to 256 MiB,
whichever binds first, permutes the block once, and emits it. That costs nothing extra to read.
It's also not a global permutation. A row never crosses a block boundary, so a corpus written in
label order stays clumped. Pass `epoch` each epoch, together with a fixed `seed`, or every epoch
replays the same order.

`dtypes` casts the yielded tensors, either one name for every column (`"float16"`, or an
abbreviation such as `"bf16"`) or a `{column: dtype}` mapping. `drop_last=True` drops a final
batch narrower than `batch_size`, so a ragged tail never reaches DDP. It needs an explicit
`batch_size`.

Two options change what comes back. `collate_fn` receives the `{column: ndarray}` batch, with
every column included, and its return value is yielded in place of the default dict. It's the
way through for string labels and ragged sequences. `zero_copy=True` hands the Arrow buffer to
torch through DLPack and saves a CPU copy before the device move. Use it only for read-only
inference, because training mutates batches in place.

## The framework converters

When you drive the loop yourself, the converters sit over *any* iterable of Arrow batches rather
than over a `Dataset`: `iter_batches()`, a reader, or the output of
{py:class}`InferencePool <batcher.ml.InferencePool>` or `run_pipeline`. Use
`ds.ml.iter_torch_batches` when you want tensors straight out of a dataset.

{py:func}`to_numpy_batches(batches, columns=...) <batcher.ml.to_numpy_batches>` is the base of
the other two. It yields one `{column: ndarray}` dict per batch, and numeric non-null columns
convert zero-copy. A tensor column, or a fixed-size list of numbers, comes back with its real
`(n, width...)` shape rather than as an object array, so an embedding or image column reaches the
model as a matrix. It needs nothing but NumPy:

```python
from batcher.ml import to_numpy_batches

arrays = next(to_numpy_batches(ds.iter_batches(batch_size=2), columns=["f0", "label"]))
print({name: array.tolist() for name, array in arrays.items()})
# {'f0': [0.1, 0.2], 'label': [0, 1]}
```

{py:func}`to_torch_iterable(batches, columns=...) <batcher.ml.to_torch_iterable>` wraps that in a
`torch.utils.data.IterableDataset` of `{column: tensor}` dicts, which replaces the hand-written
wrapper class most projects start with. Under `DataLoader(num_workers=k)` it strides the batches
across the workers, so each batch comes from exactly one worker. A naive `IterableDataset` runs in
full in every worker and trains on each sample *k* times per epoch. Non-numeric columns are
skipped, so keep text and ids in the engine. The dataset is single-pass unless `batches` is
itself re-iterable.

```python
# docs: skip
from torch.utils.data import DataLoader

from batcher.ml import to_torch_iterable

stream = to_torch_iterable(prepared.iter_batches(batch_size=256), columns=["f0", "f1", "label"])
for batch in DataLoader(stream, batch_size=None):  # batches are already sized
    loss = loss_fn(model(batch["f0"]), batch["label"])
    loss.backward()
```

{py:func}`to_tf_dataset(batches, columns=...) <batcher.ml.to_tf_dataset>` is the TensorFlow
equivalent. It returns a `tf.data.Dataset` of `{column: tensor}` dicts.

```python
# docs: skip
from batcher.ml import to_tf_dataset

tf_ds = to_tf_dataset(prepared.iter_batches(batch_size=256), columns=["f0", "f1", "label"])
model.fit(tf_ds.map(lambda row: (row["f0"], row["label"])), epochs=3)
```

## Distributed training with DDP and FSDP

For data-parallel training, use
{py:meth}`ds.ml.stream_loader(...) <batcher.api.dataset.ml.DatasetML.stream_loader>`. It gives
each rank a `torch.utils.data.IterableDataset` over its slice of one seed-reproducible global
order, and it's the only shard authority. Disable any framework auto-sharding, including a
`DistributedSampler`, or the splits overlap.

Every rank yields the *same* number of batches, so none finishes early and stalls the others at
the all-reduce barrier. `drop_last` only chooses how the epoch's tail becomes divisible by
`world_size`. The default `True` drops the remainder, and `False` pads by repeating a few samples,
as `DistributedSampler` does. A `collate_fn` here receives each batch as a `pyarrow.Table`.

```python
# docs: skip
import batcher as bt
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()

ds = bt.read.parquet("s3://bucket/train/*.parquet")
iterable = ds.ml.stream_loader(
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    columns=["features", "label"],
)
model = DistributedDataParallel(model.cuda())
for batch in DataLoader(iterable, batch_size=None):  # batches are already sized
    x = batch["features"].cuda()
    y = batch["label"].cuda()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

The same iterator drives FSDP unchanged. FSDP shards the *model* and the loader shards the
*data*, and the two don't interact. The global order depends on `(seed, epoch)` and not on
`world_size`, so a job can checkpoint `global_consumed` and resume mid-epoch on a differently
sized cluster. {doc}`Distributed training </ml/training/distributed-training>` covers the
ordering and resume contract.

For inference rather than training, use
{py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>`, described in
{doc}`Inference </ml/inference/inference>`.

## See also

- {doc}`Data loaders </ml/training/data-loaders>`: which loader to use, larger-than-RAM shards, and TensorFlow.
- {doc}`Distributed training </ml/training/distributed-training>`: balanced, deterministic, resumable ranks.
- {doc}`Streaming for training </ml/inference/streaming>`: which plans `iter_batches` streams.
- {doc}`GPU scheduling </ml/inference/gpu>`: run transforms on GPU workers.
- {doc}`The ML accessor </api/models/ml>`: `map_batches`, `infer`, and `embed`.
