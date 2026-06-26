# PyTorch

Batcher does not replace PyTorch's data loading; it feeds it. The engine produces
Arrow `RecordBatch`es through `iter_batches` and `map_batches`, and you convert
those batches to tensors at the edge of your training code. The heavy work
(reading, filtering, joining, feature engineering) runs in the engine; PyTorch
sees ready batches.

Two entry points on the `.ml` accessor turn a dataset straight into tensor batches,
so most training loops never write a `Dataset` or `DataLoader` wrapper of their own:

- **`ds.ml.iter_torch_batches(...)`** — the bounded-memory streaming path. It consumes
  `iter_batches()` incrementally and yields `{column: tensor}` dicts, with device
  transfer, prefetch, and an optional local shuffle. Use it for single-process
  training and for larger-than-memory or streaming sources.
- **`ds.ml.stream_loader(...)`** — a `torch.utils.data.IterableDataset` for
  *distributed* (DDP/FSDP/DeepSpeed) training, with a deterministic, balanced,
  resumable global sample order across ranks. Covered in
  [Streaming for training](streaming.md).

## The pattern

1. Build and shape the dataset with the DataFrame API and `map_batches`.
2. Stream batches with `iter_batches()` (or directly as tensors via
   `iter_torch_batches`).
3. Convert each Arrow batch to tensors inside an `IterableDataset` or directly in
   the loop.

Shaping runs in the engine and is runnable here; the torch conversion is not.

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

`ds.ml.iter_torch_batches` yields a `{column: tensor}` dict per batch, converting the
numeric columns and dropping the rest. This runs here on CPU — no GPU or model
needed — because the conversion is the only torch dependency:

```python
import batcher as bt

ds = bt.from_pydict(
    {"f0": [0.1, 0.2, 0.3, 0.4], "f1": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1]}
)

batches = list(ds.ml.iter_torch_batches(batch_size=2, device="cpu"))
print(len(batches))
# 2
first = batches[0]
print(sorted(first), first["label"].shape[0])
# ['f0', 'f1', 'label'] 2
```

In real training you leave `device="auto"` (the default), which picks the best
available accelerator — CUDA, ROCm, Intel XPU, or Apple MPS — or falls back to CPU,
and moves each batch there. The device move overlaps the next batch's host work when
`prefetch_batches > 0` (the default), and `pin_memory=True` page-locks the CPU tensors
for faster host→device copies.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/train/*.parquet")
loader = ds.ml.iter_torch_batches(
    batch_size=256,
    device="auto",          # CUDA / ROCm / XPU / MPS / CPU
    pin_memory=True,         # faster async host→device copies
    prefetch_batches=2,      # overlap the device move with compute
    local_shuffle_buffer_size=8192,  # streaming approximation of a shuffle
)
for batch in loader:
    logits = model(batch["features"])
    loss = loss_fn(logits, batch["label"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

`local_shuffle_buffer_size` shuffles within a rolling window of that many rows before
batching — a streaming approximation of a global shuffle that keeps memory bounded.
For full control over batch assembly, pass a `collate_fn`, which receives the
`{column: ndarray}` batch and whose return is yielded in place of the default dict.
For read-only **inference** (never training, which mutates batches in place) set
`zero_copy=True` to hand the Arrow buffer to torch via DLPack and save a CPU copy
before the device move.

## Feeding a DataLoader

When you want torch's own batching, shuffling buffer, and multi-worker prefetch, wrap
the batch stream in an `IterableDataset`. Each Arrow batch becomes a tensor; the
`DataLoader` handles the rest. This requires torch, so it is shown but not run.

```python
# docs: skip
import torch
from torch.utils.data import IterableDataset, DataLoader


class BatcherDataset(IterableDataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for batch in self.dataset.iter_batches(
            batch_size=self.batch_size
        ):
            features = torch.tensor(
                [batch.column(c).to_pylist() for c in ("f0", "f1")]
            ).T
            labels = torch.tensor(batch.column("label").to_pylist())
            for i in range(batch.num_rows):
                yield features[i], labels[i]


loader = DataLoader(BatcherDataset(prepared, batch_size=256), batch_size=64)
for features, labels in loader:
    # forward, loss, backward, step ...
    pass
```

## Per-batch tensors without the wrapper

For full-batch training steps you can skip the per-row `IterableDataset` and
convert a whole Arrow batch to a tensor directly, which is faster.

```python
# docs: skip
import torch

for batch in prepared.iter_batches(batch_size=256):
    features = torch.tensor(
        [batch.column(c).to_pylist() for c in ("f0", "f1")]
    ).T
    labels = torch.tensor(batch.column("label").to_pylist())
    # forward, loss, backward, step ...
```

## Distributed training: DDP and FSDP

For data-parallel training across ranks, use `ds.ml.stream_loader`, which gives each
rank a `torch.utils.data.IterableDataset` over its slice of a single, seed-reproducible
global order. It is the one shard authority — **disable any framework auto-sharding**
(`DistributedSampler`, a DataLoader sampler) so the splits do not overlap. Every rank
yields the *same* number of batches (`drop_last=True`), so no rank finishes early and
stalls the others at the all-reduce barrier — the property DDP and FSDP both depend on.

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

The same iterator drives FSDP unchanged — sharding the *model* (FSDP) is orthogonal to
sharding the *data* (the loader), and the loader only owns the data split. Because the
global order is deterministic in `(seed, epoch)` and independent of `world_size`, a job
can checkpoint `global_consumed` and resume mid-epoch on a differently-sized cluster
with no repeated or skipped samples. See [Streaming for training](streaming.md) for the
ordering contract and resumption in detail.

## Notes

- `iter_batches()` pulls batches incrementally for a breaker-free pipeline, so
  memory stays bounded for datasets larger than RAM (no flag needed).
- Do feature engineering in `map_batches` and expressions, not in `__getitem__`;
  the engine vectorizes it and runs it in parallel.
- `iter_torch_batches` returns CPU 64-bit tensors as-is, but downcasts 64-bit columns
  to 32-bit when targeting Apple MPS (which has no 64-bit dtype), so `device="auto"`
  works on a dev box without a crash.
- For inference rather than training, use `ds.ml.infer`; see
  [Inference](inference.md).

## Next steps

- [Streaming](streaming.md): the `iter_batches()` contract and distributed
  `stream_loader`.
- [GPU scheduling](gpu.md): run transforms on GPU workers.
- [The ML accessor](../api/ml.md): `map_batches` / `infer` / `embed`.
