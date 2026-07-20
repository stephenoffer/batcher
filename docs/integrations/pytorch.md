# PyTorch

Batcher does not replace PyTorch's data loading. It replaces the part of it that is a data engine.
Reading, filtering, joining, feature engineering, shuffling, sharding: those run in Rust over
Arrow. What reaches the training loop is `{column: tensor}` dicts, already batched, already on the
device.

| | |
| --- | --- |
| **Tensors in** | `bt.from_torch(tensor)` |
| **Tensors out** | `ds.ml.iter_torch_batches(...)` |
| **Distributed** | `batcher.ml.streaming_split(...)`, or `ds.ml.stream_loader` |
| **Inference** | `ds.ml.map_batches(SomeClass, batch_format="torch")` |
| **Extra** | `pip install 'batcher-engine[torch]'` |

The measured stake, on 10 M rows x 32 float features, `batch_size=1024`, `prefetch=2`: 1.76 Mrows/s
through `iter_torch_batches` against Ray Data's 0.58, a 3.0x gap, and 3.5x on a 4-rank DDP
`streaming_split` (`benchmarks/BENCHMARK_RESULTS.md`). Last-mile ingest is where a training job
quietly starves its GPUs, so measure it rather than assume it.

## Tensors in

`bt.from_torch` adapts a tensor, a tuple of tensors, or a map-style `Dataset` into the engine.
Tensors are moved to CPU and adapted through NumPy in bulk, with no per-row Python.

```python
import torch

import batcher as bt

features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
ds = bt.from_torch(features)
print(ds.to_pydict())
# {'data': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]}
```

:::{dropdown} What each tensor shape becomes as a column
An `(n, dim)` tensor becomes a fixed-size-list column of width `dim`, which is the embedding
convention, and a higher-rank tensor becomes a fixed-shape-tensor column that keeps its per-row
shape. A tuple of tensors becomes one column each, named `col_0`, `col_1`, and so on.
:::

This is for adapting something you already have in memory. It is not the ingest path; for that,
read the corpus with `bt.read.parquet` and never build the tensors twice.

## Tensors out

`ds.ml.iter_torch_batches(...)` streams the dataset to the training loop, consuming
`iter_batches()` incrementally. Nothing is materialized, so it scales past memory and works on an
unbounded source. Numeric columns convert; strings and other types are dropped (keep ids and text
in the engine, not in the trainer's hot path).

::::{tab-set}

:::{tab-item} The shape of a batch

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "f0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
    }
)

batches = list(ds.ml.iter_torch_batches(batch_size=4, device="cpu"))
print(len(batches), sorted(batches[0]), tuple(batches[0]["f0"].shape))
# 2 ['f0', 'label'] (4,)
```
:::

:::{tab-item} A real training loop

In a real job leave `device="auto"` (the default): it picks CUDA, ROCm, Intel XPU, or Apple MPS,
falls back to CPU, and overlaps the host-to-device move with the next batch's work when
`prefetch_batches > 0`. `pin_memory=True` page-locks the CPU tensors first, which is what makes
that copy asynchronous.

```python
# docs: skip
import batcher as bt

train = bt.read.parquet("s3://lake/train/*.parquet")
for batch in train.ml.iter_torch_batches(
    batch_size=256,
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
:::

::::

:::{warning}
`local_shuffle_buffer_size` permutes within a rolling window of that many rows, a streaming
approximation of a global shuffle with bounded memory. It is not a global shuffle: if your data is
sorted by label, a window will not save you. Shuffle the corpus at write time, or use
`ds.ml.stream_loader`, which owns a real global order.
:::

## Distributed training

`batcher.ml.streaming_split(ds, world_size, rank=...)` gives each DDP rank a disjoint shard of the
same stream. It emits only **complete rounds** of `world_size` batches, so every rank yields the
same number of batches and none stalls the others at the all-reduce barrier. That equal-count
property is the whole reason to use it rather than slicing the stream yourself.

```python
import batcher as bt
from batcher.ml import streaming_split

ds = bt.from_pydict(
    {
        "f0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
    }
)

rank0 = streaming_split(ds, world_size=2, rank=0, batch_size=2, device="cpu")
print([batch["label"].tolist() for batch in rank0])
# [[0, 1], [0, 1]]
```

:::{warning}
Called *without* `rank`, it returns a list of `world_size` iterators: one reader consumes the
dataset once and fans batches out round-robin to bounded per-rank queues, so the data is read once
total rather than once per rank. All the ranks must then be drained **concurrently**, because the
reader blocks when any rank's queue fills. That is the DDP norm, and it deadlocks if you consume
them one after another in a single thread.
:::

For separate DDP processes over a *bounded* corpus, prefer `ds.ml.stream_loader`, whose indexed
split is exactly balanced, deterministic in `(seed, epoch)`, and resumable mid-epoch from
`global_consumed`, even onto a differently-sized cluster.

:::{important}
`stream_loader` is the one shard authority, so **turn off any framework auto-sharding**
(`DistributedSampler`, a DataLoader sampler) or your ranks will overlap.
[Streaming for training](../ml/streaming.md) has the ordering contract.
:::

## Inference: load the model once

:::{important}
For batch inference, pass a **class** to `map_batches`. It is instantiated once per worker and the
instance handles every batch. A plain function is rebuilt per batch, which reloads the model every
time; that is the single most common inference foot-gun, and Batcher raises a `PerformanceWarning`
when it sees a GPU stage given a function.
:::

```python
# docs: skip
import batcher as bt


class Classifier:
    def __init__(self):
        self.model = load_model().cuda().eval()

    def __call__(self, batch):
        # batch is a {column: tensor} dict on the GPU
        with torch.inference_mode():
            scores = self.model(batch["image"])
        return {"id": batch["id"], "score": scores.cpu().numpy()}


scored = bt.read.parquet("s3://lake/images/*.parquet").ml.map_batches(
    Classifier,
    batch_format="torch",
    batch_size=64,
    num_gpus=1,
    concurrency=8,
    model_memory_gb=4.0,
)
scored.write.parquet("s3://lake/scores")
```

`batch_format="torch"` hands the `fn` tensors instead of an Arrow batch; the engine boundary stays
Arrow either way, and the conversion happens only around the call. `model_memory_gb` lets the
resource layer budget host RAM per worker and pack small models onto a shared GPU.

## Failure modes worth knowing

:::{important}
**Tensors own their memory, on purpose.** `column_to_tensor` copies out of the Arrow buffer.
Sharing it would be faster and is undefined behavior per torch, since a training loop mutates
batches in place and the Arrow buffer is immutable. For **read-only inference** you can opt into
`zero_copy=True` on `iter_torch_batches`, which hands the buffer over via DLPack and saves a copy.
Do not set it for training.
:::

**Apple MPS has no 64-bit dtypes.** `device="auto"` downcasts float64/int64 to 32-bit when it
targets MPS, so a dev box works. Nothing downcasts on CUDA, so a float64 feature column will move 8
bytes per value to the GPU forever. Cast in the plan.

**A `map_batches` retry re-runs your `fn`.** Under `distributed=True`, a preempted worker's
partition is recomputed. Side effects (writing to a feature store, POSTing to a service) can happen
twice. Make them idempotent.

**Don't do feature engineering in `__getitem__`.** Every row that goes through Python is a row the
engine could have vectorized. Express it as a `map_batches` or an `Expr` and the work runs in
parallel, in Rust, before it becomes a tensor.

## See also

- [PyTorch (ML guide)](../ml/pytorch.md): converters, DataLoader wrapping, the full loop.
- [Streaming for training](../ml/streaming.md): the sample-order contract and resumption.
- [Feature pipeline](../examples/ml/feature-pipeline.md): the engineering that happens before
  a row becomes a tensor.
- [Train/test split](../examples/ml/train-test-split.md): a deterministic split that survives
  a re-run.
- [ML API](../api/ml.md): `iter_torch_batches`, `streaming_split`, `stream_loader`.
- [Ray](ray.md): what a distributed run actually schedules.
- [Hugging Face](huggingface.md): model ids, and the corpus that feeds this loop.
