# PyTorch

Batcher does not replace PyTorch's data loading. It replaces the part of it that is a data engine.
Reading, filtering, joining, feature engineering, shuffling and sharding run in Rust over Arrow,
and what reaches the training loop is `{column: tensor}` dicts, already batched and already on the
device.

| | |
| --- | --- |
| **Tensors in** | {py:func}`bt.from_torch(tensor) <batcher.from_torch>` |
| **Tensors out** | {py:meth}`ds.ml.iter_torch_batches(...) <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` |
| **Distributed** | {py:func}`batcher.ml.streaming_split(...) <batcher.ml.streaming_split>`, or {py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` |
| **Inference** | {py:meth}`ds.map_batches(SomeClass, batch_format="torch") <batcher.Dataset.map_batches>` |
| **Extra** | `pip install 'batcher-engine[torch]'` |

The measured stake, on 10 M rows x 32 float features, `batch_size=1024`, `prefetch=2`: 1.76 Mrows/s
through {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`, and 1.28 Mrows/s on a 4-rank DDP `streaming_split`
(`benchmarks/BENCHMARK_RESULTS.md`). Ingest throughput is easy to assume and cheap to check,
so measure it against your own data rather than taking either figure on trust.

## Tensors in

`bt.from_torch` adapts a tensor, a mapping of tensors, a tuple of tensors, or a map-style
{py:class}`Dataset <batcher.Dataset>` into the engine. Tensors are moved to CPU and adapted through NumPy in bulk, with
no per-row Python.

```python
import torch

import batcher as bt

features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
ds = bt.from_torch(features)
print(ds.to_pydict())
# {'data': [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]}
```

A `{name: tensor}` mapping keeps its keys as column names, which makes it the exact inverse of
the loader: what `iter_torch_batches` yields, `from_torch` reads back.

```python
batch = {"x": torch.arange(3), "y": torch.ones(3)}
print(bt.from_torch(batch).schema.names)
# ['x', 'y']
```

:::{dropdown} What each tensor shape becomes as a column
An `(n, dim)` tensor becomes a fixed-size-list column of width `dim`, which is the embedding
convention, and a higher-rank tensor becomes a fixed-shape-tensor column that keeps its per-row
shape. A tuple or list of tensors becomes one column each, named `col_0`, `col_1`, and so on, and
each one goes through the same shape rules, so a `(features, labels)` pair keeps its feature matrix
as a vector column.
:::

**Low-precision dtypes widen to `float32` on the way in, and say so.** `bfloat16` is what nearly
every LLM checkpoint carries and `float8_e4m3fn`/`float8_e5m2` are what quantized inference emits.
Neither NumPy nor Arrow has a dtype for any of them, so the tensor is widened and a `UserWarning`
names the column's new width. No value moves, because `float32` has more mantissa bits and no fewer
exponent bits than all three. What changes is four bytes a value instead of one or two, so cast the
tensor yourself to `torch.float16` first if the width matters more than the precision.

This is for adapting something you already have in memory. It is not the ingest path; for that,
read the corpus with {py:meth}`bt.read.parquet <batcher.api.io_namespace.reader.Reader.parquet>` and never build the tensors twice.

## Tensors out

`ds.ml.iter_torch_batches(...)` streams the dataset to the training loop, consuming
{py:meth}`iter_batches() <batcher.Dataset.iter_batches>` incrementally. Nothing is materialized, so it scales past memory and works on an
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
{py:meth}`stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` is the one shard authority, so **turn off any framework auto-sharding**
(`DistributedSampler`, a DataLoader sampler) or your ranks will overlap.
{doc}`Streaming for training </ml/inference/streaming>` has the ordering contract.
:::

## Inference: load the model once

:::{important}
For batch inference, pass a **class** to `map_batches`. It is instantiated once per worker and the
instance handles every batch. A plain function is rebuilt per batch, so it reloads
the model every time. That is the single most common inference foot-gun here, and Batcher raises
a `PerformanceWarning` when it sees a GPU stage given a function.
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


scored = bt.read.parquet("s3://lake/images/*.parquet").map_batches(
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

### When the class is just a module

For a module with nothing to wire beyond its inputs, {py:func}`torch_predictor <batcher.ml.torch_predictor>` writes the class for
you. It applies `eval()` and `torch.inference_mode()`, casts each input to the weights'
precision, and loads once per worker.

```python
# docs: skip
from batcher.ml import torch_predictor

udf = torch_predictor(model, input_columns=["features"], output_columns=["logits"])
scored = ds.map_batches(udf, num_gpus=1).collect()
```

`model` may be a TorchScript path, a pickled-module path, a zero-arg factory, or an `nn.Module`
you already hold. Prefer a path or a factory for a distributed run: each worker opens it, where a
built module is pickled to every one of them.

:::{note}
{py:meth}`ds.ml.predict() <batcher.api.dataset.ml.DatasetML.predict>` does not score an `nn.Module`. It assembles a feature matrix and calls
`predict`/`predict_proba`/`decision_function`/`transform`, which is the scikit-learn contract that
XGBoost, LightGBM, CatBoost, ONNX and MLflow models also follow. A deep model's entry point is its
forward, so it goes through `torch_predictor` instead. Passing one to `predict` raises a
{py:class}`PlanError <batcher.PlanError>` that says so and names the function to use.
:::

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
engine could have vectorized. Express it as a `map_batches` or an {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` instead. The work then
runs in parallel, in Rust, before it becomes a tensor.

## See also

- {doc}`PyTorch (ML guide) </ml/inference/pytorch>`: converters, DataLoader wrapping, the full loop.
- {doc}`Streaming for training </ml/inference/streaming>`: the sample-order contract and resumption.
- {doc}`Feature pipeline </cookbook/ml/pipelines/features/feature-pipeline>`: the engineering that happens before
  a row becomes a tensor.
- {doc}`Train/test split </cookbook/ml/pipelines/features/train-test-split>`: a deterministic split that survives
  a re-run.
- {doc}`ML API </api/models/ml>`: `iter_torch_batches`, `streaming_split`, `stream_loader`.
- {doc}`Ray </integrations/compute/ray>`: what a distributed run actually schedules.
- {doc}`Hugging Face </integrations/compute/huggingface>`: model ids, and the corpus that feeds this loop.
