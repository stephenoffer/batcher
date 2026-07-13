# Batch inference

Run a function over a dataset in whole Arrow batches, through the `.ml` accessor. Your
function receives a `pyarrow.RecordBatch`, never a row, so per-element work stays
vectorized and out of the Python hot path. The tiny example here runs as written; the
real-model parts are shown but not executed.

:::{note}
**What you'll build.** A batch scoring function, then the class form of it that loads a model
once per worker instead of once per batch. The toy example needs nothing but
`pip install batcher-engine`.
:::

| Step | Needs |
|---|---|
| The batch function | `pip install batcher-engine` |
| The `Classifier` class | `torch` and a saved model; shown, not run |
| `num_gpus=1.0` | A GPU |

## The shape of a batch function

`ds.ml.map_batches(fn)` applies `fn` to each Arrow `RecordBatch` and expects a
`RecordBatch` back. Here a trivial function scores each row by a column, standing in
for a model's forward pass.

```python
import batcher as bt
import pyarrow as pa

ds = bt.from_pydict(
    {
        "id": [1, 2, 3, 4],
        "feature": [0.5, 1.5, 2.5, 3.5],
    }
)


def score(batch: pa.RecordBatch) -> pa.RecordBatch:
    feature = batch.column("feature").to_pylist()
    preds = [round(x * 2.0, 1) for x in feature]
    return batch.append_column("score", pa.array(preds))


scored = ds.ml.map_batches(score)
print(scored.to_pydict())
# {'id': [1, 2, 3, 4], 'feature': [0.5, 1.5, 2.5, 3.5], 'score': [1.0, 3.0, 5.0, 7.0]}
```

:::{warning}
The `.to_pylist()` here turns one batch's column into Python values for the toy computation.
Do not copy that into a real pipeline. It materializes every element as a Python object,
which is exactly the per-row cost the batch interface exists to avoid. A real model consumes
the Arrow buffers directly (`to_numpy`, or DLPack into torch) and no per-row Python work
happens at all.
:::

## Loading a model once per worker

:::{tip}
Pass the **class**, not an instance, and not a closure over a loaded model. When `fn` is a
class, it is constructed once per worker and reused across batches, so an expensive model
load is amortized across every batch that worker sees. A model loaded per batch is the
single most common reason an inference pipeline is slow, and on the benchmark the warm pool
is what turns a 7-second gpt2 load from the dominant cost into a rounding error.
:::

The class is callable: its `__call__` takes a batch and returns a batch.

```python
# docs: skip
import batcher as bt
import pyarrow as pa
import torch


class Classifier:
    def __init__(self) -> None:
        # Loaded once per worker, not once per batch.
        self.model = torch.load("model.pt").eval()

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        features = torch.tensor(batch.column("feature").to_numpy())
        with torch.no_grad():
            preds = self.model(features).argmax(dim=1)
        return batch.append_column("label", pa.array(preds.tolist()))


ds = bt.read.parquet("s3://bucket/features.parquet")
labeled = ds.ml.map_batches(
    Classifier,
    batch_size=1024,
    num_gpus=1.0,
    concurrency=4,
)
labeled.write.parquet("output/labeled.parquet")
```

## Controlling batching and resources

`map_batches` accepts knobs that tune throughput and placement:

- `batch_size`: rows per batch handed to `fn`.
- `output_columns`: declares the columns `fn` adds, when the engine should know the
  output schema ahead of time.
- `num_gpus`: fractional GPUs reserved per worker.
- `concurrency`: number of parallel workers.

The same accessor also offers `ds.ml.infer(model, ...)` and
`ds.ml.embed(model, ...)` for the common inference and embedding cases. See the
[ML guide](../ml/index.md) and [inference reference](../ml/inference.md).

:::{tip}
Leave `batch_size` unset unless you have measured a reason to set it. Adaptive batch sizing
picks a VRAM-safe default and halves the batch on a CUDA OOM, which is why
`ds.map_batches(Model, num_gpus=1)` with no knobs runs at 2,451 img/s and 82% GPU
utilization on the [AI and GPU benchmark](../benchmarks/ai-and-gpu.md). Ray Data rejects that
same call outright.
:::

## What you learned

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`search;1.1em` RAG from scratch
:link: rag-from-scratch
:link-type: doc
Chunk, embed, retrieve, generate: the same accessor, four times.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Synthetic data
:link: synthetic-data-generation
:link-type: doc
Build inputs to test a pipeline before the real corpus arrives.
:::

:::{grid-item-card} {octicon}`zap;1.1em` GPU inference
:link: ../ml/gpu
:link-type: doc
Fractional GPUs, stage overlap, and the warm pool.
:::
::::

## See also

- [Inference guide](../ml/inference.md): `infer`, `embed`, `generate`, and the pool.
- [PyTorch integration](../ml/pytorch.md): zero-copy tensors, device transfer, prefetch.
- [UDFs](../user-guide/udfs.md): the batch-callback contract, and its cost.
- [GPU execution](../deep-dives/gpu-execution.md): why stage overlap lifts a two-stage
  pipeline from 942 to 2,504 img/s.
- [ML API reference](../api/ml.md): the full `.ml` accessor surface.
