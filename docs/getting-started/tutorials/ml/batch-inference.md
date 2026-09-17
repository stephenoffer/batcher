# Batch inference

Run a function over a dataset in whole Arrow batches, through the `.ml` accessor. Your
function receives a `pyarrow.RecordBatch`, never a row, so per-element work stays
vectorized and out of the Python hot path. You write a batch scoring function first, then
the class form of it that loads a model once per worker instead of once per batch. The toy
example runs as written. The real-model parts are shown rather than executed.

| Step | Needs |
|---|---|
| The batch function | `pip install batcher-engine` |
| The `Classifier` class | `torch` and a saved model; shown, not run |
| `num_gpus=1.0` | A GPU |

## The shape of a batch function

{py:meth}`ds.map_batches(fn) <batcher.Dataset.map_batches>` applies `fn` to each Arrow `RecordBatch` and expects a
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


scored = ds.map_batches(score)
print(scored.to_pydict())
# {'id': [1, 2, 3, 4], 'feature': [0.5, 1.5, 2.5, 3.5], 'score': [1.0, 3.0, 5.0, 7.0]}
```

:::{warning}
The pyarrow `.to_pylist()` here turns one batch's column into Python values for the toy computation.
Don't copy that into a real pipeline. It materializes every element as a Python object,
which is exactly the per-row cost the batch interface exists to avoid. A real model reads
the Arrow buffers directly, through the column's `to_numpy()` or DLPack into torch, and no
per-row Python work happens at all.
:::

## Loading a model once per worker

:::{tip}
Pass the class, not an instance and not a closure over a loaded model. When `fn` is a
class, Batcher constructs it once per worker and reuses it across batches, so an expensive
model load is paid once for every batch that worker sees. A model loaded per batch is the
most common reason an inference pipeline is slow. On the {doc}`AI and GPU benchmark
</benchmarks/results/ai-and-gpu>`, a gpt2 load takes about 7 seconds against roughly 1 second
of generation, so loading once is most of the win.
:::

The difference is where the load sits relative to the batches:

![On the left, a plain function such as def score(batch) is used as-is on every batch, so a function that loads its model pays the load on batch 1, again on batch 2, and again on batch 3. On the right, a class such as Classifier is built once per worker, its constructor loads the model a single time, and that one instance's call method scores batch 1, batch 2, and batch 3 with the model it already holds. A gpt2 load takes about 7 seconds against about 1 second of generation, so the load is most of the cost.](/_static/diagrams/model_load_once.svg)

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
labeled = ds.map_batches(
    Classifier,
    batch_size=1024,
    num_gpus=1.0,
    concurrency=4,
)
labeled.write.parquet("output/labeled.parquet")
```

## Controlling batching and resources

`map_batches` accepts any of the following arguments to tune throughput and placement:

- `batch_size`: rows per batch handed to `fn`.
- `output_columns`: the columns `fn` returns, when the engine should know the output
  schema ahead of time.
- `num_gpus`: fractional GPUs reserved per worker.
- `concurrency`: the size of the distributed actor pool, as an int or a `(min, max)` range.
- `fn_constructor_args` and `fn_constructor_kwargs`: arguments for a class's `__init__`,
  such as a model path.
- `max_retries` and `max_errored_rows`: how many times a failing batch is retried, and how
  many per-row errors the job tolerates before it fails.

The same accessor also offers {py:meth}`ds.ml.infer(model, ...) <batcher.api.dataset.ml.DatasetML.infer>` and
{py:meth}`ds.ml.embed(model, ...) <batcher.api.dataset.ml.DatasetML.embed>` for the common inference and embedding cases. See the
{doc}`ML guide </ml/index>` and {doc}`inference reference </ml/inference/inference>`.

:::{tip}
Leave `batch_size` unset unless you have measured a reason to set it. Adaptive batch sizing
picks a VRAM-safe default and halves the batch on a CUDA OOM, which is why
{py:meth}`ds.map_batches(Model, num_gpus=1) <batcher.Dataset.map_batches>` with no knobs runs at 2,451 img/s and 82% GPU
utilization over 131k images on 8xT4 in the {doc}`AI and GPU benchmark </benchmarks/results/ai-and-gpu>`,
within 2% of the hand-tuned `batch_size=128` run.
:::

## Where to go next

The same batch contract carries every model workload that follows:

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`search;1.1em` RAG from scratch
:link: /getting-started/tutorials/ml/rag-from-scratch
:link-type: doc
Chunk, embed, retrieve, generate: the same accessor, four times.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Synthetic data
:link: /getting-started/tutorials/pipelines/synthetic-data-generation
:link-type: doc
Build inputs to test a pipeline before the real corpus arrives.
:::

:::{grid-item-card} {octicon}`zap;1.1em` GPU inference
:link: /ml/inference/gpu
:link-type: doc
Fractional GPUs, stage overlap, and the warm pool.
:::
::::

## See also

- {doc}`Inference guide </ml/inference/inference>`: `infer`, `embed`, `generate`, and the pool.
- {doc}`PyTorch integration </ml/inference/pytorch>`: zero-copy tensors, device transfer, prefetch.
- {doc}`UDFs </user-guide/transform/columns/udfs>`: the batch-callback contract, and its cost.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: why stage overlap lifts a two-stage
  pipeline from 942 to 2,504 img/s.
- {doc}`ML API reference </api/models/ml>`: the full `.ml` accessor surface.
