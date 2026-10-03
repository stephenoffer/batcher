# Batch inference

Run a function over a dataset in whole Arrow batches. Your function receives a `pyarrow.RecordBatch`, never a row, so per-element work stays vectorized. You write a batch scoring function first, then the class form that loads a model once per worker.

| Step | Needs |
|---|---|
| The batch function and the stub class | `pip install batcher-engine` |
| The `Classifier` class | `torch` and a saved model; shown, not run |
| `num_gpus=1.0` | A GPU |

## The shape of a batch function

{py:meth}`ds.map_batches(fn) <batcher.Dataset.map_batches>` applies `fn` to each Arrow `RecordBatch` and expects a `RecordBatch` back. Here a vectorized multiply stands in for a model's forward pass:

```python
import batcher as bt
import pyarrow as pa
import pyarrow.compute as pc

ds = bt.from_pydict({"id": [1, 2, 3, 4], "feature": [0.5, 1.5, 2.5, 3.5]})


def score(batch: pa.RecordBatch) -> pa.RecordBatch:
    return batch.append_column("score", pc.multiply(batch.column("feature"), 2.0))


scored = ds.map_batches(score)
print(scored.to_pydict())
# {'id': [1, 2, 3, 4], 'feature': [0.5, 1.5, 2.5, 3.5], 'score': [1.0, 3.0, 5.0, 7.0]}
```

The function reads Arrow buffers directly, so no Python object is built per element. A real model does the same through the column's `to_numpy()` or DLPack into torch.

Prefer NumPy or pandas? Set `batch_format` and the batch arrives in that form:

::::{tab-set}
:::{tab-item} NumPy
```python
def shift(batch):
    batch["feature"] = batch["feature"] + 1
    return batch


print(ds.map_batches(shift, batch_format="numpy").to_pydict()["feature"])
# [1.5, 2.5, 3.5, 4.5]
```
:::

:::{tab-item} pandas
```python
doubled = ds.map_batches(lambda df: df.assign(double=df["feature"] * 2), batch_format="pandas")
print(doubled.to_pydict()["double"])
# [1.0, 3.0, 5.0, 7.0]
```
:::
::::

## Loading a model once per worker

Pass the class, not an instance. Batcher constructs it once per worker and reuses it across batches, so an expensive model load is paid once per worker instead of once per batch:

![On the left, a plain function such as def score(batch) is used as-is on every batch, so a function that loads its model pays the load on batch 1, again on batch 2, and again on batch 3. On the right, a class such as Classifier is built once per worker, its constructor loads the model a single time, and that one instance's call method scores batch 1, batch 2, and batch 3 with the model it already holds. A gpt2 load takes about 7 seconds against about 1 second of generation, so the load is most of the cost.](/_static/diagrams/model_load_once.svg)

The class is callable: `__init__` loads, `__call__` scores. Constructor arguments go through `fn_constructor_kwargs`:

```python
class Scaler:
    def __init__(self, factor: float) -> None:
        self.factor = factor  # stands in for an expensive model load

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch.append_column("score", pc.multiply(batch.column("feature"), self.factor))


print(ds.map_batches(Scaler, fn_constructor_kwargs={"factor": 10.0}).to_pydict()["score"])
# [5.0, 15.0, 25.0, 35.0]
```

The real model has the same shape, with a GPU reservation and an actor pool:

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

`batch_size` sets the rows per call, and `output_columns` declares the schema `fn` returns:

```python
small = ds.map_batches(score, batch_size=2, output_columns=["id", "feature", "score"])
print(small.to_pydict()["score"])
# [1.0, 3.0, 5.0, 7.0]
```

:::{dropdown} Every throughput and placement argument

- `batch_size`: rows per batch handed to `fn`.
- `output_columns`: the columns `fn` returns, when the engine should know the output
  schema ahead of time.
- `num_gpus`: fractional GPUs reserved per worker.
- `concurrency`: the size of the distributed actor pool, as an int or a `(min, max)` range.
- `fn_constructor_args` and `fn_constructor_kwargs`: arguments for a class's `__init__`,
  such as a model path.
- `max_retries` and `max_errored_rows`: how many times a failing batch is retried, and how
  many per-row errors the job tolerates before it fails.
:::

The same accessor also offers {py:meth}`ds.ml.infer(model, ...) <batcher.api.dataset.ml.DatasetML.infer>` and
{py:meth}`ds.ml.embed(model, ...) <batcher.api.dataset.ml.DatasetML.embed>` for the common inference and embedding cases. See the
{doc}`ML guide </ml/index>` and {doc}`inference reference </ml/inference/inference>`.

:::{tip}
On a GPU, leave `batch_size` unset. Adaptive batch sizing picks a VRAM-safe default and halves the batch on a CUDA OOM. {py:meth}`ds.map_batches(Model, num_gpus=1) <batcher.Dataset.map_batches>` with no knobs reaches 2,451 img/s at 82% GPU utilization on 8xT4, within 2% of a hand-tuned run. See the {doc}`AI and GPU benchmark </benchmarks/results/ai-and-gpu>`.
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
- {doc}`UDFs </user-guide/transform/columns/udfs>`: the batch-callback contract.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: why stage overlap lifts a two-stage
  pipeline from 942 to 2,504 img/s.
- {doc}`ML API reference </api/models/ml>`: the full `.ml` accessor surface.
