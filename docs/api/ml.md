# The ML accessor

ML work attaches to a `Dataset` through the `.ml` accessor:

| Method | Use |
| --- | --- |
| `ds.ml.map_batches(fn, ...)` | Apply an arbitrary function to each Arrow batch. |
| `ds.ml.infer(model, ...)` | Run batch inference with a model callable. |
| `ds.ml.embed(model, ...)` | Generate embeddings with a model callable. |
| `ds.ml.embed_text(col, model, ...)` | Embed text with a sentence-transformers model. |
| `ds.ml.download(url_col, ...)` | Fetch bytes at each URL/path into a column. |
| `ds.ml.upload(data_col, dir, ...)` | Write a bytes column out to object storage. |
| `ds.ml.iter_torch_batches(...)` | Stream the dataset to PyTorch as tensor batches. |
| `ds.ml.stream_loader(...)` | A distributed-training `IterableDataset` for one rank. |

These operate on whole `pyarrow.RecordBatch` objects, never on individual rows.
They are lazy like every other transformation and return a new `Dataset` (except the
loaders, which return a torch iterator).

## Whole-batch semantics

A function passed to `map_batches` takes one `pyarrow.RecordBatch` and returns one
`pyarrow.RecordBatch`. Because it sees the whole batch, it can use vectorized
Arrow compute rather than per-row Python loops.

```python
import batcher as bt
import pyarrow.compute as pc

ds = bt.from_pydict({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})


def add_sum(batch):
    total = pc.add(batch.column("x"), batch.column("y"))
    return batch.append_column("sum", total)


print(ds.ml.map_batches(add_sum).to_pydict())
# {'x': [1, 2, 3, 4], 'y': [10, 20, 30, 40], 'sum': [11, 22, 33, 44]}
```

## Class-based functions load once per worker

A plain function is re-imported on each worker. A class is instantiated once per
worker and then called per batch, so any expensive setup (loading a model,
opening a tokenizer) happens once and is reused across all batches that worker
processes. The class implements `__call__(self, batch) -> batch`.

```python
import pyarrow as pa


class Scale:
    def __init__(self, factor):
        self.factor = pa.scalar(factor)

    def __call__(self, batch):
        scaled = pc.multiply(batch.column("x"), self.factor)
        return batch.set_column(0, "x", scaled)


print(ds.ml.map_batches(Scale(10)).to_pydict())
# {'x': [10, 20, 30, 40], 'y': [10, 20, 30, 40]}
```

For a real model, the constructor loads the weights and `__call__` runs the
forward pass. That needs a GPU and a model, so it is shown but not run here.

```python
# docs: skip
class Classifier:
    def __init__(self):
        import torch

        self.model = torch.load("model.pt").cuda().eval()

    def __call__(self, batch):
        import torch

        x = torch.tensor(batch.column("features").to_pylist()).cuda()
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1).cpu().tolist()
        return batch.append_column("prediction", pa.array(preds))


labelled = ds.ml.map_batches(Classifier(), num_gpus=1, concurrency=4)
```

## Common arguments

All three methods share these keywords:

| Argument | Meaning |
| --- | --- |
| `batch_size` | Rows per batch handed to `fn`. Defaults to the engine morsel size. |
| `output_columns` | Names of the columns the function produces, when they differ from the input. |
| `batch_format` | What `fn` receives/returns: `"pyarrow"` (default), `"numpy"`, `"pandas"`, or `"torch"`. |
| `num_gpus` | GPUs to reserve per worker (a fraction packs several workers onto one GPU). |
| `concurrency` | Actor-pool size: an `int` for a fixed pool, or a `(min, max)` tuple to autoscale to the workload. |
| `accelerator_type` | Pin GPU actors to a device model (a `ray.util.accelerators` name such as `"NVIDIA_A100"`). |
| `model_memory_gb` | The model's GB footprint — budgets host memory per worker (OOM protection) and VRAM-packs small models onto a shared GPU. |
| `num_workers` | Number of workers (`map_batches`). |

`num_gpus` and `concurrency` together describe a GPU actor pool: each actor holds
`num_gpus` of a device, and `concurrency` actors run in parallel. `batch_format`
converts only around the call; the engine boundary stays Arrow. See
[GPU scheduling](../ml/gpu.md).

## infer and embed

`ds.ml.infer(model, ...)` and `ds.ml.embed(model, ...)` are the inference-shaped
calls. `model` is a callable (typically a class that loads weights once per
worker) applied to each batch; `infer` appends predictions and `embed` appends
vectors. Both take `batch_size`, `output_columns`, `num_gpus`, and `concurrency`.
Real models need GPUs, so these are not run here.

```python
# docs: skip
scored = ds.ml.infer(Classifier(), batch_size=512, num_gpus=1, concurrency=4)
vectors = ds.ml.embed(Embedder(), batch_size=256, num_gpus=1, concurrency=2)
```

See [Inference](../ml/inference.md) for the inference workflow and
[Streaming](../ml/streaming.md) for feeding training loops.

## The `batcher.ml` package

Operators that are not `Dataset` methods live in `batcher.ml` — the standalone
`embed` / `llm_generate` functions, the [preprocessors](../ml/preprocessors.md),
the [serving adapters](../ml/serving.md), [vector search](../ml/multimodal.md), and
the [LLM engines](../ml/llm.md). A model passed to `map_batches`/`infer` still
receives the whole batch and picks its own columns (no `input_columns=` keyword);
`embed_text` is the exception that takes a `text_column`.

## Next steps

- [Inference](../ml/inference.md): batch prediction and embeddings.
- [Preprocessors](../ml/preprocessors.md): fit/transform feature engineering.
- [Multimodal](../ml/multimodal.md): download, decode, tensors, vector search.
- [Serving](../ml/serving.md) and [LLM inference](../ml/llm.md).
- [PyTorch](../ml/pytorch.md) and [streaming](../ml/streaming.md) training loaders.
- [GPU scheduling](../ml/gpu.md): how `num_gpus` and `concurrency` map to actors.
