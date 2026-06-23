# The ML accessor

ML work attaches to a `Dataset` through the `.ml` accessor. It exposes three
methods that run Python code over whole Arrow batches inside the engine:

| Method | Use |
| --- | --- |
| `ds.ml.map_batches(fn, ...)` | Apply an arbitrary function to each Arrow batch. |
| `ds.ml.infer(model, ...)` | Run batch inference with a model callable. |
| `ds.ml.embed(model, ...)` | Generate embeddings with a model callable. |

All three operate on whole `pyarrow.RecordBatch` objects, never on individual
rows. They are lazy like every other transformation and return a new `Dataset`.

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
| `num_gpus` | GPUs to reserve per worker (a fraction packs several workers onto one GPU). |
| `concurrency` | Number of parallel actors in the worker pool. |
| `num_workers` | Number of workers (`map_batches`). |

`num_gpus` and `concurrency` together describe a GPU actor pool: each actor holds
`num_gpus` of a device, and `concurrency` actors run in parallel. See
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

## What is not here

There is no `StreamingDataLoader`, no `from batcher.ml import ...`, and no
top-level `ds.embed(...)` / `ds.infer(...)` methods. Everything ML-related goes
through the `.ml` accessor, and the model functions take no `text_column=` or
`input_columns=` keywords; they receive the whole batch and pick columns
themselves.

## Next steps

- [Inference](../ml/inference.md): batch prediction and embeddings.
- [Streaming](../ml/streaming.md): streaming consumption for training.
- [GPU scheduling](../ml/gpu.md): how `num_gpus` and `concurrency` map to actors.
