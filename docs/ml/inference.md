# Inference

Batch inference applies a model to every row of a dataset. In Batcher this runs
through the `.ml` accessor: `ds.ml.infer(model, ...)` for predictions and
`ds.ml.embed(model, ...)` for vectors. Both are lazy, return a new `Dataset`, and
operate on whole Arrow batches, so the model sees a batch of inputs at a time and
the engine handles parallelism, batching, and GPU placement.

## The model is a callable over batches

`model` is a callable applied to each `pyarrow.RecordBatch`. The recommended form
is a class: the constructor loads the weights once per worker, and `__call__` runs
the forward pass on each batch and returns a batch with the results appended. This
amortizes model loading across every batch the worker handles.

A model needs a GPU and weights, so the inference call itself is shown but not
run. The mechanics, a class that sets up once and is called per batch, are the
same as the runnable `map_batches` example below.

```python
# docs: skip
import batcher as bt
import pyarrow as pa


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


ds = bt.read.parquet("data/features.parquet")
scored = ds.ml.infer(Classifier(), batch_size=512, num_gpus=1, concurrency=4)
scored.write.parquet("output/scored.parquet")
```

## Arguments

| Argument | Meaning |
| --- | --- |
| `batch_size` | Rows per batch handed to the model. Larger batches improve GPU utilization up to memory limits. |
| `num_gpus` | GPUs reserved per worker. A fraction (for example `0.5`) packs several workers onto one device. |
| `concurrency` | Number of parallel inference actors. |
| `output_columns` | Names of the columns the model adds, when they differ from the input. |

`num_gpus` and `concurrency` together size the GPU actor pool. See
[GPU scheduling](gpu.md).

## Embeddings

`ds.ml.embed(model, ...)` is the same call shaped for embedding generation: the
model returns a vector per row, appended as a column. Use it to build inputs for
vector search or downstream models.

```python
# docs: skip
class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    def __call__(self, batch):
        texts = batch.column("text").to_pylist()
        vectors = self.model.encode(texts)
        return batch.append_column("embedding", pa.array(vectors.tolist()))


docs = bt.read.parquet("data/docs.parquet")
embedded = docs.ml.embed(Embedder(), batch_size=256, num_gpus=1, concurrency=2)
```

## A runnable batch transform

The inference and embedding calls follow the same contract as `map_batches`: a
function (or class) that takes one Arrow batch and returns one. This in-memory
example proves the shape without a model or a GPU.

```python
import batcher as bt
import pyarrow as pa
import pyarrow.compute as pc

ds = bt.from_pydict({"score": [0.2, 0.8, 0.5, 0.9]})


class Threshold:
    def __init__(self, cutoff):
        self.cutoff = pa.scalar(cutoff)

    def __call__(self, batch):
        label = pc.greater_equal(batch.column("score"), self.cutoff)
        return batch.append_column("label", label)


print(ds.ml.map_batches(Threshold(0.5)).to_pydict())
# {'score': [0.2, 0.8, 0.5, 0.9], 'label': [False, True, True, True]}
```

Swap the threshold for a model forward pass and the structure is identical; that
is what `infer` and `embed` run.

## Next steps

- [The ML accessor](../api/ml.md): the full `map_batches` / `infer` / `embed` reference.
- [GPU scheduling](gpu.md): how `num_gpus` and `concurrency` allocate devices.
- [Streaming](streaming.md): stream results into a training loop.
