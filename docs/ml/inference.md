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
| `column` | Input column to score, when `model` is a model id (a HuggingFace pipeline). |
| `output_column` | Name of the appended prediction column (default `"prediction"`; `"embedding"` for `embed`). |
| `batch_size` | Rows per batch handed to the model. Larger batches improve GPU utilization up to memory limits. |
| `num_gpus` | GPUs reserved per worker. A fraction (for example `0.5`) packs several workers onto one device. |
| `concurrency` | Size of the GPU actor pool: an `int`, or a `(min, max)` tuple for an autoscaling pool. |
| `batch_format` | What the callable sees and returns: `"pyarrow"` (default), `"numpy"`, `"pandas"`, or `"torch"`. |
| `accelerator_type` | Pin actors to a GPU model, e.g. `"NVIDIA_A100"` (a `ray.util.accelerators` name). |
| `model_memory_gb` | The model's GB footprint, so the resource layer can budget host RAM and VRAM-pack small models. |
| `output_columns` | Names of the columns the model adds, when they differ from the input. |

`num_gpus` and `concurrency` together size the GPU actor pool. See
[GPU scheduling](gpu.md).

## The model-id shortcut

When the model is a HuggingFace `transformers` pipeline, you can skip the wrapper
class: pass the model id as a string and the `column` to score. The pipeline loads
once per worker and its prediction is appended as `output_column`. `task` selects
the pipeline kind when it cannot be inferred from the model. This path needs the
`transformers` extra (`pip install 'batcher-engine[transformers]'`).

```python
# docs: skip
import batcher as bt

reviews = bt.read.parquet("data/reviews.parquet")  # has a "text" column
scored = reviews.ml.infer(
    "distilbert-base-uncased-finetuned-sst-2-english",
    column="text",
    output_column="sentiment",
    task="sentiment-analysis",
    batch_size=64,
    num_gpus=1,
    concurrency=(1, 4),  # autoscale the actor pool between 1 and 4 GPUs
)
```

`ds.ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="text")` is the same
shortcut for embedding models, appending a vector column; it needs the `st` extra.

## Batch formats and tensor columns

By default the callable receives and returns a `pyarrow.RecordBatch` (zero-copy,
no conversion). `batch_format` switches that to whatever the model code is written
against, converting only around the call — the engine boundary stays Arrow:

- `"numpy"` — a `{column: ndarray}` dict, the most natural shape for a NumPy or
  pure-array model.
- `"pandas"` — a `DataFrame`.
- `"torch"` — a `{column: tensor}` dict over the numeric columns, ready to move to
  a device.

A tensor column (every row a same-shape N-d array, e.g. decoded images) arrives as
a stacked `ndarray` under `"numpy"`/`"torch"`, so a `(batch, H, W, 3)` block feeds
straight into a vision model. See [multimodal](multimodal.md) for building those
columns.

```python
import batcher as bt
import numpy as np

ds = bt.from_pydict({"recency": [0.9, 0.1, 0.6], "frequency": [0.7, 0.2, 0.5]})


def score(batch):  # batch is a {column: ndarray} dict
    logit = 3.0 * batch["recency"] + 2.0 * batch["frequency"] - 2.5
    batch["score"] = 1.0 / (1.0 + np.exp(-logit))
    return batch


out = ds.ml.map_batches(
    score, batch_format="numpy", output_columns=["recency", "frequency", "score"]
)
print(out.to_pydict()["score"])
# [0.8320183851339245, 0.14185106490048782, 0.5744425168116589]
```

## GPU placement

Inference does not have to run on a GPU, but when it does the placement is declared
on the same call: `num_gpus` reserves a device per actor and `concurrency` sizes the
pool. Preprocessing stays on CPU workers while the model runs on GPU actors — the
heterogeneous pipeline. The full mechanics, fractional packing, and how to keep the
devices fed live in [GPU scheduling](gpu.md).

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
