# Inference

This page covers batch inference: applying a deep-learning or HuggingFace model to every row of a dataset. Two calls on the `.ml` accessor do it, {py:meth}`ds.ml.infer(model, ...) <batcher.api.dataset.ml.DatasetML.infer>` for predictions and {py:meth}`ds.ml.embed(model, ...) <batcher.api.dataset.ml.DatasetML.embed>` for vectors. Both are lazy and return a new {py:class}`Dataset <batcher.Dataset>`. Both hand the model whole Arrow batches, so parallelism, batch sizing, and GPU placement stay in the engine. For a fitted XGBoost, LightGBM, or scikit-learn model, use {doc}`ds.ml.predict </ml/inference/tabular-models>` instead.

## The model is a callable over batches

`model` is a callable applied to each `pyarrow.RecordBatch`. Write it as a class. The constructor loads the weights once per worker, and `__call__` runs the forward pass on each batch and returns it with the results appended, so the load cost is paid once rather than per batch.

The call below needs a GPU and real weights, so it doesn't run here. The runnable `map_batches` examples further down follow the same contract.

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
scored = ds.ml.infer(Classifier, batch_size=512, num_gpus=1, concurrency=4)
scored.write.parquet("output/scored.parquet")
```

## Arguments

The table lists the `ds.ml.infer` arguments you reach for most. The signature also takes `task`, `device`, `dtype`, `model_kwargs`, and the failure-handling keywords `max_errored_rows`, `timeout`, `max_retries`, `retry_backoff`, and `retry_on`:

| Argument | Meaning |
| --- | --- |
| `column` | Input column to score, when `model` is a model id (a HuggingFace pipeline). |
| `output_column` | Name of the appended prediction column (default `"prediction"`; `"embedding"` for `embed`). |
| `batch_size` | Rows per batch handed to the model. Larger batches improve GPU utilization up to memory limits. |
| `num_gpus` | GPUs reserved per worker. A fraction (for example `0.5`) packs several workers onto one device. |
| `concurrency` | Size of the GPU actor pool: an `int`, or a `(min, max)` tuple for an autoscaling pool. |
| `batch_format` | What the callable sees and returns: `"pyarrow"` (default), `"numpy"`, `"pandas"`, or `"torch"`. |
| `accelerator_type` | Pin actors to a GPU model such as `"NVIDIA_A100"`, a `ray.util.accelerators` name. |
| `model_memory_gb` | The model's GB footprint, so the resource layer can budget host RAM and VRAM-pack small models. |
| `output_columns` | Names of the columns the model adds, when they differ from the input. |

## The model-id shortcut

A HuggingFace `transformers` pipeline needs no wrapper class. Pass the model id as a string and the `column` to score. The pipeline loads once per worker and its prediction is appended as `output_column`. `task` selects the pipeline kind when the model doesn't imply it. This path needs the `transformers` extra, `pip install 'batcher-engine[transformers]'`.

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
shortcut for embedding models. It appends a vector column and needs the `st` extra.

## Batch formats and tensor columns

By default the callable receives and returns a `pyarrow.RecordBatch`, with no copy and no conversion. `batch_format` converts around the call to whatever your model code expects, and the engine boundary stays Arrow. The alternatives are the following:

- `"numpy"` gives a `{column: ndarray}` dict, the natural shape for a NumPy or
  pure-array model.
- `"pandas"` gives a `DataFrame`.
- `"torch"` gives a `{column: tensor}` dict over the numeric columns, ready to move
  to a device.

A tensor column holds a same-shape N-d array in every row, such as a set of decoded
images. It arrives as a stacked `ndarray` under `"numpy"` and `"torch"`, so a
`(batch, H, W, 3)` block feeds straight into a vision model. See
{doc}`multimodal </ml/preparing/multimodal/index>` for building those columns.

```python
import batcher as bt
import numpy as np

ds = bt.from_pydict({"recency": [0.9, 0.1, 0.6], "frequency": [0.7, 0.2, 0.5]})


def score(batch):  # batch is a {column: ndarray} dict
    logit = 3.0 * batch["recency"] + 2.0 * batch["frequency"] - 2.5
    batch["score"] = 1.0 / (1.0 + np.exp(-logit))
    return batch


out = ds.map_batches(
    score, batch_format="numpy", output_columns=["recency", "frequency", "score"]
)
print(out.to_pydict()["score"])
# [0.8320183851339245, 0.14185106490048782, 0.5744425168116589]
```

## Running on CPU

Most inference runs on CPU. A model stage with no `num_gpus` is an ordinary CPU stage: it runs on the cluster's workers, `concurrency` sizes the pool, and the load-once class matters just as much.

A hand-written CPU stage usually gets three things wrong that Batcher gets right by default.

The forward runs under `torch.inference_mode()`. Without it every forward builds a backward
graph nobody reads and holds each layer's activations alive for the whole call. On a 12-layer forward over 8,192 rows that measured 409.5 MB of peak resident memory against 39.9 MB. Speed doesn't change. What changes is how large a batch fits and whether the worker survives. A stage whose result is a gradient, such as a saliency map or an influence score, opts out with `batcher_inference_mode = False` on the class.

Half precision is *not* applied on CPU. It changes the numbers, it needs tensor cores to pay for that, and the probe that decides costs an extra forward. It stays on the GPU path.

Thread counts aren't divided among concurrent calls. Dividing them is the tempting move and the wrong one. Torch's intra-op pool is work-stealing, so oversubscription costs far less than starving each operator. The engine caps only the pool size. An explicit `OMP_NUM_THREADS` wins, and otherwise the cap is the *container's* usable cores, where torch would size itself to the host.

For an exported model on CPU, {py:func}`bt.ml.openvino_predictor <batcher.ml.openvino_predictor>`
and {py:func}`bt.ml.onnx_predictor <batcher.ml.onnx_predictor>` are usually faster than the
framework. See {doc}`/ml/inference/runtimes`.

## GPU placement

On a GPU, placement goes on the same call. `num_gpus` reserves a device per actor and `concurrency` sizes the pool, while preprocessing stays on CPU workers. {doc}`GPU scheduling </ml/inference/gpu>` covers fractional packing and keeping the devices fed.

## Embeddings

`ds.ml.embed(model, ...)` is the same call shaped for embeddings. The model returns a vector per row, appended as a column, ready for {doc}`vector search </ml/retrieval/vector-search>` or a downstream model. {doc}`/ml/retrieval/embeddings` covers `normalize`, `fp16`, and the output types.

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
embedded = docs.ml.embed(Embedder, batch_size=256, num_gpus=1, concurrency=2)
```

## Driving the pool yourself

`ds.ml.infer` runs on a `Dataset`. When you hold a bare stream of Arrow batches instead, such as the output of {py:meth}`iter_batches() <batcher.Dataset.iter_batches>` or a previous stage, {py:class}`InferencePool <batcher.ml.InferencePool>` gives you the same worker pool with no plan around it.

Two callables define it. A `Worker` maps one `pyarrow.RecordBatch` to one
`RecordBatch`: the forward pass, the tokenizer, whatever the batch has to go through. A
`WorkerFactory` is a zero-argument callable that builds a `Worker`. The pool calls the
factory exactly `num_workers` times, once per slot, so the weights load once and every
batch that slot handles reuses them. Build the model inside the `Worker` instead and it
reloads on every batch.

```python
import pyarrow as pa
import pyarrow.compute as pc
from batcher.ml import InferencePool


def make_worker():  # a WorkerFactory, called once per pool slot
    scale = pa.scalar(2.0)  # stands in for the weights you would load here

    def worker(batch):  # a Worker, called once per batch
        return batch.append_column("scaled", pc.multiply(batch.column("x"), scale))

    return worker


batches = [pa.record_batch({"x": [1.0, 2.0]}), pa.record_batch({"x": [3.0]})]
pool = InferencePool(make_worker, num_workers=2, target_batch_rows=2)
print([b.column("scaled").to_pylist() for b in pool.run(batches)])
# [[2.0, 4.0], [6.0]]
```

`run` re-chunks the input to `target_batch_rows`, coalescing small batches and splitting
large ones. It dispatches across the workers concurrently and yields results in input
order, so concurrency never reorders your rows. Set `target_latency_ms` to retune the
batch size online toward a per-batch latency, bounded by `min_batch_rows` and
`max_batch_rows`. Leave it unset for a fixed size. `objective="throughput"` hill-climbs
the batch size for rows per second under a VRAM cap instead, which is what offline batch
work wants.

A batch that exhausts accelerator memory doesn't fail the job. The pool frees the cache, runs the two halves, and concatenates them, and only a single row that still runs out of memory raises. Two cases skip the split. A fragmented allocator is retried once at the same size after its cache is released, and a device whose memory another process holds re-raises at once, since shrinking this batch can't free it.

## Overlapping stages with `run_pipeline`

A real inference job is a chain of read, decode, and forward pass. In lockstep, the GPU
idles while the CPU decodes the next batch. `run_pipeline` runs each {py:class}`Stage <batcher.ml.Stage>` on
its own thread with a bounded queue between them, so the GPU stage works on batch *k*
while the CPU stage prepares *k+1*.

![An inference pipeline with each stage on its own thread. Arrow batches flow from the source through a CPU stage that decodes and tokenizes, then through a GPU stage running the model, then out. A bounded credit window sits between each pair of stages, so a stage blocks once its window to the next stage is full. That bound keeps a fast stage from running ahead into memory, lets stages overlap so the GPU is fed while the CPU decodes, and preserves output order while the run streams.](/_static/diagrams/inference_stages.svg)

Each `Stage` carries a factory, a `credits` count, and a `num_gpus` placement hint. The factory is built once on that stage's thread, the same load-once contract as `WorkerFactory`. Single-node execution ignores `num_gpus`, so treat it as a declaration rather than a placement.

Credits are the backpressure. They cap how many finished batches may sit between one stage and the next, so a slow consumer blocks its producer. Peak memory is the sum of the stages' credits, counted in batches, not the length of the stream. A fast decoder can't fill memory ahead of a slow model, and the run streams. See
{doc}`/architecture/deep-dives/distribution/credit-flow-control` for the same mechanism
applied to the distributed shuffle.

```python
import pyarrow as pa
import pyarrow.compute as pc
from batcher.ml import Stage, run_pipeline


def decode():  # a CPU stage
    return lambda b: b.append_column("x2", pc.multiply(b.column("x"), pa.scalar(2)))


def forward():  # the stage that would hold the model
    return lambda b: b.append_column("y", pc.add(b.column("x2"), pa.scalar(1)))


batches = [pa.record_batch({"x": [1, 2]}), pa.record_batch({"x": [3]})]
stages = [Stage(decode, credits=2, name="decode"), Stage(forward, num_gpus=1, name="model")]
print([b.column("y").to_pylist() for b in run_pipeline(batches, stages)])
# [[3, 5], [7]]
```

The result is exactly what applying the stages in sequence would produce, in the same
order. Overlapping them changes the schedule, never the answer. An exception in any
stage stops the others and is re-raised to the consumer.

## A runnable batch transform

`infer` and `embed` follow the same contract as `map_batches`: a function or class that takes one Arrow batch and returns one. This example runs that shape without a model or a GPU.

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


print(ds.map_batches(Threshold(0.5)).to_pydict())
# {'score': [0.2, 0.8, 0.5, 0.9], 'label': [False, True, True, True]}
```

Swap the threshold for a forward pass and you have what `infer` and `embed` run.

## See also

- {doc}`The ML accessor </api/models/ml>`: the full `map_batches` / `infer` / `embed` reference.
- {doc}`GPU scheduling </ml/inference/gpu>`: how `num_gpus` and `concurrency` allocate devices.
- {doc}`/ml/inference/batch-scoring`: the end-to-end read, score, and write job.
- {doc}`/ml/inference/pytorch`: PyTorch models, datasets, and data loaders.
- {doc}`Streaming </ml/inference/streaming>`: stream results into a training loop.
- {doc}`/ml/inference/tabular-models`: fitted XGBoost, LightGBM, and scikit-learn models.
- {doc}`/ml/inference/runtimes`: ONNX Runtime, OpenVINO, and TensorRT for exported models.
