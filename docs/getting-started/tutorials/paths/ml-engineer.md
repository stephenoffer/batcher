# ML engineer learning path

Run models over large data: batch inference, embeddings, and GPUs. Your function sees a whole Arrow batch, never a row, and a class-based model loads once per worker.

## Reading order

1. {doc}`Getting started </getting-started/index>`: install and run a first query.
1. {doc}`Your first pipeline </getting-started/tutorials/foundations/first-pipeline>`: the data flow a model
   plugs into.
1. {doc}`Batch inference </getting-started/tutorials/ml/batch-inference>`: the `.map_batches`
   pattern.
1. {doc}`Feature engineering </getting-started/tutorials/ml/feature-engineering>`: build a model-ready
   feature matrix with fit/transform preprocessors.
1. {doc}`ML overview </ml/index>`: the accessor and its operations.
1. {doc}`Inference </ml/inference/inference>`: {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` and {py:meth}`ds.ml.embed <batcher.api.dataset.ml.DatasetML.embed>`.
1. {doc}`GPU execution </ml/inference/gpu>`: reserving and sharing GPUs.
1. {doc}`PyTorch integration </ml/inference/pytorch>`.
1. {doc}`Streaming </ml/inference/streaming>`: processing batches as a stream.
1. {doc}`ML API reference </api/models/ml>`.

## Example: map a function over batches

```python
import batcher as bt
import pyarrow as pa
import pyarrow.compute as pc

ds = bt.from_pydict({"id": [1, 2, 3, 4], "feature": [0.5, 1.5, 2.5, 3.5]})


def score(batch: pa.RecordBatch) -> pa.RecordBatch:
    preds = pc.multiply(batch.column("feature"), 2.0)
    return batch.append_column("score", preds)


print(ds.map_batches(score).to_pydict())
# {'id': [1, 2, 3, 4], 'feature': [0.5, 1.5, 2.5, 3.5], 'score': [1.0, 3.0, 5.0, 7.0]}
```

Prefer NumPy? Ask for `batch_format="numpy"` and return a dict of arrays:

```python
doubled = ds.map_batches(
    lambda b: {"id": b["id"], "z": b["feature"] * 2}, batch_format="numpy"
)
print(doubled.to_pydict())
# {'id': [1, 2, 3, 4], 'z': [1.0, 3.0, 5.0, 7.0]}
```

## Example: stream batches into a training loop

`iter_batches` yields Arrow batches without materializing the whole dataset:

```python
print([batch.num_rows for batch in ds.iter_batches(batch_size=2)])
# [2, 2]
```

## Example: load a model once per worker

Pass a class instead of a function and Batcher constructs it once per worker, then calls it on every batch:

```python
class Scale:
    def __init__(self) -> None:
        self.weight = 10.0  # stands in for an expensive model load

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        return batch.append_column("y", pc.multiply(batch.column("feature"), self.weight))


print(ds.map_batches(Scale).to_pydict())
# {'id': [1, 2, 3, 4], 'feature': [0.5, 1.5, 2.5, 3.5], 'y': [5.0, 15.0, 25.0, 35.0]}
```

With a real model, declare GPUs and concurrency on the call. Replace `load_model()` with your own loader:

```python
# docs: skip
import batcher as bt
import pyarrow as pa


class Embedder:
    def __init__(self) -> None:
        self.model = load_model()  # once per worker

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        vectors = self.model.encode(batch.column("text").to_pylist())
        return batch.append_column("embedding", pa.array(vectors))


(
    bt.read.parquet("s3://bucket/docs.parquet")
    .map_batches(Embedder, batch_size=512, num_gpus=1.0, concurrency=4)
    .write.parquet("output/embeddings.parquet", mode="overwrite")
)
```

## Runnable examples

- `ml_inference.py` is a batch-inference pipeline built on {py:meth}`ds.map_batches <batcher.Dataset.map_batches>`, and it
  runs as written.
- `feature_engineering.py` prepares model-ready features.
- `preprocessors.py` builds the same features from fit/transform preprocessor objects
  and {py:class}`Chain <batcher.ml.preprocessors.Chain>`.
- `streaming_pipeline.py` sketches the shape of a streaming inference pipeline. It
  needs a broker to run.

See also the {doc}`performance guide </user-guide/operate/tuning/performance>` for caching feature
tables, and the {doc}`GPU guide </ml/inference/gpu>` for accelerator placement.

## Recipes and deeper reading

The {doc}`ML cookbook </cookbook/ml/pipelines/index>` covers the applied path: embeddings, batch scoring, RAG indexes, feature pipelines, and leak-free splits.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`search;1.1em` Text embeddings
:link: /cookbook/ml/pipelines/text/text-embeddings
:link-type: doc
Encode a corpus, then retrieve from it.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` LLM batch scoring
:link: /cookbook/ml/pipelines/text/llm-batch-scoring
:link-type: doc
Structured output from an LLM over a whole table.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Train/test split
:link: /cookbook/ml/pipelines/features/train-test-split
:link-type: doc
Split by entity so nothing leaks.
:::

:::{grid-item-card} {octicon}`zap;1.1em` GPU execution
:link: /architecture/deep-dives/distribution/gpu-execution
:link-type: doc
Keep the device busy with stage overlap.
:::
::::

## See also

- {doc}`/getting-started/tutorials/paths/data-scientist`: the analysis path that feeds this one.
- {doc}`/getting-started/tutorials/paths/platform-engineer`: sizing, scheduling, and observability for the jobs you build.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: ten workload families, measured.
- {doc}`PyTorch </integrations/compute/pytorch>` and {doc}`Hugging Face </integrations/compute/huggingface>`: the framework bindings behind the model calls.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: how an image becomes a column.
- {doc}`/cookbook/ml/pipelines/index`: runnable versions of the pipelines above.
