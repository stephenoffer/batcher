# ML engineer learning path

This path covers running models over large data: batch inference, embeddings, and GPUs,
all through the `.ml` accessor. Your function sees a whole Arrow batch rather than a
row, so the data path stays vectorized. The model loads once per worker, not once per
batch.

## Reading order

1. [Getting started](../getting-started/index.md): install and run a first query.
1. [Your first pipeline](../tutorials/first-pipeline.md): the data flow a model
   plugs into.
1. [Batch inference](../tutorials/batch-inference.md): the `.ml.map_batches`
   pattern.
1. [Feature engineering](../tutorials/feature-engineering.md): build a model-ready
   feature matrix with fit/transform preprocessors.
1. [ML overview](../ml/index.md): the accessor and its operations.
1. [Inference](../ml/inference.md): `ds.ml.infer` and `ds.ml.embed`.
1. [GPU execution](../ml/gpu.md): reserving and sharing GPUs.
1. [PyTorch integration](../ml/pytorch.md).
1. [Streaming](../ml/streaming.md): processing batches as a stream.
1. [ML API reference](../api/ml.md).

## Example: map a function over batches

```python
import batcher as bt
import pyarrow as pa

ds = bt.from_pydict({"id": [1, 2, 3, 4], "feature": [0.5, 1.5, 2.5, 3.5]})


def score(batch: pa.RecordBatch) -> pa.RecordBatch:
    preds = [round(x * 2.0, 1) for x in batch.column("feature").to_pylist()]
    return batch.append_column("score", pa.array(preds))


print(ds.ml.map_batches(score).to_pydict())
# {'id': [1, 2, 3, 4], 'feature': [0.5, 1.5, 2.5, 3.5], 'score': [1.0, 3.0, 5.0, 7.0]}
```

## Example: a model loaded once per worker (sketch)

A class function is constructed once per worker and reused across batches. GPUs and
concurrency are declared on the call itself. This one needs a real model, so it is
shown rather than run.

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
    .ml.map_batches(Embedder, batch_size=512, num_gpus=1.0, concurrency=4)
    .write.parquet("output/embeddings.parquet")
)
```

## Runnable examples

- `ml_inference.py` is a batch-inference pipeline built on `ds.ml.map_batches`, and it
  runs as written.
- `feature_engineering.py` prepares model-ready features.
- `preprocessors.py` builds the same features from fit/transform preprocessor objects
  and `Chain`.
- `streaming_pipeline.py` sketches the shape of a streaming inference pipeline. It
  needs a broker to run.

See also the [performance guide](../user-guide/performance.md) for caching feature
tables, and the [GPU guide](../ml/gpu.md) for accelerator placement.


## Recipes and deeper reading

The [ML cookbook](../examples/ml/index.md) covers the applied path: embeddings, batch
scoring, RAG indexes, feature pipelines, and the train/test leak you get for free from a
naive random split.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`search;1.1em` Text embeddings
:link: ../examples/ml/text-embeddings
:link-type: doc
Encode a corpus, then retrieve from it.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` LLM batch scoring
:link: ../examples/ml/llm-batch-scoring
:link-type: doc
Structured output, and why the engine you pick barely matters.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Train/test split
:link: ../examples/ml/train-test-split
:link-type: doc
The leak a naive random split hands you.
:::

:::{grid-item-card} {octicon}`zap;1.1em` GPU execution
:link: ../deep-dives/gpu-execution
:link-type: doc
Why the device idles, and what stage-overlap does about it.
:::
::::

:::{seealso}
- [AI and GPU benchmarks](../benchmarks/ai-and-gpu.md): ten workload families, measured.
- [PyTorch](../integrations/pytorch.md) and [Hugging Face](../integrations/huggingface.md).
- [Tensor columns](../deep-dives/tensor-columns.md): how an image becomes a column.
:::
