# ML engineer learning path

For running models over data at scale: batch inference, embeddings, and GPU
execution through the `.ml` accessor. Functions operate on whole Arrow batches, so
the data path stays vectorized and a model loads once per worker rather than once per
batch.

## Reading order

1. [Getting started](../getting-started/index.md): install and run a first query.
2. [Your first pipeline](../tutorials/first-pipeline.md): the data flow a model
   plugs into.
3. [Batch inference](../tutorials/batch-inference.md): the `.ml.map_batches`
   pattern.
4. [ML overview](../ml/index.md): the accessor and its operations.
5. [Inference](../ml/inference.md): `ds.ml.infer` and `ds.ml.embed`.
6. [GPU execution](../ml/gpu.md): reserving and sharing GPUs.
7. [PyTorch integration](../ml/pytorch.md).
8. [Streaming](../ml/streaming.md): processing batches as a stream.
9. [ML API reference](../api/ml.md).

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

A class function constructs once per worker and is reused across batches, with GPUs
and concurrency declared on the call. This needs a real model, so it is shown but
not run.

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
