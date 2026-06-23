# Batch inference

This tutorial runs a function over a dataset in whole Arrow batches using the `.ml`
accessor. The function receives a `pyarrow.RecordBatch`, never a row, so per-element
work stays vectorized and out of the Python hot path. The tiny example here runs as
written; the real-model parts are shown but not executed.

## The shape of a batch function

`ds.ml.map_batches(fn)` applies `fn` to each Arrow `RecordBatch` and expects a
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


scored = ds.ml.map_batches(score)
print(scored.to_pydict())
# {'id': [1, 2, 3, 4], 'feature': [0.5, 1.5, 2.5, 3.5], 'score': [1.0, 3.0, 5.0, 7.0]}
```

The `.to_pylist()` here turns one batch's column into Python values for the toy
computation. A real model consumes the Arrow buffers directly (for example via
`to_numpy`) so no per-row Python work happens.

## Loading a model once per worker

When `fn` is a class, it is constructed once per worker and reused across batches, so
an expensive model load is amortized. The class is callable: its `__call__` takes a
batch and returns a batch.

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
labeled = ds.ml.map_batches(
    Classifier,
    batch_size=1024,
    num_gpus=1.0,
    concurrency=4,
)
labeled.write.parquet("output/labeled.parquet")
```

## Controlling batching and resources

`map_batches` accepts knobs that tune throughput and placement:

- `batch_size`: rows per batch handed to `fn`.
- `output_columns`: declares the columns `fn` adds, when the engine should know the
  output schema ahead of time.
- `num_gpus`: fractional GPUs reserved per worker.
- `concurrency`: number of parallel workers.

The same accessor also offers `ds.ml.infer(model, ...)` and
`ds.ml.embed(model, ...)` for the common inference and embedding cases. See the
[ML guide](../ml/index.md) and [inference reference](../ml/inference.md).

## Next steps

- [Synthetic data generation](synthetic-data-generation.md): build inputs to test a
  pipeline.
- [GPU inference](../ml/gpu.md) and [PyTorch integration](../ml/pytorch.md).
