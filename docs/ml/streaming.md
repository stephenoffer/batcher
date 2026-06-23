# Streaming for training

A training loop wants a stream of batches, not one materialized result. Batcher
produces that with `iter_batches()`: the engine yields Arrow
`RecordBatch`es as they are produced, so memory stays bounded and the loop starts
consuming before the full dataset is read.

Transforms applied before the stream (`map_batches`, `select`, `filter`) run
inside the engine, so the batches arrive already shaped for training.

## Streaming consumption

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4, 5, 6], "label": [0, 1, 0, 1, 0, 1]})

seen = 0
for batch in ds.iter_batches():
    seen += batch.num_rows
print(seen)
# 6
```

`iter_batches` picks the execution mode automatically: a breaker-free pipeline (and
a top-level aggregate / distinct / top-N over one) is delivered incrementally in
bounded memory, so large or unbounded inputs stream without materializing; other
plans materialize first. Set `batch_size` to control rows per batch.

```python
for batch in ds.iter_batches(batch_size=2):
    print(batch.num_rows)
# 2
# 2
# 2
```

## Shaping batches before the stream

Do feature engineering with expressions and `map_batches` so the work runs in the
engine, not the training loop. The loop then receives ready-to-use Arrow batches.

```python
import pyarrow.compute as pc


def normalize(batch):
    scaled = pc.divide(pc.cast(batch.column("x"), "float64"), 6.0)
    return batch.set_column(0, "x", scaled)


prepared = ds.map_batches(normalize)
first = next(prepared.iter_batches())
print(first.column("x").to_pylist())
# [0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0]
```

## Building a training-data pipeline

The pattern is: shape the data with the DataFrame and `map_batches` API, then
stream batches into the framework. Each Arrow batch converts to tensors with zero
or one copy. The framework-specific part (a PyTorch `IterableDataset`, a Ray
training actor) is outside the engine, so it is shown but not run here.

```python
# docs: skip
import torch


def to_tensors(batch):
    x = torch.tensor(batch.column("x").to_pylist())
    y = torch.tensor(batch.column("label").to_pylist())
    return x, y


for batch in prepared.iter_batches(batch_size=256):
    x, y = to_tensors(batch)
    # forward, loss, backward, step ...
```

See [PyTorch integration](pytorch.md) for wiring batches into a `DataLoader`, and
[Inference](inference.md) for the prediction-time counterpart.

## Next steps

- [PyTorch integration](pytorch.md): feed Arrow batches to a `DataLoader`.
- [Inference](inference.md): batch prediction and embeddings.
- [GPU scheduling](gpu.md): run transforms on GPU workers.
