# PyTorch

Batcher does not replace PyTorch's data loading; it feeds it. The engine produces
Arrow `RecordBatch`es through `iter_batches` and `map_batches`, and you convert
those batches to tensors at the edge of your training code. The heavy work
(reading, filtering, joining, feature engineering) runs in the engine; PyTorch
sees ready batches.

## The pattern

1. Build and shape the dataset with the DataFrame API and `map_batches`.
2. Stream batches with `iter_batches()`.
3. Convert each Arrow batch to tensors inside an `IterableDataset` or directly in
   the loop.

Shaping runs in the engine and is runnable here; the torch conversion is not.

```python
import batcher as bt
import pyarrow.compute as pc

ds = bt.from_pydict(
    {
        "f0": [0.1, 0.2, 0.3, 0.4],
        "f1": [1.0, 2.0, 3.0, 4.0],
        "label": [0, 1, 0, 1],
    }
)


def scale(batch):
    f1 = pc.divide(batch.column("f1"), 4.0)
    return batch.set_column(1, "f1", f1)


prepared = ds.map_batches(scale)
print(prepared.to_pydict())
# {'f0': [0.1, 0.2, 0.3, 0.4], 'f1': [0.25, 0.5, 0.75, 1.0], 'label': [0, 1, 0, 1]}
```

## Feeding a DataLoader

Wrap the batch stream in a torch `IterableDataset`. Each Arrow batch becomes a
tensor; the `DataLoader` handles shuffling within its buffer and prefetch. This
requires torch, so it is shown but not run.

```python
# docs: skip
import torch
from torch.utils.data import IterableDataset, DataLoader


class BatcherDataset(IterableDataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for batch in self.dataset.iter_batches(
            batch_size=self.batch_size
        ):
            features = torch.tensor(
                [batch.column(c).to_pylist() for c in ("f0", "f1")]
            ).T
            labels = torch.tensor(batch.column("label").to_pylist())
            for i in range(batch.num_rows):
                yield features[i], labels[i]


loader = DataLoader(BatcherDataset(prepared, batch_size=256), batch_size=64)
for features, labels in loader:
    # forward, loss, backward, step ...
    pass
```

## Per-batch tensors without the wrapper

For full-batch training steps you can skip the per-row `IterableDataset` and
convert a whole Arrow batch to a tensor directly, which is faster.

```python
# docs: skip
import torch

for batch in prepared.iter_batches(batch_size=256):
    features = torch.tensor(
        [batch.column(c).to_pylist() for c in ("f0", "f1")]
    ).T
    labels = torch.tensor(batch.column("label").to_pylist())
    # forward, loss, backward, step ...
```

## Notes

- `iter_batches()` pulls batches incrementally for a breaker-free pipeline, so
  memory stays bounded for datasets larger than RAM (no flag needed).
- Do feature engineering in `map_batches` and expressions, not in `__getitem__`;
  the engine vectorizes it and runs it in parallel.
- For inference rather than training, use `ds.ml.infer`; see
  [Inference](inference.md).

## Next steps

- [Streaming](streaming.md): the `iter_batches()` contract.
- [GPU scheduling](gpu.md): run transforms on GPU workers.
- [The ML accessor](../api/ml.md): `map_batches` / `infer` / `embed`.
