# Serve and train

This section covers the two ends of the model lifecycle that sit outside inference: feeding a training loop, and reaching a model that is served somewhere else.

Batcher doesn't run your training loop or host your endpoints. It owns the data on both sides of them. Every rank of a training run needs a disjoint slice, the same number of batches, and an order it can reproduce after a crash. A backfill against a served model needs batching and retries. Batcher does both, so the trainer and the server see ready tensors and well-sized requests.

## Training ingest that survives a restart

{py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` hands each PyTorch rank an `IterableDataset` over its slice of one global order. Every rank yields the same number of batches, the order depends only on `seed` and `epoch` rather than `world_size`, and passing `global_consumed` from a checkpoint resumes mid-epoch with no repeated or skipped samples.

The order is a computed permutation, not a stored index list, so it costs constant memory at any corpus size:

```python
from batcher.ml import epoch_order

print(epoch_order(8, seed=42))
# [6, 4, 7, 3, 2, 5, 0, 1]
print(epoch_order(8, seed=42, epoch=1))
# [4, 0, 6, 5, 7, 3, 1, 2]
```

Each rank gets a disjoint, equal-sized share of that order:

```python
from batcher.ml import rank_index_batches

for rank in (0, 1):
    print(rank, list(rank_index_batches(8, batch_size=2, world_size=2, rank=rank, seed=42)))
# 0 [[6, 7], [2, 0]]
# 1 [[4, 3], [5, 1]]
```

Resuming after four samples skips exactly what was consumed:

```python
print(list(rank_index_batches(8, batch_size=2, world_size=2, rank=0, seed=42, global_consumed=4)))
# [[2, 0]]
```

A framework-free loop takes NumPy batches:

```python
import batcher as bt

ds = bt.from_pydict({"x": list(range(6)), "y": [0, 1] * 3})
for batch in ds.ml.to_numpy_batches(batch_size=4):
    print(batch["x"].tolist())
# [0, 1, 2, 3]
# [4, 5]
```

Once a slice outgrows memory, {py:meth}`ds.ml.write_shards <batcher.api.dataset.ml.DatasetML.write_shards>` writes Arrow IPC shards and {py:func}`shard_stream_loader <batcher.ml.shard_stream_loader>` streams them back with the same balanced, resumable per-rank order. A single-process loop gets tensors from {py:meth}`ds.ml.iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`, which reaches 1.06 M rows/s with zero-copy DLPack views on an 8xT4 cluster.

## Calling a served model

When another team owns the model, or the same endpoint also serves online traffic, call it instead of loading it. The adapters in `batcher.ml.serving`, such as `triton_client`, are load-once class UDFs for {py:meth}`ds.map_batches <batcher.Dataset.map_batches>`. Requests are sized to the server's batch window and results come back in input order. When you can load the weights in the worker, prefer that: it skips a network round trip per batch.

## In this section

The following table lists the pages in this section, training first:

| Page | Covers |
|---|---|
| {doc}`/ml/training/data-loaders` | Which loader to use for single-process, multi-rank, larger-than-RAM, streaming, NumPy, and TensorFlow training. |
| {doc}`/ml/training/distributed-training` | The four guarantees, the computed order, checkpointing the position, and how it compares to other loaders. |
| {doc}`/ml/training/training-corpus` | Mixing text sources at declared weights, filtering, decontaminating, and ordering to cut padding. |
| {doc}`/ml/training/ensembling` | Averaging, voting, and stacking several models into one prediction. |
| {doc}`/ml/training/serving` | Adapters for Triton, TorchServe, and HTTP servers, request sizing, errors, and the path to online serving. |
| {doc}`/ml/training/model-serving-patterns` | Loading in-process against calling a served model, overlapping stages, and adaptive batching. |

## See also

- {doc}`/getting-started/tutorials/ml/distributed-training-pipeline`: build a balanced, resumable loader for data-parallel PyTorch.
- {doc}`/ml/inference/pytorch`: tensors, DataLoader integration, and the framework converters.
- {doc}`/ml/preparing/tokenization`: tokenize once as a pipeline stage and pack sequences for pretraining.
- {doc}`/ml/inference/batch-scoring`: the offline scoring job, when the model loads in the worker.

```{toctree}
:hidden:

data-loaders
distributed-training
training-corpus
ensembling
serving
model-serving-patterns
```
