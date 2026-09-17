# Serve and train

This section covers the two ends of the model lifecycle that sit outside inference: feeding a training loop, and reaching a model that is served somewhere else.

Batcher doesn't run your deep-learning training loop or host your endpoints. It owns the data on both sides of them. That is where most of the operational pain lives. A training run at 512 ranks is mostly a data problem, because every rank needs a disjoint slice, the same number of batches, and an order it can reproduce after a crash. A backfill against a served model is a batching and retry problem. Both are data-plane work, and the trainer and the server see ready tensors and well-sized requests.

## Training ingest that survives a restart

{py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` hands each PyTorch rank an `IterableDataset` over its slice of one global order, and holds four guarantees. Every rank yields the same number of batches, so none stalls the all-reduce. The order depends only on `seed` and `epoch`, not on `world_size`, so a job that dies on 64 GPUs resumes on 32 against the same permutation. Passing `global_consumed` from a checkpoint resumes mid-epoch with no repeated or skipped samples. No coordinator sits in the middle.

The shuffle behind it is a keyed pseudorandom bijection, not a stored index list. A shuffled list of 10 billion indices would need about 280 GB of driver RAM before a row is read. The bijection needs constant memory. Seeking is one computation. You can inspect the order directly:

```python
from batcher.ml import epoch_order

print(epoch_order(8, seed=42))
# [6, 4, 7, 3, 2, 5, 0, 1]
print(epoch_order(8, seed=42, epoch=1))
# [4, 0, 6, 5, 7, 3, 1, 2]
```

The shuffle is exact over the whole corpus, where WebDataset and MosaicML Streaming shuffle shards plus a local buffer.

`stream_loader` keeps one rank's slice resident. Once a slice outgrows memory, {py:meth}`ds.ml.write_shards <batcher.api.dataset.ml.DatasetML.write_shards>` writes the corpus to Arrow IPC shards and {py:func}`shard_stream_loader <batcher.ml.shard_stream_loader>` streams them back with the same balanced, resumable per-rank order and a bounded shard cache, trading the global shuffle for a seeded shuffle of shards and of rows within them. A single-process loop gets tensors from {py:meth}`ds.ml.iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`, which reached 1.06 M rows/s with zero-copy DLPack views and no shuffle on an 8xT4 cluster.

## Calling a served model

When another team owns the model, or the same endpoint must also serve online traffic, call it instead of loading it. The adapters in `batcher.ml.serving`, such as `triton_client`, are load-once class UDFs for {py:meth}`ds.ml.map_batches <batcher.api.dataset.ml.DatasetML.map_batches>`. Preprocessing stays on CPU workers, requests are sized to the server's batch window, and results come back in input order. Still, load the weights in the worker whenever you can. A 10-million-row backfill through an HTTP endpoint is 10 million round trips.

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
