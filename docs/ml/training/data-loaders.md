# Data loaders

This page maps the training data loaders: which one to reach for in which situation, and
what each one guarantees.

A training loop that waits on data is a loop with an expensive GPU sitting idle, and the
usual cause is a loader doing per-row Python work that should have been a columnar
operator. Shape the data in the engine, and let the loader do nothing but hand tensors to
the step function.

## Which loader

Find the row that matches your training setup. Each loader hands batches to a different
consumer, and picking the wrong one is what puts Python back on the hot path:

| Situation | Reach for |
| --- | --- |
| Anything, in Arrow, no framework | {py:meth}`ds.iter_batches() <batcher.Dataset.iter_batches>` |
| Single-process PyTorch training | {py:meth}`ds.ml.iter_torch_batches(...) <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` |
| Multi-rank DDP/FSDP over a bounded corpus | {py:meth}`ds.ml.stream_loader(...) <batcher.api.dataset.ml.DatasetML.stream_loader>` |
| Corpus larger than RAM | {py:meth}`ds.ml.write_shards(...) <batcher.api.dataset.ml.DatasetML.write_shards>` then {py:func}`batcher.ml.shard_stream_loader(...) <batcher.ml.shard_stream_loader>` |
| Unbounded / streaming source, no global length | {py:func}`batcher.ml.streaming_split(...) <batcher.ml.streaming_split>` |
| A batch iterator you already have, and torch tensors | {py:func}`batcher.ml.to_torch_iterable(...) <batcher.ml.to_torch_iterable>` |
| NumPy, no torch | {py:func}`batcher.ml.to_numpy_batches(...) <batcher.ml.to_numpy_batches>` |
| TensorFlow | {py:func}`batcher.ml.to_tf_dataset(...) <batcher.ml.to_tf_dataset>` |
| The whole result in memory as NumPy / JAX arrays | {py:meth}`ds.to_numpy() <batcher.Dataset.to_numpy>` / {py:meth}`ds.to_jax() <batcher.Dataset.to_jax>` |

`ds.to_numpy()` and `ds.to_jax()` materialize the *entire* result as a
`{column: array}` dict, where a tensor or embedding column comes back shaped
`(n, *shape)`. Use them when the result fits in memory and you want arrays rather than a
streaming loader.

The three that get confused with each other are worth stating plainly.
{py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` is the single-process loop, and it owns the read, the device
transfer, and the batching. {py:meth}`stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` is the multi-rank one, and it owns the
*shard*, which is why nothing else may. `to_torch_iterable` owns nothing at all. It is a
converter you hand an existing batch iterator, for a pipeline you assembled yourself.

## iter_batches: the base

Everything else is built on this. It yields ordinary `pyarrow.RecordBatch`es as they are
produced, so memory stays bounded and the loop starts before the read finishes.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "f0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "label": [0, 1, 0, 1, 0, 1],
    }
)

for batch in ds.iter_batches(batch_size=2):
    print(batch.num_rows, batch.schema.names)
# 2 ['f0', 'f1', 'label']
# 2 ['f0', 'f1', 'label']
# 2 ['f0', 'f1', 'label']
```

A breaker-free pipeline is delivered incrementally. A plan that must materialize, such as
one containing a sort or a global aggregate, does that first. So {py:meth}`iter_batches <batcher.Dataset.iter_batches>` over a
filter-and-project chain streams a 10 TB source in bounded memory, and over a sort it does
not. That is a property of the plan, not of the loader.

## iter_torch_batches: tensors, single process

It folds the tensor conversion into the stream and yields `{column: tensor}` dicts over
the numeric columns. `device="auto"` picks CUDA, ROCm, XPU, MPS, or CPU. Here it is CPU
so the example runs with no GPU.

```python
batches = list(ds.ml.iter_torch_batches(batch_size=2, device="cpu"))
print(len(batches), sorted(batches[0]))
# 3 ['f0', 'f1', 'label']
print(batches[0]["f0"].shape, batches[0]["label"].dtype)
# torch.Size([2]) torch.int64
```

Non-numeric columns are dropped, since a string column has no tensor. `columns=[...]`
selects explicitly, which is what you want anyway, because it lets projection pushdown
prune the scan.

Three options matter in a real loop. `pin_memory=True` page-locks the host tensors so the
copy to the device can be asynchronous. `prefetch_batches`, which is 2 by default,
overlaps that copy with the next batch's host work, so the GPU is not waiting on the PCIe
bus. `local_shuffle_buffer_size` is a streaming approximation of a shuffle. It keeps a
reservoir and draws from it, which is not a global permutation but costs nothing extra to
read.

```python
# docs: skip
for batch in ds.ml.iter_torch_batches(
    batch_size=256,
    columns=["features", "label"],
    device="auto",
    pin_memory=True,
    prefetch_batches=2,
    local_shuffle_buffer_size=8192,
):
    loss = loss_fn(model(batch["features"]), batch["label"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## stream_loader: one shard per rank

For DDP, each rank needs a disjoint slice of a single global order. `ds.ml.stream_loader`
returns a `torch.utils.data.IterableDataset` for one rank, already batched.

```python
# docs: skip
from torch.utils.data import DataLoader

iterable = ds.ml.stream_loader(
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    seed=42,
    columns=["features", "label"],
    global_consumed=resume_offset,  # 0 for a fresh epoch
)
# stream_loader is the only shard authority: do NOT also add a DistributedSampler.
for batch in DataLoader(iterable, batch_size=None):  # already sized
    train_step(batch["features"].cuda(), batch["label"].cuda())
```

:::{warning}
Wrapping this in a `DistributedSampler` shards an already-sharded stream, so each rank
sees a fraction of its fraction and most of your corpus is never read. The job runs, the
loss falls, and you are training on a quarter of the data. `stream_loader` is the only
shard authority. While you are there, remember that passing `batch_size` to the `DataLoader`
re-batches batches that are already the right size.
:::

See {doc}`distributed training </ml/training/distributed-training>` for the balance, determinism, and
resume guarantees.

## shard_stream_loader: a corpus larger than RAM

`stream_loader` keeps one rank's whole slice of the corpus resident. That is a factor of
`world_size` better than materializing the corpus per rank, and it still stops working once
a rank's slice exceeds memory. Past that point the corpus goes to storage in a layout that
supports random access by row, and the loader reads only the rows a batch needs.

Write it once with {py:meth}`ds.ml.write_shards <batcher.api.dataset.ml.DatasetML.write_shards>`.
Rows stream out of the engine into fixed-size Arrow IPC shards plus an `index.json`, so
writing a corpus larger than memory needs no more memory than one shard:

```python
import batcher as bt
import os, tempfile

corpus = bt.from_pydict({"f": [float(i) for i in range(100)], "label": [i % 2 for i in range(100)]})
path = os.path.join(tempfile.mkdtemp(), "train-shards")

index = corpus.ml.write_shards(path, rows_per_shard=32)
print(index.total_rows, index.shard_rows)
# 100 (32, 32, 32, 4)
```

Then stream it. The loader holds at most `cache_size` decoded shards, whatever the corpus
size, and gives the same deterministic, balanced, resumable per-rank order `stream_loader`
does:

```python
from batcher.ml import shard_stream_loader

loader = shard_stream_loader(path, batch_size=10, world_size=1, rank=0, seed=42)
print(len(loader), sorted(next(iter(loader))))
# 10 ['f', 'label']
```

The corpus also reads back as an ordinary relation, which is what the questions asked
*around* a training run need — class balance, null labels, a join against the source table,
a check that this corpus is the one the features were fitted on:

```python
corpus = bt.read.training_shards(path)
print(corpus.count(), corpus.schema.names)
# 100 ['f', 'label']
print(corpus.group_by("label").agg(n=bt.col("f").count()).sort("label").to_pydict())
# {'label': [0, 1], 'n': [50, 50]}
```

The row count comes from the corpus index, so `count()` is answered without reading a shard,
and each shard is its own read task — so a scan of the corpus fans out across a cluster the
same way any other source does. The shards are plain Arrow IPC underneath, so
`bt.read.arrow(f"{path}/*.arrow")` works too. This is a layout, not a private format.

### Why the shuffle is blocked, not global

A globally shuffled epoch and a bounded shard cache cannot both work. Every sample of a
global shuffle is uniform over the whole corpus, so a batch of 1,024 samples lands in up to
1,024 different shards. The cache misses on nearly all of them, and each miss reads a whole
shard to use one row of it. Over a corpus of ten thousand shards the epoch reads the data
thousands of times over. The cache is not too small; a global shuffle has no working set for
it to hold.

So the default shuffle is *blocked*: the shards are shuffled, and the rows inside each shard
are shuffled, which keeps a batch inside one shard while still giving a different seeded
order every epoch. This is the trade MosaicML Streaming makes with its `py1s`/`py1b`
algorithms and the reason WebDataset pairs a shard shuffle with a sample buffer.

`shuffle_block_size` sets the window. It defaults to one shard, and widening it decorrelates
further at a proportional cost in `cache_size`:

```python
# docs: skip
loader = shard_stream_loader(
    path,
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    seed=42,
    shuffle_block_size=8 * 65_536,  # eight shards wide
    cache_size=9,                   # ...so nine shards stay resident
)
```

The window is deliberately **not** derived from `cache_size`. The sample order must be a
property of the corpus and the seed alone, or two ranks whose caches were sized differently
would silently train on different orders. When a requested block is wider than the cache can
hold, the loader says so rather than quietly picking one for you.

Pass `shuffle_block_size=0` for a true global shuffle. It is correct, and it is only
affordable when `cache_size` covers the whole corpus.

### Epochs and resume

Both indexed loaders carry the two conventions a training loop already knows.
`set_epoch(n)` is `torch.utils.data.DistributedSampler`'s: call it once per epoch, on every
rank, with the same value. Skip it and every epoch replays one order, which costs
convergence without ever failing.

`state_dict()` and `load_state_dict()` are MosaicML Streaming's. Take the state between
steps, where every rank has consumed the same count, and a resumed run continues the same
epoch with no sample repeated and none skipped:

```python
loader = shard_stream_loader(path, batch_size=10, seed=42)

for epoch in range(3):
    loader.set_epoch(epoch)
    for step, batch in enumerate(loader):
        if step == 2:
            checkpoint = loader.state_dict()  # save this next to the model weights
            break

print(checkpoint)
# {'epoch': 0, 'global_consumed': 30}
```

```python
resumed = shard_stream_loader(path, batch_size=10, seed=42)
resumed.load_state_dict(checkpoint)
print(len(resumed))  # the batches this rank had not reached
# 7
```

Because the global order is independent of `world_size`, that checkpoint also restores onto
a **differently sized cluster**: `global_consumed` is a position in the global order, not a
per-rank one.

:::{warning}
Read `state_dict()` from the object your loop iterates. Under `DataLoader(num_workers=k)`
the loader is pickled into *k* worker processes, so the copy left in the parent never
advances and reports a resume point of zero — which resumes by replaying the whole epoch.
Checkpoint from a loop over the loader itself, or use `num_workers=0`.
:::

:::{note}
`set_epoch` on {py:meth}`stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` re-reads the corpus, because a new epoch is a new
permutation and *which rows belong to this rank* changes with it. On `shard_stream_loader` it
is arithmetic. That difference is one more reason to write shards once the corpus is large.
:::

### Surviving a crash, and scaling past a million shards

A corpus write is hours of work and a training read is days of it, so both are built to
lose as little as possible when something fails.

The manifest is republished **as the write proceeds**, not once at the end, so the corpus on
disk is readable at every moment. A write that dies leaves a shorter but complete corpus
rather than a directory of orphaned shards, and `resume=True` continues it — the rows already
written are skipped from the source rather than re-encoded:

```python
try:
    corpus.ml.write_shards(path, rows_per_shard=32)
except KeyboardInterrupt:
    pass

# Whatever landed is already readable...
print(bt.read.training_shards(path).count() % 32)
# 0
# ...and the write continues from there.
index = corpus.ml.write_shards(path, rows_per_shard=32, resume=True)
print(index.total_rows)
# 100
```

Only whole shards count as written. A partial one from the previous attempt is redone, so a
corpus never ends up with a short shard in the middle — which would break global indexing
without failing.

Shards are published concurrently, which is what makes a large corpus write in reasonable
time against object storage: each shard is a round trip of tens of milliseconds, so
publishing them one at a time left the encoder idle and the write latency-bound. The number
in flight is sized from the destination and the measured shard size, so the memory held is a
few shards rather than a function of the corpus. Override it with `write_concurrency` if you
need to.

Reads and writes both retry the failures worth retrying. A throttle, a 503, or a dropped
connection is a blip that ends a multi-day training run if nothing catches it; a 404 or a 403
is a fact, and is surfaced immediately rather than backed off.

The manifest also does not grow with the shard count. A petabyte corpus in 256 MB shards is
around four million shards, and naming each one would be a 170 MB JSON document parsed on
every rank at startup plus four million resident strings. Because `write_shards` produces
shards of a known width under generated names, the manifest records a *count*: locating a row
is integer division, and the file is a few hundred bytes whatever the corpus size.

:::{note}
A checkpoint records the corpus size and the seed it came from, and `load_state_dict`
refuses one that does not match. Resuming against a corpus that has since grown keeps
`global_consumed` meaningful as a *position* while making the order something else, so the
samples it records as consumed are not the ones the run actually saw. That is invisible in a
loss curve, so it is refused rather than discovered later.
:::

## streaming_split: an unbounded source

A stream has no length, so there is no index to shard on. `streaming_split` fans a
*single* read of the source out to `world_size` rank iterators, round-robin, with
bounded queues. The data is read once total, not once per rank.

```python
from batcher.ml import streaming_split

train = bt.from_pydict({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "label": [0, 1, 0, 1, 0, 1]})

rank0 = streaming_split(train, world_size=2, rank=0, batch_size=1, device="cpu")
print([b["f"].tolist() for b in rank0])
# [[1.0], [3.0], [5.0]]
```

Called without `rank` it returns a list of `world_size` iterators instead, and those must
be consumed **concurrently**, because one reader is feeding all of them through bounded
queues. Consume them serially and the first one blocks forever waiting for a queue the
others are not draining.

Only complete rounds of `world_size` batches are emitted; a trailing partial round is
dropped, so every rank yields the same count and none of them stalls the all-reduce.

## Framework converters

The converters take a batch iterator and yield framework-native batches. {py:meth}`to_numpy_batches <batcher.api.dataset.ml.DatasetML.to_numpy_batches>`
is the NumPy one:

```python
from batcher.ml import to_numpy_batches

for arrays in to_numpy_batches(ds.iter_batches(batch_size=3), columns=["f0", "label"]):
    print(sorted(arrays), arrays["f0"].shape)
# ['f0', 'label'] (3,)
# ['f0', 'label'] (3,)
```

`to_torch_iterable` and `to_tf_dataset` are the same idea for the other two frameworks.
All three take a batch iterator, so they compose with anything upstream that yields
batches, including a hand-built pipeline.

### TensorFlow

{py:meth}`ds.ml.to_tf <batcher.api.dataset.ml.DatasetML.to_tf>` returns a `tf.data.Dataset` and takes the same stream options the
PyTorch loader does, because it prepares its stream with the same code:

```python
tf_ds = ds.ml.to_tf(
    batch_size=2,
    local_shuffle_buffer_size=4,
    seed=42,
    drop_last=True,
    dtypes={"f0": "float32"},
)
first = next(iter(tf_ds))
print(sorted(first), first["f0"].dtype.name, int(first["f0"].shape[0]))
# ['f0', 'f1', 'label'] float32 2
```

`drop_last` matters more here than under PyTorch: a fixed-shape Keras graph cannot take a
short final batch. Add `.prefetch(tf.data.AUTOTUNE)` to the returned dataset for the
overlap `prefetch_batches` gives the torch path — that knob belongs to `tf.data`, so this
does not duplicate it.

:::{note}
`dtypes` is applied in NumPy, before TensorFlow sees the batch, which halves what has to
be copied when you narrow a float64 column. That also means it accepts only dtypes NumPy
represents numerically: `bfloat16` is refused with a `PlanError` rather than silently
producing an opaque column. Cast to bf16 inside the model instead.
:::

## Do the work upstream

:::{tip}
Anything you can express as an operator should be an operator. Filters, projections,
feature arithmetic, casts, and a fitted preprocessor's transform all run inside the
engine, vectorized and parallel, and out of the training process entirely.
:::

```python
from batcher import col

prepared = (
    ds.filter(col("f1") > 1.0)
    .with_columns(f_scaled=col("f1") / 6.0)
    .select("f0", "f_scaled", "label")
)
first = next(prepared.iter_batches())
print(first.schema.names, first.num_rows)
# ['f0', 'f_scaled', 'label'] 5
```

For learned statistics such as standardization, one-hot encoding, or imputation, fit a
{doc}`preprocessor </ml/preparing/preprocessors/index>` on the train split and `transform` the stream. The fit
is one mergeable pass over the data, and the transform is an engine stage. Neither one
runs in the training loop.

## Diagnosing an idle GPU

If the GPU is starved, the loader is rarely the culprit. The stage feeding it usually is.
In order of frequency:

| What is actually happening | What to do about it |
| --- | --- |
| Per-row Python in a UDF | make it a vectorized batch function |
| A tokenizer or decoder running every epoch | run it once, write the column out |
| `batch_size` too small to amortize the kernel launch | size the batch to the model, not to the file |
| `prefetch_batches=0`, so the device copy is serialized against compute | prefetch, and pin the host memory |
| A global shuffle where a local buffer would have done | `local_shuffle_buffer_size` |

## See also

- {doc}`PyTorch </ml/inference/pytorch>`: device transfer, collate, zero-copy, and DDP in full.
- {doc}`Distributed training </ml/training/distributed-training>`: the multi-rank sharding contract.
- {doc}`Streaming for training </ml/inference/streaming>`: the bounded-memory ingest path.
- {doc}`Preprocessors </ml/preparing/preprocessors/index>`: the fit-on-train, transform-the-stream contract.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: how a fixed-shape tensor reaches the
  loop with its shape intact.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: what the device is waiting on when it
  is waiting.
- {doc}`Distributed training pipeline </tutorials/ml/distributed-training-pipeline>`: the
  whole path, from files to a loop.
- {doc}`ML API </api/models/ml>`: the loader and converter reference.
