# Distributed training

Batcher does not train models. It feeds them, which at 512 ranks is most of the
problem. The data path has to give every rank a disjoint slice, give them all the *same
number* of batches (or the fast ranks sit at the all-reduce barrier waiting for the slow
one), reproduce the epoch order after a crash, and do all of that without materializing
an index list that would not fit in driver RAM.

This page is the data-side contract. The training loop itself is yours.

## The four guarantees

`ds.ml.stream_loader` gives each rank a `torch.utils.data.IterableDataset` over its
slice of one global order, and holds four properties a distributed loop actually needs.

**Balanced.** Every rank yields the same number of batches. `drop_last=True` (the
default) trims the epoch's tail to a multiple of `world_size`; `drop_last=False` keeps
every sample and pads by repeating a few. Neither hands the ranks unequal counts, so no
rank finishes early and stalls the barrier.

**Deterministic and elastic.** The global order is a function of `(seed, epoch)` alone,
not of `world_size` and not of how the data is partitioned. So a job that dies on 64 GPUs
can resume on 32 and see the same permutation.

**Resumable.** Pass `global_consumed` (the sample count already processed this epoch,
read from your checkpoint) and the rank picks up mid-epoch with no repeated and no
skipped samples.

**Independent.** Each rank computes its own index slice with no central coordinator, so
a slow rank never blocks the others.

```python
# docs: skip
import batcher as bt
from torch.utils.data import DataLoader

ds = bt.read.parquet("s3://bucket/train/*.parquet")

iterable = ds.ml.stream_loader(
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    seed=42,
    columns=["features", "label"],
    global_consumed=resume_offset,  # 0 for a fresh epoch
)
# stream_loader is the only shard authority: do not add a DistributedSampler.
for batch in DataLoader(iterable, batch_size=None):  # already batched
    train_step(batch["features"].cuda(), batch["label"].cuda())
```

:::{warning}
Bump `epoch` at the top of each epoch so the shuffle reseeds, and keep `seed` fixed for
the life of the run. A job that re-draws its seed at startup resumes against a
*different* permutation, so it re-shows the model data it already trained on this epoch
and skips data it never saw. Nothing in the loss curve will tell you.
:::

## The order is computed, not stored

A shuffled list of every sample index costs about 28 bytes per sample in CPython. For a
10-billion-sample corpus that is 280 GB of driver RAM, before a single row is read. So
the order is not a list: `epoch_permutation` is a keyed pseudorandom bijection on
`[0, n)`. Index in, shuffled index out, no state.

| Corpus samples | Shuffled index list | Batcher (computed) |
| --- | --- | --- |
| 10 M | 280 MB | 0.5 MB |
| 1 B | 28 GB | 0.8 MB |
| 1 T | 28 TB | 0.8 MB |

Because the order is a function rather than a table, seeking is O(1): resuming at sample
900,000,000,000 of a trillion is a modular-arithmetic step, not a walk.

The two functions the whole contract rests on are usable on their own, which is the
easiest way to see exactly what a rank will read.

```python
from batcher.ml import epoch_order, usable_length

print(epoch_order(8, seed=42))
# [6, 4, 7, 3, 2, 5, 0, 1]
print(epoch_order(8, seed=42, epoch=1))  # the next epoch reshuffles
# [4, 0, 6, 5, 7, 3, 1, 2]

print(usable_length(8, 3), usable_length(8, 3, drop_last=False))
# 6 9
```

`usable_length` is how many sample positions the epoch spans: always a multiple of
`world_size`, rounded down when `drop_last=True` (the remainder is dropped) or up when
`False` (the remainder is padded).

`epoch_order` materializes the whole order, so it costs O(n) memory, which is fine for a
test or a corpus that fits in driver RAM. For the real thing, `rank_index_batches`
streams one batch of indices at a time, in constant memory, and is what the
larger-than-RAM loader draws from.

```python
from batcher.ml import rank_index_batches

# Rank 3 of 1024, over a trillion-sample corpus, in constant memory.
for indices in rank_index_batches(10**12, batch_size=8, world_size=1024, rank=3, seed=1):
    print(len(indices))
    break
# 8
```

## Checkpointing the position

`ResumableSampler` owns the `(epoch, global_consumed)` pair and speaks the
`state_dict` / `load_state_dict` protocol your checkpoint already uses, so the loop never
computes a sample offset by hand.

```python
from itertools import islice

from batcher.ml import ResumableSampler

sampler = ResumableSampler(1000, world_size=2, rank=0, seed=42)
seen = list(islice(sampler, 3))   # three steps in
state = sampler.state_dict()      # checkpoint, between steps

resumed = ResumableSampler(1000, world_size=2, rank=0, seed=42)
resumed.load_state_dict(state)
print(len(resumed), set(seen) & set(resumed))
# 497 set()
```

497 remaining of the rank's 500, and no overlap with what it already saw.
`set_epoch(n)` reshuffles and rewinds, the `DistributedSampler` protocol.

:::{tip}
Take the `state_dict` **between steps**, where every rank has consumed the same count. A
checkpoint written mid-step captures ranks at different offsets, and the resume is then
skewed by exactly that difference.
:::

Restoring a state from a different corpus size or seed raises rather than silently
reshuffling, which is the guard against the failure above.

## Larger than RAM, and unbounded

`stream_loader` materializes the dataset once, which is fine up to RAM. Past that, the
source decides which loader you get.

::::{tab-set}
:::{tab-item} A corpus larger than RAM

Write the corpus into shards with `batcher.io.formats.ml.write_shards` and read it with
`shard_stream_loader`, which keeps a bounded shard cache and holds the *identical*
sample-order contract.

```python
# docs: skip
from batcher.ml import shard_stream_loader

iterable = shard_stream_loader(
    "s3://bucket/shards/",
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    seed=42,
    cache_size=4,
)
```

:::

:::{tab-item} An unbounded source

There is no global length to index, so there is nothing to permute. `streaming_split`
fans a single read out to `world_size` rank iterators instead: read once, distributed
round-robin, with backpressure. It emits only complete rounds, so the ranks stay
balanced.

```python
import batcher as bt
from batcher.ml import streaming_split

live = bt.from_pydict({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "label": [0, 1, 0, 1, 0, 1]})

rank1 = streaming_split(live, world_size=2, rank=1, batch_size=1, device="cpu")
print([b["f"].tolist() for b in rank1])
# [[2.0], [4.0], [6.0]]
```

:::
::::

:::{warning}
Called without `rank`, `streaming_split` hands back a list of `world_size` iterators, and
they must be consumed **concurrently**. One reader feeds all of them through bounded
queues, so draining one serially deadlocks on the others: no error, no progress.
:::

## How this compares

| System | Global shuffle | Mid-epoch resume | Balanced ranks | Elastic world size |
| --- | --- | --- | --- | --- |
| `DistributedSampler` | in-RAM index list, O(n) per rank | no | yes (pads) | no |
| WebDataset | shard order + local buffer (approximate) | no | heuristic | no |
| MosaicML Streaming | shard/block shuffle, bounded | yes | yes | yes |
| Ray Data `streaming_split` | local buffer only | no | not guaranteed | n/a |
| Batcher | exact, O(1) memory | yes | yes (drop or pad) | yes |

The distinction worth being precise about: WebDataset and MosaicML shuffle
*approximately*, with a shard permutation plus a local buffer, so two samples in the same
shard stay correlated. Batcher's is an exact permutation of the whole corpus, and it uses
less memory than either, because it is never stored.

## Preparing the data, once

Everything upstream of the loader is a `Dataset`, so the split, the feature transform,
and the dedup all run as engine stages: distributed, vectorized, and out of the training
process.

```python
import batcher as bt
from batcher.ml import Chain, SimpleImputer, StandardScaler

raw = bt.from_pydict(
    {
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "age": [20.0, 30.0, None, 50.0, 25.0, 35.0, 45.0, 55.0],
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
    }
)

train, test = raw.ml.train_test_split(0.25, seed=0, key="id")

pipeline = Chain(SimpleImputer(["age"]), StandardScaler(["age"]))
pipeline.fit(train)                      # statistics from train only
train_ready = pipeline.transform(train)  # lazy; runs in the engine
test_ready = pipeline.transform(test)

print(train.count() + test.count(), train_ready.columns)
# 8 ['id', 'age', 'label']
```

:::{important}
Fit on train only. Fitting the scaler on train+test leaks the test distribution into the
model, and it is the easiest leak in the world to ship without noticing: every metric
improves, and the improvement is not real.
:::

## See also

- [Data loaders](data-loaders.md): the loader map and the framework converters.
- [PyTorch](pytorch.md): DDP/FSDP wiring on the training side.
- [Preprocessors](preprocessors.md): the fit/transform contract.
- [Streaming for training](streaming.md): the bounded-memory ingest path in depth.
- [Distributed training pipeline](../tutorials/distributed-training-pipeline.md): the
  tutorial, from raw files to a multi-rank loop.
- [Distributed scheduling](../deep-dives/distributed-scheduling.md): what the engine is
  doing while the ranks are reading.
- [Train/test split recipe](../examples/ml/train-test-split.md): the leak-free split, on
  its own.
- [Scaling benchmarks](../benchmarks/scaling.md): what the ingest path holds up to.
- [ML API](../api/ml.md): the `stream_loader`, `ResumableSampler`, and `epoch_order`
  reference.
