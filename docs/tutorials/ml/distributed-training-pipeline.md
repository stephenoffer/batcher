# A distributed training pipeline

Feed a data-parallel PyTorch job without starving it. The training loop is not the hard part. The
hard part is the last mile: shaping features, splitting the data, and handing each rank a
balanced, deterministic, resumable stream of tensors that keeps the GPU busy.

The shaping and the loader run here on CPU. The blocks that need a cluster or GPUs are shown
and marked.

:::{note}
**What you'll build.** A 64-row feature table, a leak-free hash split, a fitted scaler, a
tensor loader, and a per-rank stream with a deterministic, resumable global sample order. All
of that runs on CPU with `pip install batcher-engine`. The DDP loop and the sharded corpus
need GPUs and a cluster, so they are shown and not run.
:::

| Step | Runs here | Needs |
|---|---|---|
| Shape, split, fit, {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` | Yes | `pip install batcher-engine` |
| `epoch_order`, {py:meth}`stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` | Yes | Nothing more |
| The DDP training loop | No | GPUs, NCCL, `torch.distributed` |
| Sharded corpus, distributed preprocessing | No | A cluster and object storage |

## 1. Shape the features in the engine

Feature work belongs in the engine, not in the training loop. Expressions run in Rust across
every core; a `map_batches` in your `DataLoader` runs in Python, one worker at a time, while
the GPU waits.

```python
import batcher as bt

n = 64
events = bt.from_pydict(
    {
        "user_id": list(range(n)),
        "clicks": [float(i % 7) for i in range(n)],
        "spend": [float(i % 13) for i in range(n)],
        "label": [i % 2 for i in range(n)],
    }
)

featured = events.with_columns(
    spend_per_click=bt.col("spend") / (bt.col("clicks") + 1.0),
)
print(featured.columns)
# ['user_id', 'clicks', 'spend', 'label', 'spend_per_click']
```

## 2. Split before you fit

:::{important}
The split comes first, because a preprocessor fitted on the test rows has already leaked. No
error is raised and no metric looks wrong. The model scores better offline than it
ever will in production. Split, then fit, in that order, every time.
:::

Each row is assigned by a reproducible hash of its own values, so the two parts are disjoint
and identical however the data is partitioned, on one core or on a cluster.

Pass `key=` on a real corpus. Hashing only the identifying column keeps the split stable when
the *other* columns change: recompute a feature and the same rows stay in train. Hash every
column (the default) and the split moves the moment a value does.

```python
train, test = featured.ml.train_test_split(test_size=0.25, seed=7, key="user_id")
print(train.count(), test.count())
# 44 20
```

Sizes are binomial around `test_size × n` rather than exact, which is what a hash-keyed split
buys you: no shuffle, no materialization, and the same assignment on every node.

## 3. Fit the preprocessor on train, transform both

A {py:class}`StandardScaler <batcher.ml.preprocessors.StandardScaler>` is a fit/transform pair, and the fit is one mergeable pass over the data,
the same `partial → combine → finalize` algebra the aggregates use, so it runs on one core or
a cluster with the same result. The transform stays inside the engine.

```python
from batcher.ml import StandardScaler

scaler = StandardScaler(["clicks", "spend", "spend_per_click"])
scaler.fit(train)

train_x = scaler.transform(train)
test_x = scaler.transform(test)
print(train_x.columns)
# ['user_id', 'clicks', 'spend', 'label', 'spend_per_click']
```

The statistics come from `train` only. `test_x` is transformed with them, never with its own.

## 4. Stream tensors, single process

{py:meth}`ds.ml.iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` yields `{column: tensor}` dicts. It consumes the stream
incrementally, so memory stays bounded and the loop starts before the whole dataset is read,
and it overlaps the host→device copy of one batch with the host work of the next.

```python
loader = train_x.select("clicks", "spend", "spend_per_click", "label").ml.iter_torch_batches(
    batch_size=16,
    device="cpu",
)
batches = list(loader)
print(len(batches), sorted(batches[0]))
# 3 ['clicks', 'label', 'spend', 'spend_per_click']
print(tuple(batches[0]["clicks"].shape))
# (16,)
```

In real training, leave `device="auto"` (it picks CUDA, ROCm, XPU, or MPS and falls back to
CPU) and set `pin_memory=True` so the copies are async. `local_shuffle_buffer_size=` gives a
streaming approximation of a shuffle without materializing the dataset.

On the benchmark this loader streams **1.06 M rows/s**, because it is zero-copy through
DLPack rather than a per-batch Arrow-to-tensor conversion.

## 5. The sample order, before you trust it

For data-parallel training you need four things from the loader, and it is worth checking
them rather than assuming them: the ranks must be **balanced** (nobody stalls at the
all-reduce barrier), the order must be **deterministic** (so a resume is exact), it must be
**elastic** (the same seed and epoch give the same global order at *any* `world_size`), and
the ranks must be **independent** (no central coordinator).

The ordering functions are usable on their own, which is the easiest way to see what a resumed
epoch will actually read:

```python
from batcher.ml import epoch_order, usable_length

print(epoch_order(8, seed=42))
# [6, 4, 7, 3, 2, 5, 0, 1]
print(epoch_order(8, seed=42, epoch=1))
# [4, 0, 6, 5, 7, 3, 1, 2]
print(usable_length(8, 3), usable_length(8, 3, drop_last=False))
# 6 9
```

The next epoch reshuffles. `usable_length` is how many sample positions the epoch spans:
always a multiple of `world_size`, trimmed with `drop_last=True` (the default) or padded
without.

That order is **computed, not materialized**. A shuffled index list costs about 28 bytes per
sample in CPython, so a 10-billion-sample corpus would want roughly 280 GB of driver RAM
before a single row is read. `epoch_permutation` is a keyed pseudorandom bijection on
`[0, n)` instead: index in, shuffled index out, no state. Its memory is constant whatever the
corpus size, and seeking to sample 900,000,000,000 is modular arithmetic rather than a walk.

## 6. One iterable per rank

{py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` returns a `torch.utils.data.IterableDataset` over this rank's slice of
that global order.

:::{warning}
`stream_loader` is the *only* shard authority. Do not add a `DistributedSampler` on top of
it: the two will shard the same data twice, each rank will see a slice of a slice, and most
of your corpus will silently never reach the model. Nothing errors. The loss curve gets
worse for a reason you cannot see.
:::

```python
rank_stream = train_x.ml.stream_loader(
    batch_size=8,
    world_size=2,
    rank=0,
    epoch=0,
    seed=1,
    columns=["clicks", "spend", "spend_per_click", "label"],
    global_consumed=0,  # a checkpointed offset resumes mid-epoch
)
first = next(iter(rank_stream))
print(sorted(first), tuple(first["label"].shape))
# ['clicks', 'label', 'spend', 'spend_per_click'] (8,)
```

Rank 1 constructs the identical object with `rank=1` and reads a disjoint slice. Bump `epoch`
at the top of each epoch to reseed the shuffle; pass the checkpointed `global_consumed` on
restart and the rank resumes exactly where it stopped, with no sample repeated or skipped.

## 7. The training loop

Everything above is engine work. This part is yours, and it needs a GPU, so it is shown and
not run.

:::{dropdown} The DDP training loop, in full
```python
# docs: skip
import torch
import torch.distributed as dist


def train(rank: int, world_size: int, epoch: int, resume_offset: int = 0) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = torch.nn.Linear(3, 2).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    opt = torch.optim.Adam(model.parameters())

    stream = train_x.ml.stream_loader(
        batch_size=256,
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        seed=42,
        columns=["clicks", "spend", "spend_per_click", "label"],
        global_consumed=resume_offset,
    )

    for batch in torch.utils.data.DataLoader(stream, batch_size=None):
        features = torch.stack(
            [batch["clicks"], batch["spend"], batch["spend_per_click"]], dim=1
        ).cuda()
        loss = torch.nn.functional.cross_entropy(model(features), batch["label"].cuda())
        loss.backward()
        opt.step()
        opt.zero_grad()
```
:::

:::{tip}
`batch_size=None` on the `DataLoader` is not a typo. The stream already yields sized batches;
letting torch re-batch them would collate twice.
:::

## 8. Larger than memory

`stream_loader` materializes the dataset once, which is fine up to RAM. Past that, write the
corpus as shards and stream from disk. The sample-order contract is the same either way; only
the residency changes.

::::{tab-set}
:::{tab-item} Fits in memory
`stream_loader` over a resident dataset. This is the object step 6 already built.

```python
# docs: skip
stream = train_x.ml.stream_loader(
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    seed=42,
)
```
:::

:::{tab-item} Larger than memory
Shards on object storage, with a bounded shard cache instead of a resident dataset.

```python
# docs: skip
from batcher.io.formats.ml import write_shards
from batcher.ml import shard_stream_loader

write_shards(train_x, "s3://corpus/train/", rows_per_shard=100_000)

stream = shard_stream_loader(
    "s3://corpus/train/",
    batch_size=256,
    world_size=world_size,
    rank=rank,
    epoch=epoch,
    seed=42,
)
```
:::
::::

For a source with no global length (a Kafka topic, an unbounded file feed),
{py:func}`batcher.ml.streaming_split <batcher.ml.streaming_split>` fans one read of the stream out to `world_size` rank iterators,
consumed concurrently with backpressure.

## 9. Preprocessing on the cluster

When the corpus lives in object storage and the features are expensive, run the shaping
distributed. It is the same plan; only the scheduling changes, and the result is the same rows,
column names, and column types as the single-node one. A floating-point reduction is identical
up to reassociation, since the partition count sets the summation order.

```python
# docs: skip
featured = (
    bt.read.parquet("s3://corpus/events/")
    .with_columns(spend_per_click=bt.col("spend") / (bt.col("clicks") + 1.0))
    .ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="title", num_gpus=1)
)
featured.write.parquet("s3://corpus/features/", distributed=True)
```

## What you learned

- Shape features in the engine; hand the loop tensors, not work.
- Split, then fit. A scaler fitted before the split has leaked the test set.
- `stream_loader` owns the sharding, the shuffle, and the resume. Do not stack a
  `DistributedSampler` on top of it.
- The epoch order is a function, not a table, which is why it costs the same at a trillion
  samples as at ten million.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Streaming for training
:link: /ml/inference/streaming
:link-type: doc
Every loader option, in full.
:::

:::{grid-item-card} {octicon}`plug;1.1em` PyTorch integration
:link: /ml/inference/pytorch
:link-type: doc
Device transfer, prefetch, collate, zero-copy.
:::

:::{grid-item-card} {octicon}`gear;1.1em` Feature engineering
:link: /tutorials/ml/feature-engineering
:link-type: doc
Preprocessors and {py:class}`Chain <batcher.ml.preprocessors.Chain>`, the step-3 story in full.
:::
::::

## See also

- {doc}`Distributed training guide </ml/training/distributed-training>`: DDP, elasticity, and the
  resume contract.
- {doc}`Data loaders </ml/training/data-loaders>`: `iter_torch_batches` and `stream_loader` side by
  side.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: the DLPack path behind the zero-copy
  claim in step 4.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why the `fit` in step 3 gives the
  same statistics on a cluster.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: the 3.0× on `iter_torch_batches`, and
  the configurations it was measured under.
- {doc}`Scaling out </benchmarks/results/scaling>`: what the distributed preprocessing in step 9 costs.
