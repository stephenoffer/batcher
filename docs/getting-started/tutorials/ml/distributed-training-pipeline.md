# A distributed training pipeline

Feed a data-parallel PyTorch job a balanced, deterministic, resumable stream of tensors per rank, with the feature shaping and splitting done in the engine. The shaping and the loader run here on CPU. Blocks that need a cluster or GPUs are marked.

| Step | Runs here | Needs |
|---|---|---|
| Shape, split, fit, {py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` | Yes | `pip install batcher-engine` |
| `epoch_order`, {py:meth}`stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` | Yes | Nothing more |
| The DDP training loop | No | GPUs, NCCL, `torch.distributed` |
| Sharded corpus, distributed preprocessing | No | A cluster and object storage |

The engine side of this page is one flow, from shaped features to one stream per rank:

![The featured dataset is split by a hash of its key into train, 44 rows in the example, and test, 20 rows held out. The scaler is fitted on train only, so its statistics come from train. Those statistics transform train into train_x and also transform test into test_x, so both parts are scaled with train's statistics. train_x then feeds stream_loader, where rank 0 and rank 1 each read a disjoint slice. Split before you fit, because a fit on test rows raises no error and still leaks.](/_static/diagrams/training_data_flow.svg)

## 1. Shape the features in the engine

Feature work belongs in the engine. Expressions run in Rust across every core, ahead of the GPU.

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

Split first, then fit, so no test statistic reaches the features. Each row is assigned by a reproducible hash, so the parts are disjoint and identical on one core or a cluster. `key=` hashes only the identifying column, so recomputing a feature keeps the same rows in train.

```python
train, test = featured.ml.train_test_split(test_size=0.25, seed=7, key="user_id")
print(train.count(), test.count())
# 44 20
```

Sizes land near `test_size * n` with no shuffle and no materialization. `random_split` and `kfold` use the same hashing:

```python
parts = featured.ml.random_split([0.5, 0.25, 0.25], seed=7, key="user_id")
print([p.count() for p in parts])
# [26, 18, 20]
folds = featured.ml.kfold(4, seed=7, key="user_id")
print([(tr.count(), va.count()) for tr, va in folds])
# [(47, 17), (55, 9), (46, 18), (44, 20)]
```

## 3. Fit the preprocessor on train, transform both

A {py:class}`StandardScaler <batcher.ml.preprocessors.StandardScaler>` fit is one mergeable pass, so it gives the same statistics on one core or a cluster:

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

{py:meth}`ds.ml.iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` yields `{column: tensor}` dicts from a bounded-memory stream, overlapping each host-to-device copy with the next batch's host work.

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

In real training, leave `device="auto"` (CUDA, ROCm, XPU, or MPS, else CPU) and set `pin_memory=True`. `local_shuffle_buffer_size=` shuffles within a streaming buffer. The loader streams at 1.06 M rows/s through zero-copy DLPack in the {doc}`AI and GPU benchmark </benchmarks/results/ai-and-gpu>`.

## 5. The sample order

The loader's order is *balanced* across ranks, *deterministic* for exact resume, *elastic* across `world_size`, and computed per rank with no coordinator. The ordering functions are usable on their own:

```python
from batcher.ml import epoch_order, usable_length

print(epoch_order(8, seed=42))
# [6, 4, 7, 3, 2, 5, 0, 1]
print(epoch_order(8, seed=42, epoch=1))
# [4, 0, 6, 5, 7, 3, 1, 2]
print(usable_length(8, 3), usable_length(8, 3, drop_last=False))
# 6 9
```

The next epoch reshuffles. `usable_length` is the epoch's sample positions, a multiple of `world_size`, trimmed with `drop_last=True` (the default) or padded without.

The order is computed, never materialized. `epoch_permutation` is a keyed bijection on `[0, n)`, so ten billion samples cost constant memory and any position is a direct lookup:

```python
from batcher.ml import epoch_permutation

perm = epoch_permutation(10_000_000_000, seed=42)
print(len(perm), perm[123_456_789])
# 10000000000 6584942393
```

## 6. One iterable per rank

{py:meth}`ds.ml.stream_loader <batcher.api.dataset.ml.DatasetML.stream_loader>` returns a `torch.utils.data.IterableDataset` over this rank's slice of
that global order.

:::{important}
`stream_loader` is the shard authority. Don't add a `DistributedSampler` on top of it, or each rank reads a slice of a slice.
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

Rank 1 passes `rank=1` and reads a disjoint slice. Bump `epoch` each epoch, and pass the checkpointed `global_consumed` on restart to resume exactly where the rank stopped.

## 7. The training loop

Everything above is engine work. The loop is yours, and it needs a GPU.

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
Pass `batch_size=None` to the `DataLoader`. The stream already yields sized batches.
:::

## 8. Larger than memory

`stream_loader` holds the dataset in memory. Past RAM, write the corpus as shards and stream from disk with the same sample-order contract.

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
from batcher.ml import shard_stream_loader

train_x.ml.write_shards("s3://corpus/train/", rows_per_shard=100_000)

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

When the corpus lives in object storage, run the shaping distributed. It's the same plan with the same rows, column names, and types as the single-node run.

```python
# docs: skip
featured = (
    bt.read.parquet("s3://corpus/events/")
    .with_columns(spend_per_click=bt.col("spend") / (bt.col("clicks") + 1.0))
    .ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="title", num_gpus=1)
)
featured.write.parquet("s3://corpus/features/", distributed=True)
```

## Where to go next

The loop itself, the framework boundary, or the features feeding it:

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
:link: /getting-started/tutorials/ml/feature-engineering
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
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: the 1.06 M rows/s quoted in
  step 4, and the configuration it was measured under.
- {doc}`Scaling out </benchmarks/results/scaling>`: what the distributed preprocessing in step 9 costs.
