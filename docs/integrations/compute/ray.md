# Ray

Ray is how Batcher gets onto a cluster, and it is used for exactly one thing: **scheduling**.
Tasks, actors, placement groups, and small control-plane messages (file paths, worker addresses,
metrics) go through Ray. Bulk Arrow batches do not. Shuffle data moves worker to worker over Arrow
Flight (`bc-transport`) with credit-based flow control, where one credit is one in-flight batch
slot and a producer blocks when its credits reach zero. The batches bypass the Ray object store
entirely.

That distinction is the whole integration. An object-store shuffle serializes every batch into
plasma, spills it under pressure, and makes the driver a funnel. Keeping bulk data off it is what
lets a distributed query move at line rate. Single-node execution never imports Ray at all.

| | |
| --- | --- |
| **Ray carries** | Tasks, actors, placement groups, and small control-plane messages |
| **Ray does not carry** | Bulk Arrow batches. Those move over Arrow Flight (`bc-transport`). |
| **Extra** | `pip install 'batcher-engine[ray]'` |
| **Entry point** | `collect(distributed=True)`, or `"auto"` (the default) |
| **From a Ray Dataset** | `bt.from_ray_dataset(rds)`, streamed block by block |
| **Cluster config** | `config.distributed` |

## Going distributed

There is no distributed API. It is the same plan.

::::{tab-set}

:::{tab-item} On one node
```python
# docs: skip
import batcher as bt

result = (
    bt.read.parquet("s3://lake/events/*.parquet")
    .filter(bt.col("status") == "purchase")
    .group_by("region")
    .agg(bt.col("amount").sum().alias("revenue"))
    .collect()
)
```
:::

:::{tab-item} On a cluster
```python
# docs: skip
import batcher as bt

result = (
    bt.read.parquet("s3://lake/events/*.parquet")
    .filter(bt.col("status") == "purchase")
    .group_by("region")
    .agg(bt.col("amount").sum().alias("revenue"))
    .collect(distributed=True)
)
```
:::

::::

`collect(distributed="auto")`, the default, uses Ray when it detects a multi-node cluster and runs
in-process otherwise. `True` and `False` force it. The result is identical either way, because the
distributed path composes the *same* mergeable primitives that the single-node parallel executor
uses: `partial`, then shuffle, then `combine`, then `finalize`. There is no second semantics to
disagree with the first.

:::{tip}
Do not reach for `distributed=True` reflexively. On a single node the in-process engine wins by a
wide margin; distribution is for scale-out and larger-than-memory, and it costs a shuffle.
:::

## Attaching to a cluster

`config.distributed` is where the cluster lives. `ray_address=None` (the default) attaches to a
running cluster when `RAY_ADDRESS` is set, or when a managed control plane is detected, and falls
back to starting a local Ray only when nothing is reachable.

Detection is env-var only, never a metadata-service call on a hot path. It covers Anyscale, any
KubeRay-operated cluster, which is the on-prem, self-hosted, and any-cloud Kubernetes case, and an
explicit `BATCHER_RAY_CLUSTER=1` escape hatch for a platform none of those name. No platform is
privileged; batcher behaves identically wherever it runs. If detection is wrong, `_ensure_ray`
still falls back to a local start when no cluster turns out to be reachable, so a false positive
degrades rather than fails.

A head that does not answer on the first try is usually still starting rather than absent. The
driver and the head come up concurrently in every orchestrated environment: a KubeRay driver pod is
admitted before the head passes its readiness probe, and a Slurm job's `ray start --head` races the
step that runs the query. Batcher retries the attach with exponential backoff for
`cluster_connect_timeout_s` seconds, 30 by default, before concluding anything. Set it to 0 to
attach once and fall back immediately.

What happens when that window is exhausted depends on how the address was found, and the difference
matters:

| How the address was found | Cluster unreachable |
|---|---|
| `ray_address`, or `RAY_ADDRESS` | Raises. You named a cluster, and running single-node in its place is a wrong answer, not a degraded one. |
| Detected from the environment | Starts a local single-node Ray, so a dev run inside a workspace whose cluster is down still works. |

You do not need to pre-install batcher on the workers: when batcher initializes Ray against a
cluster it ships its own package (including the compiled extension) via `runtime_env`. Set
`trust_cluster_image=True` to skip that upload when your image already bakes batcher in.

```python
from batcher import Config
from batcher.config import DistributedConfig

cfg = Config().replace(distributed=DistributedConfig(namespace="nightly-etl"))
print(cfg.distributed.namespace)
# nightly-etl
```

:::{dropdown} The four fields that matter on day one
`runtime_env` ships an environment to the workers. Every worker needs `batcher` **and its compiled
extension** importable; a cluster whose workers have a different wheel than the driver fails at the
first task, usually with an import error naming the native module.

`namespace` isolates a job's shuffle actors. Two jobs in one namespace can see each other's actors.

`transport` picks the shuffle. `"auto"` chooses Flight on a genuine multi-node cluster and a disk
shuffle on a single node. The disk shuffle writes to a driver-local `work_dir`, so it is correct
only on one node or with `shared_filesystem=True`, which is why `"auto"` will not choose it across
nodes. Forcing `transport="disk"` on a multi-node cluster without a shared filesystem is a way to
get tasks that cannot find their input.

`shuffle_token` (also read from `BATCHER_SHUFFLE_TOKEN`) authenticates Flight fetches, and
`distributed.tls` turns on TLS/mTLS between workers. On a shared or untrusted network, set both.
The shuffle is a data plane on the wire.
:::

The rest (retries, straggler speculation, skew salting, adaptive credits) is in
{doc}`configuration options </configuration/options>`, and the defaults fill a cluster with no
tuning: one worker per node, an even share of each node's cores, reducer count scaled to workers.

## Bringing in a Ray Dataset

`bt.from_ray_dataset(rds)` takes a Ray Dataset and streams its Arrow blocks into the engine, one
block per batch, lazily. It does not collect to the driver, so memory stays bounded.

```python
# docs: skip
import ray
import batcher as bt

rds = ray.data.read_parquet("s3://lake/events")
events = bt.from_ray_dataset(rds)
print(events.group_by("region").agg(bt.col("amount").sum()).sort("region").to_pydict())
```

Treat this as an on-ramp, not a destination. Whatever built the incoming dataset still costs what
it costs, and the hand-off cannot claw that back. Where the source is a plain read, read it with
Batcher instead: `bt.read.parquet` takes a 20 M-row, 64-file read-and-sum in **72 ms**
(`benchmarks/BENCHMARK_RESULTS.md`), because files decode concurrently in-process with no per-file
task scheduling and no object-store hop.

The inference idiom carries over unchanged. `ds.ml.map_batches` takes a class, loads the model once
per worker, and runs the batches through an actor pool, so a ported pipeline keeps its shape while
picking up warm pools and stage overlap. See {doc}`/ml/inference/inference`.

## Failure modes worth knowing

:::{warning}
**A preempted worker recomputes its partition.** Spot node reclaimed mid-batch? The partition is
reassigned and recomputed from its durable input. That makes idempotency a requirement, not a
nicety: a `map_batches` `fn` with an external side effect (a vector-DB insert, a REST POST) can
apply it twice. Make the sink an upsert on a stable key and recompute becomes exactly-once. A pure
transform is already safe.
:::

**`resilience="spot"`** hardens the retry and restart budgets as a bundle for a churning cluster.
Use it rather than tuning six knobs by hand.

**The autoscaler is asked, then waited for.** Batcher requests the cores a query wants and waits
(bounded by `placement_timeout_s`) for the nodes to arrive before sizing the fan-out, so a big
query runs on the grown cluster instead of clamping to the pre-scale size. If your jobs
consistently run on too few workers, that timeout is where to look.

**Ray started locally by accident.** On a managed workspace that exports no `RAY_ADDRESS`, an
explicit `ray.init()` in your own code before Batcher's can strand the job on a local single-node
Ray while the cluster sits idle. Either let Batcher attach, or set `ray_address` explicitly.

:::{important}
**Do not route bulk data through Ray objects.** If you find yourself calling `ray.put` on a
`RecordBatch` to move it between stages, stop. That is the object-store tax the architecture exists
to avoid. Batches move over Flight; Ray moves the paths.
:::

## See also

- {doc}`Execution architecture </architecture/execution>`: morsels, breakers, the shuffle.
- {doc}`Shuffle over Arrow Flight </deep-dives/distribution/shuffle-flight>`: the transport that replaces the
  object store, and its credits.
- {doc}`Distributed scheduling </deep-dives/distribution/distributed-scheduling>`: how a plan becomes Ray tasks.
- {doc}`Configuration options </configuration/options>`: every distributed knob.
- {doc}`Fault tolerance </architecture/fault-tolerance>`: retries, recovery, speculation.
- {doc}`PyTorch </integrations/compute/pytorch>`: distributed training ingest.
