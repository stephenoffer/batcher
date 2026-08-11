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
| **Entry point** | {py:meth}`collect(distributed=True) <batcher.Dataset.collect>`, or `"auto"` (the default) |
| **From a Ray Dataset** | {py:func}`bt.from_ray_dataset(rds) <batcher.from_ray_dataset>`, streamed block by block |
| **Back to a Ray Dataset** | {py:meth}`ds.to_ray_dataset() <batcher.Dataset.to_ray_dataset>`, coalesced into Ray-sized blocks |
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

:::{warning}
**A Ray Dataset can carry Arrow types nothing else reads.** Ray Data stores tensor and
opaque-Python columns as its own extension types (`ray.data.arrow_tensor_v2`,
`ray.data.arrow_pickled_object`), which standard Parquet tooling, Polars, and DuckDB either
error on or silently skip. Plain columns cross unchanged; a tensor column is worth checking with
`rds.schema()` before you rely on it. Batcher's own tensor columns are ordinary Arrow
(`FixedSizeList`, or a `struct<data, shape, dtype>` for ragged shapes), so the return leg does
not introduce them.
:::

Treat this as an on-ramp, not a destination. Whatever built the incoming dataset still costs what
it costs, and the hand-off cannot claw that back. Where the source is a plain read, read it with
Batcher instead: {py:meth}`bt.read.parquet <batcher.api.io_namespace.reader.Reader.parquet>` takes a 20 M-row, 64-file read-and-sum in **72 ms**
(`benchmarks/BENCHMARK_RESULTS.md`), because files decode concurrently in-process with no per-file
task scheduling and no object-store hop.

The inference idiom carries over unchanged. {py:meth}`ds.ml.map_batches <batcher.api.dataset.ml.DatasetML.map_batches>` takes a class, loads the model once
per worker, and runs the batches through an actor pool, so a ported pipeline keeps its shape while
picking up warm pools and stage overlap. See {doc}`/ml/inference/inference`.

## Handing a result back to Ray

{py:meth}`ds.to_ray_dataset() <batcher.Dataset.to_ray_dataset>` is the return leg, for the case
where the next stage of the job is Ray's rather than Batcher's: Ray Train, Ray Tune, or a Serve
deployment that wants a `ray.data.Dataset`.

```python
# docs: skip
import batcher as bt

features = (
    bt.read.parquet("s3://lake/events/*.parquet")
    .filter(bt.col("label").is_not_null())
    .select("user_id", "features", "label")
)
train_ds = features.to_ray_dataset()
print(train_ds.count())
```

Output batches are coalesced into blocks near Ray Data's own `target_max_block_size` and put into
the object store one block at a time, so the driver holds one block rather than the whole result,
and the Ray Dataset that comes out blocks the way a `read_parquet` one does. An empty result keeps
its schema, so a filter that matches nothing hands Ray a typed dataset rather than an untyped one.

The blocks are produced on the driver, which is the right shape for a result that has already been
reduced: a training set, a scored table, an embedding index. For a result the size of the input,
write Parquet and give Ray the path instead. That keeps the data on the workers that produced it
and costs one pass over storage rather than a funnel through one process.

## Placement, and what it costs

Batcher gang-schedules a shuffle fleet as a placement group, one bundle per worker, so the whole
fleet exists before the shuffle starts. Two things about that placement are worth knowing because
they are the difference between a cluster bill and a cheaper one.

**A fleet is pinned to one availability zone when it fits in one.** A shuffle moves nearly all of
its bytes worker to worker, and every cloud prices and delays those bytes by whether the two
workers share a zone. This is not a rounding item: Anyscale's field engineering reports cross-AZ
transfer exceeding 40% of total AWS spend on distributed workloads, and 20-40% added latency on
synchronous ones. A fleet spread evenly over three zones sends about two thirds of its shuffle
across that boundary for no benefit, since the bundles are interchangeable. Batcher picks the zone
with the most free capacity that can host the whole fleet and reserves the bundles there. It is a
no-op on a single-zone cluster, on nodes with no zone label, and whenever no one zone has room —
and because the pin is on the bundles rather than the tasks, a group that cannot form is abandoned
at the placement timeout and the stage falls back to ordinary scheduling. Set
`distributed.zone_aware_placement=False` when zone diversity is being bought deliberately for
availability.

If you control how the cluster is provisioned, pinning it there is better still: a single-AZ
compute config (`STRICT_ZONAL_PACK` on Anyscale) removes the cross-zone traffic rather than
routing around it, and also covers the head node. The runtime pin exists for the case where that
is not available — most obviously an accelerator fleet, where scarce instance types make
cross-zone autoscaling the only way to get capacity at all, and the cluster therefore spans zones
whether the job wants it to or not.

**A shuffle replica avoids the primary's failure domain, not just its node.** With
`distributed.shuffle_replication` above 1, each mapper's output is copied to a survivor so a
worker loss costs a re-fetch rather than a recompute. The copy has always gone to a different
node; it now also prefers a node that is *not* spot when the primary's is. A spot reclamation
takes an instance group rather than a machine, so a second copy on another spot node goes away
in the same wave as the first — on exactly the fleet the `spot` resilience profile turns
replication on for. Spot is read from `ray.io/market-type` and from the Karpenter, EKS, and GKE
capacity labels, since a KubeRay fleet is labelled by whichever provisioner brought the node up.
It is a preference, never an exclusion: an all-spot fleet, or one with no capacity labels at
all, places its copies exactly as before.

**A reservation that does not form now says why.** A gang that cannot be satisfied used to fall
back silently, and the tasks it fell back to ask for the same resources the bundles did — so the
query would hang at a barrier with nothing anywhere explaining it. Batcher now compares the ask
against the live topology and reports the outcome: an ask no node can host names the binding
resource and the widest node's figure, an ask every candidate node is too busy for says the cluster
is full and by how much, and a cluster with room says nothing at all. The same diagnosis is
attached to the map and inference barrier's stall warning, so a stage waiting on capacity that
will never arrive is distinguishable from one that is merely slow. The shuffle barrier still
prints the older, unresolved warning, because the reader that answers this lives in the
distributed layer and that barrier's loop is owned by Carbonite, which must not import it.

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
for the nodes to arrive before sizing the fan-out, so a big query runs on the grown cluster
instead of clamping to the pre-scale size. The wait is bounded by `autoscale_wait_s` (`"auto"`
by default: a bounded wait on an autoscaling cluster, off on a fixed one), and it stops early
once capacity has been flat for `autoscale_stall_s` or never grew at all within
`autoscale_startup_grace_s`. If your jobs consistently run on too few workers, those are the
knobs — `placement_timeout_s` is a different thing, and it bounds the gang reservation that
happens *after* the fan-out is chosen.

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
- {doc}`Shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: the transport that replaces the
  object store, and its credits.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: how a plan becomes Ray tasks.
- {doc}`Configuration options </configuration/options>`: every distributed knob.
- {doc}`Fault tolerance </architecture/fault-tolerance>`: retries, recovery, speculation.
- {doc}`PyTorch </integrations/compute/pytorch>`: distributed training ingest.
