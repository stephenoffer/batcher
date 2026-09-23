# Ray

This page covers running Batcher on a Ray cluster. Ray does one job here: scheduling. Tasks, actors, placement groups, and small control-plane messages such as file paths, worker addresses, and metrics go through Ray. Bulk Arrow batches don't. Shuffle data moves worker to worker over Arrow Flight (`bc-transport`) with credit-based flow control, where one credit is one in-flight batch slot and a producer blocks when its credits reach zero.

Keeping bulk data out of the Ray object store is the whole integration. An object-store shuffle serializes every batch, spills it under pressure, and turns the driver into a funnel. Batcher's shuffle does none of that, and single-node execution never imports Ray at all.

The figure draws the two lanes. Ray sits between the driver and the workers and carries only small messages, while the batches of a shuffle go straight from mapper to reducer.

![Three layers. At the top, the driver running collect(distributed=True) hands tasks, actors, and placement groups to Ray. Ray carries those plus small control-plane messages such as file paths, worker addresses, and metrics, never the rows themselves, and schedules each stage onto the workers. The Ray object store is bypassed and sits off the data path. At the bottom, the workers run the Rust engine over Arrow, and two mappers send Arrow batches to a reducer over Arrow Flight (bc-transport), credit-bounded: one credit is one in-flight batch slot, and a producer blocks when its credits reach zero.](/_static/diagrams/ray_two_lanes.svg)

The following table summarizes the integration:

| | |
| --- | --- |
| Ray carries | Tasks, actors, placement groups, and small control-plane messages |
| Ray doesn't carry | Bulk Arrow batches, which move over Arrow Flight |
| Extra | `pip install 'batcher-engine[ray]'` |
| Entry point | {py:meth}`collect(distributed=True) <batcher.Dataset.collect>`, or `"auto"`, the default |
| From a Ray Dataset | {py:func}`bt.from_ray_dataset(rds) <batcher.from_ray_dataset>`, streamed block by block |
| Back to a Ray Dataset | {py:meth}`ds.to_ray_dataset() <batcher.Dataset.to_ray_dataset>`, coalesced into Ray-sized blocks |
| Cluster config | `config.distributed` |

## Go distributed

There's no distributed API. It's the same plan with one argument changed:

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

`collect(distributed="auto")`, the default, uses Ray on a multi-node cluster and runs in-process otherwise. `True` and `False` force the choice. The rows are the same either way, because the distributed path composes the same mergeable primitives the single-node parallel executor uses: `partial`, then shuffle, then `combine`, then `finalize`. There's no second semantics to disagree with the first.

Distribution is for scale-out and larger-than-memory work, and it costs a shuffle. On data that fits one node, the in-process engine avoids that cost, so don't set `distributed=True` by reflex.

## Attach to a cluster

`config.distributed` holds the cluster settings. With `ray_address=None`, the default, Batcher attaches to a running cluster when `RAY_ADDRESS` is set or when it detects a managed control plane, and starts a local Ray only when nothing is reachable.

Detection reads environment variables only, never a metadata service. It recognizes Anyscale (`ANYSCALE_SESSION_ID`, `ANYSCALE_CLUSTER_ID`), any KubeRay-operated cluster on any cloud or on-premises, and an explicit `BATCHER_RAY_CLUSTER=1` for a platform neither covers. A false positive degrades rather than fails, because the attach still falls back to a local start when no cluster answers.

A head that doesn't answer on the first try is usually still starting. The driver and the head come up concurrently in every orchestrated environment: a KubeRay driver pod is admitted before the head passes its readiness probe, and a Slurm job's `ray start --head` races the step that runs the query. Batcher retries the attach with exponential backoff for `cluster_connect_timeout_s` seconds, 30 by default. Set it to 0 to attach once.

What happens when that window runs out depends on how the address was found:

| How the address was found | Cluster unreachable |
|---|---|
| `ray_address`, or `RAY_ADDRESS` | Raises. You named a cluster, and running single-node in its place would be a wrong answer. |
| Detected from the environment | Starts a local single-node Ray, so a dev run in a workspace whose cluster is down still works. |

You don't need to pre-install Batcher on the workers. When Batcher initializes Ray against a cluster it ships its own package, compiled extension included, through `runtime_env`. Set `trust_cluster_image=True` to skip that upload when your image already has Batcher.

```python
from batcher import Config
from batcher.config import DistributedConfig

cfg = Config().replace(distributed=DistributedConfig(namespace="nightly-etl"))
print(cfg.distributed.namespace)
# nightly-etl
```

:::{dropdown} The four fields that matter on day one
`runtime_env` ships an environment to the workers. Every worker needs `batcher` and its compiled extension importable. A worker with a different wheel from the driver fails at the first task, usually with an import error naming the native module.

`namespace` isolates a job's shuffle actors. It defaults to `batcher`, and two jobs in one namespace can see each other's actors.

`transport` picks the shuffle. `"auto"` chooses Flight on a multi-node cluster and a disk shuffle on a single node. The disk shuffle hands paths between tasks rather than bytes, so every path must resolve on whichever node runs the task. That holds on one node, and on a cluster where `shared_filesystem=True` says every worker mounts the same scratch at the same path. Anywhere else, forcing `transport="disk"` produces tasks that can't find their input.

`shuffle_token`, also read from `BATCHER_SHUFFLE_TOKEN`, authenticates Flight fetches, and `distributed.tls` turns on TLS or mTLS between workers. The shuffle is a data plane on the wire, so set both on a shared or untrusted network.
:::

Retries, straggler speculation, skew salting, and adaptive credits are in {doc}`configuration options </configuration/options>`. The defaults fill a cluster with no tuning: each node is cut into several workers with an even share of its cores, and the reducer count scales with the workers.

## Bring in a Ray Dataset

{py:obj}`bt.from_ray_dataset(rds) <batcher.from_ray_dataset>` streams a Ray Dataset's Arrow blocks into the engine lazily, one block per batch. Nothing is collected to the driver, so memory stays bounded.

```python
# docs: skip
import ray
import batcher as bt

rds = ray.data.read_parquet("s3://lake/events")
events = bt.from_ray_dataset(rds)
print(events.group_by("region").agg(bt.col("amount").sum()).sort("region").to_pydict())
```

:::{warning}
Ray Data stores tensor and opaque-Python columns as its own Arrow extension types, `ray.data.arrow_tensor_v2` and `ray.data.arrow_pickled_object`, which standard Parquet tooling, Polars, and DuckDB either reject or skip. Plain columns cross unchanged. Check a tensor column with `rds.schema()` before you rely on it. Batcher's own tensor columns are ordinary Arrow, `FixedSizeList` or a `struct<data, shape, dtype>` for ragged shapes, so the return leg doesn't introduce extension types.
:::

Treat this as an on-ramp. Whatever built the incoming dataset still costs what it costs. Where the source is a plain read, read it with Batcher instead. {py:meth}`bt.read.parquet <batcher.api.io_namespace.reader.Reader.parquet>` reads and sums 20 M rows across 64 files in 72 ms on one node ([`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md), "Data connectors"), because files decode concurrently in-process with no per-file task and no object-store hop.

The inference idiom carries over unchanged. {py:meth}`ds.map_batches <batcher.Dataset.map_batches>` takes a class, loads the model once per worker, and runs the batches through an actor pool, so a ported pipeline keeps its shape while picking up warm pools and stage overlap. See {doc}`/ml/inference/inference`.

## Hand a result back to Ray

{py:meth}`ds.to_ray_dataset() <batcher.Dataset.to_ray_dataset>` is the return leg, for when the next stage belongs to Ray Train, Ray Tune, or a Serve deployment that wants a `ray.data.Dataset`.

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

Output batches are coalesced into blocks near Ray Data's own `target_max_block_size` and put into the object store one block at a time. The driver holds one block, not the whole result, and the Ray Dataset blocks the way a `read_parquet` one does. An empty result keeps its schema, so a filter that matches nothing still hands Ray a typed dataset.

The blocks are produced on the driver, which suits a result that's already reduced: a training set, a scored table, an embedding index. For a result the size of the input, write Parquet and give Ray the path. The data then stays on the workers that produced it.

## Placement

Batcher gang-schedules a shuffle fleet as a placement group, one bundle per worker, so the whole fleet exists before the shuffle starts. Several placement decisions are aimed at the cloud bill.

A fleet is pinned to one availability zone when one zone can host it. A shuffle moves nearly all its bytes worker to worker, and clouds charge for and delay bytes that cross a zone boundary. A fleet spread evenly over three zones sends about two thirds of its shuffle across that boundary for no benefit. Batcher picks the zone with the most free capacity that fits the whole fleet and reserves the bundles there. The pin is a no-op on a single-zone cluster, on nodes with no zone label, and when no zone has room, and a group that can't form falls back to ordinary scheduling at the placement timeout. Set `distributed.zone_aware_placement=False` when you buy zone diversity deliberately.

If you control provisioning, pin the cluster itself to one zone, which also covers the head node. The runtime pin is for clusters that must span zones, such as an accelerator fleet whose scarce instance types force cross-zone autoscaling.

A shuffle replica avoids the primary's failure domain, not just its node. With `distributed.shuffle_replication` above 1, each mapper's output is copied to another node, so losing a worker costs a re-fetch rather than a recompute. When the primary sits on spot capacity, the copy prefers a node that isn't spot, because a reclamation takes a whole instance group. Spot is read from `ray.io/market-type` and the Karpenter, EKS, and GKE capacity labels. It's a preference, never an exclusion.

Each stage can also run on the capacity its failure model fits. A stateless map partition re-derives from a durable partition descriptor, so a preempted one is resubmitted. A shuffle worker holds partial state its peers haven't fetched yet. Set `distributed.capacity_aware_placement=True` and Batcher asks Ray for spot capacity for map stages and on-demand capacity for shuffle fleets, with a fallback so a fleet that finds no on-demand capacity runs on spot rather than pending. It emits nothing unless the live fleet is genuinely mixed and labelled under a single key. Placement never changes which rows a task processes, so results are identical either way.

```python
# docs: skip
import dataclasses

import batcher as bt

base = bt.active_config()
bt.set_config(
    base.replace(distributed=dataclasses.replace(base.distributed, capacity_aware_placement=True))
)

# The scan and the filter run on spot; the aggregate's shuffle fleet runs on on-demand.
out = (
    bt.read.parquet("s3://<your-bucket>/events/")
    .filter(bt.col("status") == "ok")
    .group_by("user_id")
    .agg(events=bt.col("event_id").count())
    .collect(distributed=True)
)
```

Very wide stages stop reporting each task to the Ray Dashboard. Above `distributed.task_events_fanout_cap` tasks in one stage, 10,000 by default, Batcher turns Ray's per-task events off, because a hundred-thousand-partition stage would flood the control plane every driver shares. Set `distributed.task_events="always"` to keep them or `"never"` to drop them everywhere. Batcher's own progress reporting reads the engine's event bus, so it's unaffected.

When a reservation can't form, Batcher compares the ask against the live topology and says why. An ask no node can host names the binding resource and the widest node's figure, and an ask every node is too busy for says the cluster is full and by how much. The map and inference stall warnings carry the same diagnosis, so a stage waiting on capacity that will never arrive looks different from a slow one.

## Requirements and limitations

A preempted worker recomputes its partition from its durable input. A `map_batches` function with an external side effect, such as a vector-database insert or a REST POST, can therefore apply it twice. Make the sink an upsert on a stable key. A pure transform is already safe.

`distributed.resilience="spot"` hardens the retry and restart budgets as a bundle for a churning cluster. Use it rather than tuning each knob by hand. Batcher selects it automatically on a preemptible node.

On an autoscaling cluster Batcher requests the cores a query wants and waits for the nodes before sizing the fan-out, so a big query runs on the grown cluster. `autoscale_wait_s` bounds the wait. Its default resolves to a bounded wait on an autoscaling cluster and to no wait on a fixed one. The wait ends early once capacity has been flat for `autoscale_stall_s`, 90 seconds by default, or hasn't grown within `autoscale_startup_grace_s`, 12 seconds by default. `placement_timeout_s`, 60 seconds by default, bounds the gang reservation that follows.

On a managed workspace that exports no `RAY_ADDRESS`, calling `ray.init()` in your own code before Batcher does can strand the job on a local Ray while the cluster sits idle. Let Batcher attach, or set `ray_address`.

Don't route bulk data through Ray objects. Calling `ray.put` on a `RecordBatch` to move it between stages reintroduces the object-store cost this design removes.

## See also

- {doc}`/integrations/compute/schedulers`: bringing Ray up across a Slurm, PBS, or Kubernetes allocation.
- {doc}`Execution architecture </architecture/execution>`: morsels, breakers, and the shuffle.
- {doc}`Shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: the transport and its credits.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: how a plan becomes Ray tasks.
- {doc}`Fault tolerance </architecture/fault-tolerance>`: retries, recovery, and speculation.
- {doc}`Configuration options </configuration/options>`: every distributed setting.
- {doc}`PyTorch </integrations/compute/pytorch>`: distributed training ingest.
