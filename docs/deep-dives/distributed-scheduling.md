# Distributed scheduling

This page describes how Batcher decides where distributed work runs, how many pieces it runs in, and what does and doesn't travel through Ray.

A cluster gives you more cores and more RAM. It also gives you a scheduler, a serialization boundary, and a network, none of which a single-node engine pays for. Batcher's distributed path exists to buy the cores and the RAM without paying much for the rest, and the way it does that is by refusing to be a second engine.

:::{important}
There is one set of operator semantics. `dist/` decides *where* work runs and *how many pieces* it runs in. It doesn't decide what an aggregate means. The mergeable algebra of `partial -> combine -> finalize`, described in [Mergeable algebra](mergeable-algebra.md), already guarantees that a result assembled from partitions equals the single-node result, so distribution is a scheduling problem and nothing else.
:::

```text
   DRIVER
     │   composes stages out of ordinary plans. the engine never sees a "stage"
     │
     ├── fan-out:  sum over nodes of floor(node_cores / num_cpus)     ← cluster topology,
     │             the Ray head excluded unless it is the whole cluster  not the driver's
     │             (dist/executor.py::_cluster_fill_workers)             cpu_count()
     │
     ├── partition count:  max(rows / target_rows_per_task,
     │                         rows * width / target_bytes_per_task), clamped
     │                     (api/tuning/decisions.py)
     ▼
   MAP TASKS                                                   REDUCE TASKS
   ┌────────────┐                                              ┌────────────┐
   │  worker 0  │  nat.partial_aggregate    ═══ Flight ═══►    │  reducer 0 │  nat.combine
   ├────────────┤  nat.partition_batches    (the bulk bytes)   ├────────────┤  nat.combine_
   │  worker 1  │                                              │  reducer 1 │      finalize
   ├────────────┤  ─────────── via Ray ──────────►             ├────────────┤
   │  worker 2  │  paths, addresses, tickets, row counts,      │  reducer 2 │
   └────────────┘  and a metrics JSON string                   └────────────┘

   each worker's rayon width is pinned to its CPU grant
```

## What Ray does and does not carry

Ray schedules tasks and actors, and it carries control-plane metadata. The bulk shuffle bytes don't go through the Ray object store. Mapper output is written to Arrow IPC files or served from a Flight endpoint, and what crosses Ray is paths, addresses, tickets, row counts, and a metrics JSON string. You can see it in the return types. The shuffle map task in `python/batcher/dist/executors/aggregate.py` returns a `list[str]` of file paths, and the Flight worker in `python/batcher/dist/flight_worker.py` returns an address, not batches.

The short slogan overstates it slightly, so here is the precise version. The table lists each kind of traffic against whether it transits the object store.

| Path | Through the Ray object store? |
|---|---|
| mapper to reducer shuffle traffic | no, Arrow IPC files or a Flight endpoint |
| a distributed `map_batches` result | yes, via `ray.get` (`dist/executors/map.py::_map_udf_task`) |
| a non-splittable in-memory source | yes, shipped as task arguments |

Both of the latter are bounded. A map result is the query's output, and a map-then-aggregate returns only the partial, whose size is the group cardinality. Neither is zero, though. The claim that holds without qualification is the one about mapper-to-reducer traffic.

## The fan-out decision

The default worker count used to be the *driver's* `os.cpu_count()`, so a 16-core driver attached to a 128-CPU cluster fanned out to 16. Worse, when Ray was already initialized the cluster-fill was skipped entirely and queries ran on 2 of 16 workers. Both were control-plane bugs, both are fixed, and fixing them was worth more than every kernel optimization in the same period.

`dist/executor.py::_cluster_fill_workers` sizes the fan-out from cluster topology instead. `num_cpus` is the smallest worker node's core count, so a worker is placeable on any node, and the worker count is the sum of `floor(node_cores / num_cpus)` across nodes. One worker lands per core-slice, so a heterogeneous node gets proportionally more. A 64-core node next to 32-core nodes used to run at half utilization under a uniform one-worker-per-node grant.

`_worker_node_cpus` excludes the Ray head whenever at least one other node exists, because the head runs the GCS, the dashboard, and the job supervisor, and scheduling data operators there causes contention. A single-node cluster keeps the head, since it has to run the work. Many managed clusters already give the head zero schedulable CPUs, and a fan-out that counts it schedules onto nothing.

Each worker's rayon width is then pinned to its grant. `dist/executors/ray_runtime/lifecycle.py` fills in the engine config's `parallelism` from the worker's CPU grant when the driver left it unset, which is what stops a worker from either single-threading or oversubscribing. Any worker count is result-correct under the mergeable algebra, so all of this affects saturation and never the answer.

:::{warning}
Rayon's *global* pool is built before Ray applies the actor's cgroup affinity, so on a Ray worker it sizes itself to 1 thread. Every parallel execution therefore runs inside an explicitly-sized scoped pool (`bc_interp::par::pool_for`), never the global one. Missing that made the whole parallel executor single-threaded on every worker in the cluster, and nothing about the results looked wrong.
:::

## Task sizing

A stage's partition count comes from data volume, not from `cpu_count`. `optimizer.target_rows_per_task` (4M) and `optimizer.target_bytes_per_task` (256 MiB) set the target, and `api/tuning/decisions.py` takes the larger of the row-derived and byte-derived counts. A relation of a few very wide rows, such as video frames or embeddings, therefore still shards finely enough to fit memory. `distributed.max_shuffle_partitions` (2048) caps the result.

Per-task CPU is adaptive rather than a flat `1.0`. `dist/executors/map.py::_adaptive_task_cpus` asks for `descriptor_rows * weight / rows_per_cpu` cores, clamped to `[_MIN_TASK_CPU, node_cores]` with `_MIN_TASK_CPU` at 0.125. `rows_per_cpu` is half `target_rows_per_task`, so a full target-sized partition asks for roughly two cores. A tiny partition gets a fraction of a core and Ray packs many onto one. A UDF stage carries `_MAP_COMPUTE_WEIGHT`, which defaults to 4.0, because a single-threaded Python UDF can only be parallelized by *more tasks*, not by more cores per task. That weight is then scaled by a measured per-core busy fraction learned for the plan family, so a family that ran CPU-underutilized reserves fewer cores next run. Because the share is per-partition, a heavier partition gets proportionally more CPU, which absorbs the residual skew that split-balancing leaves behind. Reserving more or fewer cores only changes packing, never the rows a task processes.

Measured at sf10 on the 8-node cluster, a UDF-plus-aggregate pipeline went from 1.89 s to 0.88 s, and cluster utilization rose from 9% to 52% mean across 9 nodes.

## Skew

Two mechanisms handle skew, and they're separate.

Scan splits are balanced up front. Parquet `splits()` returns one split per row group, and `dist/executors/partition_io/_sources.py::_balance` greedily bin-packs them by row count. That evens the *read*.

Join skew is different, because it's a property of the key distribution and you can't see it in the file layout. `dist/executors/join.py::_detect_hot_keys` runs a Misra-Gries heavy-hitters pass per partition using `nat.heavy_hitters`, which is backed by `bc-sketches`, and a value is hot when its summed count clears `distributed.skew_join_fraction` (0.10) of the rows. Hot keys are then salted. `nat.salted_partition_batches` fans the probe-side hot rows across `salt` reducers and *replicates* the build-side hot rows to all of them. Cold keys hash exactly as before, so the joined relation is unchanged.

The detection pass costs a scan, so it runs only when you opt in through `distributed.skew_join_salt`, which defaults to 0. What makes it pay for itself is that the result is learned. `dist/skew.py` fingerprints the join shape with `join_skew_key`, a hash of both side IRs, the keys, and the join type, and persists the hot-key list in the `MetadataHub`. A shape with learned hot keys salts automatically on the next run, with no pre-pass. An empty learned list means "measured, not skewed", which is distinct from never-measured, so a non-skewed shape never re-runs the probe.

:::{warning}
Salting is result-preserving only when each reducer's output is concatenated. `salting_is_safe` refuses it for a fused join-plus-aggregate, where the reducer finalizes its bucket locally. Salted reducers would each finalize a *partial* group and the union would carry several half-summed rows for the hot key. Nothing raises. The query returns a wrong answer.
:::

## What the driver still does

The driver composes the stages. For a distributed aggregate (`dist/executors/aggregate.py::_distributed_aggregate`) that means partitioning the source, running map tasks that call `nat.partial_aggregate` and `nat.partition_batches` and write one IPC file per bucket, then running reduce tasks that fold their inputs with `nat.combine` incrementally and call `nat.combine_finalize` once. The Rust functions are the same ones the single-node parallel executor uses, and they live in `crates/bc-interp/src/dist.rs`.

Some shapes avoid the shuffle entirely. A shuffle join co-partitions both sides by the join key, so when a group-by's keys include the join key every group lies entirely within one bucket and each reducer's bucket is already complete. `_distributed_join_aggregate` gives the reducer an IR of `aggregate(hash_join(...))` and there's no second exchange. That exchange elimination took a distributed join-then-aggregate from 71.6 s to 1.75 s, because the old path collected the whole join to the driver.

:::{warning}
Some shapes shouldn't distribute at all. When a plan has no distributed path and any source it actually reads is splittable, `_unsupported` raises a `PlanError` rather than quietly running the query on the driver. The silent fallback is how the join and UDF cliffs hid for as long as they did. A query that says `distributed=True` and runs on one node is a perf cliff and an OOM risk wearing a correct result. When every source is in-memory or non-splittable there's no distributed data to speak of, so one node is the correct plan rather than a fallback.
:::

## Cost, and when not to use it

Distribution is for scale-out and for larger-than-memory. It isn't free and it isn't always faster.

::::{tab-set}
:::{tab-item} Single-node
```text
TPC-H sf1 (6M rows), the udf-map workload:   86 ms

no actor startup, no network shuffle, no serialization boundary
this is what distributed.distribute_min_rows (1M) protects
```
:::

:::{tab-item} Distributed
```text
TPC-H sf1 (6M rows), the udf-map workload:   92 ms   (batcher, 4 workers)

the result is bit-identical to the single-node one. at this size the shuffle
plus actor startup costs more than the whole query, and taking the distributed
path anyway costs about 7%.
```
:::
::::

The warm session fleet (`distributed.reuse_session_fleet`, on by default) exists because spawning and tearing down the Flight fleet per `collect()` cost about 1.5 s of a roughly 3 s query. It's health-checked and idle-auto-released after `session_fleet_idle_s` (30 s).

## Code map

| Concern | File |
|---|---|
| Entry point, fan-out, plan-shape dispatch | `python/batcher/dist/executor.py` |
| Partition-count sizing | `python/batcher/api/tuning/decisions.py` |
| Per-operator executors | `python/batcher/dist/executors/{aggregate,join,sort,map,window,union,distinct}.py` |
| Ray tasks/actors, placement, autoscale, fault policy | `python/batcher/dist/executors/ray_runtime/` |
| Split balancing | `python/batcher/dist/executors/partition_io/_sources.py` |
| Learned sizing (partitions, actor pool, straggler factor) | `python/batcher/dist/adaptive_sizing/sizing.py` |
| Join-skew learning | `python/batcher/dist/skew.py` |
| Rust mergeable primitives | `crates/bc-interp/src/dist.rs` |

## See also

:::{seealso}
- [Architecture](../architecture/index.md): distribution as a backend, never a second semantics
- [Fault tolerance](../architecture/fault-tolerance.md): straggler speculation and worker loss
- [Carbonite](../internals/carbonite.md): the envelope each worker runs inside
- [Ray integration](../integrations/ray.md): setting up the cluster this schedules onto
- [Configuration options](../configuration/options.md): every `distributed.*` knob named here
- [Scaling benchmarks](../benchmarks/scaling.md): the node-count curves, and the cluster grid
- [Mergeable algebra](mergeable-algebra.md): why a partitioned result equals a single-node one
- [Shuffle over Arrow Flight](shuffle-flight.md): how the bytes actually move
- [Credit-based flow control](credit-flow-control.md): what keeps a reducer from drowning
- [Learned metadata](learned-metadata.md): where the learned partition counts live
:::
