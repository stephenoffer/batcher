# Distributed scheduling

A cluster gives you more cores and more RAM. It also gives you a scheduler, a
serialization boundary, and a network, none of which a single-node engine pays for.
Batcher's distributed path exists to buy the cores and the RAM without paying much for
the rest, and the way it does that is by refusing to be a second engine.

:::{important}
There is one set of operator semantics. `dist/` decides *where* work runs and *how many pieces*
it runs in. It does not decide what an aggregate means. The mergeable algebra
(`partial → combine → finalize`, see [Mergeable algebra](mergeable-algebra.md)) already
guarantees that a result assembled from partitions equals the single-node result, so
distribution is a scheduling problem and nothing else.
:::

```text
   DRIVER  (python/batcher/dist/executor.py)
     │   composes stages out of ordinary plans — the engine never sees a "stage"
     │
     ├── fan-out:  sum over nodes of floor(node_cores / num_cpus)     ← cluster topology,
     │             the Ray head excluded unless it is the whole cluster  not the driver's
     │                                                                   cpu_count()
     ├── task sizing:  max(rows / target_rows_per_task,
     │                     bytes / target_bytes_per_task), capped at 2048
     ▼
   MAP TASKS                                                   REDUCE TASKS
   ┌────────────┐                                              ┌────────────┐
   │  worker 0  │  nat.partial_aggregate    ═══ Flight ═══►    │  reducer 0 │  nat.combine
   ├────────────┤  nat.partition_batches    (the bulk bytes)   ├────────────┤  nat.combine_
   │  worker 1  │                                              │  reducer 1 │      finalize
   ├────────────┤  ─────────── via Ray ──────────►             ├────────────┤
   │  worker 2  │  paths, addresses, tickets, row counts,      │  reducer 2 │
   └────────────┘  and a metrics JSON string                   └────────────┘

   each worker's rayon width is pinned to its grant: cfg["parallelism"] = num_cpus
```

## What Ray does and does not carry

Ray schedules tasks and actors, and it carries control-plane metadata. The bulk shuffle
bytes do not go through the Ray object store. Mapper output is written to Arrow IPC files
(or served from a Flight endpoint). What crosses Ray is *paths*, *addresses*, *tickets*,
row counts, and a metrics JSON string. You can see it in the return types: the shuffle map
task in `python/batcher/dist/executors/aggregate.py` returns `list[str]` of file paths, and
the Flight worker in `python/batcher/dist/flight_worker.py` returns an address, not batches.

Being precise, because the short slogan overstates it:

| Path | Through the Ray object store? |
|---|---|
| mapper → reducer shuffle traffic | no — Arrow IPC files or a Flight endpoint |
| a distributed `map_batches` result | yes, via `ray.get` (`dist/executors/map.py::_map_udf_task`) |
| a non-splittable in-memory source | yes, shipped as task arguments |

Both of the latter are bounded — a map result is the query's output, and a map-then-aggregate
returns only the partial, whose size is the group cardinality — but neither is zero. The claim
that holds without qualification is about mapper→reducer traffic.

## The fan-out decision

The default worker count used to be the *driver's* `os.cpu_count()`, so a 16-core driver
attached to a 128-CPU cluster fanned out to 16. Worse, when Ray was already initialized the
cluster-fill was skipped entirely and queries ran on 2 of 16 workers. Both were
control-plane bugs, both are fixed, and fixing them was worth more than every kernel
optimization in the same period.

`dist/executor.py::_cluster_fill_workers` sizes the fan-out from cluster topology:
`num_cpus` is the smallest node's core count, and the worker count is the sum of
`floor(node_cores / num_cpus)` across nodes, so one worker lands per core-slice and a
heterogeneous node gets proportionally more. The Ray head is excluded (`_worker_node_cpus`)
unless it is the whole cluster. Many managed clusters reserve the head with zero
schedulable CPUs, and a fan-out that counts it schedules onto nothing.

Each worker's rayon width is then pinned to its grant: `ray_runtime/lifecycle.py` sets
`cfg["parallelism"] = int(env.num_cpus)` in the engine config it ships.

:::{warning}
Rayon's *global* pool is built before Ray applies the actor's cgroup affinity, so on a Ray
worker it sizes itself to **1 thread**. Every parallel execution therefore runs inside an
explicitly-sized scoped pool (`bc_interp::par::pool_for`), never the global one. Missing that
made the whole parallel executor single-threaded on every worker in the cluster, and nothing
about the results looked wrong.
:::

## Task sizing

A stage's partition count comes from data volume, not from `cpu_count`.
`optimizer.target_rows_per_task` (4M) and `target_bytes_per_task` (256 MiB) set the target,
and the fan-out takes the max of the row- and byte-derived counts, so a relation of a few
very wide rows (video frames, embeddings) still shards finely enough to fit memory.
`distributed.max_shuffle_partitions` (2048) caps it.

Per-task CPU is adaptive rather than a flat `1.0`.
`dist/executors/map.py::_adaptive_task_cpus` asks for `descriptor_rows × weight /
rows_per_cpu` cores, clamped to `[0.125, node_cores]`. A tiny partition gets a fraction of a
core and Ray packs many onto one; a UDF stage carries `_MAP_COMPUTE_WEIGHT = 4.0`, because a
single-threaded Python UDF can only be parallelized by *more tasks*, not by more cores per
task. The share is per-partition, so a heavier partition gets proportionally more CPU,
which absorbs the residual skew that split-balancing leaves behind.

Measured on an 8-node cluster at sf10: a UDF-plus-aggregate pipeline went from 1.89 s to
0.88 s and cluster utilization from 9% to 52% mean.

## Skew

Two mechanisms, and they are separate.

Scan splits are balanced up front: Parquet `splits()` returns one split per row group and
`partition_io/_sources.py::_balance` greedily bin-packs them by row count. That evens the
*read*.

Join skew is different, because it is a property of the key distribution and you cannot see
it in the file layout. `dist/executors/join.py::_detect_hot_keys` runs a Misra-Gries
heavy-hitters pass per partition (`nat.heavy_hitters`, backed by `bc-sketches`), and a value
is hot when its summed count clears `distributed.skew_join_fraction` (0.10) of the rows. Hot
keys are then salted: `nat.salted_partition_batches` fans the probe-side hot rows across
`salt` reducers and *replicates* the build-side hot rows to all of them. Cold keys hash
exactly as before, so the joined relation is unchanged.

The detection pass costs a scan, so it only runs when you opt in. What makes it pay for
itself is that the result is learned: `dist/skew.py` fingerprints the join shape
(`join_skew_key`, a hash of both side IRs, keys, and join type) and persists the hot-key
list in the `MetadataHub`. A shape with learned hot keys salts automatically on the next
run, with no pre-pass. An empty learned list means "measured, not skewed", distinct from
never-measured, so a non-skewed shape never re-runs the probe.

## What the driver still does

The driver composes the stages. For a distributed aggregate
(`dist/executors/aggregate.py::_distributed_aggregate`) that is: partition the source, run
map tasks that call `nat.partial_aggregate` and `nat.partition_batches` and write one IPC
file per bucket, then run reduce tasks that fold their inputs with `nat.combine`
incrementally and call `nat.combine_finalize` once. The Rust functions are the same ones the
single-node parallel executor uses; they live in `crates/bc-interp/src/dist.rs`.

Some shapes avoid the shuffle entirely. When a group-by's keys are a superset of the join
key, each reducer's bucket is already complete, so `_distributed_join_aggregate` gives the
reducer an IR of `aggregate(hash_join(...))` and there is no second exchange. That one
rewrite took a distributed join-then-aggregate from 71.6 s to 1.75 s, because the old path
collected the whole join to the driver.

:::{warning}
Some shapes should not distribute at all, and `_dispatch` raises a `PlanError` when a plan has
no distributed path *and* its source is splittable, rather than quietly running it on the
driver. The silent fallback is how the join and UDF cliffs hid for as long as they did: a query
that says `distributed=True` and runs on one node is a perf cliff and an OOM risk wearing a
correct result.
:::

## Cost, and when not to use it

Distribution is for scale-out and for larger-than-memory. It is not free and it is not
always faster.

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
TPC-H sf1 (6M rows), the udf-map workload:   92 ms   (batcher)
                                          4,284 ms   (Ray Data)

the result is bit-identical to the single-node one. at this size the shuffle
plus actor startup simply costs more than the whole query.
even distributed-versus-distributed, the gap is 46×.
```
:::
::::

The warm session fleet (`distributed.reuse_session_fleet`, on by default) exists because
spawning and tearing down the Flight fleet per `collect()` cost ~1.5 s of a ~3 s query. It is
health-checked and idle-auto-released after `session_fleet_idle_s` (30 s).

## Code map

| Concern | File |
|---|---|
| Entry point, fan-out, plan-shape dispatch | `python/batcher/dist/executor.py` |
| Per-operator executors | `python/batcher/dist/executors/{aggregate,join,sort,map,window,union,distinct}.py` |
| Ray tasks/actors, placement, autoscale, fault policy | `python/batcher/dist/executors/ray_runtime/` |
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
- [Scaling benchmarks](../benchmarks/scaling.md): the node-count curves
- [vs Ray Data](../benchmarks/vs-ray-data.md): the 46× comparison above, in context
- [Mergeable algebra](mergeable-algebra.md): why a partitioned result equals a single-node one
- [Shuffle over Arrow Flight](shuffle-flight.md): how the bytes actually move
- [Credit-based flow control](credit-flow-control.md): what keeps a reducer from drowning
- [Learned metadata](learned-metadata.md): where the learned partition counts live
:::
