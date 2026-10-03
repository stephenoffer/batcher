# Distributed scheduling

This page describes how Batcher decides where distributed work runs, how many pieces it runs in, and what does and doesn't travel through Ray.

:::{important}
There is one set of operator semantics. `dist/` decides *where* work runs and *how many pieces* it runs in. It doesn't decide what an aggregate means. The mergeable algebra of `partial -> combine -> finalize`, described in {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`, guarantees that a result assembled from partitions equals the single-node result, so distribution is a scheduling problem and nothing else.
:::

From the user's side it's one argument. The same pipeline runs single-node or on a cluster, and `distributed="auto"`, the default, distributes only past a size floor:

```python
# docs: skip
import batcher as bt

ds = bt.from_pydict({"k": [i % 10 for i in range(100_000)], "v": list(range(100_000))})
q = ds.group_by("k").agg(s=bt.sum("v")).sort("k")
local = q.collect(distributed=False)
cluster = q.collect(distributed=True, num_workers=4)
print(local.equals(cluster))  # True
```

```python
import batcher as bt

print(bt.Config().distributed.distribute_min_rows)  # 20000000
```

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

Ray schedules tasks and actors and carries control-plane metadata. Mapper-to-reducer shuffle bytes never go through the Ray object store: the shuffle map task in [`python/batcher/dist/executors/aggregate.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/executors/aggregate.py) returns a `list[str]` of file paths, and the Flight worker in [`python/batcher/dist/flight_worker.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/flight_worker.py) returns an address.

| Path | Through the Ray object store? |
|---|---|
| mapper to reducer shuffle traffic | no, Arrow IPC files or a Flight endpoint |
| a distributed `map_batches` result | yes, via `ray.get` (`dist/executors/map.py::_map_udf_task`) |
| a non-splittable in-memory source | yes, shipped as task arguments |

Both of the latter are bounded: a map result is the query's output, and a map-then-aggregate returns only the partial, sized by the group count.

## The fan-out decision

`dist/executor.py::_cluster_fill_workers` sizes the fan-out from cluster topology, not the driver's `os.cpu_count()`. `num_cpus` is the smallest worker node's core count, so a worker is placeable anywhere, and the worker count is the sum of `floor(node_cores / num_cpus)` across nodes, so a larger node gets proportionally more workers.

`scaling.node_classes` is the single definition of "worker-eligible" every sizing path reads. It excludes the Ray head whenever another node exists, since the head runs the GCS and dashboard, and excludes nodes Ray has marked for drain, so a query sized during scale-in targets the nodes that will remain.

:::{dropdown} Placement, packing and per-worker threads
A chosen fan-out is checked against what single nodes can host, because Ray gang-schedules the fleet and an unsatisfiable placement group would hang. `capacity.placeable_workers` sums each node's own capacity, bounded by every resource a bundle reserves: cores, GPUs, the per-worker memory grant, and the node class when a relational fleet is held off accelerator nodes.

Carbonite prefers `PACK` for a small-shuffle breaker; `dist` downgrades it to `SPREAD` when no single node can hold the gang. `STRICT_PACK`, requested only by a GPU collective whose actors must be co-located, is never downgraded.

Each worker's rayon width is pinned to its CPU grant by `dist/executors/ray_runtime/lifecycle.py`. Rayon's global pool is built before Ray applies the actor's cgroup affinity, so on a Ray worker it sizes itself to 1 thread. Every parallel execution therefore runs inside an explicitly-sized scoped pool (`bc_interp::par::pool_for`). Missing that once made the whole parallel executor single-threaded on every worker, with nothing wrong in the results. Any worker count is result-correct under the mergeable algebra, so all of this affects saturation, never the answer.
:::

## Task sizing

A stage's partition count comes from data volume. `api/tuning/decisions.py` takes the larger of the row-derived and byte-derived counts, so a relation of a few very wide rows, such as video frames or embeddings, still shards finely enough to fit memory:

```python
cfg = bt.Config()
print(cfg.optimizer.target_rows_per_task, cfg.optimizer.target_bytes_per_task >> 20)  # 4000000 256
print(cfg.distributed.max_shuffle_partitions)  # 2048
```

Per-task CPU is adaptive. `dist/executors/map.py::_adaptive_task_cpus` asks for `descriptor_rows * weight / rows_per_cpu` cores, clamped to `[0.125, node_cores]`, so a tiny partition gets a fraction of a core and Ray packs many onto one. A UDF stage carries `_MAP_COMPUTE_WEIGHT` (4.0), because a single-threaded Python UDF parallelizes only by more tasks, scaled by a per-core busy fraction learned for the plan family. Measured at sf10 on the project cluster, a UDF-plus-aggregate pipeline went from 1.89 s to 0.88 s, and mean cluster utilization rose from 9% to 52%.

A Flight shuffle's map stage cuts its input into `workers x distributed.map_partition_multiplier` partitions (four per worker by default), capped at the splits the source has, and `map_barrier` deals them to actors as they go idle with exactly `workers` tasks in flight. A slow node takes fewer partitions, and a lost worker loses one small partition rather than a node's whole share.

### How many reducers

An aggregate exchanges partial state sized by its group count, so `adaptive_sizing/sizing.py::aggregate_reducer_count` sizes its reduce from the learned or estimated number of groups, `ceil(groups / optimizer.target_rows_per_task)`. The count is floored at the worker count, and that floor is bounded by whether the groups can keep those workers busy, `_MIN_GROUPS_PER_REDUCER` (50,000) per reducer. Measured on the project cluster, warm, median of seven, every case checked against DuckDB:

| groups | workers | floored at `workers` | bounded by work |
|---|---|---|---|
| 64 | 64 | 302 ms | 91 ms (1 reducer) |
| 200,000 | 64 | 348 ms | 249 ms (4) |
| 1,000,000 | 64 | 880 ms | 434 ms (20) |

A join, sort or window exchanges the rows themselves, so `row_shuffle_reducer_count` may only raise the fan-out above one bucket per worker, never lower it.

:::{dropdown} The sliding reduce window
A bucket is reduced by the one worker it hashes to (`bucket % workers`), so tasks launched past the worker count can't start. `dist/executors/ray_runtime/reduce.py::gather_in_windows` keeps at most `distributed.pending_window_factor` times the worker count outstanding, or `distributed.max_pending_tasks` when set. The window slides: one completion launches one new task, so a slow bucket never holds a whole chunk of idle actors. Results return in submission order either way.
:::

## Skew

Scan splits are balanced up front. `dist/executors/partition_io/assignment.py::_balance` bin-packs Parquet row-group splits by uncompressed bytes from the footer when every split carries them, and by row count otherwise.

Join skew is a property of the key distribution. `dist/executors/join.py::_detect_hot_keys` runs a Misra-Gries heavy-hitters pass (`nat.heavy_hitters`, backed by `bc-sketches`), and a value is hot when its count clears `distributed.skew_join_fraction` of the rows. `nat.salted_partition_batches` then fans the probe-side hot rows across `salt` reducers and *replicates* the build-side hot rows to all of them. Cold keys hash as before, so the joined relation is unchanged.

```python
print(cfg.distributed.skew_join_fraction, cfg.distributed.skew_join_salt)  # 0.1 0
```

`distributed.skew_join_salt` is the fan-out: 0 leaves the decision to measurement, a positive value forces the pre-pass and pins the fan-out, a negative value never salts. `dist/skew.py::resolve_hot_keys` asks the cheap sources first, the hot-key set learned for this join shape and Kyber's column statistics, and runs the pre-pass on its own once the estimated input clears about 8.4M rows. There one pass costs around 4% on a uniform join, while an undetected 40% hot key costs 5.8x.

:::{dropdown} Learned skew verdicts
`dist/skew.py` fingerprints the join shape with `join_skew_key`, a hash of both side IRs, the keys and the join type, and persists the hot-key list in the `MetadataHub`. A learned hot list salts with no pre-pass on the next run. An empty list means "measured, not skewed" and expires after a week (`_UNIFORM_VERDICT_TTL_S`) or once the estimated input moves 4x (`_UNIFORM_VERDICT_DRIFT`), since a table can drift into skew under an unchanged query. A hot list doesn't expire, because salting a cooled key costs some replication and never an answer.

`salting_is_safe` refuses salting for a fused join-plus-aggregate, where each reducer finalizes its bucket locally and salted reducers would each finalize a partial group.
:::

## What the driver does

The driver composes the stages. For a distributed aggregate (`dist/executors/aggregate.py::_distributed_aggregate`) it partitions the source, runs map tasks calling `nat.partial_aggregate` and `nat.partition_batches`, then reduce tasks folding with `nat.combine` and calling `nat.combine_finalize` once. These are the same Rust functions the single-node parallel executor uses, in [`crates/bc-interp/src/dist.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/dist.rs).

Some shapes avoid a second shuffle entirely. When a group-by's keys include the join key, every group lies within one join bucket, so `_distributed_join_aggregate` gives the reducer `aggregate(hash_join(...))` with no second exchange. That took a distributed join-then-aggregate from 71.6 s to 1.75 s.

When a plan has no distributed path and any source it reads is splittable, `_unsupported` raises a {py:exc}`PlanError <batcher.PlanError>` rather than quietly running the query on the driver. When every source is in-memory or non-splittable, one node is the correct plan.

### Staging a UDF so the operator above it can shuffle

A `map_batches` pipeline runs in Python with no engine IR, so no shuffle can see through it. Batcher cuts the query in two: the UDF pipeline runs as its own distributed stage and lands its output as Parquet on cluster-shared scratch, and the breaker above runs as the ordinary distributed operator over a scan of that scratch. Staging follows the plan's operands, so `map_batches(...).join(other).group_by(...)` stages only the UDF branch and reaches the fused join-aggregate reducer. An empty staged operand is passed as a zero-row input of the UDF's output schema, so an outer join still applies its own empty-input semantics. The operand is declined only when no shard ran the UDF at all and there is no schema to give it.

## What never reaches the driver

A stage can leave its result where it was computed. An aggregate, `distinct`, hash join, sort and partitioned window each publish one bucket per reducer and return handles: an Arrow IPC file (`MaterializedSource`) on the disk transport, or a bucket resident on the producing actor (`FlightMaterializedSource`) on Flight.

![Where a distributed query is cut, and what crosses a cut. The cut set is a plan property: plan_analysis._has_breaker names Aggregate, Sort, Join, Distinct and Limit, and every other node runs inside the stage it is already in, so a scan, filter and project chain feeding an aggregate, then a sort, then a limit is cut three times and the cluster does not enter into it. One cut is one stage. Inside a stage the input is cut into the worker count times four partitions, each a durable descriptor of splits with the projection already pushed into it, and a barrier deals them to whichever actor just went idle, keeping exactly workers tasks in flight so a slow node takes fewer. The map tasks compute partials, emit one bucket per reducer, and each bucket is reduced by the one worker it hashes to, through combine and combine_finalize. What crosses to the next stage is one handle per reducer bucket, scanned in place as an ordinary scan: the rows stay on the worker that computed them, and a multi-join query never round-trips an intermediate through the driver.](/_static/diagrams/distributed_stages.svg)

Three things consume those handles. The adaptive executor scans one stage's buckets as the next stage's input. {py:meth}`iter_batches(distributed=True) <batcher.Dataset.iter_batches>` reads one bucket at a time, so peak driver memory is one reducer's output. And an unpartitioned distributed write hands the buckets to the workers, so only file locators travel back.

```python
# docs: skip
for batch in q.iter_batches(distributed=True, num_workers=4):
    print(batch.num_rows)  # one reducer bucket at a time
```

A sort's buckets are *ranges* of the leading key, listed in range order, so reading them in sequence is the sorted relation with no merge. A `Filter` or `Project` above a sort runs inside each reducer. Any other operator above a breaker is applied to the assembled result on the driver, as is a sort whose `limit` is too large for the shuffle-free top-N (up to 1,000,000 rows).

## Practical limits

- **Small inputs.** Actor startup and the shuffle are fixed costs, which is what `distributed.distribute_min_rows` (20M) protects under `"auto"`. On the TPC-H sf1 udf-map workload (6M rows), single-node ran in 86 ms and four workers in 92 ms.
- **Fleet startup.** The warm session fleet (`distributed.reuse_session_fleet`, on by default) keeps the Flight fleet across {py:meth}`collect() <batcher.Dataset.collect>` calls, health-checked and released after `session_fleet_idle_s` (30 s) idle.
- **Split balancing.** It evens bytes or rows, not codec cost or a slow remote store.

## Code map

| Concern | File |
|---|---|
| Entry point, fan-out, plan-shape dispatch | [`python/batcher/dist/executor.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/executor.py) |
| Partition-count sizing | [`python/batcher/api/tuning/decisions.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/api/tuning/decisions.py) |
| Per-operator executors | `python/batcher/dist/executors/{aggregate,join,sort,map,window,union,distinct}.py` |
| Ray tasks/actors, placement, autoscale, fault policy | [`python/batcher/dist/executors/ray_runtime/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/dist/executors/ray_runtime) |
| Split balancing | [`python/batcher/dist/executors/partition_io/assignment.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/executors/partition_io/assignment.py) |
| Learned sizing (partitions, actor pool, straggler factor) | [`python/batcher/dist/adaptive_sizing/sizing.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/adaptive_sizing/sizing.py) |
| Join-skew learning | [`python/batcher/dist/skew.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/skew.py) |
| Rust mergeable primitives | [`crates/bc-interp/src/dist.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/dist.rs) |

## See also

- {doc}`Architecture </architecture/index>`: distribution as a backend, never a second semantics.
- {doc}`Fault tolerance </architecture/fault-tolerance>`: straggler speculation and worker loss.
- {doc}`Carbonite </architecture/internals/carbonite>`: the envelope each worker runs inside.
- {doc}`Ray integration </integrations/compute/ray>`: setting up the cluster this schedules onto.
- {doc}`Configuration options </configuration/options>`: every `distributed.*` knob named here.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: the node-count curves, and the cluster grid.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why a partitioned result equals a single-node one.
- {doc}`Shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: how the bytes actually move.
- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`: what keeps a reducer from drowning.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: where the learned partition counts live.
