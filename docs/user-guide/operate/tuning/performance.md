# Performance and memory

This page covers the levers you reach for when a correct query needs to be faster or leaner: caching a reused result, cutting the fixed cost of small queries, sizing morsels, giving the engine a memory budget so it spills instead of dying, and reading back what the query did.

Every knob lives on one frozen {py:class}`Config <batcher.Config>`, applied process-wide with {py:func}`set_config <batcher.set_config>` or scoped to a block with {py:func}`config_context <batcher.config_context>`. For how Batcher compares against DuckDB, Polars, and Daft, see the {doc}`benchmarks </benchmarks/index>`, which carry the methodology behind every figure.

## Setup

```python
import batcher as bt

events = bt.from_pydict(
    {
        "region": ["us", "eu", "us", "eu", "us", "eu"],
        "status": ["active", "active", "churned", "active", "active", "churned"],
        "amount": [10.0, 5.0, 99.0, 7.0, 3.0, 8.0],
    }
)
```

## Cache a result you reuse

A `Dataset` is a plan, so two terminals run it twice. `cache()` marks a result to be kept the first time it is computed, and every later terminal on that same dataset is served from it. Use it when an expensive filter, join, or aggregation feeds several downstream queries.

```python
hot = events.filter(bt.col("status") == "active").cache()

first = hot.to_pydict()  # computed once, then stored
second = hot.to_pydict()  # cache hit, no re-execution
print(first == second)
# True
print(sorted(first["region"]))
# ['eu', 'eu', 'us', 'us']
print(hot.count())
# 4
print(hot.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict())
# {'region': ['eu', 'us'], 'total': [12.0, 13.0]}
```

The cache is process-wide and bounded by `memory.result_cache_max_bytes`, 256 MiB by default. It evicts by cost rather than recency, demotes to a local disk tier instead of dropping, and yields memory back to running queries under pressure. It marks *this* result only: a further transform on a cached dataset is a new plan. {doc}`caching` covers storage levels, the shared cross-process cache, and how to tell whether the cache is earning its budget.

## What a query costs before any row moves

Every terminal pays a fixed control-plane cost before the engine touches data: building the plan, optimizing it, deciding a memory envelope, and reporting what happened. Over millions of rows it is invisible. On a workload of thousands of *small* queries, such as an interactive session, a serving endpoint, or a test suite, it is most of the bill.

Two savings come without asking. Optimized plans are memoized: re-issuing a query whose plan lowers to the same IR, over the same sources and configuration, reuses the plan Kyber already chose. `explain()` reads the same memo, so inspecting a plan and then running it optimizes once, and what `explain()` shows is what `collect()` runs.

```python
plan = events.filter(bt.col("amount") > 5).group_by("region").agg(n=bt.count())
print(plan.explain().splitlines()[0].split()[0])
# aggregate
print(plan.sort("region").to_pydict())
# {'region': ['eu', 'us'], 'n': [2, 3]}
```

Connectors also load on first use. Importing Batcher doesn't import every database, warehouse, and message-broker connector; a format family is imported the first time you name one of its formats, so a process that never opens a Snowflake table never pays for it.

### Turn off the event log for serving workloads

One default does cost per query: the JSON *event log*. Each terminal writes one profile document under `$BATCHER_HOME/logs` (`~/.batcher/logs` when the variable is unset), capped at `observability.event_log_max_files`, 200, with the oldest pruned. That's what makes a finished query inspectable later. A workload issuing many small queries that nobody will inspect can switch it off:

```python
import dataclasses

from batcher.config import active_config, config_context

current = active_config()
quiet = current.replace(observability=dataclasses.replace(current.observability, event_log=False))
with config_context(quiet):
    print(events.count())
# 6
```

Setting `BATCHER_OBSERVABILITY_EVENT_LOG=false` before the process starts does the same. No result changes. Only the archived profile goes.

The document's cost is roughly fixed while the query's isn't, so it matters in proportion to how small the queries are. Measured on a 50,000-row SQLite table with the release engine, in `benchmarks/BENCHMARK_RESULTS.md`:

| Workload | Event log on | Off |
| --- | --- | --- |
| A terminal op over a one-row table, no operators | 1.945 ms | **0.733 ms** (-62%) |
| A point lookup pushed to the database | 3.503 ms | **2.930 ms** (-16%) |
| Point lookups per second, one process | 309 | **374** |

On a serving-shaped workload, switch it off. On anything that reads more than a few thousand rows, don't bother.

### The small-query fast path

On a query that returns in about a millisecond, the orchestration *is* the cost. Measured on a filter over 10,000 in-memory rows with the event log already off, the engine call and the Arrow table build account for roughly a fifth of the query. The rest is admission, morsel sizing, pressure classification, and profile assembly.

`execution.fast_path` skips all of that for plans that provably don't need it. The plan is still optimized through the same plan cache and runs through the same engine call, so the result is identical. It's off by default:

```python
fast = current.replace(execution=dataclasses.replace(current.execution, fast_path=True))
with config_context(fast):
    print(events.filter(bt.col("amount") > 5).count())
# 4
```

The path is taken only when the query is single-node, on the CPU backend, reads sources already in memory, contains no `map_batches` UDF, and stays under a row and plan-node cap. Anything else takes the ordinary path, so turning the flag on is always safe.

A query on the fast path still feeds the cross-query learning loop: its measured cardinality, selectivity, per-operator metrics, and column statistics are all recorded. It skips only the *resource* half, since it consults no pressure monitor and holds no resource manager.

```{warning}
The fast path gives up observability. A query answered on it doesn't appear in `explain(analyze=True)`, the JSON event log, or the dashboard, because the profile those read is assembled by the orchestration it skips. Use it on a latency-sensitive serving path whose plan shape you already trust, and leave it off where you need to see what ran.
```

## Morsel-driven execution

The engine's unit of work is a *morsel*: a small Arrow `RecordBatch` that closes at 16,384 rows or 1 MiB, whichever comes first. `execution.morsel_rows` and `execution.morsel_bytes` set the two bounds. The byte bound is what keeps wide rows carrying large strings, embeddings, or blobs memory-bounded at a fixed row count. You rarely change either, and neither changes the output.

The figure follows a scan, filter and project chain from its input batches to the cores:

![Three zoom levels of morsel-driven execution. First, whatever batches the source emitted are split or coalesced into morsels, each full at 16,384 rows or 1 MiB, whichever trips first, and a single over-budget row becomes a one-row morsel. Second, the morsel vector is handed to one worker pool with par_iter(), one morsel per task; the pool width W is operator_cores(), capped by the number of morsels the input can produce, and an idle worker steals work through rayon's scheduler rather than a Batcher-owned queue. Third, one worker takes one morsel through the filter, compiled once by the JIT, and the project, which is never materialized, in a single pass, and the output morsels are collected in index order. Filter and project therefore preserve row order; the hash operators do not.](/_static/diagrams/morsel_scheduling.svg)

```python
from batcher.config import Config, ExecutionConfig, config_context

small_morsels = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
with config_context(small_morsels):
    out = events.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict()
print(out)
# {'region': ['eu', 'us'], 'total': [20.0, 112.0]}
```

`execution.adaptive_morsel_sizing`, on by default, shrinks the per-morsel target under memory pressure and leaves it alone otherwise. For per-batch UDF and inference workloads, the `pid` section tunes the controller that grows or shrinks the batch toward a target latency.

## Adaptive re-optimization

Every cost estimate is a guess until the query runs. At a pipeline breaker, meaning a sort, an aggregate, or a join build, the engine has *measured* what it just produced. When an estimate missed by more than `optimizer.reoptimize_error`, 2x by default, it re-plans the rest of the query on the measured numbers. This is stage-boundary re-optimization, the same grain as Spark's AQE, and Batcher runs it on a single node as well as on a cluster.

{py:meth}`collect(adaptive=...) <batcher.Dataset.collect>` controls it. `True` and `False` force it. The default, `"auto"`, turns it on only where it can change a decision: the query has a join whose input size is a pure estimate, and its input clears a per-stage floor of 5M rows or about 320 MB for each pipeline breaker the loop would cut at, so roughly 10M rows for the simplest joined shape. Below that, the one-shot plan is cheaper than staging it. A distributed plan that can only run staged takes that path at any size. The result is identical whichever way it runs.

```python
dim = bt.from_pydict({"region": ["us", "eu"], "tier": ["gold", "silver"]})
joined = (
    events.join(dim, on="region").group_by("tier").agg(total=bt.col("amount").sum()).sort("tier")
)
print(joined.collect(adaptive=True).to_pydict())
# {'tier': ['gold', 'silver'], 'total': [112.0, 20.0]}
```

## A repeated top-N gets faster on its second run

`ORDER BY x DESC LIMIT 10` over a wide table decodes every projected column of every row, then throws all but ten away. What would make it cheap is a value separating the ten from the rest, and before the scan nothing knows one.

After the scan, Batcher does. It remembers the tenth-best value and applies it on the next run of the same query as a filter, which the reader answers by skipping row groups whose bounds exclude it and decoding the other columns only for rows that survive.

```python
top = events.sort("amount", descending=True).limit(3)

print(top.collect().to_pydict()["amount"])  # first run: learns where the cut falls
# [99.0, 10.0, 8.0]
print(top.collect().to_pydict()["amount"])  # second run: starts from it
# [99.0, 10.0, 8.0]
```

There's nothing to turn on, and the answer never depends on it. The filter removes only rows strictly worse than the remembered value, so whenever the requested number of rows survives, those rows *are* the true top-N however stale the value was. If too few survive, which is what happens after the data moves, the engine notices the short result and re-runs the query as written. A stale value costs one extra cheap scan, never a wrong row.

The first run gets no speedup, because it is the run that learns the bound. Two shapes opt out entirely. A `nulls_first` ordering is never seeded, because a bound predicate would drop the nulls it wants on top. Very large limits are skipped too, since a bound far out in the tail excludes almost nothing.

## What the engine learns is per machine

Adaptive re-optimization improves one query while it runs. A second loop improves the *next* run: Batcher fits per-row costs, memory per group, batch sizes, and VRAM footprints from what it measured, so a query that runs often is planned better each time.

Each of those numbers describes a workload **on a machine**. A cost fitted on a large server core is wrong on a small ARM core, and a VRAM figure measured on one GPU model is wrong on another. So each is stored under a fingerprint of the machine that measured it: core and cache counts, memory capacity, vector width, NUMA nodes, the scratch device's class, and the attached accelerators.

Machines that are alike share a fingerprint, so a fleet of identical nodes pools everything it learns. Machines that differ don't, so a cluster mixing instance types converges per shape on its own share of the runs. That's slower, and it's the right trade: a model averaged across unlike hardware is wrong everywhere. Batcher logs a line when it notices the cluster mixes machine classes.

Changing the machine resets the learning for that class. Adding memory, attaching a GPU, or moving scratch from a spinning disk to NVMe produces a new fingerprint, and the engine starts from its priors. {py:func}`bt.start_ui() <batcher.start_ui>` shows a node's fingerprint in its system panel.

Statistics about the *data* are not scoped this way. Distinct counts, quantiles, column widths, and selectivities are the same wherever the data is read, so every machine that touches the dataset shares them.

## Out-of-core spilling

Stateful operators spill to disk when they would exceed the memory envelope. That covers aggregation, distinct, sort, join build, and windows with a `PARTITION BY`, so a query that doesn't fit in memory slows down rather than dying. Spilling is a property of the runtime primitive, not a separate plan, and the result is identical to the in-memory run.

Which mechanism an operator uses follows from what its state is keyed on, as the figure shows:

![Two out-of-core mechanisms, chosen by the shape of an operator's state. Keyed state takes grace partitioning: rows are routed by a hash of the key into buckets, one Arrow IPC file per bucket; one bucket at a time is read back and run through the in-memory kernel, and the union of the buckets is the whole answer; a bucket still over budget is re-split under a salted hash, with a fan-out of at most 256 and a depth of at most 3. Aggregate spills partial state, one row per group per morsel; join co-partitions both sides by the join key and keeps only the build bucket resident while the probe streams through it; window spills whole rows keyed by PARTITION BY; DISTINCT ON reduces to one row per key per morsel before writing. Ordered state takes external merge sort: the input is sorted into runs cut by size, 64 MiB by default and never by key, so skew cannot defeat it; up to 16 runs are merged at a time with one batch per run resident, and passes repeat until one run is left. Median and n_unique stream one pass over the sorted run. Both mechanisms return exactly what the in-memory kernel returns, and only peak memory differs.](/_static/diagrams/spill_ladder.svg)

You don't ask an operator to spill. The engine has a budget and decides. By default each query senses the memory cap from host RAM or a container or cgroup limit, and you can pin one with `memory.max_memory_bytes`. The data plane receives a per-operator budget of that cap times `memory.hard_limit`, which defaults to 0.9. The example below forces the out-of-core path with a deliberately tiny budget so it runs anywhere. In production, set a cap only when the OS reports the wrong one, and set `memory.unbounded_memory` to turn spilling off.

```python
from batcher.config import MemoryConfig

big = bt.from_pydict({"k": [i % 50 for i in range(2000)], "v": list(range(2000))})


def totals(ds: bt.Dataset) -> dict:
    return ds.group_by("k").agg(total=bt.col("v").sum()).sort("k").to_pydict()


in_memory = totals(big)

tiny_budget = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
with config_context(tiny_budget):
    spilled = totals(big)

print(in_memory == spilled)  # the out-of-core result is identical to the in-memory one
# True
print(len(spilled["k"]))
# 50
```

On a big job, the local spill tier overflows to `memory.spill_remote_uri`, any fsspec URL, once local disk fills. An aggregate bucket that overflows `memory.spill_bucket_max_bytes`, 128 MiB by default, is re-partitioned and reduced one piece at a time, so a skewed aggregate degrades gracefully.

A few operators can't spill because they need one global order over the whole relation, and those raise rather than risk the process. {doc}`Skewed keys and hostile data shapes <skew>` lists them, along with what Batcher does about a join key concentrated on one value.

## Read back what the query did

`explain()` renders the optimized plan with a row estimate per operator, without executing. It's how you confirm a predicate landed at the scan or a join was reordered the way you expected. Each estimate carries its provenance: `exact` when the source knows, `learned` from a previous run, `default` from a heuristic.

```python
print(events.filter(bt.col("status") == "active").select("region", "amount").explain())
```

```text
query plan (planned)                            3 operators
───────────────────────────────────────────────────────────
OPERATOR                     ESTIMATE  NOTES
project                         est≈4  (learned)
└─ filter  [status = active]    est≈4  (learned)
   └─ scan  [source 0]          est≈6  (exact)  pushed[status = active]

decisions:
  - [core/io] source read at 40 MB/s (learned)
```

The throughput under `decisions:` is measured, so it moves between runs. `stats()` runs the query and reports what the engine *measured*: rows in and out, wall time, peak bytes, spill, and the operator that dominated wall time.

```python
run = events.group_by("region").agg(total=bt.col("amount").sum()).stats()
print(run.rows)
# 2
print(run.bottleneck is not None)  # the operator that took the most wall time
# True
```

For a per-column read of the data itself before a load, covering counts, null fraction, and approximate distinct count, {py:meth}`profile() <batcher.Dataset.profile>` executes a one-row-per-column summary.

```python
print(events.profile().columns)
# ['column', 'count', 'null_count', 'null_fraction', 'approx_distinct']
```

{doc}`explain-plans` walks through every column of `explain(analyze=True)` and the checklist for a slow query.

## Measure your own change

Benchmark numbers live in one place, {doc}`/benchmarks/index`, so a figure is never restated in two pages that can drift apart. Every number there is correctness-gated: the engines must return the identical result before any timing is recorded. {doc}`/benchmarks/results/analytics` covers operators, TPC-H, and connectors, {doc}`/benchmarks/results/scaling` covers distributed scaling and spilling, and {doc}`/benchmarks/methodology` covers hardware and reproduction.

To measure a change the way the project does, run the harness:

```bash
python benchmarks/run.py --benchmark operators               # single-node operator mix (sf1)
python benchmarks/run.py --benchmark operators --scale 10    # at 60M rows (sf10)
```

## Tuning checklist

Reach for these in order. Most workloads need none of them.

- A query slower than expected: `explain()` to check the plan, then `stats()` to find the operator that dominated wall time.
- A result reused across queries: `cache()` the shared upstream.
- Many tiny queries: turn off the event log, then consider `execution.fast_path`.
- Bounded or container memory: set `memory.max_memory_bytes` to the real ceiling, so stateful operators spill instead of running out of memory.
- Wide rows such as blobs or embeddings: lower `execution.morsel_bytes` and leave `morsel_rows` alone.
- Data in object storage on a cluster: see {doc}`object-storage`.
- A large reducing query and GPUs available: see {doc}`gpu`.
- A cluster shuffle under memory pressure: the credit-based backpressure in `flow_control` and `distributed`. See {doc}`Fault tolerance </architecture/fault-tolerance>`.

Every field, with its default and meaning, is in {doc}`Configuration options </configuration/options>`.

## See also

- {doc}`caching`: storage levels, the shared cache, and cache statistics.
- {doc}`explain-plans`: reading `explain(analyze=True)` line by line.
- {doc}`large-tables`: what changes once planning a table costs more than reading it.
- {doc}`Configuration options </configuration/options>`: the full `Config` reference.
- {doc}`Fault tolerance </architecture/fault-tolerance>`: how a distributed query survives task, worker, and node failures.
- {doc}`Aggregations </user-guide/analyze/aggregations>`: the breakers that spill and re-optimize.
- {doc}`Agent skills </agents>`: `optimize-a-slow-query` covers the measure-first method and the ordered fix checklist.
- {doc}`/cookbook/operations/memory_and_caching`: caching and spilling under a tight budget, as a script.
