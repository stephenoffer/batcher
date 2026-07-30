# Performance and memory

Batcher is built to stay fast on a laptop and survive on a cluster. The levers a
user actually reaches for are few: cache a result you reuse, size the morsels, give
the engine a memory budget so it spills instead of dying, and read back what the
query did. Every knob lives on one frozen `Config`, applied process-wide with
`set_config` or scoped to a block with `config_context`.

This page is about making *your* query faster. For how Batcher compares against DuckDB,
Polars, and Daft, see the {doc}`benchmarks <../benchmarks/index>`, which carry the
methodology behind every figure.

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

## Result caching

`cache()` marks a dataset's result to be stored in memory the first time it is
computed. A later terminal on the *same* cached dataset returns the stored result
instead of re-running the plan. It is the Spark/Polars `cache` pattern. Use it when
an expensive upstream such as a filter, a join, or an aggregation feeds several downstream
queries.

```python
hot = events.filter(bt.col("status") == "active").cache()

first = hot.to_pydict()   # computed once, then stored
second = hot.to_pydict()  # cache hit — no re-execution
print(first == second)
# True
print(sorted(first["region"]))
# ['eu', 'eu', 'us', 'us']
```

The cache is process-wide and memory-bounded by `memory.result_cache_max_bytes`, which
defaults to 256 MiB. It holds results LRU and yields their memory back to running
queries under pressure, so caching never grows the process without bound. And it
marks *this* result: a further transform on a cached dataset is a new, uncached one.

## Reusing a cached dataset

Because a cached dataset is a reusable handle, run as many terminals on it as you
like. Each is served from the one materialized result.

```python
print(hot.count())
# 4
print(hot.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict())
# {'region': ['eu', 'us'], 'total': [12.0, 13.0]}
```

## What a query costs before any row moves

Every terminal operation pays a fixed control-plane cost before the engine touches
data: building the plan, optimizing it, deciding a memory envelope, and reporting what
happened. On a query over millions of rows this is invisible. On a workload made of
thousands of *small* queries, such as an interactive session, a serving endpoint, or a test
suite, it is the whole bill, so it is worth knowing what is amortized for you and what you can
switch off.

Two things are already reused across calls, and you get them without asking:

Optimized plans are memoized. Re-issuing a query whose plan lowers to the same IR, over
the same sources and configuration, reuses the plan Kyber already chose instead of
re-deriving it. `explain()` reads the same memo, so inspecting a plan and then running it
optimizes once, and what `explain()` shows is what `collect()` runs.

```python
plan = events.filter(bt.col("amount") > 5).group_by("region").agg(n=bt.count())
print(plan.explain().splitlines()[0].split()[0])
# aggregate
print(plan.sort("region").to_pydict())
# {'region': ['eu', 'us'], 'n': [2, 3]}
```

Connectors load on first use. Importing Batcher does not import every database,
warehouse, and message-broker connector it can reach; a format family is imported the
first time a format from it is named. Reading Parquet, CSV, or JSON costs nothing extra,
and a process that never opens a Snowflake table never pays to be able to.

One thing is on by default and does cost per query: the JSON *event log*. Each terminal
operation writes one profile document under `$BATCHER_HOME/logs`, capped at
`observability.event_log_max_files` (200) with the oldest pruned. That is what makes a
finished query inspectable after the fact. A workload issuing many small queries that
nobody will inspect can turn it off:

```python
import dataclasses

from batcher.config import active_config, config_context

current = active_config()
quiet = current.replace(
    observability=dataclasses.replace(current.observability, event_log=False)
)
with config_context(quiet):
    print(events.count())
# 6
```

Or set `BATCHER_OBSERVABILITY_EVENT_LOG=false` in the environment before
the process starts. Turning it off changes no result, only whether the profile is
archived to disk.

## Morsel-driven execution

The engine's unit of work is a *morsel*: a small Arrow `RecordBatch`, 16,384 rows by
default, sized to fit cache so scheduling stays granular and parallelism stays even
across cores. You rarely change it. When you do, `execution.morsel_rows` and
`execution.morsel_bytes` are the levers, and a morsel splits at whichever bound trips
first, so wide rows carrying large strings, embeddings, or blobs stay memory-bounded even
at a fixed row count. The setting is result-invariant. A morsel batches data; it never
changes the output.

```python
from batcher.config import Config, ExecutionConfig, config_context

small_morsels = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
with config_context(small_morsels):
    out = events.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict()
print(out)
# {'region': ['eu', 'us'], 'total': [20.0, 112.0]}
```

`execution.adaptive_morsel_sizing` (on by default) shrinks the per-morsel target
under memory pressure so the streaming working set stays bounded when memory is
tight, and leaves it at the configured target otherwise. For per-batch UDF and
inference workloads, the `pid` section tunes the controller that grows or shrinks
the batch toward a target latency.

## Adaptive re-optimization

Every cost estimate is a guess until the query runs. At a pipeline breaker, meaning a
sort, an aggregate, or a join build, the engine has *measured* the real size of what it
just processed. When an estimate was off by more than `optimizer.reoptimize_error`, which
defaults to 2x, it re-plans the rest of the query on the measured numbers before
continuing. This is the part static optimizers cannot match. `collect(adaptive=...)`
controls it: `"auto"` (the default) turns it on only when a join's input size is a
pure estimate, and `True`/`False` force it. The result is identical whichever way it
runs.

```python
dim = bt.from_pydict({"region": ["us", "eu"], "tier": ["gold", "silver"]})
joined = (
    events.join(dim, on="region")
    .group_by("tier")
    .agg(total=bt.col("amount").sum())
    .sort("tier")
)
print(joined.collect(adaptive=True).to_pydict())
# {'tier': ['gold', 'silver'], 'total': [112.0, 20.0]}
```

## What the engine learns is per machine

Adaptive re-optimization improves one query while it runs. A second loop improves the *next*
run: Batcher fits per-row costs, memory per group, batch sizes, and VRAM footprints from what
it measured last time, so a query that runs often is planned better each time.

Every one of those numbers is a property of a workload **on a machine**. A per-row coefficient
fitted on a 3 GHz AVX-512 core is wrong on a small ARM core by several times over, and a VRAM
figure measured on an A100 is wrong on a T4 by five. So each is stored under a fingerprint of
the machine that measured it: core and cache counts, memory capacity, vector width, NUMA
nodes, the scratch device's class, the attached accelerators.

Two consequences worth knowing.

Machines that are alike share a fingerprint, so a fleet of identical nodes pools everything it
learns and converges as fast as the whole fleet can produce feedback. Machines that differ do
not, so on a cluster that mixes instance types each shape converges on its own share of the
runs. That is slower, and it is the correct trade: a model averaged across unlike hardware is
wrong everywhere rather than slow anywhere. Batcher logs a line the first time it notices a
mixed fleet, so the slower convergence has an explanation.

Changing the machine resets the learning for that machine class. Adding memory, attaching a
GPU, or moving from a spinning disk to NVMe all produce a different fingerprint, and the
engine starts from its priors rather than from measurements of hardware that no longer exists.
The fingerprint is in `bt.start_ui()`'s system panel if you need to confirm which class a node
belongs to.

Statistics about the *data* are deliberately not scoped this way. Distinct counts, quantiles,
column widths, and selectivities describe the data and are identical wherever it is read, so
they are shared across every machine that touches the dataset.

## Out-of-core spilling

Stateful operators spill to disk when they would exceed the memory envelope, which covers
aggregation, distinct, sort, join build, and windowed-by-partition, so a query that does not
fit in memory slows down rather than dying. Spilling is a property of the runtime
primitive, not a separate plan: the result is bit-identical to the in-memory run.

You do not ask an operator to spill. You set a memory budget and the engine decides.
Setting `memory.max_memory_bytes` is what opts the in-memory engine into spilling,
and the data plane receives a per-operator budget of `max_memory_bytes x hard_limit`.
A deliberately tiny budget forces the out-of-core path here so the example runs
anywhere. In production you set it to the real ceiling, honoring a container or cgroup
limit.

```python
from batcher.config import MemoryConfig

big = bt.from_pydict({"k": [i % 50 for i in range(2000)], "v": list(range(2000))})


def totals(ds: bt.Dataset) -> dict:
    return ds.group_by("k").agg(total=bt.col("v").sum()).sort("k").to_pydict()


in_memory = totals(big)

tiny_budget = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
with config_context(tiny_budget):
    spilled = totals(big)

print(in_memory == spilled)
# True — the out-of-core result is identical to the in-memory one
print(len(spilled["k"]))
# 50
```

On a big job the local (NVMe) spill tier overflows to `memory.spill_remote_uri` (any
fsspec URL) once local disk fills. A skewed aggregate bucket that overflows
`memory.spill_bucket_max_bytes` is re-partitioned and reduced one piece at a time, so
a large or skewed query degrades gracefully instead of running out of memory.

## Running a query on the GPU

A supported relational query can run on the GPU (cuDF) instead of the CPU engine by
passing `backend=` to `collect()`. It is the same query and the same result. Only
*where* it runs changes.

```python
# docs: skip
ds = bt.read.parquet("s3://warehouse/events/")
q = ds.group_by("country").agg(revenue=bt.col("amount").sum())

q.collect(backend="cpu")   # the native engine (default)
q.collect(backend="gpu")   # force the cuDF GPU backend for any supported shape
q.collect(backend="auto")  # let Kyber decide GPU vs CPU by estimated size
```

`backend="auto"` is the adaptive choice. Kyber sends a query to the GPU only when the
estimated input is large enough to amortize the device overhead of host-to-device
transfer, cuDF import, and task dispatch, and only when it fits the cluster's GPU
memory. A working set that exceeds one GPU is sharded across many. One that exceeds
them all stays on the spillable CPU engine. Below the crossover a small query stays on
the CPU engine, and anything unsupported or a GPU-less cluster falls back
transparently.

The crossover itself is learned rather than fixed. Each GPU or CPU group-by run
records its estimated rows and wall time to the metadata hub, Kyber fits a cost line
per backend and solves for their intersection, so the threshold self-corrects to the
hardware you have. Until enough runs are seen it uses the measured default,
`distributed.gpu_min_rows`.

## Measured results

Benchmark numbers live in one place, {doc}`../benchmarks/index`, so a figure is never
restated in two pages that can drift apart. Every number there is correctness-gated:
the engines must return the identical result before any timing is recorded, and a run
that disagrees with the oracle produces no number at all.

Start with the page that matches your workload:

| Page | Covers |
|---|---|
| {doc}`../benchmarks/analytics` | Operators, TPC-H, connectors, and the lazy control plane. |
| {doc}`../benchmarks/ai-and-gpu` | Batch inference, embeddings, LLM generation, training ingest. |
| {doc}`../benchmarks/multimodal-ingest` | Image, point-cloud, audio, and video decode. |
| {doc}`../benchmarks/scaling` | Distributed scaling, spilling, and the memory-bound regime. |
| {doc}`../benchmarks/methodology` | Hardware, the correctness gate, and how to reproduce a run. |

To measure your own change, run the harness the same way the project does:

```bash
python benchmarks/run.py --benchmark operators               # single-node operator mix (sf1)
python benchmarks/run.py --benchmark operators --scale 10    # at 60M rows (sf10)
```

## Reading a query plan

`explain()` runs the optimizer and renders the optimized plan with per-operator
cardinality estimates, without executing. It is how you confirm a predicate landed at
the scan, or that a join was reordered the way you expected.

```python
print(events.filter(bt.col("status") == "active").select("region", "amount").explain())
```

One operator per line, indented by depth, each with its row estimate and where that
estimate came from: `exact` when the source knows, `learned` from a previous run's
measurements, `default` from a heuristic. Under `decisions:` are the calls the engine
made along the way.

```text
project                         est≈4 (learned)
  filter                        est≈4 (learned)
    scan                        est≈6 (exact)

decisions:
  - [core/io] source read at 40 MB/s (learned)
```

The throughput in `decisions:` is measured, so it moves from run to run.

Where `explain()` shows the *planned* shape, `stats()` runs the query and reports
what the engine *measured*: rows in/out, wall time, peak bytes, spill, and the
operator that dominated wall time.

```python
run = events.group_by("region").agg(total=bt.col("amount").sum()).stats()
print(run.rows)
# 2
print(run.bottleneck is not None)
# True — the operator that took the most wall time
```

For a quick per-column read of the data itself (counts, null fraction, approximate
distinct count) before a load, `profile()` executes a one-row-per-column summary.

```python
print(events.profile().columns)
# ['column', 'count', 'null_count', 'null_fraction', 'approx_distinct']
```

## Tuning checklist

Reach for these in order. Most workloads need none of them.

- A result reused across queries: `cache()` the shared upstream.
- Bounded or container memory: set `memory.max_memory_bytes` to the real ceiling, so
  stateful operators spill instead of OOMing.
- Wide rows (blobs, embeddings): lower `execution.morsel_bytes` to keep the working set
  bounded, and leave `morsel_rows` alone.
- A query slower than expected: `explain()` to check the plan, then `stats()` to find the
  operator that dominated wall time.
- A cluster shuffle under memory pressure: the credit-based backpressure in `flow_control`
  and `distributed`. See {doc}`Fault tolerance <../architecture/fault-tolerance>`.

Every field, with its default and meaning, is in
{doc}`Configuration options <../configuration/options>`.

## See also

- {doc}`Benchmarks <../benchmarks/index>`: the measured results, and how to reproduce them.
- {doc}`Configuration options <../configuration/options>`: the full `Config` reference.
- {doc}`Fault tolerance <../architecture/fault-tolerance>`: how a distributed query
  survives task, worker, and node failures.
- {doc}`Aggregations <aggregations>`: the breakers that spill and re-optimize.
- {doc}`Agent skills <../agents/index>`: `optimize-a-slow-query` covers the measure-first
  methodology and the ordered fix checklist.
