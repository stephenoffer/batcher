# Metrics

This page describes the counters Batcher keeps for a scrape loop, and the two shapes it
serves them in. For the per-query detail behind them, see
{doc}`Observability <observability>`.

The event bus is the right tool when you want every detail of one query. When you want a
handful of numbers scraped every fifteen seconds forever, use the counters instead.
`metrics_snapshot` returns them as a nested dict of plain numbers, with no Batcher types in
it and nothing to close:

```python
from batcher.observe import metrics_snapshot

snap = metrics_snapshot()
print(sorted(snap))
# ['backends', 'bytes', 'cpu', 'data_quality', 'gpu', 'inference', 'io', 'logs', 'memory',
#  'node', 'operators', 'partitions', 'queries', 'recovery', 'resources', 'rows',
#  'skipped', 'spills', 'streaming', 'uptime_seconds', 'writes']
```

Several of those fill in only for work that reports them: `partitions`, `skipped`, and
`recovery` for a distributed read, `inference` and `gpu` for a batch-inference pass,
`data_quality` for a run that checks a contract. A single-node relational query leaves those
at zero rather than absent, so a scraper never has to handle a changing key set.

Two sections are the exception and stay empty until something fills them. `resources` needs
a query to have completed under a resource manager, and `streaming` needs a continuous query
to have run a micro-batch. Both hold *levels* rather than totals, and a zero level is a
claim — "the pool is empty", "the query is idle" — that would be wrong rather than merely
uninformative.

## Resource utilization

`queries`, `rows`, and `operators` say what the process *did*. `cpu`, `memory`, and `io`
say what it *cost the machine*, which is the half that answers whether a slow job is
compute-bound, memory-bound, or waiting on a disk:

```python
import batcher as bt
from batcher.observe import metrics_snapshot, start_metrics

start_metrics()
rows = 200_000
ds = bt.from_pydict({"a": list(range(rows))})
ds.filter(bt.col("a") > 1).group_by(k=bt.col("a") % 1000).agg(n=bt.col("a").sum()).collect()

snap = metrics_snapshot()
print(snap["cpu"]["time_ms_total"] > 0, snap["rows"]["scanned_total"] == rows)
# True True
```

The fields worth knowing:

| Field | Meaning |
|---|---|
| `cpu.time_ms_total` | CPU milliseconds summed across every worker thread. |
| `cpu.execution_ms_total` | Wall milliseconds spent inside the engine, planning excluded. |
| `cpu.cores_busy` | The two above divided: mean cores kept busy. |
| `cpu.involuntary_context_switches_total` | Times the scheduler preempted the engine, which is contention for cores this process was told it had. |
| `memory.peak_rss_bytes_max` | The largest resident-set growth any one query forced. |
| `memory.major_faults_total` | Page faults that needed disk. Any meaningful count means the box is paging against the query. |
| `io.read_bytes_total` / `io.write_bytes_total` | Bytes that reached a block device, page-cache hits excluded. |
| `spills.bytes_total` | Logical bytes routed to disk, by a spilling operator or an out-of-core phase. |
| `backends` | Operators per execution tier: `interp`, `jit`, or `interp+jit`. |

`cores_busy` is the one to plot first. A query at 1.0 on a sixteen-core box did not
parallelize; one at 14 with a high `involuntary_context_switches_total` is fighting
something else on the machine, and those two have opposite fixes.

Two conventions matter when reading any of this. `bytes.scanned_total` is the Arrow
in-memory volume the scans produced, while `io.read_bytes_total` is what actually came off
a device, so a warm scan and a cold scan of the same file are identical in the first and
orders of magnitude apart in the second. And **zero means unmeasured, not zero**: not every
platform reports every counter, and the engine's streaming executor interleaves its
operators, so per-operator hardware figures are zero there by design. The `cpu`, `memory`,
and `io` roll-ups are measured across the whole execution and hold on every tier.

## What the job produced

`rows` and `bytes` count what a job read. `writes` counts what it committed, which for an
ETL pipeline is the thing it exists to do and the thing no counter reported: a run that read
its inputs correctly and wrote half of them looked, from the metrics, exactly like a healthy
run.

| Field | Meaning |
|---|---|
| `commits_total` | Writes committed. One per `ds.write(...)`, not one per file. |
| `files_total` | Data files committed. |
| `rows_total` | Rows committed. |
| `bytes_total` | Size on storage, after encoding and compression. |
| `by_format` | The same three per sink format, so "the parquet sink stopped producing" and "the delta sink did" are separable. |

```python
import tempfile

import batcher as bt
from batcher.observe import metrics_snapshot, start_metrics

start_metrics()
with tempfile.TemporaryDirectory() as out:
    bt.from_pydict({"a": [1, 2, 3]}).write.parquet(f"{out}/table")

print(metrics_snapshot()["writes"]["rows_total"] >= 3)
# True
```

`bytes_total` is **not** comparable with `bytes.scanned_total`. One is size on storage after
compression, the other is Arrow's in-memory size of what was read, and dividing them gives a
compression ratio only because the two are labelled apart.

## What each execution path reports

Batcher runs a query one of several ways, and they do not all measure the same things. The
gaps are stated here rather than left for you to infer from a zero, because **a zero in this
export means unmeasured, not none**.

| Path | Per-operator detail | Machine cost |
|---|---|---|
| Single node, in memory | Yes | Yes |
| `iter_batches` (streaming to you) | No: the result is produced lazily, so there is no profile to replay | No |
| `map_batches` / ML pipeline | Yes, per stage, against the logical plan | Yes |
| Out-of-core (spilling) | No: it streams the plan through unmetered dispatches | Yes, measured around the phase, plus the spill volume |
| Distributed, disk shuffle | Yes, from the workers' documents | Yes, summed across workers |
| Distributed, Arrow Flight | **No** | **No** |

`iter_batches` is the one path that reports *more* than the others in one respect: it is
the only one that can count rows while the query runs, so it fills `rows.streamed_total` and
`bytes.streamed_total` per batch. Those count rows **delivered to you**, not rows read — a
filter upstream means the two differ — which is why they are their own fields rather than
folded into `rows.scanned_total`.

The Flight row is the one to know about, because Flight is what a genuine multi-node cluster
uses by default. Its workers call the engine's unmetered entry point, so they produce no
per-operator record at all: nothing reaches the profile, and nothing reaches the optimizer's
learned statistics either. Live progress still works there, so `partitions` counts and the
dashboard fills in. Pass `transport="disk"` to `collect()` if you need the per-operator
numbers from a distributed run today.

## Streaming queries

A continuous query reports a full micro-batch record to any
{py:class}`StreamingQueryListener <batcher.plan.streaming.StreamingQueryListener>` you
register, and the same numbers land in `streaming`, keyed by query name. That matters
because the streaming failure modes develop over hours rather than announcing themselves: a
query that falls a little further behind every batch, or a state store that grows because a
watermark never advances, both look healthy in any single reading and obvious in a chart.

| Field | Meaning |
|---|---|
| `batches`, `input_rows`, `output_rows`, `duration_ms` | Counters, summed across micro-batches. |
| `behind_by_ms` | How much longer the last micro-batch took than its trigger cadence. Growing means the query is falling behind its source. |
| `input_rows_per_second` / `processed_rows_per_second` | The last batch's throughput in and out. |
| `state_rows` / `state_bytes` | What the stateful operators currently retain. |
| `duration_<phase>_ms` | The per-phase breakdown, summed. `duration_walCommit_ms` climbing while `duration_addBatch_ms` is flat means the checkpoint is the bottleneck, not the query. |

The section is empty until a stream completes a batch. That is deliberate: a zero here
would be indistinguishable from a query that has stopped, and every other section is
always-on precisely so a scrape config never has to be conditional.

In the Prometheus rendering each series is labelled by query, so
`batcher_streaming_behind_by_ms{query="ingest"}` is the alert to write first.

## What the engine is holding

`resources` is a different kind of number: a level rather than a total. It carries
Carbonite's own readings, grouped by what they measure.

| Group | What it holds |
|---|---|
| `memory` | The envelope, its soft and hard budgets, live headroom, the buffer pool's usage and high-water mark, and the kernel's own limits. |
| `admission` | Slots, how many queries are running and queued, and how many were shed. |
| `spill` | Bytes and buckets per tier, the high-water mark, the lifetime volume written, and free disk. |
| `result_cache` | Hits, misses, hit rate, evictions, and fill. |

Each reading replaces the previous one for its group, so differencing successive scrapes
gives noise rather than a rate. Read them as gauges. In the Prometheus rendering they are
`batcher_memory_pool_used_bytes`, `batcher_admission_waiting`, `batcher_spill_bytes_written`,
and so on, with an enumerated level such as memory pressure exposed the conventional way as
`batcher_memory_pressure_level{state="NORMAL"} 1`.

The pairing is what makes a diagnosis. A query that spilled with an empty pool and a full
cache was starved by storage; one that spilled with a full pool and no cache was simply too
big.

## Data-quality counters

Every constraint {py:meth}`ds.dq.validate <batcher.api.dataset.dq.DatasetDQ.validate>` evaluates is published to the bus and folded
into `data_quality`, whether it passed or failed. Passing checks are counted too, on
purpose: a series that appears only when something breaks has no baseline, and the useful
question is not "did the contract fail today" but "has this constraint's violation count
been climbing all week".

```python
import batcher as bt
from batcher.observe import metrics_snapshot, start_metrics

start_metrics()
bt.from_pydict({"amount": [10.0, -2.0]}).dq.positive("amount").validate()
dq = metrics_snapshot()["data_quality"]
print(dq["checks_total"] >= 1, dq["violations_total"] >= 1)
# True True
```

`checks_total`, `failed_total`, and `violations_total` are the roll-ups, and
`by_constraint` breaks them down by the constraint's name. Failures and violations are
separate on purpose: one failed check can carry a million violating rows, and one check
inside its `mostly` tolerance carries violations while passing. The breakdown is capped at
256 names, folding the rest into `other`, so a `check()` whose name is built from a row
value cannot grow it without bound; the roll-ups stay exact either way.

The Prometheus rendering exposes the same numbers as `batcher_dq_checks_total`,
`batcher_dq_failed_total`, `batcher_dq_violations_total`, and a labelled
`batcher_dq_constraint_violations_total{constraint="..."}`.

If the dashboard is already running, the same counters are served from it, so a scrape
loop needs no code of yours at all:

```bash
curl -s http://127.0.0.1:4040/metrics        # Prometheus text exposition
curl -s http://127.0.0.1:4040/api/metrics    # the same numbers as JSON
```

The whole read-only API lists itself at `/api`, and every route there is a `GET` over what
the dashboard is showing.

Counters are cumulative from the moment collection starts, the convention every metrics
backend expects, so a scrape loop differences successive snapshots to get rates. Collecting
costs a few integer adds per event.

The first snapshot starts collection, which means it reports only what happened after it.
Call `start_metrics()` once during startup when the first scrape should also cover the
queries that ran before it:

```python
from batcher.observe import start_metrics

start_metrics()
```

Collection is opt-in rather than always-on for a reason. Attaching any sink to the event
bus tells the engine that per-query profiles are being consumed, so it assembles one on
every query. A process that exports no metrics shouldn't pay that on the sub-second path.

`prometheus_text` renders the same numbers in the Prometheus text exposition format. Serve
it from the `/metrics` endpoint your application already has and Batcher joins whatever you
scrape today. Batcher runs no HTTP server for this and pulls in no client library:

```python
from batcher.observe import prometheus_text

print("batcher_queries_total" in prometheus_text())
# True
```

Series are prefixed `batcher_`, counters carry the conventional `_total` suffix, and query
duration is a real histogram with `_bucket`, `_sum`, and `_count` series. OpenTelemetry and
StatsD users can map the same dict onto their own instruments. `reset_metrics` zeroes the
counters, for tests and for a service that would rather report per-interval numbers itself.

## See also

- {doc}`Observability <observability>`: the event bus, the dashboard, and the per-query event log.
- {doc}`Monitoring a stream </user-guide/moving-data/streaming-monitoring>`: the per-query listener behind the `streaming` counters.
- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: the same measurements for one query rather than the process.
- {doc}`Troubleshooting <troubleshooting>`: symptom-first debugging.
