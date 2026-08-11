# Scaling out

Distribution in Batcher is a scheduling decision, not a second engine. One core or a
hundred, the same mergeable operators (`partial → combine → finalize`) run, and a
multi-node result is bit-identical to the single-node one. This page is what that buys.

:::{important}
Every pipeline's result signature is compared across engines before any timing is kept, and
a distributed Batcher result is compared against the single-node one. The mergeable algebra
is what makes that check pass rather than a coincidence: `combine` is associative and
commutative, so partials merge in any order and a multi-node answer is bit-identical to a
one-core answer.
:::

:::{note}
The tables below come from three different clusters: a 9-node 128-CPU Ray cluster, a 16
worker node x 8 CPU cluster, and an 8xT4 GPU cluster. Compare engines *within* a table.
{doc}`/benchmarks/methodology` lists the shapes.
:::

## Small data should not distribute

At TPC-H scale factor 1 (6M rows), distributing a query is a mistake, and the benchmark
shows it. On a 9-node, 128-CPU Ray cluster:

| Path | Time |
|---|---:|
| Batcher, single node | 86 ms |
| Batcher, distributed (4 workers) | 92 ms |

Batcher's distributed path is about 7% behind its own single-node path here, which is the
right answer: the network shuffle and actor startup cost more than they save at this size.
The point of the row is that the distributed path works, is correct, and costs almost
nothing to take when it turns out you did not need it.

This is also why `distributed="auto"` is size-aware. It used to fan every query out on a
multi-node cluster based on topology alone, paying a ~2 s Ray dispatch on an 80k-row filter.
It now distributes only when the estimated input is large enough (or a GPU stage forces it):

| 80k-row filter, 8×T4 cluster | Before | After |
|---|---:|---:|
| {py:meth}`collect(distributed="auto") <batcher.Dataset.collect>` | ~2,150 ms | **~67 ms** |

Same 48,886 rows out. An explicit `distributed=True` still overrides.

## Cluster against cluster

16 worker nodes x 8 CPUs (128 CPUs) plus a 0-CPU head. Both engines attach to the *same*
live Ray cluster and read TPC-H parquet directly from S3, so the distributed read is part
of the measured work. Daft runs its Ray runner (flotilla), not its local engine. Each
pipeline's result signature is compared across engines before any timing is kept.

Ratios are `daft_ms / batcher_ms`, so **above 1 means Batcher is faster**.

| Pipeline | sf1 | sf10 | sf100 |
|---|---:|---:|---:|
| `scan_count` | **162x** | **208x** | **250x** |
| `join` | **2.23x** | **1.73x** | **1.72x** |
| `groupby` | 1.03x | **1.18x** | **1.30x** |
| `filter_count` | **1.18x** | 0.92x | 0.84x |

Batcher takes the join at every scale, and the group-by lead widens as the data grows. The
metadata count is answered without a scan at all, which is where the three-order-of-magnitude
rows come from.

:::{note}
`filter_count` is the most purely S3-bound pipeline in the grid: both engines read the same
bytes from the same bucket, so that row measures object-store read throughput rather than
execution. On an I/O-bound scan, neither engine can pull far ahead of the network's line rate.
:::

## How much of the suite runs distributed

A distributed path that refuses a query shape is not slower, it is absent, and that is worth
reporting separately from any timing. Measured 2026-08-01 over splittable Parquet on shared
storage, which is the configuration an in-memory source never exercises, because a
{py:func}`from_arrow <batcher.from_arrow>` source is not splittable and the dispatcher runs it on one node:

| Suite | Shape | Ran end to end |
|---|---|---|
| TPC-H sf1, 22 queries | 4 workers, 16 partitions | 13 → **19** |
| ClickBench, 43 queries | 2 workers, 8M-row `hits` mirror | 37 |

Every TPC-H result was compared against DuckDB **row by row, in order**, not as an unordered
multiset. That distinction matters here more than usual, because one of the two fixes is a
change to how the distributed sort routes rows, and an order-independent comparison cannot
see a sort bug.

Two causes accounted for the gap. A distributed `ORDER BY` on a string column had no path at
all: the sort routes rows against sampled quantile boundaries compared as `f64`, and a KLL
sketch is numeric-only, so the dispatcher refused the shape. Refusing is harmless only while
the refusal can fall back to one node, and once an earlier stage leaves its result on the
workers every source is splittable and the fallback is withdrawn. Four queries end in a
string `ORDER BY` over a materialized aggregate. Separately, `WHERE EventDate >= '2013-07-01'`
against a `date32` column is a comparison Arrow does not implement, and the Parquet scanner
raised rather than declining it, which killed six ClickBench queries that run fine
single-node. The scanner now types each literal against the fragment schema and drops only
the conjunct it cannot push.

:::{warning}
The ClickBench row reads 37, not 43. The six failures' cause is fixed and verified at the
scanner against a real shard, but the full 43-query distributed re-run has not completed, so
no distributed 43/43 is claimed here. The single-node 43/43 on {doc}`/benchmarks/results/analytics` is a
different measurement. TPC-H q15 also still fails distributed, with every worker marked dead
at the map barrier; its CTE is referenced by both a join and a scalar subquery, and the
suspected cause is a materialized intermediate outliving the fleet that holds it.
:::

## What the first cluster run found

The first version of this benchmark pointed at distributed-scan throughput. That was right
about the neighborhood and wrong about the depth. Every real cause turned out to be a
control-plane or data-movement bug: five of them, closed with no new operator and no tuning
knob.

:::{dropdown} The five bugs, in full
The cluster-fill fan-out was dead. A derived `num_workers` was being read as an explicit
user override, which suppresses the one-worker-per-node fill, so any query that ran with Ray
already initialized used **2 of 16 workers**. Worse, the fan-out it did compute was sized
from the query's *output* rows: an sf10 join emitting 5 rows after a `GROUP BY` asked for
~2 workers to chew through 7.5M input rows.

Then there was the map path. Any stage containing a UDF went to the *single-node*
orchestrator, ignoring `distributed=True`, and since adaptive execution is on by default the
whole batch-inference path ran on **1 of 17 nodes**. The distributed map also never pushed a
projection into its scan, so a UDF over one column of `lineitem` pulled all 17 from S3 on
every task.

The last one was the largest single win. A join reducer sent its whole output back through
Python: 3.75M rows, about 106 MB of Python `RecordBatch` objects, straight back into Rust for
the aggregate. A new FFI entry now runs the join and folds the aggregate inside the engine.
:::

## Thread-pool sizing on a Ray worker

The reducer combine and the map-side shuffle on a Ray worker were running on the
global rayon pool, which is **1 thread** on a Ray actor (it is built before the actor's cgroup
affinity lands). Wrapping them in the worker's width-sized pool means the reduce and shuffle
compute now spread across every core the worker owns.

A/B on the live 8-worker cluster, sf10 high-cardinality distributed group-by (60M rows →
15M groups), best of 3:

| Reduce/shuffle pool | Time |
|---|---:|
| Global pool | 1,540 ms |
| Worker pool | 1,605 ms |

:::{tip}
**The two are equivalent, and that is the useful finding.** A distributed group-by at this
shape is bound by network and I/O rather than per-worker compute, so widening the reducer's
compute pool has nothing to move. The change is kept for correctness and consistency, not
claimed as a speedup.

The lever that does move a distributed group-by is data movement: coalesced range reads and
a faster shuffle. If you are tuning your own cluster, profile the transfer before you widen
a thread pool.
:::

## Beyond one GPU's memory

The clearest case for distribution is the one where the alternative does not run at all. A
group-by sum on 8×T4, cuDF as the per-GPU data plane:

| Rows | Single-GPU cuDF | Batcher distributed over 8 GPUs |
|---|---:|---:|
| 200M (fits one GPU) | **1,983 M rows/s** | 768 M rows/s |
| 600M | **OOM** | 10,731 M rows/s |
| 1.2B | **OOM** | 13,358 M rows/s |
| 2.0B | **OOM** | 10,799 M rows/s |

Below one GPU's memory, single-GPU cuDF wins, because the cross-device combine is not free.
Above it, distribution is the only thing that runs. That is a distribution win over a compute win,
and it is exactly why a data engine should integrate a GPU dataframe rather than reimplement
one.

## Memory stays bounded

Everything above rests on the mergeable algebra. A stateful operator reduces its partition
*before* anything leaves it, so per-node memory is a function of the partition, not the
relation. When the partition still does not fit, aggregation, distinct, sort, join build and
partitioned windows all spill, and a skewed bucket is recursively re-partitioned. The query
gets slower. It does not die.

A `flat_map → count` over 120M rows would materialize 480M rows on one node. Distributed, it
runs about **5.8× faster**, because each partition reduces before anything leaves it.

## Reproduce

```bash
python benchmarks/cluster/vs_ray_daft.py 10     # sf1 / sf10 / sf100, all three engines
python benchmarks/scenarios/dist_bench.py --workers 4
python benchmarks/scenarios/scale_bench.py
```

## See also

- {doc}`/benchmarks/comparisons/vs-daft`: the single-node half.
- {doc}`/benchmarks/comparisons/vs-spark`: the architectural comparison these measurements stand in for.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra`: why one core and a hundred nodes
  give the same answer.
- {doc}`/architecture/deep-dives/distribution/distributed-scheduling`: what `distributed="auto"`
  is deciding, and on what.
- {doc}`/architecture/deep-dives/distribution/shuffle-flight` and
  {doc}`/architecture/deep-dives/distribution/credit-flow-control`: the data movement the page
  names as the real lever.
- {doc}`/architecture/deep-dives/memory/spilling`: what keeps per-node memory bounded when a partition
  does not fit.
- {doc}`/architecture/fault-tolerance`: how a distributed query survives
  a lost worker.
- {doc}`/benchmarks/methodology`: the cluster shapes; rows measured on different hardware are
  not comparable.
