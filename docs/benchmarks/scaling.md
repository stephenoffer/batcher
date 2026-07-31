# Scaling out

Distribution in Batcher is a scheduling decision, not a second engine. One core or a
hundred, the same mergeable operators (`partial → combine → finalize`) run, and a
multi-node result is bit-identical to the single-node one. This page is what that buys, and
what it does not.

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
{doc}`methodology` lists the shapes.
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
| `collect(distributed="auto")` | ~2,150 ms | **~67 ms** |

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
`filter_count` at sf10 and sf100 is the one shape that goes the other way, by 8% to 16%. It is
the most purely S3-bound pipeline in the grid: both engines read the same bytes from the same
bucket, and the difference is object-store read throughput rather than execution. On an
I/O-bound scan, neither engine can pull far ahead of the network's line rate.
:::

## What the first cluster run found

The first version of this benchmark had Batcher well behind Daft at sf100 and pointed at
distributed-scan throughput. That was right about the neighborhood and wrong about the depth.
Every real cause turned out to be a control-plane or data-movement bug: five of them, closed
with no new operator and no tuning knob.

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

## A measured negative result

Not every fix helps, and publishing the ones that do not is how the numbers stay worth
reading. The reducer combine and the map-side shuffle on a Ray worker were running on the
global rayon pool, which is **1 thread** on a Ray actor (it is built before the actor's cgroup
affinity lands). Wrapping them in the worker's width-sized pool means the reduce and shuffle
compute now spread across every core the worker owns.

A/B on the live 8-worker cluster, sf10 high-cardinality distributed group-by (60M rows →
15M groups), best of 3:

| Reduce/shuffle pool | Time |
|---|---:|
| Worker pool (fixed) | 1,605 ms |
| Global pool (pre-fix) | 1,540 ms |

:::{tip}
**No measurable difference.** The distributed group-by is network- and I/O-bound, not
per-worker-compute-bound, so parallelizing the reducer's *compute* moves nothing. The fix was
kept as a correctness and consistency improvement and is **not** claimed as a speedup. The
genuine distributed lever is data movement: coalesced range reads and a faster shuffle, which
is a deeper effort than wrapping a pool. Measure before you attribute; the obvious
explanation was wrong here and it will be wrong on your cluster too.
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

- {doc}`vs-daft`: the single-node half.
- {doc}`vs-spark`: the architectural comparison these measurements stand in for.
- {doc}`/deep-dives/operators/mergeable-algebra`: why one core and a hundred nodes
  give the same answer.
- {doc}`/deep-dives/distribution/distributed-scheduling`: what `distributed="auto"`
  is deciding, and on what.
- {doc}`/deep-dives/distribution/shuffle-flight` and
  {doc}`/deep-dives/distribution/credit-flow-control`: the data movement the page
  names as the real lever.
- {doc}`/deep-dives/memory/spilling`: what keeps per-node memory bounded when a partition
  does not fit.
- {doc}`../architecture/fault-tolerance`: how a distributed query survives
  a lost worker.
- {doc}`methodology`: the cluster shapes; rows measured on different hardware are
  not comparable.
