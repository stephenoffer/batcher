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
worker node × 8 CPU cluster, and an 8×T4 GPU cluster. Compare engines *within* a table.
[Methodology](methodology.md) lists the shapes.
:::

## Small data should not distribute

At TPC-H scale factor 1 (6M rows), distributing a query is a mistake, and the honest
benchmark shows it. On a 9-node, 128-CPU Ray cluster:

| Engine | Time |
|---|---:|
| Batcher, single node | 86 ms |
| Batcher, distributed (4 workers) | 92 ms |
| Ray Data (attached to the cluster) | 4,284 ms |

Batcher's distributed path is ~7% behind its own single-node path here, which is the right
answer: the network shuffle and actor startup cost more than they save at this size. The
point of the row is that the distributed path works, is correct, and does not cost much to
take. Even distributed-against-distributed it is **46× Ray Data**.

This is also why `distributed="auto"` is size-aware. It used to fan every query out on a
multi-node cluster based on topology alone, paying a ~2 s Ray dispatch on an 80k-row filter.
It now distributes only when the estimated input is large enough (or a GPU stage forces it):

| 80k-row filter, 8×T4 cluster | Before | After |
|---|---:|---:|
| `collect(distributed="auto")` | ~2,150 ms | **~67 ms** |

Same 48,886 rows out. An explicit `distributed=True` still overrides.

## All three engines on the same cluster

16 worker nodes × 8 CPUs (128 CPUs) plus a 0-CPU head. Every engine attaches to the *same*
live Ray cluster and reads TPC-H parquet directly from S3, so the distributed read is part
of the measured work. Daft runs its Ray runner (flotilla), not its local engine. Each
pipeline's result signature is compared across engines before any timing is kept.

Ratios are `engine_ms / batcher_ms`, so **above 1 means Batcher is faster**.

| Pipeline | sf1 (Ray / Daft) | sf10 (Ray / Daft) | sf100 (Ray / Daft) |
|---|---:|---:|---:|
| `scan_count` | **4944×** / **162×** | **5526×** / **208×** | **7831×** / **250×** |
| `filter_count` | **16.6×** / 1.18× | **7.7×** / 0.92× | **2.9×** / 0.84× |
| `groupby` | **33.9×** / 1.03× | **21.3×** / 1.18× | **6.6×** / 1.30× |
| `join` | **33.0×** / **2.23×** | **16.6×** / **1.73×** | (Ray OOM/err) / **1.72×** |
| `udf` (`map_batches`) | **5.6×** / n/a | **1.7×** / n/a | **2.2×** / n/a |

Batcher beats Ray Data on every pipeline at every scale.

:::{warning}
Against Daft the result is mixed. Batcher wins the join, the group-by and the metadata
count, and **loses `filter_count` at sf10 and sf100** (0.84–0.92×), the most purely S3-bound
shape there is. Both engines are reading the same bytes from the same bucket there, and the
difference is object-store read throughput, not execution. The 10× bar Batcher clears
against Ray Data is **not attainable against Daft on these shapes**, and pretending
otherwise would be dishonest. On an I/O-bound scan, no engine can be 10× another that is
already at a similar fraction of the network's line rate.
:::

## What was actually broken

The first version of this benchmark had Batcher ~10× behind Daft at sf100 and concluded the
problem was distributed-scan throughput. That conclusion was right about the neighborhood and
wrong about the depth. Every real cause turned out to be a control-plane or data-movement
bug: five of them, with no new operator and no tuning knob.

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

The last one is the most embarrassing. A join reducer sent its whole output back through
Python: 3.75M rows, ~106 MB of Python `RecordBatch` objects, straight back into Rust for the
aggregate. A new FFI entry now runs the join and folds the aggregate inside the engine.
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

- [vs Ray Data](vs-ray-data.md) and [vs Daft](vs-daft.md): the single-node halves.
- [vs Spark](vs-spark.md): the architectural comparison these measurements stand in for.
- [Mergeable algebra](../deep-dives/mergeable-algebra.md): why one core and a hundred nodes
  give the same answer.
- [Distributed scheduling](../deep-dives/distributed-scheduling.md): what `distributed="auto"`
  is deciding, and on what.
- [Shuffle over Flight](../deep-dives/shuffle-flight.md) and
  [credit flow control](../deep-dives/credit-flow-control.md): the data movement the page
  names as the real lever.
- [Spilling](../deep-dives/spilling.md): what keeps per-node memory bounded when a partition
  does not fit.
- [Fault tolerance](../architecture/fault-tolerance.md): how a distributed query survives
  a lost worker.
- [Methodology](methodology.md): the cluster shapes; rows measured on different hardware are
  not comparable.
