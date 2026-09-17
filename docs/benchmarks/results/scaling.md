# Scaling out

This page reports how Batcher's results move as the data grows, as cores are added, and as the work spreads across a cluster.

Distribution in Batcher is a scheduling decision rather than a second engine. On one core or a hundred nodes the same mergeable operators run, `partial`, then `combine`, then `finalize`, so a multi-node result holds the same rows, column names and column types as the single-node one.

:::{important}
Every pipeline's result signature is compared across engines before a timing is kept, and a distributed Batcher result is compared against the single-node one. The mergeable algebra is why that check passes: `combine` is associative and commutative, so partials merge in any order. A floating-point reduction is the one tolerated difference. IEEE addition isn't associative, so a different partition count sums in a different order and the two answers agree to the last bits rather than every bit.
:::

:::{note}
The tables below come from different machines, which each section names: a 96-core node, a 128-CPU Ray cluster, and 8xT4 GPU clusters. Compare engines within a table, not across tables.
:::

## With the data: sublinear on nine shapes of thirteen

Growing the data on one machine is the axis a scheduling change can quietly get wrong. The following table runs TPC-H at scale factor 1 and scale factor 10 on the same 96-core, 184 GiB node with the same binary, measured 2026-08-15. Ten times the rows, so ten times the time is the line to beat:

| Query | Shape | sf1 | sf10 | Batcher growth | DuckDB growth |
|---|---|---:|---:|---:|---:|
| q15 | Scan and aggregate | 2.3 ms | 3.3 ms | **1.4x** | 3.8x |
| q22 | Scan and aggregate | 18.9 ms | 43.0 ms | **2.3x** | 3.0x |
| q6 | Scan and filter | 5.8 ms | 30.3 ms | **5.2x** | 3.6x |
| q10 | Join | 28.0 ms | 152.1 ms | **5.4x** | 3.4x |
| q1 | Scan and aggregate | 15.5 ms | 93.6 ms | **6.0x** | 4.7x |
| q8 | Join | 17.7 ms | 105.8 ms | **6.0x** | 3.7x |
| q7 | Join | 20.7 ms | 134.6 ms | **6.5x** | 2.4x |
| q3 | Join | 19.9 ms | 145.3 ms | **7.3x** | 3.8x |
| q21 | Join | 73.1 ms | 540.1 ms | **7.4x** | 4.1x |
| q9 | Join | 47.9 ms | 536.1 ms | 11.2x | 3.0x |
| q18 | Join and aggregate | 31.4 ms | 392.9 ms | 12.5x | 4.2x |
| q13 | Join and aggregate | 40.2 ms | 510.3 ms | 12.7x | 2.8x |
| q5 | Join | 25.3 ms | 376.7 ms | 14.9x | 4.5x |

Nine of thirteen are sublinear. A scan, a filter and most joins cost less than ten times as much for ten times the rows, because at sf1 they don't fill the machine and at sf10 they do. Four are superlinear, and they are named rather than averaged away. q13 and q18 carry very high-cardinality `GROUP BY`s (1.5M and 15M groups at sf10), q9 builds the largest intermediate in the benchmark, and q5 is the six-way join.

DuckDB's column reads 2.4x to 4.7x throughout. That isn't a better scaling law. It is a fixed cost of about 15 ms per query that dominates DuckDB's sf1 times and disappears at sf10.

That sweep put Batcher at 0.78x DuckDB's native store at sf1 and 1.27x at sf10. The changes of 2026-08-25 moved three of the four superlinear queries, q9 from 456 ms to 233 ms, q13 from 325 ms to 174 ms and q5 from 189 ms to 122 ms, and took sf10 to **0.963x**. Ten times the data no longer costs the single-node lead.

## With cores: a gather-bound join saturates near 10x

The honest curve on more cores isn't a straight line. The following ladder is the H2O.ai `join` q5 shape, a 10M by 10M inner join emitting 9M rows and 13 columns, whose cost is dominated by materializing its own output. Worker count is pinned, measured 2026-08-15:

| Threads | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Time | 3,790 ms | 1,885 ms | 1,114 ms | 658 ms | 433 ms | 388 ms | **375 ms** | 442 ms |
| Speedup | 1.0x | 2.0x | 3.4x | 5.8x | 8.8x | 9.8x | **10.1x** | 8.6x |
| Efficiency | 100% | 101% | 85% | 72% | 55% | 31% | 21% | 13% |

It is linear to two cores, 85% efficient at four, and reaches a ceiling near 10x by sixteen, an Amdahl serial fraction of roughly 9%. Past the box's 48 physical cores it gets worse, which is why the executor's default width is every physical core plus a third of the SMT siblings rather than every hardware thread. A query that moves a gigabyte of output is bounded by the part of that work that can't be split, so the way to make it faster is to move less, not to add threads.

## Small data shouldn't distribute

Small inputs don't benefit from distribution, and the benchmark shows it. The following UDF map workload over TPC-H sf1 ran on the live Ray cluster, measured by `benchmarks/scenarios/dist_bench.py`:

| Path | Time |
|---|---:|
| Batcher, single node | 86 ms |
| Batcher, distributed over 4 workers | 92 ms |

On this workload the distributed path is about 7% behind the single-node path, and that is the right answer: the network shuffle and actor startup cost more than they save at this size. The point of the row is that the distributed path works, returns the same result, and costs little when it turns out you didn't need it.

It is also why `distributed="auto"` is size-aware. It used to fan every query out on a multi-node cluster based on topology alone, paying about 2 seconds of Ray dispatch on an 80,000-row filter. It now distributes only when the estimated input reaches `distributed.distribute_min_rows` (1,000,000 by default), or when a GPU stage requires the cluster:

| 80,000-row filter, 8xT4 cluster | Before | After |
|---|---:|---:|
| {py:meth}`collect(distributed="auto") <batcher.Dataset.collect>` | ~2,150 ms | **~67 ms** |

Both paths return the same 48,886 rows, and an explicit `distributed=True` still overrides.

## Cluster against cluster

Both engines attach to the same live Ray cluster, 16 worker nodes of 8 CPUs each (128 CPUs) plus a head node with no CPUs, and read TPC-H Parquet directly from S3, so the distributed read is part of the measured work. Daft runs its Ray runner rather than its local engine. Each pipeline's result signature is compared across engines before a timing is kept (2026-07-12).

Ratios in this table are `daft_ms / batcher_ms`, so **above 1 means Batcher is faster**:

| Pipeline | sf1 | sf10 | sf100 |
|---|---:|---:|---:|
| `scan_count` | **162x** | **208x** | **250x** |
| `join` | **2.23x** | **1.73x** | **1.72x** |
| `groupby` | 1.03x | **1.18x** | **1.30x** |
| `filter_count` | **1.18x** | 0.92x | 0.84x |

Batcher takes the join at every scale, and its group-by lead widens as the data grows. The metadata count is answered without a scan, which is where the rows of two orders of magnitude come from. `filter_count` is the most purely S3-bound pipeline in the grid: scan one column, filter, count. Both engines read the same bytes from the same bucket, so that row measures object-store throughput rather than execution.

:::{dropdown} What the first cluster run found
An earlier round of this benchmark put Batcher behind Daft, and the diagnosis pointed at distributed-scan throughput. That was right about the neighborhood and wrong about the depth. Every real cause was a control-plane or data-movement bug, and all were fixed with no new operator and no tuning knob.

The cluster-fill fan-out was dead. A derived `num_workers` was read as an explicit user override, which suppresses the one-worker-per-node fill, so any query that ran with Ray already initialized used 2 of 16 workers. The fan-out it did compute was sized from the query's output rows, so an sf10 join emitting 5 rows after a `GROUP BY` asked for about 2 workers to process 7.5M input rows.

The map path had its own problems. Any stage containing a UDF went to the single-node orchestrator and ignored `distributed=True`, so the whole batch-inference path ran on 1 of 17 nodes. The distributed map also never pushed a projection into its scan, so a UDF over one column of `lineitem` read all of them from S3 on every task.

The largest single win was a join reducer that sent its whole output back through Python, 3.75M rows or about 106 MB of `RecordBatch` objects, straight back into Rust for the aggregate. A new FFI entry now runs the join and folds the aggregate inside the engine.
:::

## How much of each suite runs distributed

A distributed path that refuses a query shape isn't slow, it's absent, and that is worth reporting apart from any timing. The following counts were measured 2026-08-01 over splittable Parquet on shared storage on a 4-GPU cluster. An in-memory {py:func}`from_arrow <batcher.from_arrow>` source isn't splittable, so the dispatcher runs it on one node and never exercises this path:

| Suite | Shape | Ran end to end |
|---|---|---|
| TPC-H sf1, 22 queries | 4 workers, 16 partitions | **19**, up from 13 |
| ClickBench, 43 queries | 2 workers, 8M-row `hits` mirror | 37 |

Every TPC-H result was compared against DuckDB row by row and in order, not as an unordered multiset. That matters here because one of the two fixes changed how the distributed sort routes rows, and an order-independent comparison can't see a sort bug.

Two causes accounted for the gap. A distributed `ORDER BY` on a string column had no path at all, because the sort routes rows against quantile boundaries from a numeric-only KLL sketch. Four TPC-H queries end in a string `ORDER BY` over an aggregate. Separately, a `date32` comparison against a string literal raised inside the Parquet scanner instead of declining pushdown, which failed six ClickBench queries that run fine on one node. The scanner now types each literal against the file schema and drops only the conjunct it can't push.

## Beyond one GPU's memory

The clearest case for distribution is the one where the alternative doesn't run. The following group-by sum over 1,000 groups ran on 8xT4 with cuDF as the per-GPU data plane:

| Rows | Single-GPU cuDF | Batcher distributed over 8 GPUs |
|---|---:|---:|
| 200M | **1,983M rows/s** | 768M rows/s |
| 600M | OOM | **10,731M rows/s** |
| 1.2B | OOM | **13,358M rows/s** |
| 2.0B | OOM | **10,799M rows/s** |

While the data fits one GPU, single-GPU cuDF is faster, because it pays no cross-device combine. Past one GPU's memory, distribution is the only thing that runs. That is a distribution result rather than a kernel result, and it is why Batcher uses cuDF as the per-GPU data plane rather than reimplementing it.

## Memory stays bounded

Everything above rests on the mergeable algebra. A stateful operator reduces its partition before anything leaves it, so per-node memory depends on the partition, not the whole relation. When a partition still doesn't fit, aggregation, distinct, sort, join build and partitioned windows spill, and the query gets slower rather than failing. A `flat_map` then `count` over 120M rows would materialize 480M rows on one node. Distributed, it runs about 5.8x faster, because each partition reduces before anything leaves it.

## Requirements and limitations

The following limits apply to the distributed results:

- **ClickBench** reads 37 of 43 because the full distributed rerun after the scanner fix hasn't been recorded. The single-node 43 of 43 is a different measurement.
- **TPC-H q15** still fails distributed, with every worker marked dead at the map barrier. Its CTE is referenced by both a join and a scalar subquery.
- **`filter_count`** against Daft at sf10 and sf100 is a loss, at 0.92x and 0.84x, on the shape bound by object-store reads.
- **Shuffle durability** relies on replication. There is no external shuffle service, so a bucket outlives its worker only if a replica does.

## Reproduce

The following commands rerun the distributed measurements. `vs_ray_daft.py` takes one scale factor per run:

```bash
python benchmarks/cluster/vs_ray_daft.py 1
python benchmarks/cluster/vs_ray_daft.py 10
python benchmarks/cluster/vs_ray_daft.py 100
python benchmarks/scenarios/dist_bench.py --workers 4
python benchmarks/scenarios/scaling/ladder.py --rungs 1,2,4
```

## See also

- {doc}`/benchmarks/comparisons/vs-daft`: the single-node half of the Daft comparison.
- {doc}`/benchmarks/comparisons/vs-spark`: the architectural comparison with Spark.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra`: why one core and a hundred nodes give the same answer.
- {doc}`/architecture/deep-dives/distribution/distributed-scheduling`: what `distributed="auto"` decides, and on what.
- {doc}`/architecture/deep-dives/distribution/shuffle-flight` and {doc}`/architecture/deep-dives/distribution/credit-flow-control`: the data movement behind the join results.
- {doc}`/architecture/deep-dives/memory/spilling`: what keeps per-node memory bounded when a partition doesn't fit.
- {doc}`/architecture/fault-tolerance`: how a distributed query survives a lost worker.
- {doc}`/benchmarks/methodology`: the cluster shapes behind each table.
