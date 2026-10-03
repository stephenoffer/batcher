# Scaling out

This page reports how Batcher's results move as the data grows, as cores are added, and as the work spreads across a cluster.

Distribution in Batcher is a scheduling decision rather than a second engine. On one core or a hundred nodes the same mergeable operators run, `partial`, then `combine`, then `finalize`, so a multi-node result holds the same rows, column names and column types as the single-node one.

Every distributed result is compared against the single-node one before a timing is kept. Floating-point reductions agree to the last bits rather than every bit, because a different partition count sums in a different order. Each table names its machine.

## With the data

The following table runs TPC-H at scale factor 1 and scale factor 10 on the same 96-core, 184 GiB node with the same binary, measured 2026-08-15. Ten times the rows, so ten times the time is the line to beat:

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

Nine of thirteen are sublinear: at sf1 they don't fill the machine and at sf10 they do. The changes of 2026-08-25 then took q9 from 456 ms to 233 ms, q13 from 325 ms to 174 ms and q5 from 189 ms to 122 ms, and put the sf10 suite at **0.963x** DuckDB's native store.

## With cores

The following ladder is the H2O.ai `join` q5 shape, a 10M by 10M inner join emitting 9M rows and 13 columns, whose cost is dominated by materializing its own output. Worker count is pinned, measured 2026-08-15:

| Threads | 1 | 2 | 4 | 8 | 16 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Time | 3,790 ms | 1,885 ms | 1,114 ms | 658 ms | 433 ms | 388 ms | **375 ms** | 442 ms |
| Speedup | 1.0x | 2.0x | 3.4x | 5.8x | 8.8x | 9.8x | **10.1x** | 8.6x |
| Efficiency | 100% | 101% | 85% | 72% | 55% | 31% | 21% | 13% |

It is linear to two cores and reaches about 10x at 48 threads. The executor's default width is every physical core plus a third of the SMT siblings rather than every hardware thread.

## Distributing only when it pays

`distributed="auto"` is size-aware. It distributes when the estimated input reaches `distributed.distribute_min_rows`, or when a GPU stage requires the cluster, and otherwise runs on the driver. On an 80,000-row filter on an 8xT4 cluster:

| 80,000-row filter, 8xT4 cluster | Fan out on topology | Size-aware |
|---|---:|---:|
| {py:meth}`collect(distributed="auto") <batcher.Dataset.collect>` | ~2,150 ms | **~67 ms** |

Both return the same 48,886 rows. You can read the threshold and pin either path explicitly:

```python
import batcher as bt

print(bt.active_config().distributed.distribute_min_rows)
# 20000000

q = bt.from_pydict({"k": [1, 2, 1], "v": [10, 20, 30]}).group_by("k").agg(s=bt.col("v").sum())
print(q.sort("k").collect(distributed=False).to_pydict())
# {'k': [1, 2], 's': [40, 20]}
```

On a Ray cluster, `collect(distributed=True, num_workers=4)` runs the same plan over four workers and returns the same rows.

## Cluster against cluster

Both engines attach to the same live Ray cluster, 16 worker nodes of 8 CPUs each (128 CPUs) plus a head node with no CPUs, and read TPC-H Parquet directly from S3, so the distributed read is part of the measured work. Daft runs its Ray runner rather than its local engine. Each pipeline's result signature is compared across engines before a timing is kept (2026-07-12).

Ratios in this table are `daft_ms / batcher_ms`, so **above 1 means Batcher is faster**:

| Pipeline | sf1 | sf10 | sf100 |
|---|---:|---:|---:|
| `scan_count` | **162x** | **208x** | **250x** |
| `join` | **2.23x** | **1.73x** | **1.72x** |
| `groupby` | 1.03x | **1.18x** | **1.30x** |
| `filter_count` | **1.18x** | 0.92x | 0.84x |

Batcher takes the join at every scale, and its group-by lead widens as the data grows. The count is answered from metadata without a scan.

## Suite coverage on a cluster

Measured 2026-08-01 over splittable Parquet on shared storage on a 4-GPU cluster. Every TPC-H result was compared against DuckDB row by row and in order:

| Suite | Shape | Ran end to end |
|---|---|---|
| TPC-H sf1, 22 queries | 4 workers, 16 partitions | **19** |
| ClickBench, 43 queries | 2 workers, 8M-row `hits` mirror | 37 |

## Beyond one GPU's memory

The clearest case for distribution is the one where the alternative doesn't run. The following group-by sum over 1,000 groups ran on 8xT4 with cuDF as the per-GPU data plane:

| Rows | Single-GPU cuDF | Batcher distributed over 8 GPUs |
|---|---:|---:|
| 200M | **1,983M rows/s** | 768M rows/s |
| 600M | OOM | **10,731M rows/s** |
| 1.2B | OOM | **13,358M rows/s** |
| 2.0B | OOM | **10,799M rows/s** |

Past one GPU's memory, distribution keeps the query running at over 10 billion rows/s. Batcher uses cuDF as the per-GPU data plane rather than reimplementing it.

## Memory stays bounded

A stateful operator reduces its partition before anything leaves it, so per-node memory depends on the partition, not the whole relation. When a partition still doesn't fit, aggregation, distinct, sort, join build and partitioned windows spill. A `flat_map` then `count` over 120M rows, which would materialize 480M rows on one node, runs about 5.8x faster distributed.

## Reproduce

`vs_ray_daft.py` takes one scale factor per run.

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
