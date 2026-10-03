# Planning on the layout a table already has

This page describes how Batcher skips a shuffle entirely when the table on disk is already partitioned by the columns a query groups on, why that decision is verified rather than assumed, and which shapes it covers.

## What does a partitioned table already do?

A partitioned table stores each value's rows apart from the others. A Hive-partitioned Parquet tree uses a directory per value:

```python
import os
import batcher as bt

ds = bt.from_pydict({"day": [1, 1, 2, 3], "amount": [10.0, 5.0, 7.0, 3.0]})
ds.write.parquet("sales", partition_by=["day"])
print(sorted(d for d in os.listdir("sales") if d.startswith("day=")))  # ['day=1', 'day=2', 'day=3']
```

Batcher reads such a table one directory per split, and a split goes to one worker, whole. So by the time the read finishes, every row for a day is on one worker, which is precisely what a shuffle by `day` would have arranged. A `GROUP BY day` needs no exchange: each worker folds its own directories to final groups, and the driver concatenates them.

```python
daily = bt.read.parquet_dataset("sales").group_by("day").agg(revenue=bt.col("amount").sum())
print(daily.sort("day").to_pydict())  # {'day': [1, 2, 3], 'revenue': [15.0, 7.0, 3.0]}
```

Nothing in the query asks for this. On a cluster, the scheduler decides from the layout it finds:

```python
# docs: skip
daily.collect(distributed=True, num_workers=8)  # no shuffle: the table is already partitioned by day
```

A Delta or Iceberg table records partition values in its metadata and splits per data file, so a partition of 300 files is 300 splits. The scheduler groups splits by partition value and assigns whole groups, keeping the fine per-file splits inside each group for the read.

## The conditions

The elimination applies when three conditions hold. The first two are correctness conditions, and the third is a scheduling judgment.

- **The layout guarantees co-location.** Every split declares the same clustering columns, *checked* by `io/splits/clustering.py::declared_clustering`, and splits sharing a value are assigned together, *established* by `group_by_clustering`.
- **Every clustering column is a group key.** Grouping by `(day, region)` is fine, because those groups sit inside `day` groups. Grouping by `region` alone is not, because it repeats in every directory.
- **Enough parallelism survives.** The aligned plan runs `min(groups, workers)` tasks and the shuffle `min(splits, workers)`. The aligned plan needs at least two tasks, unless the shuffle would not have had two either, and at least a quarter of the shuffle's count.

![The tests that let a read replace a shuffle, and what each one costs when it is wrong. A directory per value, assigned whole, puts every row for a value on one worker, which is exactly what a shuffle by that column arranges. Four tests then decide whether the exchange may go. Does every split declare the same clustering columns, or none do, checked by declared_clustering. Are a value's splits assigned together as one unit, established by group_by_clustering, which no individual split can promise. Do the group keys contain the clustering columns, the containment held by properties.satisfies. And are enough tasks left to be worth it, at least two and at least a quarter of the shuffle's count, read off scan_clustering_for. The first three are correctness: a group split across two workers returns two partial sums, each labelled final. The fourth is a judgment about speed. Any one test failing falls back to the hash shuffle, where every row crosses the network and the answer is never wrong; passing all four runs with no exchange, each worker folding its own directories, and publishes the decision to explain(analyze=True) as a core / exchange entry. Three things unclaim the layout: a glob path, whose per-file splits record no partition value; grouping below the split, since month= sits under every year= directory; and a Limit in the chain, because clustering places rows and does not finish them.](/_static/diagrams/partition_aware_planning.svg)

## Verified, not declared

A group split across two workers would come back as two rows, each labelled finished, so the guarantee is verified against the split set the read will actually use, planned with the executor's own partition count, projection and predicate. The executor checks it a second time against the splits it is about to assign, and **raises** if they no longer declare what the plan was chosen on, since the plan by then has no combine in it. Values are compared typed, so `x=01` and `x=1` are one partition.

Kyber owns the question "what distribution does this relation already have". `kyber/properties.py::clustered_on` propagates a clustering through `Filter`, `Limit`, `Distinct` and `Project`, which cannot move a row between workers. `dist` supplies only what the split set guarantees. Both ask `kyber/properties.py::satisfies`, the same containment test that lets an aggregate on a superset of a join key skip its shuffle after a co-partitioned join.

## What it covers

Aggregation, deduplication and windowing, on both the disk and Flight transports:

- **`group_by` / `agg`**, including **non-mergeable** aggregates such as `median`, because each group is finalized where it was read.
- **`DISTINCT`**, which groups on every column and so always contains the partition columns, and **`DISTINCT ON`** when its keys cover the clustering. A `DISTINCT` with a limit is excluded.
- **Windows** partitioned by the clustering, such as `ROW_NUMBER() OVER (PARTITION BY day ...)`. Frames, ordering and `rank_limit` are all within a partition.
- **`COUNT(DISTINCT)`**, which lowers to an aggregate over a `Distinct`: over a clustered relation the per-partition dedup is already the global one.

Only `Filter`, `Project` and an unlimited `Distinct` may sit between the scan and the operator. A `Limit` keeps rows in place but makes a per-partition computation a different query.

## Seeing that it fired

The shuffle path returns the same rows, so the scheduler publishes a decision that surfaces in `explain(analyze=True)` and the live job view:

```text
core / exchange   aggregate needs no shuffle: the table is already partitioned by day
```

If the line is missing, check the parallelism condition first. A lakehouse table with many small files per partition and few partitions relative to the fleet is the shape where shuffling wins, and the scheduler chose it.

## Measured

On 8,000,000 rows grouped by the partition column on an eight-worker local cluster, against the same query forced through the shuffle, best of three warm runs each:

| Layout | Splits | Aligned | Shuffled | Ratio |
|---|---:|---:|---:|---:|
| Hive Parquet, one directory per partition | 16 | 200 ms | 850 ms | 4.2x |
| Delta, four data files per partition | 64 | 310 ms | 780 ms | 2.3x |
| Hive Parquet, `COUNT(DISTINCT v)` | 16 | 490 ms | 1,020 ms | 2.1x |

Produced by [`benchmarks/internals/partition_aligned.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/internals/partition_aligned.py), which checks both paths return the same rows before reporting either time, on a debug engine build. Its `--sweep` mode varies the partition count against a fixed fleet, which is where the parallelism floors were set from.

## Practical limits

- **Nested trees.** A `year=/month=` tree is clustered on `year`. Grouping by `(year, month)` aligns; grouping by `month` alone can't, because `month=1` exists under every year.
- **Supported layouts.** Hive-partitioned Parquet trees read with `bt.read.parquet_dataset`, Delta tables and Iceberg tables. A glob path such as `sales/day=*/*.parquet` doesn't recover partition columns, and Batcher raises a `DataWarning` saying so. Another layout can join in by exposing `clustering_columns` and `clustering_value` on its splits.
- **Iceberg spec evolution.** The scan clusters on the fields common to every spec its files were written under, matched on source column and transform (`IcebergSource._common_clustering`), so a table that evolved from `day` to `(day, region)` still clusters on `day`. A split declares a partition field's source column, which lets `GROUP BY ts` over a `days(ts)`-partitioned table align. Transformed fields are tested against in-memory specs in [`tests/unit/test_iceberg_transformed_clustering.py`](https://github.com/stephenoffer/batcher/blob/main/tests/unit/test_iceberg_transformed_clustering.py).
- **Joins.** Partition-aligned planning covers single-input operators; joins still shuffle.

## See also

- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: the fan-out, task sizing and skew decisions this one sits beside.
- {doc}`Shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: what the eliminated exchange would have cost.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why a partitioned result equals the single-node one when a combine *is* needed.
- {doc}`Physical properties </architecture/deep-dives/query/physical-properties>`: the partitioning and ordering properties `satisfies` compares.
