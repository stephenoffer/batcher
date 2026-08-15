# Planning on the layout a table already has

This page describes how Batcher skips a shuffle entirely when the table on disk is already partitioned by the columns a query groups on, why that decision is verified rather than assumed, and what it does not yet cover.

## What a partitioned table already did for you

A partitioned table stores each value's rows apart from the others. A Hive-partitioned Parquet tree does it with a directory per value:

```text
sales/
  day=2026-08-01/part-0.parquet
  day=2026-08-02/part-0.parquet
  day=2026-08-03/part-0.parquet
```

Every row for the 1st is inside one directory. Batcher reads such a table one directory per split, and a split is the unit of assignment: it goes to one worker, whole, never cut in half. So by the time the read finishes, every row for the 1st is on one worker.

That is precisely what a shuffle by `day` would have arranged. A `GROUP BY day` over this table therefore needs no exchange at all. Each worker folds its own directories to final groups, and the driver concatenates them.

A Delta or Iceberg table records the same thing differently, and it splits differently too. Its metadata names one data file per split and records that file's partition values, so a partition of 300 files is 300 splits rather than one. Assigning them individually would scatter a partition across the fleet, so the scheduler groups splits by partition value and assigns whole groups. The fine per-file splits survive inside the group, so nothing is lost from the read; only the *assignment* is coarsened, to exactly the degree the guarantee needs.

Without this, the same query hash-shuffles every row across the network to rediscover a partitioning the storage layout already had.

```python
# docs: skip
import batcher as bt
from batcher import col, count

# day= directories on disk, and a group_by over the partition column
daily = (
    bt.read.parquet("s3://warehouse/sales")
    .group_by("day")
    .agg(revenue=col("amount").sum(), orders=count())
    .collect(distributed=True, num_workers=8)
)
```

Nothing in the query says so. The decision is made by the scheduler from the layout it finds.

## The three conditions

The elimination applies when all three hold. Each one is load-bearing, and the third is a scheduling judgment rather than a correctness one.

Every split declares the same clustering columns. A set where one split names `day` and the next names nothing has no column every row can be located by, and guarantees nothing. This is the half that is *checked*, by `io/splits/clustering.py::declared_clustering`.

Splits sharing a value are assigned together. This is the half that is *established*, by grouping. It is the one that matters and the one no individual split can promise: two splits at `day=2026-08-01` on two different workers would make a query that skipped its shuffle report the day twice.

Every clustering column is a group key. Grouping by a *superset* is fine, because `(day, region)` groups are inside `day` groups, which are inside one directory. Grouping by a column the table is not partitioned by is not: `region` repeats in every directory, so its groups straddle every worker.

The grouping keeps enough of the read's parallelism to be worth the exchange it saves.

This is the only thing the aligned plan gives up, and how much it gives up depends entirely on the reader. A Hive tree already splits one-per-partition, so grouping changes nothing and the aligned plan runs exactly the tasks the shuffle would have. A Delta or Iceberg table splits per data file, so a partition of eight files becomes one assignable unit, and a table with fewer partitions than the fleet has workers ends up running on a fraction of it while the shuffle uses all of it.

So the test is written against the two task counts the plans would actually run: `min(groups, workers)` for the aligned plan, `min(splits, workers)` for the shuffle. Both are capped by the fleet because neither plan can use more workers than exist, which is why the same layout can be aligned on a small fleet and shuffled on a large one. The floors come from measurement rather than taste, and the section below shows what they were set from.

## Why the check is exact and not a declaration

Getting this condition wrong does not make a query slow. It makes it wrong, in a way that is close to undetectable.

A group split across two workers comes back as two rows, each carrying a partial sum, each labelled as a finished group. Single-node execution cannot produce that result, so no single-node test catches it. Any test that does not run a real fleet cannot catch it either. The query returns plausible numbers at PB scale and nobody has a reason to look.

So the guarantee is verified against the split set the read will actually use, rather than declared by the source and trusted. That costs one pass over a list of strings on the driver.

The split set is planned with the same arguments the executor will use, including the same partition count and the same pushed projection and predicate, so what the check inspects is the set the read gets and not a lookalike. It is then checked a second time, inside the executor, against the splits it is about to assign: if they no longer declare what the plan was chosen on, the read **raises** rather than falling back. A fallback would be a wrong answer, because by that point the plan has no combine in it.

Values are compared *typed* rather than as the strings a directory name gives, so `x=01` and `x=1` are not mistaken for two partitions of one value.

## Who decides what

The decision splits across two layers along the line the architecture already draws.

Kyber owns the question "what distribution does this relation already have". `kyber/properties.py::clustered_on` propagates a clustering up through the operators that cannot disturb it: a `Filter`, a `Limit` and a `Distinct` only remove rows, and removing a row never moves another one to a different worker; a `Project` carries the clustering forward under its output names. Anything else leaves it unclaimed, which costs at most a needless shuffle.

`dist` supplies the one part of the answer only it can see, which is what the split set actually guarantees, and then schedules against the result. It does not re-derive the containment rule, so the two cannot drift.

This is the storage-side twin of an elimination Batcher already made. A shuffle join co-partitions both sides by the join key, so an aggregate grouping on a superset of that key skips its shuffle too. Both are the same question asked of `kyber/properties.py::satisfies`, and both are answered by the same containment: the delivered partitioning must be a *subset* of the required grouping.

## What it covers

Aggregation, deduplication and windowing, in both the disk and Flight transports.

A dedup is a group-by that keeps one row per group, so it eliminates the same exchange under the same condition. A whole-row `DISTINCT` groups on every column, which contains the partition columns by definition, so any clustered layout aligns it. A `DISTINCT ON` aligns when its keys cover the clustering.

One shape is excluded outright. A `DISTINCT` carrying a limit would keep `n` rows per partition and concatenate them, which is `n x partitions` rows.

A window computes each partition independently, so co-locating a partition's rows is the only thing its shuffle establishes, and a `ROW_NUMBER() OVER (PARTITION BY day ...)` over a directory-per-day table has that already. The frame and the ordering need no attention because both are *within* a partition, and a `rank_limit` is per-partition too.

`COUNT(DISTINCT)` benefits twice over, and it is the most expensive shape here. It lowers to an aggregate over a `Distinct`, so the shuffle path has to dedup globally before it can count anything. A dedup only ever collapses rows that agree, and rows that agree on the partition columns are already on one worker, so over a clustered relation the per-partition dedup is already the global one.

That is what the chain check is for, and it is also where the clustering property alone would mislead. A `Limit` between the scan and the aggregate does not move a row between workers, so the relation is still clustered by every measure this page has given. But `limit(100).group_by(day)` run per partition keeps a hundred rows on *each* of them, which is a different query. Clustering says where rows are; it does not say that a per-partition computation is complete. Only `Filter`, `Project` and an unlimited `Distinct` are allowed in the chain.

Because nothing is combined across partitions, a **non-mergeable** aggregate is correct here too. `median` and `n_unique` carry per-group partial state that a shuffle has to merge; aligned, each group is finalized where it was read and there is nothing to merge.

## Seeing that it fired

The shuffle path returns exactly the same rows, so nothing about a result tells you which plan ran. The scheduler therefore publishes a decision when it eliminates an exchange, which surfaces in `explain(analyze=True)` and in the live job view alongside Kyber's and Carbonite's:

```text
core / exchange   aggregate needs no shuffle: the table is already partitioned by day
```

If you expect the line and it isn't there, work down the three conditions. The most common reason is the third, and it is worth checking before the others: a lakehouse table with many small files per partition and few partitions relative to the fleet is the shape where shuffling genuinely wins, and the scheduler chose it.

## Measured

On 8,000,000 rows grouped by the partition column on an eight-worker local cluster, against the same query forced through the shuffle. Best of three warm runs each, repeated three times; the spread was under 4% for Parquet and about 15% for Delta.

| Layout | Splits | Aligned | Shuffled | Ratio |
|---|---:|---:|---:|---:|
| Hive Parquet, one directory per partition | 16 | 200 ms | 850 ms | 4.2x |
| Delta, four data files per partition | 64 | 310 ms | 780 ms | 2.3x |
| Hive Parquet, `COUNT(DISTINCT v)` | 16 | 490 ms | 1,020 ms | 2.1x |

Produced by `benchmarks/internals/partition_aligned.py`, which checks the two paths return the same rows before reporting either time. The engine under it was a *debug* build, which slows the local aggregation both paths do while leaving the shuffle's orchestration alone, so a release engine should widen these rather than close them.

Delta's smaller ratio is the read, not the elimination: four files per partition against one, through pyarrow's per-file open rather than one directory scan. The `COUNT(DISTINCT)` row understates its own case, because `v` here holds only a thousand distinct values; the shuffle it removes carries the *deduped* rows, so the gap grows with the column's cardinality.

### Where it stops paying

`--sweep` varies the partition count against a fixed fleet, forcing the aligned path so the losing cases are visible:

| partitions (8 workers) | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Hive tree, one split per partition | 0.95x | 1.33x | 2.35x | 4.44x | 4.10x |
| Delta, four data files per partition | **0.62x** | 1.30x | 2.44x | 2.69x | 2.64x |

Both losses are the column where the aligned plan has **one task**: the whole query runs on one worker while the shuffle spreads the read across the fleet. A Hive tree at one partition is a wash rather than a loss, because its shuffle has one split to work with too and is equally serial.

A ratio cannot see that, which is why the rule is not one. Delta at one partition keeps a *quarter* of the shuffle's parallelism and loses 1.6x; Delta at two partitions keeps the same quarter and wins 1.3x. What separates them is the absolute task count, not the fraction.

So there are two conditions, and the scheduler applies both: at least two tasks unless the shuffle would not have had two either, and at least a quarter of what the shuffle would have run. The first rules out the serial plan; the second keeps a two-partition table off a five-hundred-worker fleet, which a fixed-width sweep cannot reach but arithmetic can.

## Requirements and limitations

A nested `year=/month=` tree is clustered on the **year**, and that is the complete guarantee rather than a partial one. Grouping by `(year, month)` is aligned, because those groups sit inside `year` groups and the containment does the work. Grouping by `month` alone is not, and no split granularity would change that: `month=1` exists under every year, so its rows are spread across every top-level directory, and splitting per leaf directory would only make a split's value `(year, month)` while `month` alone still straddles them.

Hive-partitioned Parquet trees, Delta tables and Iceberg tables declare a clustering. A Parquet path containing a glob falls back to per-file splits that record no partition value, so it guarantees nothing about where equal values live. Any other layout carrying the same guarantee can join in by exposing `clustering_columns` and `clustering_value` on its splits.

Iceberg needs one extra care that the other two don't, because it is the only one whose partitioning can change under an existing table. Its partition spec can evolve, and a file written before the change carries a partition record holding the *old* spec's fields. Reading that against the current spec's columns groups by the wrong thing entirely, so a file whose `spec_id` is not the current one declares no clustering, which makes the whole set refuse rather than be half-trusted.

Iceberg also stores `transform(column)` rather than the column, so what a split declares is the partition field's **source column**. That is the sound claim and the useful one: every transform is a deterministic function of its column, so equal column values always produce equal partition values and land in the same group, which is what lets `GROUP BY ts` over a `days(ts)`-partitioned table skip its shuffle. The tests exercise the identity transform; writing a `bucket`- or `days`-partitioned table needs the `pyiceberg-core` extra, which this repository does not depend on.

Joins do not use it yet. A join whose both sides are partitioned by the join key is co-partitioned on disk and could skip its shuffle on the same argument, but it needs both sides to agree on the layout, which is a stronger condition than either single-input operator has to meet.

## See also

- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: the fan-out, task sizing and skew decisions this one sits beside.
- {doc}`Shuffle over Arrow Flight </architecture/deep-dives/distribution/shuffle-flight>`: what the eliminated exchange would have cost.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why a partitioned result equals the single-node one when a combine *is* needed.
