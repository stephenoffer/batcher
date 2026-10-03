# Reading a very large table

This page covers what changes when a table is large enough that *planning* it costs real time: hundreds of thousands of files, a directory per day going back years, more rows than any single machine will hold. The levers on {doc}`performance` all still apply. The difference is that the work you most want to avoid happens before a single row is read.

Batcher settles three things at plan time: how much of the table it can rule out without opening it, how well it estimates the rest, and how finely it divides the work.

## What "plan time" costs

Every metadata operation the driver performs is `O(files)` and happens before any task launches. On object storage each is a network round trip, so a million-file table can spend twenty minutes listing footers while the cluster sits idle.

Above `BATCHER_MAX_FOOTER_PLAN_FILES` (10,000 by default), Batcher stops reading a footer per file and falls back to methods whose cost doesn't grow with the file count. Every fallback reads *more* data, never less, so results are unchanged.

## Partition pruning happens before the tasks exist

A Hive-partitioned directory tree is the standard layout at this size:

```text
events/
  day=2024-01-01/part-0.parquet
  day=2024-01-02/part-0.parquet
  ...
```

The driver lists only the top-level `day=` directories, and each worker lists only its own subtree. A filter on the partition column prunes that directory list *before* splits are built, so a directory that can't match never becomes a task:

```python
import datetime
import os
import tempfile

import batcher as bt

root = os.path.join(tempfile.mkdtemp(), "events")
bt.from_pydict(
    {"day": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-02"], "user_id": [3, 1, 2, 4]}
).write.parquet(root, partition_by=["day"])

events = bt.read.parquet(root)
print(events.filter(bt.col("day") == datetime.date(2024, 1, 2)).count())
# 2
```

Over a table with a directory per day for ten years, that filter is the difference between 3,650 tasks and one.

The figure traces that plan-time work in order:

![A four-step flow that runs at plan time, before any task exists. Step 1: the driver lists only the top-level day= directories, in one cheap listing. Step 2: the directory list is pruned by a predicate on the partition column, which comes either from a filter you wrote on day or, through dynamic partition pruning, from the smaller join side's key range; pruning is exact because the directory name records the partition value. Step 3: each surviving directory goes to a worker, which lists only its own subtree. Step 4: splits, and so tasks, are built from the survivors only. A directory the predicate rules out is never listed, opened, or turned into a task. With a directory per day for ten years, 3,650 tasks become one with the same rows, and where the layout can't decide, every directory survives and the rows are filtered as usual.](/_static/diagrams/partition_pruning_flow.svg)

Pruning is exact, because the directory name records the partition value. Where the layout can't decide, such as a predicate over a data column, every directory survives and rows are filtered as usual. A date or a string literal both prune:

```python
print(events.filter(bt.col("day") == "2024-01-02").count())
# 2
```

The same machinery prunes a Delta or Iceberg table from its transaction log, which records
each data file's partition values and per-column bounds.

### A join can prune too

When a query joins a partitioned table to a smaller one, the smaller side's key range already says which partitions can match. Batcher turns that into a filter on the partition column before the scan is planned. This is *dynamic partition pruning*:

```python
campaigns = bt.from_pydict({"day": [datetime.date(2024, 1, 3)], "name": ["spring"]})
joined = events.join(campaigns, on="day", how="inner")
print(joined.select("user_id", "name").to_pydict())
# {'user_id': [2], 'name': ['spring']}
print("pushed[day IS NOT NULL AND day = 19725]" in joined.explain())
# True
```

Nobody wrote a `filter`, yet the scan received one. It works in both directions. It needs the join key to *be* the partition column, and the smaller side's range to be genuinely narrower.

```{note}
The bounds are deliberately not treated as exact. A partition directory can outlive its rows:
deleting a day's files leaves `day=...` standing, so the lowest directory name may name a day
that holds nothing. That is harmless for pruning, which may only ever keep too much, but it
means an exact `MIN(day)` still reads data rather than answering from the layout.
```

### Making pruning possible

Partition on a column queries filter on, at a granularity that leaves a useful number of directories. Partitioning by a timestamp to the second makes a directory per row, and Batcher warns when a write is about to do that. For a column too fine to partition on, sort by it and rely on file-level bounds:

```python
manifest = events.write.parquet(
    os.path.join(tempfile.mkdtemp(), "sorted"), partition_by=["day"], sort_by=["user_id"]
)
print(len(manifest.files))
# 3
```

## Estimates at a size that cannot be counted

Above the footer ceiling, Batcher samples instead of counting. It reads 64 footers spread evenly across the file list, measures rows per byte, and scales by the table's on-disk size. That's 64 round trips whether the table has 20,000 files or ten million. Spreading the sample matters: the first 64 files of a date-partitioned listing are all the same day.

`meta.source_stats()` says which kind of count you hold:

```python
import pyarrow as pa

table = bt.from_arrow(pa.table({"user": ["a", "b"], "v": [1, 2]}))
stats = table.meta.source_stats()[0]
print(stats.row_count, stats.exact_rows)
# 2 True
```

Above the ceiling the same call reports an estimated `row_count` with `exact_rows=False`. The estimate sizes plans. An exact `count()` still reads the data.

## Dividing the work

A shuffle divides its input twice: into *map partitions*, which are the unit of scheduling
and of recovery, and into *hash buckets*, which are the unit a reducer holds at once.

The bucket count decides whether a large query fits in memory, because a join, sort or window holds one bucket at a time. Batcher sizes buckets from the volume being exchanged: measured for a shape that has run before, estimated from source statistics on a first run, and never below one bucket per worker. Any bucket count returns the same rows.

### Why more buckets don't flood the scheduler

Batcher launches buckets within a submit-ahead window and refills a slot as each reduce finishes, so the scheduler never holds a queue of tasks that can't start. The same bound applies to an aggregate's combiner tree. Two settings control the depth:

| Setting | Meaning |
|---|---|
| `distributed.max_pending_tasks` | A hard cap on outstanding tasks. `0` (the default) derives one instead. |
| `distributed.pending_window_factor` | When no cap is set, the window is this multiple of the tasks the cluster's schedulable cores can run at once. Default `4`. |

A stage smaller than the window submits everything before its first wait, so ordinary queries
are unaffected.

```{note}
More buckets do not fix skew. A hash bucket is the unit a key cannot be split below, so a
single dominant key stays on one reducer however fine the hash. Splitting one key across
reducers is salting, which Batcher applies from measured hot keys. See
{doc}`skew` and `distributed.skew_join_salt` in {doc}`/configuration/distributed-options`.
```

## Requirements and limitations

- The plan-time ceiling is a file count, not a byte count. A table of 500 very large files
  reads every footer; one of 50,000 small files does not.
- The sampled row count assumes rows per byte is roughly stable across files. A bad estimate costs a worse plan, never a wrong answer.
- Partition pruning needs the predicate to reach the scan. `explain()` shows where the filter ended up.
- Directory-level pruning applies to the top-level partition column. Deeper levels are pruned
  by the worker that lists the subtree, not on the driver.

## See also

- {doc}`performance`: the levers that apply at every size.
- {doc}`explain-plans`: confirming where a filter was pushed to.
- {doc}`/user-guide/moving-data/writing-data`: choosing a partition layout on the way in.
- {doc}`/user-guide/operate/running/unstable-nodes`: what happens when a query this long
  loses a worker.
