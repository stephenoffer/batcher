# Reading a very large table

This page covers what changes when a table is large enough that *planning* it costs real
time: hundreds of thousands of files, a directory per day going back years, more rows than
any single machine will hold. The levers on {doc}`performance` all still apply. What is
different here is that the work you most want to avoid happens before a single row is read.

Three things decide how a table of this size behaves, and all three are settled at plan
time: how much of it the engine can rule out without opening it, how well it can estimate
what is left, and how finely it divides the work that survives.

## What "plan time" costs

Every metadata operation the driver performs is `O(files)` and happens before any task
launches. On object storage each one is a network round trip, so a million-file table can
spend twenty minutes listing and reading footers while the whole cluster sits idle.

Batcher refuses that sweep past a ceiling. Above `BATCHER_MAX_FOOTER_PLAN_FILES` (10,000 by
default) it stops reading a footer per file for split planning, exact row counts, and column
bounds, and falls back to something whose cost does not grow with the file count.

Nothing about this is a correctness question. Every fallback reads *more* data, never less,
and the engine's own filter re-checks every row.

## Partition pruning happens before the tasks exist

A Hive-partitioned directory tree is the standard layout at this size:

```text
events/
  day=2024-01-01/part-0.parquet
  day=2024-01-02/part-0.parquet
  ...
```

Batcher enumerates only the top-level `day=` directories on the driver — one cheap,
non-recursive listing — and hands each one to a worker, which lists only its own subtree. So
the listing cost is `O(subtree)` per worker rather than `O(whole table)` on the driver.

A filter on the partition column is then applied to that directory list *before* the splits
are built, so a directory that cannot match is never listed, never opened, and never becomes
a task:

```python
# docs: skip
import datetime
import batcher as bt

events = bt.read.parquet("s3://bucket/events/")
one_day = events.filter(bt.col("day") == datetime.date(2024, 1, 2))
```

Over a table with a directory per day for ten years, that filter is the difference between
3,650 tasks and one. Both return the same rows.

Pruning is decided from the directory name, which records the partition value exactly, so it
is exact rather than approximate. Where it cannot decide — a predicate over a data column, or
one side of an `OR` that nothing in the layout can rule on — every directory survives and the
engine filters the rows as usual.

Both spellings of a date predicate prune. These are equivalent:

```python
# docs: skip
events.filter(bt.col("day") == datetime.date(2024, 1, 2))
events.filter(bt.col("day") == "2024-01-02")
```

The same machinery prunes a Delta or Iceberg table from its transaction log, which records
each data file's partition values and per-column bounds.

### A join can prune too

A partition predicate does not have to be written by you. When a query joins a partitioned
table to a smaller one, the smaller side's range of key values already says which partitions
can possibly match, and Batcher turns that into a filter on the partition column before the
scan is planned:

```python
# docs: skip
facts = bt.read.parquet("s3://bucket/events/")       # a directory per day, ten years of them
recent = bt.read.parquet("s3://bucket/campaigns/")   # names four days

facts.join(recent, on="day", how="inner")
```

The join reads four partitions, not 3,650, and nobody wrote a `filter`. This is *dynamic
partition pruning*, and it works because the partition column carries min/max bounds derived
from the directory names — bounds that cost nothing, since the directory listing already
happened.

It applies in both directions: the fact table's own range equally rules out dimension rows
whose key falls outside every partition.

Two things stop it. The join key has to *be* the partition column, and the smaller side's
range has to be genuinely narrower than the partitioned side's — a dimension spanning the
whole table's date range implies nothing. `explain()` shows the filter when it fires.

```{note}
The bounds are deliberately not treated as exact. A partition directory can outlive its rows:
deleting a day's files leaves `day=…` standing, so the lowest directory name may name a day
that holds nothing. That is harmless for pruning, which may only ever keep too much, but it
means an exact `MIN(day)` still reads data rather than answering from the layout.
```

### Making pruning possible

Partition on a column queries actually filter on, and at a granularity that leaves a useful
number of directories. Partitioning by a timestamp to the second produces a directory per
row, which inverts the trade: the write becomes a `PUT` per directory and every later query
pays a listing over all of them. Batcher warns when a write is about to do this.

For a column too fine to partition on, sort by it instead and rely on file-level bounds:

```python
# docs: skip
events.write.parquet("s3://bucket/events/", partition_by=["day"], sort_by=["user_id"])
```

## Estimates at a size that cannot be counted

Above the footer ceiling the engine cannot state an exact row count, and a planner with no
size at all makes every downstream decision on a default: join order, which side to build,
how much memory to reserve, how many workers to ask for.

So instead of reporting nothing, Batcher samples. It reads a fixed 64 footers spread evenly
across the file list, measures their rows per byte, and scales that by the table's total
on-disk size. The cost is 64 metadata round trips whether the table has 20,000 files or ten
million.

The result is advisory and marked as such. `meta.source_stats()` returns one
`SourceStatistics` per source, and `exact_rows` says which kind of count you are holding:

```python
import pyarrow as pa
import batcher as bt

table = bt.from_arrow(pa.table({"user": ["a", "b"], "v": [1, 2]}))
stats = table.meta.source_stats()[0]
print(stats.row_count, stats.exact_rows)
```

```text
2 True
```

Above the ceiling the same call reports an estimated `row_count` with `exact_rows=False`. An
exact `count()` still reads the data; the estimate sizes plans and never answers a question
that was asked exactly.

The sample is spread across the listing rather than taken from the front, because the front
of a partitioned table's listing is one partition. Sixty-four files from the head of a
date-partitioned table are all the same day.

## Dividing the work

A shuffle divides its input twice: into *map partitions*, which are the unit of scheduling
and of recovery, and into *hash buckets*, which are the unit a reducer holds at once.

The bucket count is the one that decides whether a large query fits in memory. A join builds
its hash table from one bucket, a sort sorts one, a window materializes one partition-run of
one. Fixing that count to the size of the cluster makes the working set grow with the data,
so the same query on the same cluster spills once the table doubles.

Batcher sizes buckets from the volume being exchanged — measured for a shape that has run
before, estimated from source statistics on its first run — and never below one bucket per
worker, since a bucket is reduced by exactly one of them. The surplus buckets cost
scheduling, not memory: they are queued across the same workers.

Any bucket count returns the same rows. The mergeable algebra makes partial states
combinable in any grouping, which is why this is a scheduling knob and not a semantic.

### Why more buckets don't flood the scheduler

A bucket is reduced by exactly one worker, so a stage with far more buckets than workers
cannot run them all at once. Batcher launches them within a submit-ahead window and refills a
slot as each reduce finishes, so the scheduler never holds a queue of tasks that cannot start.
The same bound applies to an aggregate's combiner tree, whose levels are a product of two
fan-outs and so grow fastest of all.

Two settings control the depth, shared with the map stage:

| Setting | Meaning |
|---|---|
| `distributed.max_pending_tasks` | A hard cap on outstanding tasks. `0` (the default) derives one instead. |
| `distributed.pending_window_factor` | How many tasks per worker may be outstanding when no cap is set. Default `4`. |

A stage smaller than the window submits everything before its first wait, so ordinary queries
are unaffected.

```{note}
More buckets do not fix skew. A hash bucket is the unit a key cannot be split below, so a
single dominant key stays on one reducer however fine the hash. Splitting one key across
reducers is salting, which Batcher applies from measured hot keys. See
{doc}`/user-guide/analyze/joins`.
```

## Requirements and limitations

- The plan-time ceiling is a file count, not a byte count. A table of 500 very large files
  reads every footer; one of 50,000 small files does not.
- The sampled row count assumes rows per byte is roughly stable across files. A table whose
  files differ wildly in schema or compression will be estimated less well. It stays
  advisory, so a bad estimate costs a worse plan and never a wrong answer.
- Partition pruning needs the predicate to reach the scan. A filter the optimizer cannot push
  below a join or through a non-deterministic expression will not prune. `explain()` shows
  where the filter ended up.
- Directory-level pruning applies to the top-level partition column. Deeper levels are pruned
  by the worker that lists the subtree, not on the driver.

## See also

- {doc}`performance`: the levers that apply at every size.
- {doc}`explain-plans`: confirming where a filter was pushed to.
- {doc}`/user-guide/moving-data/writing-data`: choosing a partition layout on the way in.
- {doc}`/user-guide/operate/running/unstable-nodes`: what happens when a query this long
  loses a worker.
