# Partition backfill

The enrichment step had a bug on Tuesday. Tuesday's revenue is wrong, Monday and
Wednesday are fine, and you need to reload one day into a table that a dozen dashboards
read from.

:::{warning}
Two reflexes, both bad. Append the corrected rows: now Tuesday exists twice and every
`SUM` is inflated. Delete Tuesday and then insert it: two writes, and between them the
table is missing a day. If the job dies in that gap (and it will, the one time the fix
runs against production at 2am), Tuesday is gone until somebody notices.
:::

What you want is one operation that replaces a *slice*, defined by a predicate, and
commits atomically.

## replace_where

```python
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
sales = os.path.join(work, "sales")

bt.from_pydict(
    {
        "day": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
        "region": ["us", "us", "eu", "us"],
        "revenue": [100.0, 55.0, 60.0, 300.0],
    }
).write.delta(sales, mode="overwrite", partition_by=["day"])
```

`2024-01-02` is the bad day. The recomputed rows for it:

```python
fixed = bt.from_pydict(
    {
        "day": ["2024-01-02", "2024-01-02"],
        "region": ["us", "eu"],
        "revenue": [200.0, 150.0],
    }
)
fixed.write.delta(sales, replace_where=bt.col("day") == "2024-01-02", partition_by=["day"])

print(bt.read.delta(sales).sort("day", "region").to_pydict())
# {'day': ['2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03'],
#  'region': ['us', 'eu', 'us', 'us'], 'revenue': [100.0, 150.0, 200.0, 300.0]}
```

Every row matching the predicate is replaced by the rows you wrote. Everything else is
untouched, and it lands as one Delta commit, so no reader ever sees a table without
Tuesday in it.

## Run it again. And again.

The property that matters for a backfill is not that it works once. It is that running it
twice does not change the answer:

```python
fixed.write.delta(sales, replace_where=bt.col("day") == "2024-01-02", partition_by=["day"])
print(bt.read.delta(sales).count())
# 4
```

Still four rows. Which means the backfill can be a plain scheduled job with retries, run
over a range of days in a loop, killed halfway and restarted, without anybody having to
reason about what already ran. That is the whole point.

This is why `replace_where` and `merge` are different tools. `merge` reconciles rows *by
key*: a target row whose key the source never mentions survives. `replace_where`
reconciles a *region of the table*: everything matching the predicate is gone, whether or
not the source has a replacement for it. Backfills are region-shaped. Upserts are
key-shaped. Using the wrong one produces a table that looks fine and is wrong.

| Write | What it reconciles | Target rows the source never mentions | Use it for | It is wrong when |
|---|---|---|---|---|
| `mode="append"` | nothing; it adds rows | untouched | a fresh partition nobody has written yet | the rows are already in the table, so you get them twice |
| `merge_on=` | rows, by key | survive | upserts, CDC, dimension loads | a correct backfill must *delete* rows the recompute no longer produces |
| `replace_where=` | a region, by predicate | deleted | rebuilding a day, a partition, any slice | the predicate is wider than the slice your source actually produces |

## The predicate is a delete statement

:::{important}
The predicate deletes everything it matches. If your recomputed source only covers *part*
of what the predicate matches, the rest is deleted. Not overwritten with something wrong:
gone from the table entirely, in the same atomic commit that made the fix look like it
worked.
:::

Say the recompute job was scoped to `region = 'us'` and somebody forgot to narrow the
predicate to match:

```python
partial = bt.from_pydict({"day": ["2024-01-02"], "region": ["us"], "revenue": [201.0]})
partial.write.delta(sales, replace_where=bt.col("day") == "2024-01-02", partition_by=["day"])

print(bt.read.delta(sales).sort("day", "region").to_pydict())
# {'day': ['2024-01-01', '2024-01-02', '2024-01-03'],
#  'region': ['us', 'us', 'us'], 'revenue': [100.0, 201.0, 300.0]}
```

The EU row for Tuesday is gone. Not wrong: gone. The rule is that the predicate must
describe exactly the slice the source produces, no wider. Here that means
`(bt.col("day") == "2024-01-02") & (bt.col("region") == "us")`.

Because the target is Delta, this is recoverable. Every commit is a version, so the state
before the mistake is still queryable:

```python
print(bt.read.delta(sales, version=1).count())
# 4
```

Read that version and overwrite the table with it. On a plain Parquet target you would be
restoring from backup instead, which is a decent argument for Delta on any table a
backfill can reach.

## Hive-partitioned Parquet

:::{important}
`replace_where` needs a target it can read back and rewrite as a whole: a Delta table, or
a single Parquet file. Point it at a Hive-partitioned Parquet directory and it will not do
what you want.
:::

For that layout there is a blunter move that works, because the partition *is* the slice:
overwrite the partition directory.

```python
import glob

lake = os.path.join(work, "lake")
bt.from_pydict(
    {"day": ["2024-01-01", "2024-01-02", "2024-01-02"], "revenue": [100.0, 55.0, 60.0]}
).write.parquet(lake, partition_by=["day"])

# Recompute just that day and rewrite only its directory.
day2 = bt.from_pydict({"revenue": [200.0, 150.0]})
day2.repartition(num_files=1).write.parquet(os.path.join(lake, "day=2024-01-02"))

print(bt.read.parquet_dataset(lake).sort("day", "revenue").to_pydict())
# {'revenue': [100.0, 150.0, 200.0], 'day': ['2024-01-01', '2024-01-02', '2024-01-02']}
```

The partition column is not in the payload (it lives in the directory name, and
`parquet_dataset` recovers it on read). Two caveats, both real: the rewrite is not atomic,
so a concurrent reader can catch it mid-swap. It also only replaces the part files it
writes, so if the old partition had more parts than the new one, the leftovers are still
there and still readable. `repartition(num_files=1)` above keeps that from biting, and
`bt.compact` (see [file compaction](file-compaction.md)) cleans up after a write that did
not.

If a backfill can run against a table while anyone is reading it, use Delta and take the
atomic commit.

## Backfilling a range

Nothing about this is special-cased for one day. Loop:

::::{tab-set}

:::{tab-item} Delta target

```python
# docs: skip
for day in bt.date_range("2024-01-01", "2024-01-31").to_pydict()["date"]:
    recomputed = recompute(day)  # your pipeline, scoped to one day
    recomputed.write.delta(
        "s3://lake/sales",
        replace_where=bt.col("day") == day,
    )
```

Each iteration is one atomic commit and is independently idempotent, so a failure at day
17 costs you day 17. Restart the loop from wherever you like.
:::

:::{tab-item} Hive-partitioned Parquet target

```python
# docs: skip
for day in bt.date_range("2024-01-01", "2024-01-31").to_pydict()["date"]:
    recomputed = recompute(day)  # your pipeline, scoped to one day
    recomputed.repartition(num_files=1).write.parquet(f"s3://lake/sales/day={day}")
```

Same loop, same idempotence, but the swap of each day's directory is not atomic. A reader
running against that partition at the wrong moment sees it half-written.
:::

::::

:::{tip}
Whichever target you use, keep the recompute for one day in a function that takes the day
and returns a dataset. The loop then stays trivial, a retry is one more call, and the
predicate and the source cannot drift apart, which is the mistake that deletes the EU rows.
:::

## See also

- [Late-arriving data](late-arriving-data.md): why the day needed rebuilding in the
  first place.
- [Slowly changing dimensions](slowly-changing-dimensions.md): when history is the
  point, not the obstacle.
- [File compaction](file-compaction.md): cleaning up the parts a partial rewrite leaves.
- [Lakehouse tables](../../user-guide/lakehouse.md): commits, versions, time travel.
- [Writing data](../../user-guide/writing-data.md): `replace_where`, `merge_on`, `mode`.
- [Delta Lake](../../integrations/delta-lake.md): what makes the commit atomic.
- [IO API reference](../../api/io.md): the sink arguments in full.
