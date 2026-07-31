# File compaction

A streaming job writes a file per micro-batch. An incremental loader writes a file per
arrival. Neither is wrong, and after a month the table is 40,000 files averaging 80 KB,
and the read that used to take four seconds takes four minutes.

:::{warning}
The data did not grow. The number of files grew, and on object storage each one costs a
footer read plus a round trip before a single row is decoded. Nothing about the query
changed. The table underneath it did.
:::

## Watch it happen

Twelve arrivals, two rows each:

```python
import glob
import os
import tempfile

import batcher as bt

work = tempfile.mkdtemp()
events = os.path.join(work, "events")

for i in range(12):
    bt.from_pydict({"id": [i * 2, i * 2 + 1], "v": [i, i]}).write.parquet(
        os.path.join(events, f"part-{i:03d}.parquet")
    )

files = sorted(glob.glob(os.path.join(events, "*.parquet")))
print(len(files), bt.read.parquet(events).count())
# 12 24
```

Twelve files, 24 rows. Now look at what they cost:

```python
total_bytes = sum(os.path.getsize(f) for f in files)
print(total_bytes)
# 9300
```

9,300 bytes to store 24 small rows. Parquet's footer and per-column metadata are paid
*per file*, and at this size the metadata *is* the file. Scale that to 40,000 files and the
overhead is not a rounding error, it is the workload.

## Compact in place

`bt.compact` reads the path and rewrites it into fewer, larger files. The parts it
replaced are deleted:

```python
manifest = bt.compact(events, num_files=1, format="parquet")

remaining = glob.glob(os.path.join(events, "*.parquet"))
print(len(remaining), bt.read.parquet(events).count())
# 1 24
print(os.path.getsize(remaining[0]))
# 879
```

Same 24 rows, one file, 879 bytes instead of 9,300. The read now opens one footer.

:::{dropdown} What `compact` hands back

```python
print(manifest.num_files, manifest.total_rows, manifest.total_bytes)
# 1 24 879
for written in manifest.files:
    print(os.path.basename(written.path), written.rows, written.bytes)
# part-00000.parquet 24 879
```

The `WriteManifest` is the record of what the rewrite produced: one entry per file, with
its row count and size. It is the same object every write returns, so a scheduled
compaction job can log it, compare it against the previous run, and tell you the day the
file sizes started drifting back down.
:::

In production you size by bytes, not by file count, because you do not know the row count
in advance and you do not care:

```python
# docs: skip
bt.compact("s3://lake/events", target_size_mb=128)
bt.compact("s3://lake/events", target_size_mb=128, by=["day"])  # per Hive partition
```

128 MB to 1 GB per file is the range worth aiming at. Parquet splits are row-group
granular, so a large file still parallelizes across workers, but a file so large that a
single writer takes an hour to produce it is a file that will fail halfway through
something eventually.

## Better: do not make the mess

Compaction is a rewrite of the whole table. It is the fix, not the plan. If you know the
target file size at write time, say so and skip the second pass entirely.

::::{tab-set}

:::{tab-item} Size the files by bytes

`repartition(target_size_mb=...)` sizes the output files from the materialized result:

```python
sized = os.path.join(work, "sized")
bt.from_pydict({"id": list(range(100))}).repartition(target_size_mb=0.0005).write.parquet(sized)
print(len(glob.glob(os.path.join(sized, "*.parquet"))))
# 2
```

That target is absurdly small so the example produces more than one file. Use 128 for
real data.
:::

:::{tab-item} Size the files by rows

`max_rows_per_file` is the same idea in rows, which is the right knob when rows have a
predictable width:

```python
capped = os.path.join(work, "capped")
bt.from_pydict({"id": list(range(10))}).write.parquet(capped, max_rows_per_file=4)
print(len(glob.glob(os.path.join(capped, "*.parquet"))))
# 3
```
:::

::::

| Knob | When it acts | Reach for it when |
|---|---|---|
| `repartition(target_size_mb=...)` | at write time, from the materialized result | you know roughly how big the output should be |
| `max_rows_per_file=` | at write time, per file | rows have a predictable width |
| `bt.compact(path, target_size_mb=...)` | after the fact, rewriting the table | the small files already exist |

For a streaming sink, the honest answer is that per-micro-batch files are the price of low
latency. Write them small, and run compaction on a schedule. Trying to buffer your way out
of it in the writer moves the latency somewhere less visible instead of removing it.

## Compaction is also your chance to cluster

While you are rewriting every file anyway, decide *which rows go in which file*. Sorting
the output on the column your queries filter by makes each file's min/max bounds tight,
and tight bounds are what let the next reader skip whole files without opening them.

```python
clustered = os.path.join(work, "clustered")
bt.from_pydict({"day": ["d3", "d1", "d2", "d1"], "v": [1, 2, 3, 4]}).write.parquet(
    clustered, sort_by=["day"], max_rows_per_file=2
)
print(bt.read.parquet(clustered).to_pydict())
# {'day': ['d1', 'd1', 'd2', 'd3'], 'v': [2, 4, 3, 1]}
```

Both `d1` rows now live in one file, so `filter(col("day") == "d1")` reads one file and
proves the other cannot match. On a Delta table the same statistics land in the
transaction log and the pruning happens at *plan* time, before any file is opened. That is
covered in {doc}`lakehouse tables </user-guide/moving-data/lakehouse>`.

:::{tip}
Compaction without clustering gives you fewer files. Compaction with clustering also cuts
the bytes the next query has to read, out of the same rewrite. You are paying for the
rewrite either way, so sort on whatever your `WHERE` clause actually mentions.
:::

## On a Delta table, compaction does not delete

:::{important}
Compacting a transactional table commits the new files and retires the old ones *from the
log*. The old files stay on storage, because time travel is exactly the promise that an
older version can still be read. Your bill does not go down until you vacuum, and a
compaction job that runs nightly and never vacuums grows the bucket instead of shrinking it.
:::

`bt.vacuum` is the operation that reclaims them, and it defaults to a dry run:

```python
# docs: skip
bt.compact("s3://lake/events", target_size_mb=128)

print(bt.vacuum("s3://lake/events"))  # dry run: what it *would* delete
bt.vacuum("s3://lake/events", dry_run=False)  # actually reclaim
```

The retention window (7 days for Delta by default) is the safety argument: a file is only
removed once it has been unreferenced for longer than any reader could still be holding it
open. Shorten the window below the table's configured minimum and the backend refuses,
because "I have a scan running" and "these files are unreferenced" can both be true at once.
Do not fight it.

## The rules

:::{important}
`bt.compact` is single-writer and it rewrites in place. Do not run it against a table an
ingest job is appending to: it materializes the current contents and writes them back,
deleting the parts it replaced. Files written in between are not in the picture it took.
Schedule it in a quiet window, or point the ingest at a Delta table and take the
transaction.
:::

It is also not free. Compacting a 500 GB table reads and writes 500 GB. Run it on the
tables whose read pattern justifies it (the ones queried constantly), not on everything on
a cron out of tidiness.

And check before you act. If the files are already big enough, compaction is pure cost:

```python
def wants_compaction(path, min_mb=32.0):
    """True when the average part file is small enough that reads pay for it."""
    parts = glob.glob(os.path.join(path, "*.parquet"))
    if len(parts) < 2:
        return False
    avg_mb = sum(os.path.getsize(f) for f in parts) / len(parts) / 1e6
    return avg_mb < min_mb


print(wants_compaction(events))
# False
```

One file here, so nothing to do. Run it against the 40,000-file table and it says yes.

## See also

- {doc}`Incremental ingest </cookbook/data-engineering/incremental-ingest>`: the job that makes the small files.
- {doc}`Partition backfill </cookbook/data-engineering/partition-backfill>`: rewriting a slice rather than the table.
- {doc}`Late-arriving data </cookbook/data-engineering/late-arriving-data>`: the other reason a day gets rewritten.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: the write options in full.
- {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`: file statistics, pruning, time travel.
- {doc}`Delta Lake </integrations/lakehouse/delta-lake>`: what `vacuum` is protecting you from.
- {doc}`IO API reference </api/relational/io>`: `bt.compact`, `bt.vacuum`, and the sink arguments.
