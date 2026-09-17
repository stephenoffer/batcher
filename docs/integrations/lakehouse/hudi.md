# Apache Hudi

This page covers reading Apache Hudi tables. {py:meth}`bt.read.hudi(path) <batcher.api.io_namespace.reader.Reader.hudi>` gives you a lazy {py:class}`Dataset <batcher.Dataset>` over a Hudi table through hudi-rs, with no Spark and no JVM: snapshot reads, time travel to an instant, incremental reads between instants, file pruning, and one split per file slice on a copy-on-write table.

Hudi tables are usually produced by a Spark or Flink ingest job that already exists and consumed by whatever runs the analytics. Batcher is built to be that consumer. Writing Hudi means the Spark or Flink write stack, with its commit protocol, timeline, index, and compaction service, so {py:meth}`ds.write.hudi(...) <batcher.api.io_namespace.writer.Writer.hudi>` raises and says why.

The following table summarizes the connector:

| | |
| --- | --- |
| Read | `bt.read.hudi(path)`, with `as_of_instant=` |
| Write | Not supported. `ds.write.hudi(...)` raises {py:exc}`BackendError <batcher.BackendError>`. |
| Extra | `pip install 'batcher-engine[hudi]'` |
| Parallelism | One split per file slice on copy-on-write. Merge-on-read reads whole. |
| Pushdown | An AND of column-vs-literal comparisons, as hudi-rs filter tuples, pruning files |
| Incremental | `HudiSource.read_incremental(start, end)` |

The write error is immediate, not a silent no-op:

```python
import os
import tempfile

import batcher as bt

target = os.path.join(tempfile.mkdtemp(), "events")
try:
    bt.from_pydict({"id": [1]}).write.hudi(target)
except bt.BackendError as exc:
    print(exc)
# Hudi writes require Spark/Flink; Batcher supports Hudi reads only
```

:::{tip}
For a table format Batcher writes transactionally, use {doc}`Delta </integrations/lakehouse/delta-lake>` (append, overwrite, merge, replace-where) or {doc}`Iceberg </integrations/lakehouse/iceberg>` (append, overwrite, upsert, replace-where).
:::

## Read a table

A read is a snapshot query against the current table state. The blocks below need a real Hudi table, so they aren't run here.

```python
# docs: skip
import batcher as bt

events = bt.read.hudi("s3://lake/hudi/events")
by_day = (
    events.filter(bt.col("event_type") == "purchase")
    .group_by("day")
    .agg(bt.col("amount").sum().alias("revenue"))
)
print(by_day.sort("day").to_pydict())
```

Once the batches are in the engine, nothing about the source matters. The plan optimizes, the
operators run in Rust, and the table joins against Parquet, Delta, or a Postgres extract the same
way.

## Time travel and incremental reads

Hudi's timeline is a sequence of *instants*, the commit timestamps you see in `.hoodie/`.

::::{tab-set}

:::{tab-item} The snapshot at an instant

`as_of_instant=` reads the table as it was:

```python
# docs: skip
snapshot = bt.read.hudi("s3://lake/hudi/events", as_of_instant="20240301120000000")
```
:::

:::{tab-item} Only what changed

To read only what changed between two instants, which is the incremental query a downstream
medallion stage wants, go through the source directly. It returns an Arrow table, so wrap it with
{py:func}`bt.from_arrow <batcher.from_arrow>` to keep working in the engine.

```python
# docs: skip
from batcher.io.formats.lakehouse import HudiSource

source = HudiSource("s3://lake/hudi/events")
changed = source.read_incremental("20240301120000000", "20240302120000000")
recent = bt.from_arrow(changed)
```
:::

::::

:::{warning}
Both of these depend on the retention of the timeline. Hudi cleans old commits on a schedule that its writer controls, so an instant your reader still wants can be cleaned out from under you.
Coordinate the cleaner policy with whoever owns the ingest job.
:::

## Credentials and options

`options=` is passed straight to hudi-rs, which is where cloud storage configuration goes. The keys
are hudi-rs's own, not Batcher's.

:::{dropdown} Pointing the reader at an S3 table
```python
# docs: skip
events = bt.read.hudi(
    "s3://lake/hudi/events",
    options={"aws_region": "us-east-1"},
)
```
:::

## How it parallelizes

`HudiSource.splits()` asks the timeline for the surviving file slices and returns one
{py:class}`HudiFileSliceSplit <batcher.io.formats.lakehouse.hudi.HudiFileSliceSplit>` per slice.
A split carries locators only, the table root plus the slice's base-file path and the reader
options, so it pickles cheaply and its worker opens that one file. The pushed predicate is applied
*before* the enumeration, so a partition the filter excludes is never turned into a split at all.

Each split's row count comes from its base file's Parquet footer rather than from hudi-rs's
`HudiFileSlice.num_records`, which reports the whole table's total on every slice. That is
metadata rather than a scan, and it is what lets the distributed planner bin-pack by real size.

:::{important}
A merge-on-read table is read whole. A MoR slice is a base file plus log files holding later
updates and deletes, and the per-slice reader opens the base file only, so splitting one would
resurrect superseded rows. A table with any log files therefore falls back to a single
{py:class}`WholeSourceSplit <batcher.io.WholeSourceSplit>`, which hudi-rs merges correctly.
Correctness first. The same fallback covers a timeline that cannot be enumerated at all.
:::

So the parallel read is a copy-on-write property, and a MoR fact table with a long log tail is
still one unit of work at the source. Compaction upstream is what returns it to the split path.

Predicate pushdown prunes files, never rows. An AND of column-vs-literal comparisons becomes
hudi-rs filter tuples, which eliminate whole slices by partition path and, from hudi-rs 0.5, by a
base file's column statistics, so even an unpartitioned table skips files a predicate excludes.
Anything the translation cannot express, an `OR` or a computed term, is left to the engine's own
filter, which produces the same rows over a wider scan. If hudi-rs rejects the pushed filters
outright, on a version or format mismatch, the read retries unfiltered rather than failing. A
correct answer is never at stake, only I/O.

## Requirements and limitations

hudi-rs applies log files on the read path, so a merge-on-read result is correct, but a table with a long log-file tail reads slowly, on one split, until the writer's compaction catches up.

The row count costs one pass over the base files' Parquet footers, which is metadata rather than a scan but still a read per slice on a table with many slices.

hudi-rs tracks the Hudi spec independently of the Spark/Flink writer that
produced your table. A table written by a much newer Hudi than the installed `hudi` package can
fail to open with a `BackendError` naming the table. Pin the reader version against the writer's.

## See also

- {doc}`Lakehouse </user-guide/moving-data/lakehouse>`: the table-format guide.
- {doc}`Reading data </user-guide/moving-data/reading-data>`: sources, splits, and pushdown.
- {doc}`CDC pipeline </cookbook/data-engineering/ingest/cdc-pipeline>`: what an incremental read
  between two instants usually feeds.
- {doc}`I/O API </api/relational/io>`: the full reader reference.
- {doc}`Delta Lake </integrations/lakehouse/delta-lake>` and {doc}`Iceberg </integrations/lakehouse/iceberg>`: the writable table formats.
