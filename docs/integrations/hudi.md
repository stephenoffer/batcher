# Apache Hudi

**Hudi is read-only in Batcher.** There is no writer. `bt.read.hudi(path)` gives you a lazy
`Dataset` over a Hudi table; `ds.write.hudi(...)` raises immediately and tells you why. Writing
Hudi means the Spark or Flink write stack: the commit protocol, the timeline, the index, the
compaction service. None of that belongs in a Rust/Arrow data plane. Reading is served by hudi-rs
(`pip install 'batcher-engine[hudi]'`).

| | |
| --- | --- |
| **Read** | `bt.read.hudi(path)`, with `as_of_instant=` |
| **Write** | Not supported. `ds.write.hudi(...)` raises `BackendError`. |
| **Extra** | `pip install 'batcher-engine[hudi]'` |
| **Parallelism** | None at the source. `splits()` returns a single `WholeSourceSplit`. |
| **Pushdown** | An AND of column-vs-literal comparisons, as hudi-rs filter tuples |
| **Incremental** | `HudiSource.read_incremental(start, end)` |

That is the whole shape of this integration, and it fits the common case. Hudi tables are usually
*produced* by a Spark or Flink ingest job that already exists, and *consumed* by whatever runs the
analytics. Batcher is the consumer.

The error is deliberate and immediate, not a silent no-op:

```python
import os
import tempfile

import batcher as bt
from batcher._internal.errors import BackendError

target = os.path.join(tempfile.mkdtemp(), "events")
try:
    bt.from_pydict({"id": [1]}).write.hudi(target)
except BackendError as exc:
    print(exc)
# Hudi writes require Spark/Flink; Batcher supports Hudi reads only
```

:::{tip}
If you need a table format Batcher can write transactionally, that is [Delta](delta-lake.md)
(append, overwrite, merge, replace-where) or [Iceberg](iceberg.md) (append, overwrite).
:::

## Reading

A read is a snapshot query against the current table state. It needs a real Hudi table, so the
blocks below are not run here.

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

Hudi's timeline is a sequence of *instants* (the commit timestamps you see in `.hoodie/`).

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
`bt.from_arrow` to keep working in the engine.

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
Both of these depend on the retention of the timeline. Hudi cleans old commits on a schedule that
its *writer* controls, so an instant your reader still wants can be cleaned out from under you.
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

## How it parallelizes, and where it doesn't

:::{important}
Honestly: it doesn't, at the split level. `HudiSource.splits()` returns a single
`WholeSourceSplit`. hudi-rs owns file-group resolution, log-file merging, and the read itself, and
Batcher does not reach inside that to hand out one split per base file. So the *read* is one unit
of work, and parallelism starts after it: batches morselize into the engine, and every operator
downstream runs morsel-parallel across cores and across the cluster.
:::

For a table that is a small dimension or a filtered slice, that is fine. For a multi-terabyte fact
table, it is a real bottleneck, and the fix is upstream. Read the table's underlying Parquet with
`bt.read.parquet_dataset`, which does split per file, if and only if you can guarantee the layout
is copy-on-write with no pending log files. That guarantee is the catch, so measure before you take
it.

Predicate pushdown does work. An AND of column-vs-literal comparisons becomes hudi-rs filter tuples
and prunes at the source. Anything it cannot express (an `OR`, a computed term) is simply not
pushed, and the engine's own filter produces the same rows over a wider scan. If hudi-rs rejects
the pushed filters outright, on a version or format mismatch, the read retries unfiltered rather
than failing. A correct answer is never at stake, only I/O.

## Failure modes worth knowing

**Merge-on-read tables.** hudi-rs applies log files on the read path, so the result is correct, but
an MOR table with a long log-file tail reads slowly until the writer's compaction catches up. This
is a property of the table, not the connector.

**No exact count.** Hudi exposes no cheap row count here, so `count()` scans. Delta and Iceberg
answer it from their logs; Hudi does not.

**Version skew.** hudi-rs tracks the Hudi spec independently of the Spark/Flink writer that
produced your table. A table written by a much newer Hudi than the installed `hudi` package can
fail to open with a `BackendError` naming the table. Pin the reader version against the writer's.

## See also

- [Lakehouse](../user-guide/lakehouse.md): the table-format guide.
- [Reading data](../user-guide/reading-data.md): sources, splits, and pushdown.
- [CDC pipeline](../examples/data-engineering/cdc-pipeline.md): what an incremental read
  between two instants usually feeds.
- [I/O API](../api/io.md): the full reader reference.
- [Delta Lake](delta-lake.md) and [Iceberg](iceberg.md): the writable table formats.
