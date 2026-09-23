# Source and sink registries

`bt.read.parquet(...)` and `ds.write.parquet(...)` are thin wrappers over two registries, `SOURCES` and `SINKS` in `batcher.io`. Listing them tells you what *this* build can read and write, including the formats an optional extra adds.

The script lists both registries and looks up the `ParquetSource` and `ParquetSink` classes behind the readers. Those classes are the contract to study when you write a connector of your own.

The whole script, executed on every test run:

```{literalinclude} ../../../examples/io/sources_and_sinks.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/io/sources_and_sinks.py
```


## What a source can answer without reading rows

A source carries metadata the optimizer consults before any scan, and the same methods are
available to you. They are how a plan gets a row count, a bound, or a sum without touching
the data, and reading them directly is the quickest way to see whether a pushdown will
find what it needs.

{py:meth}`InMemorySource.column_bounds <batcher.io.InMemorySource>` returns the min/max
zone map for a column, {py:meth}`column_mean <batcher.io.InMemorySource>` and
{py:meth}`column_sum <batcher.io.InMemorySource>` return exact aggregates the source
already holds, and {py:meth}`column_cheap_stat <batcher.io.InMemorySource>` returns only
what is free to compute. {py:meth}`column_predicate_count <batcher.io.InMemorySource>`
answers how many rows satisfy a simple comparison, or `None` when the source cannot say
without scanning.

```python
import pyarrow as pa

from batcher.io import InMemorySource

table = pa.table({"a": [1, 2, 3, 4, 5], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
source = InMemorySource(table.to_batches())

bounds = source.column_bounds("a")
print(bounds.min, bounds.max)
print(source.column_sum("a"), source.column_mean("b"))
print(source.column_predicate_count(">", "a", 2))
```

A `None` from any of these means "not known cheaply", never "zero". The optimizer treats
it that way, and so should a caller.

{py:meth}`ParquetSource.row_group_bounds <batcher.io.ParquetSource>` is the file-backed
equivalent, reading per-row-group statistics straight from the footers. It is what lets a
predicate skip whole row groups without opening them.

```python
import tempfile

import pyarrow.parquet as pq

from batcher.io import ParquetSource

directory = tempfile.mkdtemp()
pq.write_table(table, f"{directory}/part.parquet")
source = ParquetSource(directory)

for group in source.row_group_bounds(["a"]):
    print(group.num_rows, group.mins["a"], group.maxs["a"])
```

Two more properties describe the source rather than its data.
{py:obj}`FileSource.node_local <batcher.io.FileSource>` says whether the data is reachable
only from the process that holds it, which is what stops the scheduler shipping a plan over
it to another node. {py:meth}`stats_version <batcher.io.FileSource>` is a digest of the
statistics the source currently reports, so a cache keyed on it invalidates when the files
change.

```python
print(source.node_local, isinstance(source.stats_version(), str))
```


## Writing a stream without holding it

Two sink methods write an iterator of batches straight to files, so a result larger than
memory never has to be materialized first.

{py:meth}`FileSink.write_stream_parts <batcher.io.FileSink>` rolls a new file every
`max_rows_per_file` rows and returns one record per file written.
{py:meth}`write_stream_shard <batcher.io.FileSink>` writes the whole iterator to a single
file whose name carries the `file_index` you pass, which is how a distributed write gives
each worker a name that cannot collide with another's.

```python
import tempfile

import pyarrow as pa

import batcher as bt
from batcher.io.formats.structured.parquet.sink import ParquetSink

table = pa.table({"a": list(range(10))})
directory = tempfile.mkdtemp()
sink = ParquetSink()

written = sink.write_stream_parts(
    iter(table.to_batches(max_chunksize=3)), f"{directory}/parts", max_rows_per_file=4
)
print([file.rows for file in written])
print(sorted(bt.read.parquet(f"{directory}/parts").to_pydict()["a"]))

shard = sink.write_stream_shard(
    iter(table.to_batches(max_chunksize=5)), f"{directory}/shard", file_index=2
)
print(shard.rows)
```

The row counts show the roll boundary falling inside a batch: ten rows at four per file
give 4, 4 and 2, not three whole batches. Each returned record also carries the file's byte
size and statistics, which is what a manifest-based table format records after a commit.

`FileSink` is abstract, so use the concrete sink for the format you want. Both methods
accept `resume=True` to skip files that already exist, which is what makes a retried write
idempotent rather than duplicating rows.

## See also

- {doc}`save_modes`: what happens when the target already exists.
- {doc}`streaming_reads`: iter_batches, limits, and lazy metadata.
- {doc}`/user-guide/moving-data/reading-data`: every source format and how paths and schemas resolve.
- {doc}`/user-guide/moving-data/writing-data`: sinks, save modes, and partitioned output.
