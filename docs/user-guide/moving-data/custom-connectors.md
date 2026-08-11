# Custom connectors

Batcher reads and writes through two small contracts. A {py:class}`Source <batcher.io.Source>` says what its
schema is and how it divides into {py:class}`Split <batcher.io.Split>`s. A {py:class}`Sink <batcher.io.Sink>` consumes Arrow tables and
reports the files it produced. Everything the engine ships (Parquet, CSV, JSON,
Delta, Kafka) is written against those contracts and registered by name, and your
own format plugs in the same way, with no fork of the engine.

Import the pieces from `batcher.io`:

```python
import batcher as bt
from batcher.io import SINKS, SOURCES, FileSink, FileSource
```

## The read side

A `Source` is a lazily-readable relation. It knows its `schema()` without touching
the data, reads on demand (`read()` materializes, {py:meth}`iter_batches() <batcher.Dataset.iter_batches>` streams), reports
`row_count()` when that is cheap (and `None` when counting would cost a scan), and
returns a stable `identity()` string that the metadata hub keys learned statistics
under.

Both read methods take a `projection`: the list of columns the scan must produce.
That parameter is the hook the optimizer's projection pushdown uses, so a columnar
source reads only the columns a query asked for. Ignoring it is correct but slow,
since the engine still selects the right columns afterwards.

:::{tip}
Honor `projection` even if your format cannot skip the read. Returning only the requested
columns keeps the rest out of every downstream buffer, and it costs you one `select` on a
batch you already have in hand.
:::

## Splits are the unit of read parallelism

The sixth method is the interesting one. `splits()` returns the independently
readable slices of the source, and a slice is what one worker gets. A split carries
*locators* only (a format name, a path, a set of row-group ids), never data, so it
pickles cheaply and the worker opens storage directly instead of receiving bytes
from the driver.

Three split types cover almost everything:

| Split | Covers | Used by |
| --- | --- | --- |
| {py:class}`FileSplit <batcher.io.FileSplit>` | one whole file, rebuilt on the worker from `(format_name, path, kwargs)` | the {py:class}`FileSource <batcher.io.FileSource>` default |
| {py:class}`RowGroupSplit <batcher.io.RowGroupSplit>` | a contiguous run of Parquet row-groups inside one file | {py:class}`ParquetSource <batcher.io.ParquetSource>` |
| {py:class}`WholeSourceSplit <batcher.io.WholeSourceSplit>` | a source that cannot subdivide, read as one slice | {py:class}`InMemorySource <batcher.io.InMemorySource>`, {py:class}`IteratorSource <batcher.io.IteratorSource>` |

This is why a directory of 1,000 files and a single Parquet file with 1,000
row-groups both parallelize: the first yields a split per file, the second a split
per row-group run. And a source that cannot be sliced is not a failure case: it
returns one `WholeSourceSplit` and reads serially, which is exactly what an
in-memory relation or a Python generator does.

```python
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from batcher.io import InMemorySource, IteratorSource, ParquetSource

pq_path = os.path.join(tempfile.mkdtemp(), "big.parquet")
pq.write_table(pa.table({"x": list(range(1000))}), pq_path, row_group_size=250)

splits = ParquetSource(pq_path).splits()
print(len(splits), type(splits[0]).__name__, splits[0].row_count())
# 4 RowGroupSplit 250

schema = pa.schema([("n", pa.int64())])
mem = InMemorySource([pa.record_batch({"x": [1, 2, 3]})])
stream = IteratorSource(lambda: iter([pa.record_batch({"n": [1, 2]}, schema=schema)]), schema)
print(type(mem.splits()[0]).__name__, type(stream.splits()[0]).__name__)
# WholeSourceSplit WholeSourceSplit
```

`RowGroupSplit` carries the footer-derived row count, so balancing the splits across
workers never re-opens the file. {py:class}`CSVSource <batcher.io.CSVSource>` and {py:class}`JSONSource <batcher.io.JSONSource>` go further and cut a
large file into newline-aligned byte ranges, so one multi-GB CSV or NDJSON file fans
across workers instead of being read on one node.

## The write side

A `Sink` takes the other direction. {py:obj}`write(table, path) <batcher.Dataset.write>` writes a single file
atomically and returns a {py:class}`WrittenFile <batcher.io.WrittenFile>` (path, rows, bytes, and any Hive partition
values). `write_partitioned(table, path, ...)` writes one shard of a directory
write and returns a `WrittenFile` per file. `commit(manifest, path)` finalizes the
write from the {py:class}`WriteManifest <batcher.io.WriteManifest>` that every shard contributed to: a no-op for file
sinks, whose data is visible as soon as it is written, and an atomic transaction-log
commit for a lakehouse sink.

The manifest is what makes a distributed write mergeable: each worker returns its
own `WrittenFile`s, the driver concatenates them (the merge is commutative), and one
commit runs at the end.

## The registries

`SOURCES` and `SINKS` map a name to a class. Registering under a name is what
makes {py:obj}`bt.read(path, format="myfmt") <batcher.read>` and {py:obj}`ds.write(path, format="myfmt") <batcher.Dataset.write>` resolve to
your class, and it is also how a worker rebuilds a reader from a `FileSplit`: the
split ships the format *name*, not the object.

Registration is a decorator, and it happens as a side effect of importing your
module, so import it once before you read. Extension-based autodetection
(`bt.read("data/events.parquet")`) uses a fixed table of the built-in extensions;
a custom format is addressed by passing `format=` explicitly.

## A custom format, end to end

{py:class}`FileSource <batcher.io.FileSource>` and {py:class}`FileSink <batcher.io.FileSink>` are Template-Method bases that already own path/glob
expansion, filesystem resolution, schema caching, multi-file reads, projection
plumbing, streaming, atomic writes, Hive partitioning, and split generation. A
concrete format is the two or three primitives they call. Here is a pipe-separated
text format:

::::{tab-set}
:::{tab-item} The source

```python
from typing import IO, Any

import pyarrow.csv as pacsv


@SOURCES.register("psv")
class PSVSource(FileSource):
    """Pipe-separated text: a header line, then `a|b` per row."""

    suffix = ".psv"  # what a directory/glob read expands to
    format_name = "psv"  # the registry key a FileSplit rebuilds through
    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        return pacsv.open_csv(fh, parse_options=pacsv.ParseOptions(delimiter="|")).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        convert = pacsv.ConvertOptions(include_columns=projection) if projection else None
        table = pacsv.read_csv(
            fh, parse_options=pacsv.ParseOptions(delimiter="|"), convert_options=convert
        )
        return table.to_batches()
```

Two primitives: read the schema from an open file, and read the batches. `FileSource`
owns the rest, including `splits()`.

:::

:::{tab-item} The sink

```python
@SINKS.register("psv")
class PSVSink(FileSink):
    """Write the same pipe-separated text."""

    suffix = ".psv"
    format_name = "psv"
    __slots__ = ()

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        pacsv.write_csv(table, fh, write_options=pacsv.WriteOptions(delimiter="|"))
```

One primitive. `FileSink` owns the rest: the atomic rename, the Hive partitioning,
the manifest.

:::
::::

Both are now first-class. The read is lazy and the projection reaches the parser;
the write is atomic and returns a manifest:

```python
path = os.path.join(tempfile.mkdtemp(), "events.psv")

manifest = bt.from_pydict({"id": [1, 2, 3], "kind": ["a", "b", "a"]}).write(path, format="psv")
print(manifest.num_files, manifest.total_rows)
# 1 3

written = manifest.files[0]  # a WrittenFile
print(written.rows, written.bytes > 0)
# 3 True

print(bt.read(path, format="psv").filter(bt.col("kind") == "a").to_pydict())
# {'id': [1, 3], 'kind': ['a', 'a']}
```

`PSVSource` inherited `splits()`, so it already parallelizes across files:

```python
print([type(s).__name__ for s in PSVSource(path).splits()])
# ['FileSplit']
```

That default is one `FileSplit` per file, which is the right answer for any format
you cannot address below file granularity: a compressed stream, or a container whose
records are only reachable by decoding from the start. To go finer, override
`_file_splits(path, target_size)` and return your own splits: `ParquetSource` returns
`RowGroupSplit`s there, `CSVSource` returns byte ranges. Two more override points are
worth knowing. `_file_row_count(path)` lets the planner size the source from a footer
instead of a scan, and `_reader_kwargs()` must return any non-path constructor
arguments your source needs (a message class, a sheet name).

:::{warning}
`_reader_kwargs()` is not optional bookkeeping. A `FileSplit` rebuilds the reader on the
worker as `SOURCES.get(format_name)(path, **kwargs)`, so an argument your constructor
needs and the split did not carry either raises on the worker, or, when the constructor
has a default, silently reads the wrong thing. A source configured with the wrong schema,
or the wrong protobuf message class, will happily return rows.
:::

## Sources that aren't files

A source does not have to subclass `FileSource`. Implement the six `Source` methods
directly (a generator, a REST API, a table in a system Batcher has never heard of),
return a single `WholeSourceSplit` if it cannot be sliced, and register it. Non-file
sources are read by name with {py:meth}`bt.read.table(name, *args, **opts) <batcher.api.io_namespace.reader.Reader.table>`, which forwards
its arguments straight to your constructor:

```python
from collections.abc import Iterator

from batcher.io import Split, WholeSourceSplit


@SOURCES.register("squares")
class SquaresSource:
    """A generated relation: `n` and `n ** 2` for n in [0, count)."""

    bounded = True  # False for an unbounded stream, so collect() refuses it

    def __init__(self, count: int) -> None:
        self._count = count

    def schema(self) -> pa.Schema:
        return pa.schema([("n", pa.int64()), ("square", pa.int64())])

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        ns = list(range(self._count))
        batch = pa.record_batch({"n": ns, "square": [n * n for n in ns]}, schema=self.schema())
        yield batch.select(projection) if projection is not None else batch

    def row_count(self) -> int | None:
        return self._count

    def identity(self) -> str:
        return f"squares:{self._count}"

    def splits(self, target_size: int | None = None) -> list[Split]:
        return [WholeSourceSplit(self)]


print(bt.read.table("squares", 4).to_pydict())
# {'n': [0, 1, 2, 3], 'square': [0, 1, 4, 9]}
```

`InMemorySource` and `IteratorSource` are the two built-ins of this shape: the first
wraps batches already in the process (what {py:func}`from_arrow <batcher.from_arrow>` and {py:func}`from_pydict <batcher.from_pydict>` build), the
second wraps a zero-argument factory that returns a fresh iterator of batches each
time it is called (what {py:func}`from_batches <batcher.from_batches>` builds, and the entry point for unbounded
streams).

## Reference points in the tree

Read the built-ins before you write your own; each is small, and each demonstrates a
different split strategy.

| Format | Source / sink | Splits into |
| --- | --- | --- |
| Parquet | {py:class}`ParquetSource <batcher.io.ParquetSource>` / {py:class}`ParquetSink <batcher.io.ParquetSink>` | {py:class}`RowGroupSplit <batcher.io.RowGroupSplit>` runs, packed toward a target size |
| CSV | {py:class}`CSVSource <batcher.io.CSVSource>` / {py:class}`CSVSink <batcher.io.CSVSink>` | newline-aligned byte ranges, or one `FileSplit` for a small file |
| JSON (NDJSON) | {py:class}`JSONSource <batcher.io.JSONSource>` / {py:class}`JSONSink <batcher.io.JSONSink>` | newline-aligned byte ranges, same rule |

`ParquetSink` streams row-groups incrementally, `CSVSink` and `JSONSink` fan their
encode across cores. None of that is required of a new format: implement `_write_file`
and the base buffers a table for you.

## Large payloads: fetch bytes late

:::{tip}
Media sources can hand back *reference* rows (a URI, a size, some metadata) rather
than the payload itself. Filter and sample those rows first, then materialize only the
survivors. A predicate over a reference column costs nothing; the same predicate after
the bytes are resident has already paid for every row it drops.
:::

`read_blob_bytes` reads each row's URI column and writes the file contents into a
`large_binary` column. Run it inside `map_batches` with a small `batch_size` so only a
few payloads are resident at once:

```python
# docs: skip
from batcher import col
from batcher.io import read_blob_bytes

clips = bt.read.video("s3://clips/", materialize_bytes=False)
small = clips.filter(col("size") < 500_000_000)
decoded = small.map_batches(read_blob_bytes, batch_size=4)
```

The same trick applies to a custom source over an object store: yield handles from
`iter_batches`, and let the query decide which payloads are worth fetching.

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>`: the built-in readers and what each one takes.
- {doc}`Writing data </user-guide/moving-data/writing-data>`: partitioned, compacted, and distributed writes.
- {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: the filesystem layer your source resolves through.
- {doc}`IO API </api/relational/io>`: the full `batcher.io` reference.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: what a worker does with the
  split you handed it.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: how splits become
  tasks, and why they carry locators rather than data.
- {doc}`Incremental ingest </cookbook/data-engineering/ingest/incremental-ingest>`: a source read
  repeatedly, without re-reading what it already saw.
- {doc}`Multimodal ingest benchmark </benchmarks/results/multimodal-ingest>`: the reference-row
  pattern measured against the alternatives.
- {doc}`Agent skills </agents>`: `add-an-io-format-or-connector`, the ordered
  procedure and the tests a new connector must carry.
- {doc}`/cookbook/io/sources_and_sinks`: the registries a new format registers into, as a script.
