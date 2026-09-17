# Writing data

This page covers writing a `Dataset` out: file formats and database sinks, partitioned layouts, file sizing, completion markers and distributed writes. A write is a terminal operation. It executes the plan and streams Arrow batches to the sink, so the memory bounds and spill behavior that govern `collect` apply here too.

The format-specific helpers are `write.parquet`, `write.csv`, and `write.json`. The
generic {py:obj}`write(path, format=...) <batcher.Dataset.write>` covers all of them and is the place to pass a
partitioning scheme.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "b", "a"],
        "price": [10.0, 20.0, 30.0],
    }
)
```

## Parquet round trip

This example writes to a temporary directory and reads the file back, so it runs
end to end. Parquet is the format to reach for: columnar, compressed, friendly to
pushdown.

```python
import os
import tempfile

out_dir = tempfile.mkdtemp()
path = os.path.join(out_dir, "out.parquet")

ds.write.parquet(path)
back = bt.read.parquet(path)
print(back.to_pydict())
# {'category': ['a', 'b', 'a'], 'price': [10.0, 20.0, 30.0]}
```

`write.parquet` defaults to `zstd` compression. Pass `compression=` to override it.

## CSV and JSON

Both take the same path-and-options shape as `write.parquet`:

```python
# docs: skip
ds.write.csv("output/data.csv")
ds.write.json("output/data.json")
```

Use CSV and JSON for interchange with tools that require them. Parquet is faster and
preserves types, so prefer it for anything that will be read back by Batcher.

CSV is a grid of scalar cells, so it has nowhere to put a list, struct, map, or
fixed-size-list column. Writing one raises a `SchemaError` naming the column rather than
letting the underlying writer report a bare type. Flatten the column first with
`.cast("string")`, or `.list.join(",")` for a list, or write a format that carries it.

JSON has no temporal type, so a date or timestamp column is written as an ISO-8601 string
at the column's own resolution, the spelling DuckDB and Spark produce. That is true wherever
the column sits, including inside a struct or a list, and it does not depend on what else
the table holds.

```python
import datetime as dt

import batcher as bt

events = bt.from_pydict({"at": [dt.datetime(2024, 2, 29, 12, 34, 56, 123456)], "amount": [10]})
print(events.schema.field("at").type)
# timestamp[us]
```

Writing that column produces `{"at":"2024-02-29 12:34:56.123456","amount":10}`. The
microseconds survive, but the *type* does not: Arrow's JSON reader infers a bare date back
as a timestamp and leaves a sub-second instant as text. Write Parquet, Arrow, or Avro when
the reader has to get a timestamp column back.

A duration column has no ISO form either writer produces, so JSON and CSV both write its
integer count in the column's own unit. The unit is not in the output, so record it
alongside, or cast the column to something self-describing before writing.

## Other file and database sinks

The write namespace mirrors the reader namespace: every format has a typed writer,
and the generic `write(path, format=...)` reaches them all, each returning a
{py:class}`WriteManifest <batcher.io.WriteManifest>`. Beyond Parquet, CSV, and JSON:

| Writer | Writes | Needs |
| --- | --- | --- |
| `write.orc(path)` | ORC files | nothing extra |
| `write.arrow(path)` | Arrow/Feather IPC files, or IPC streams with `ipc_format="stream"` | nothing extra |
| `write.avro(path)` | Avro files | `[avro]` |
| `write.fasta(path)` | FASTA, one record per row, wrapped at 60 | nothing extra |
| `write.fastq(path)` | Four-line FASTQ, one read per row | nothing extra |
| `write.bed(path)` | BED intervals, the leading run of standard columns | nothing extra |
| `write.gff(path)` | GFF3 annotations, all nine columns | nothing extra |
| `write.lance(path)` | A Lance dataset (columnar ML format) | `[lance]` |
| `write.msgpack(path)` | MessagePack files | `[msgpack]` |
| `write.text(path)` | One string column, one value per line | nothing extra |
| `write.xml(path, row_tag=..., root_tag=...)` | XML in Spark's row-element layout | nothing extra |
| `write.numpy(path, column=...)` | One column as `.npy` arrays | nothing extra |
| `write.webdataset(path)` | WebDataset `.tar` shards keyed by `__key__` | nothing extra |
| `write.tfrecord(path)` | TFRecord, one `tf.train.Example` per row | `google-crc32c` |
| `write.sql(table, uri=..., mode=...)` | A SQL table: append, or upsert/update/delete by key | a driver + reachable DB |
| `write.snowflake(table, connection_kwargs=...)` | A Snowflake table | Snowflake account |
| `write.clickhouse(table, host=...)` | An existing ClickHouse table, via `insert_arrow` | `[clickhouse]` + a server |
| `write.mongo(collection, uri=..., mode=...)` | A MongoDB collection | a running MongoDB |
| `write.dynamodb(table, region_name=...)` | A DynamoDB table via `BatchWriteItem` | AWS DynamoDB |
| `write.cassandra(table, contact_points=..., keyspace=...)` | A Cassandra / Scylla table | a running Cassandra |
| `write.redis(key_prefix, host=...)` | A Redis keyspace, one hash or string per key | a running Redis |
| `write.elasticsearch(index, hosts=...)` | An Elasticsearch index via `_bulk` | a running Elasticsearch |
| `write.hbase(table, host=...)` | An HBase table via happybase batches | an HBase Thrift server |

The database sinks take a `mode` that is not a save mode. `append` and `overwrite` load a
table. `upsert`, `update` and `delete` *maintain* one, changing only the rows whose keys
match and leaving every other row alone. See
{doc}`Writing to a database </integrations/databases/writing>` for what each mode does, which
backend serves it, and the transaction it runs in.

ORC stores timestamps as nanoseconds and nothing else, which bounds the instants it can
hold to roughly 1677-09-21 through 2262-04-11. Writing a timestamp outside that range
raises a `SchemaError` naming the column, rather than storing a different instant: the
underlying conversion wraps rather than failing, so `2300-01-01` would otherwise be read
back as `1715-06-13` from a file that reports success and is perfectly valid. Parquet,
Arrow, and Avro hold the full range, so a column with far-future sentinels such as
`9999-12-31`, or with pre-1677 historical dates, belongs in one of those.

Delta, Iceberg, and Hudi table writes (transactional append and merge/upsert) are
covered in {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`. The sinks that need an optional extra or
a live service are shown but not executed:

```python
# docs: skip
ds.write.msgpack("output/data.msgpack")
ds.write.avro("output/data.avro")
ds.write.snowflake(
    "ORDERS",
    connection_kwargs={"account": "acct", "warehouse": "WH", "database": "DB"},
)
ds.write.clickhouse("analytics.orders", host="localhost", password="env:CH_PASSWORD")
ds.write.tfrecord("output/train.tfrecord")
```

`write.clickhouse` inserts into a table that already exists and never creates one, because a
ClickHouse table needs an engine and a sort key that the data does not determine. Its `mode`
defaults to `append`, and `overwrite` truncates the table first.

## Text, XML, and training-data formats

The text and XML writers are Spark's `DataFrameWriter.text` and `.xml`. `write.text` takes
exactly one string column and writes a null as an empty line, because a text file has no way
to say "no value". `write.xml` writes one `row_tag` element per row inside a `root_tag`
element, omits a null field, repeats a list's element once per item, and turns a field named
with a leading `_` into an attribute:

```python
books = bt.from_pydict({"_id": ["b1", "b2"], "title": ["Dune", None]})
xml_path = os.path.join(out_dir, "books.xml")
books.write.xml(xml_path, row_tag="book", root_tag="books", declaration="")
print(open(xml_path).read())
# <books>
# <book id="b1"><title>Dune</title></book>
# <book id="b2"></book>
# </books>

lines_path = os.path.join(out_dir, "titles.txt")
books.select("title").write.text(lines_path)
print(bt.read.text(lines_path).to_pydict()["text"])
# ['Dune', '']
```

The NumPy, WebDataset, and TFRecord writers are the inverses of `read.numpy`,
`read.webdataset`, and `read.tfrecord`, in the layouts Ray Data writes. `write.numpy` writes
one column, so name it with `column=` when there is more than one, and it refuses nulls
because a `.npy` array cannot hold them. `write.webdataset` needs a string `__key__` column
and writes every other column as one tar member per row. Bytes are written as they are, text
as UTF-8, and numbers as decimal text:

```python
import tarfile

samples = bt.from_pydict({"__key__": ["s0", "s1"], "txt": ["cat", "dog"], "cls": [0, 1]})
shard = os.path.join(out_dir, "shard.tar")
samples.write.webdataset(shard)
print(tarfile.open(shard).getnames())
# ['s0.txt', 's0.cls', 's1.txt', 's1.cls']
print(bt.read.webdataset(shard).to_pydict()["cls"])
# [b'0', b'1']

npy_path = os.path.join(out_dir, "labels.npy")
samples.write.numpy(npy_path, column="cls")
print(bt.read.numpy(npy_path).to_pydict())
# {'data': [0, 1]}
```

`write.tfrecord` encodes each row as a `tf.train.Example`. Integer and boolean columns become
`Int64List` features, float columns `FloatList`, which is float32 by definition, and string
and binary columns `BytesList`. `record_format="raw"` writes one binary column's values as the
record payloads instead, which is what `read.tfrecord` returns. Every record carries the
checksum TensorFlow verifies, so the writer needs `google-crc32c` installed.

The text, XML, WebDataset, TFRecord, and IPC stream writers encode one batch at a time, so a
streaming write of a large result holds a single batch. A `.npy` header states its row count
before the data, so `write.numpy` encodes each file whole. Bound a large NumPy write with
`max_rows_per_file`.

## Partitioned output

`partition_by` writes one subdirectory per distinct value of the named columns, in
Hive style (`category=a/`, `category=b/`). A reader can then prune whole partitions
when a filter matches the partition key.

```python
# docs: skip
ds.write.parquet("output/events", partition_by=["category"])
```

`partition_by` is also accepted by the generic `write`:

```python
# docs: skip
ds.write("output/events", format="parquet", partition_by=["category"])
```

The partition columns live in the directory names rather than inside the files, so they
cost no bytes. Reading the directory back gives them to you as ordinary columns, and a
filter on one of them prunes whole directories before any file is opened.

Partition on a column you actually filter on, and one whose values are few. The whole
trade is a directory per distinct value in exchange for skipping directories, and it
inverts once the values are many and small: the write becomes one request per directory,
and every later query pays a listing over all of them to read a few rows each. Batcher
warns when a write is about to create more than 10,000 directories, which is well past any
layout chosen on purpose. Partition by a date rather than a timestamp, or a hash bucket
rather than an id, and use `sort_by` where you want finer skipping than a directory.

```python
part_dir = tempfile.mkdtemp()
ds.write.parquet(part_dir, partition_by=["category"])
back = bt.read.parquet(part_dir)
print(sorted(back.columns))
# ['category', 'price']
print(back.filter(bt.col("category") == "a").count())
# 2
```

### Reloading one partition

`mode="overwrite"` replaces the whole output, which is rarely what a daily reload wants:
writing one day into a table holding five years of them deletes the other five years, at
full speed and without complaint. `mode="overwrite_partitions"` replaces only the
partitions the incoming data covers and leaves the rest exactly as they were. It is
Spark's `partitionOverwriteMode="dynamic"` and Hive's `INSERT OVERWRITE`, and both of
those spellings are accepted.

The two modes differ only in what happens to the partitions the new rows never mention. The picture uses the same data as the example that follows:

![A partitioned output directory under the two overwrite modes. Before the write, out/ holds a _SUCCESS marker and three partition directories, dt=a, dt=b and dt=c, each with one part-00000 file holding v=1, v=2 and v=3. The new rows mention only dt=b, with v=99, and are written with write.parquet(out/, partition_by=["dt"]). With mode="overwrite" the whole output is replaced: dt=b holds v=99, and dt=a and dt=c are deleted. With mode="overwrite_partitions" only dt=b is replaced with v=99, while dt=a with v=1 and dt=c with v=3 are kept. overwrite_partitions needs partition_by. The _SUCCESS marker is written last, and readers skip it.](/_static/diagrams/partitioned_write_modes.svg)

```python
reload_dir = tempfile.mkdtemp()
_ = bt.from_pydict({"dt": ["a", "b", "c"], "v": [1, 2, 3]}).write.parquet(
    reload_dir, partition_by=["dt"]
)
_ = bt.from_pydict({"dt": ["b"], "v": [99]}).write.parquet(
    reload_dir, partition_by=["dt"], mode="overwrite_partitions"
)
print(bt.read.parquet(reload_dir).sort("dt").to_pydict())
# {'v': [1, 99, 3], 'dt': ['a', 'b', 'c']}
```

The mode needs `partition_by`, since without partitioning there is nothing to scope the
replacement to. On a transactional table the same intent is a scoped commit rather than a
file rewrite, so use `replace_where` there instead.

### Partition transforms

A partition key may be an expression rather than a column name, which is how you
partition by something the table does not store: the year of a timestamp, a hash bucket
of an id. This is Iceberg's `days(ts)` and `bucket(16, id)`, and Spark's generated
partition column. The expression is evaluated once and its alias becomes the directory
name, so nothing is added to the data itself.

```python
import glob

events = bt.from_pydict(
    {"ts": ["2024-01-05", "2024-02-09", "2025-03-01"], "amount": [10, 20, 30]}
).with_columns(ts=bt.col("ts").cast("date"))

year_dir = tempfile.mkdtemp()
events.write.parquet(year_dir, partition_by=[bt.col("ts").dt.year().alias("year")])
print(sorted(os.path.basename(p) for p in glob.glob(os.path.join(year_dir, "year=*"))))
# ['year=2024', 'year=2025']
```

An expression key must carry an `alias`, because the alias is the directory name. A hash
bucket is the same shape:

```python
# docs: skip
ds.write.parquet("output/by_bucket", partition_by=[(bt.col("id") % 16).alias("bucket")])
```

## Size the output files

A write's file layout is a separate decision from its directory layout. Three options
express it, and you pass one:

| Option | What it does |
|---|---|
| `write(..., max_rows_per_file=N)` | Caps each file at `N` rows. |
| `repartition(num_files=N)` | Splits the output into exactly `N` files. |
| `repartition(target_size_mb=M)` | Sizes each file to about `M` megabytes. |

`max_rows_per_file` is the direct form when you know the row count you want. The two
`repartition` options are resolved against the data's measured size, so you can ask for a
file size without knowing the rows per megabyte:

```python
size_dir = tempfile.mkdtemp()
rows = bt.from_pydict({"x": list(range(1000))})
manifest = rows.repartition(num_files=4).write(size_dir, format="parquet")
print(manifest.num_files)
# 4
```

`target_size_mb` sizes against the in-memory Arrow footprint, so a compressed format
lands somewhat under the target rather than over it. That is the safe direction: the
alternative is guessing a compression ratio that changes per column and per codec.

`max_rows_per_file` over a file source *streams*: the writer rolls over to a new file as
each one fills, so the driver holds one batch however large the result is. The other two
need the whole result, since both are computed from its total size. So a write too large
for one machine can still be cut into files of a size you choose.

All three apply on the distributed path as well. There the layout travels to the worker
and is resolved against the shard that worker holds, because a streaming distributed
write never materializes its result on the driver. `num_files` stays a total across the
whole write, so its budget is divided among the shards rather than applied to each one.

## Confirm a write finished

Every data file is published atomically, so a reader never sees a half-written one. That
does not make a *directory* safe on its own: a run that dies partway leaves a directory of
perfectly valid files, and reading it back returns however many exist, silently short.

So a directory write ends by publishing an empty `_SUCCESS` marker, and only after every
shard's files are accounted for. Its presence means the write finished. Readers skip
`_`-prefixed files, so it never joins the data.

```python
marker_dir = tempfile.mkdtemp()
_ = bt.from_pydict({"x": [1, 2, 3]}).write.parquet(marker_dir, max_rows_per_file=1)
print(os.path.exists(os.path.join(marker_dir, "_SUCCESS")))
# True
```

Pass `require_success=True` on the read to act on it. It is off by default, because a
directory Batcher did not write has no marker and is not thereby incomplete. Turn it on
for a path some other job produces, which is where a half-written directory is a real risk
and nothing else would tell you.

```python
# docs: skip
ds = bt.read.parquet("s3://bucket/upstream-export/", require_success=True)
```

## Compact small files

Incremental or streaming writes leave many tiny part files, which slow later reads. This
is the small-files problem, and `compact` fixes a dataset in place. It reads `path`,
repartitions to a target file size or to an exact `num_files`, writes the result back,
and deletes the now-stale parts. It runs on local files, so it executes here:

```python
import glob

comp_dir = tempfile.mkdtemp()
_ = bt.from_pydict({"x": [1, 2, 3, 4]}).repartition(num_files=2).write(comp_dir, format="parquet")
_ = bt.compact(comp_dir, num_files=1, format="parquet")
print(len(glob.glob(os.path.join(comp_dir, "*.parquet"))))
# 1
```

Compaction changes a table's file sizes, not how it is organized, so an existing Hive
layout is carried forward: a directory already partitioned by `dt=` comes back
partitioned by `dt=`, with fewer files inside each partition. Pass `by=` to repartition
it differently, and the directories of the previous scheme are removed rather than left
standing empty beside the new ones.

```python
part_comp = tempfile.mkdtemp()
_ = bt.from_pydict({"dt": ["x", "x", "y"], "v": [1, 2, 3]}).write.parquet(
    part_comp, partition_by=["dt"], max_rows_per_file=1
)
_ = bt.compact(part_comp, num_files=1, format="parquet")
print(sorted(os.path.basename(p) for p in glob.glob(os.path.join(part_comp, "dt=*"))))
# ['dt=x', 'dt=y']
print(len(glob.glob(os.path.join(part_comp, "dt=*", "*.parquet"))))
# 2
```

## Distributed writes

For large outputs, write across workers. Each worker writes its own files into the
target directory, and the returned `WriteManifest` lists what was produced.

```python
# docs: skip
manifest = ds.write("s3://bucket/events", format="parquet", distributed=True, num_workers=8)
```

The distributed path uses the same mergeable execution as a single-node write, so
the output is identical in content. Distribution changes only how the work is
scheduled.

### The result does not pass through the driver

An unpartitioned distributed write never assembles the result anywhere. A breaker-free
plan (`read` → `filter` → `write`) runs on each worker and that worker writes what it
produced. A plan **with** a breaker (a `group_by`, a join, a sort, a `distinct`, a window)
keeps each reducer's bucket where it was computed and the workers write those buckets, so
only the file locators travel back. Either way the driver holds a list of paths, never
rows, and a result far larger than any one machine is writable.

That matters most for the operators whose output is the size of their input. An aggregate
reduces, so its result is a summary. A sort and a window emit one row per input row, and a
join can emit more rows than either side, so for those an assembled result is the entire
relation in one process.

### Partitioned writes and the file count

Combining `partition_by` with a distributed write raises a file-count question, and the
answer depends on the shape of the plan:

- A plan with a breaker (a `group_by`, a join, a sort) is assembled on the driver and its
  shards are then cut **by partition key** rather than by row range. Each key's rows land
  wholly inside one shard, so the write emits one file per partition rather than one per
  partition per worker. Keys are packed largest-first, so a single hot partition does not
  become one overloaded worker while the rest idle. This is the one case where the result
  does reach the driver, and it is a deliberate trade: the alternative scatters every
  partition key across every worker, and the small files that produces are what makes the
  next query slow.
- A breaker-free plan (`read` → `filter` → `write`) streams straight from each worker to
  storage and never reaches the driver at all, which is what keeps a result larger than
  one node writable. Each worker therefore writes its own file in each partition it sees,
  so the count is workers times partitions. This is what Spark's `partitionBy` does too.

When that file count matters more than the streaming, pass `sort_by=` the partition
columns. The sort is a breaker, so the write takes the first path and lands one file per
partition:

```python
# docs: skip
ds.write.parquet("s3://bucket/events", partition_by=["dt"], sort_by=["dt"], distributed=True)
```

A single key can be passed on its own, without the brackets, the same way `partition_by`
takes one:

```python
sorted_dir = os.path.join(tempfile.mkdtemp(), "by-price")
_ = ds.write.parquet(sorted_dir, sort_by="price")
print(bt.read.parquet(sorted_dir).to_pydict()["price"])
# [10.0, 20.0, 30.0]
```

`resume=True` also reaches the workers, which is where it matters most: a run that lost
workers to preemption re-runs only the shards that never finished, and leaves the files
that did. Resume identifies finished work by file position, so it is exactly-once only
for a deterministic plan. See {py:obj}`ds.write <batcher.Dataset.write>` for the
precondition in full.

## See also

- {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`: transactional Delta writes, merge/upsert,
  slowly-changing dimensions.
- {doc}`Data quality </user-guide/trust/data-quality>`: validate and quarantine before you write.
- {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: write to an object store.
- {doc}`/user-guide/moving-data/streaming`: the same `ds.write` with a trigger, as a continuous query.
- {doc}`IO API </api/relational/io>`: the full {py:obj}`ds.write <batcher.Dataset.write>` writer reference.
- {doc}`/cookbook/io/save_modes`: save modes and write manifests, as a runnable script.
