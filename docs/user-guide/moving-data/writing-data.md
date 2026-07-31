# Writing data

A Dataset is written to disk with a terminal write operation. Writing executes the
plan and streams Arrow batches to the sink, so the same memory bounds and spill
behavior that govern `collect` apply here too.

The format-specific helpers are `write.parquet`, `write.csv`, and `write.json`. The
generic `write(path, format=...)` covers all of them and is the place to pass a
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

```python
# docs: skip
ds.write.csv("output/data.csv")
ds.write.json("output/data.json")
```

Use CSV and JSON for interchange with tools that require them. Parquet is faster and
preserves types, so prefer it for anything that will be read back by Batcher.

## Other file and database sinks

The write namespace mirrors the reader namespace: every format has a typed writer,
and the generic `write(path, format=...)` reaches them all, each returning a
`WriteManifest`. Beyond Parquet, CSV, and JSON:

| Writer | Writes | Needs |
| --- | --- | --- |
| `write.orc(path)` | ORC files | nothing extra |
| `write.arrow(path)` | Arrow/Feather IPC files | nothing extra |
| `write.avro(path)` | Avro files | `[avro]` |
| `write.lance(path)` | A Lance dataset (columnar ML format) | `[lance]` |
| `write.msgpack(path)` | MessagePack files | `[msgpack]` |
| `write.sql(table, driver=..., db_kwargs=...)` | A database table via ADBC/FlightSQL | driver + reachable DB |
| `write.snowflake(table, connection_kwargs=...)` | A Snowflake table | Snowflake account |
| `write.mongo(collection, ...)` | A MongoDB collection | a running MongoDB |

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
```

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

## Compacting small files

Incremental or streaming writes leave many tiny part files, which slow later reads. This
is the small-files problem, and `compact` fixes a dataset in place. It reads `path`,
repartitions to a target file size or to an exact `num_files`, writes the result back,
and deletes the now-stale parts. It runs on local files, so it executes here:

```python
import glob

comp_dir = tempfile.mkdtemp()
_ = bt.from_pydict({"x": [1, 2, 3, 4]}).repartition(num_files=2).write(
    comp_dir, format="parquet"
)
_ = bt.compact(comp_dir, num_files=1, format="parquet")
print(len(glob.glob(os.path.join(comp_dir, "*.parquet"))))
# 1
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

## See also

- {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`: transactional Delta writes, merge/upsert,
  slowly-changing dimensions.
- {doc}`Data quality </user-guide/trust/data-quality>`: validate and quarantine before you write.
- {doc}`Cloud storage </user-guide/moving-data/cloud-storage>`: write to an object store.
- {doc}`IO API </api/relational/io>`: the full `ds.write` writer reference.
- {doc}`/cookbook/io/save_modes`: save modes and write manifests, as a runnable script.
