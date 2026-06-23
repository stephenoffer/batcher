# Writing data

A Dataset is written to disk with a terminal write operation. Writing executes the
plan and streams Arrow batches to the sink, so the same memory bounds and spill
behavior that govern `collect` apply here too.

The format-specific helpers are `write.parquet`, `write.csv`, and `write.json`. The
generic `write(path, fmt=...)` covers all of them and is the place to pass a
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
end to end. Parquet is the recommended format: columnar, compressed, and pushdown
friendly.

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
ds.write("output/events", fmt="parquet", partition_by=["category"])
```

## Distributed writes

For large outputs, write across workers. Each worker writes its own files into the
target directory; the returned `WriteManifest` lists what was produced.

```python
# docs: skip
manifest = ds.write("s3://bucket/events", fmt="parquet", distributed=True, num_workers=8)
```

The distributed path uses the same mergeable execution as a single-node write, so
the output is identical in content. Distribution changes only how the work is
scheduled.
