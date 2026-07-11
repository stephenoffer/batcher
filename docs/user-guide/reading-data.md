# Reading data

A pipeline starts by building a `Dataset` from a source. Sources fall into two
groups: in-memory constructors that wrap data already in the process, and file or
path readers that load from disk or object storage. Every constructor is lazy and
returns a `Dataset`.

## In-memory constructors

### From a column dict

`from_pydict` takes a column-oriented dictionary. This is the constructor used
throughout the docs because it needs no files.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"],
        "value": [100, 200, 300],
    }
)
print(ds.to_pydict())
# {'id': [1, 2, 3], 'name': ['alice', 'bob', 'carol'], 'value': [100, 200, 300]}
```

### From Arrow

`from_arrow` wraps a `pyarrow.Table`, a `RecordBatch`, or a list of batches with
no copy of the underlying buffers.

```python
import pyarrow as pa

table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
ds = bt.from_arrow(table)
print(ds.to_pydict())
# {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
```

### From a streaming factory

`from_batches` builds a streaming source from a callable that returns a fresh
iterator of Arrow batches each time it is called, plus the schema those batches
follow.

```python
schema = pa.schema([("n", pa.int64())])


def make_batches():
    for start in (0, 3):
        yield pa.record_batch({"n": [start, start + 1, start + 2]}, schema=schema)


ds = bt.from_batches(make_batches, schema)
print(ds.to_pydict())
# {'n': [0, 1, 2, 3, 4, 5]}
```

### From items and generators

`from_items` builds a `Dataset` from a Python list — one row per item, Ray Data
style (a dict item expands to columns, a scalar becomes a single `item` column).
`date_range` generates a calendar dimension, the date-typed sibling of `range`.

```python
print(bt.from_items([1, 2, 3]).to_pydict())
# {'item': [1, 2, 3]}
print(bt.date_range("2024-01-01", "2024-01-03").count())
# 3
```

### From other frameworks

Adapters convert a frame from another library into a `Dataset`:
`from_pandas`, `from_polars`, `from_numpy`, `from_spark`, `from_dask`,
`from_huggingface`, `from_torch`, and `from_tf`. They require the corresponding
library to be installed.

```python
# docs: skip
import pandas as pd

ds = bt.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
```

## File and path readers

File readers load from local paths, glob patterns, and object-store URLs. They
need real files, so the examples below are shown but not executed here.

{py:obj}`bt.read(path, format=None, **opts) <batcher.read>` detects the format from the path when
`format` is omitted. Format-specific helpers (`read.parquet`, `read.csv`,
`read.json`, `read.table`) accept the same path and option style.

```python
# docs: skip
ds = bt.read("data/events.parquet")          # format inferred from extension
ds = bt.read("data/*.parquet")               # glob across many files
ds = bt.read("s3://bucket/events.parquet")   # object storage (needs [cloud])
```

```python
# docs: skip
ds = bt.read.parquet("data/events.parquet")
ds = bt.read.csv("data/events.csv")
ds = bt.read.json("data/events.jsonl")
```

Many more readers exist for columnar, table, and multimodal formats, including
`read.orc`, `read.arrow`, `read.avro`, `read.lance`, `read.delta`, `read.iceberg`,
`read.hudi`, `read.sql`, `read.snowflake`, `read.bigquery`, `read.kafka`,
`read.images`, `read.audio`, and `read.video`. Each takes a path or connection
plus format-specific options.

```python
# docs: skip
ds = bt.read.delta("s3://lake/events")
frames = bt.read.images("s3://bucket/photos/*.jpg")
```

## Databases, warehouses, and specialized formats

Beyond the file formats above, `bt.read` reaches databases, warehouses, and a set
of scientific/columnar container formats through the same namespace. They all share
one shape: `bt.read.<name>(path_or_uri, **opts)` returns a lazy `Dataset`, and
nothing is fetched until a terminal op runs. The database connectors (`mongo`,
`cassandra`, `dynamodb`, `elasticsearch`) take their connection as keyword options
rather than a path.

| Reader | Reads | Needs |
| --- | --- | --- |
| `read.parquet_dataset(dir)` | A Hive-partitioned Parquet directory, partition columns recovered from the layout | — |
| `read.webdataset(path)` | WebDataset `.tar` shards, one row per sample | — |
| `read.excel(path)` | Excel workbook(s) via python-calamine | `[excel]` |
| `read.hdf5(path)` | HDF5 file(s), datasets as columns | `[hdf5]` |
| `read.zarr(path)` | A Zarr store of chunked n-dimensional arrays | `[zarr]` |
| `read.delta_sharing(url)` | A Delta Sharing `<profile>#<share>.<schema>.<table>` | `[delta-sharing]` |
| `read.clickhouse(query)` | A ClickHouse query result over the Arrow-native interface | a running ClickHouse |
| `read.databricks(table)` | A Databricks/Unity Catalog table (credential vending) | a Databricks workspace |
| `read.mongo(...)` | A MongoDB collection via pymongoarrow | a running MongoDB |
| `read.cassandra(...)` | A Cassandra/Scylla table, fanned out across token ranges | a running Cassandra |
| `read.dynamodb(...)` | A DynamoDB table via native parallel scan segments | AWS DynamoDB |
| `read.elasticsearch(...)` | An Elasticsearch index via ES\|QL Arrow output | a running Elasticsearch |

Each connector needs its service reachable (or its optional extra installed), so
these are shown but not executed:

```python
# docs: skip
sales = bt.read.clickhouse("SELECT * FROM sales", host="localhost")
orders = bt.read.databricks("main.sales.orders")
events = bt.read.mongo(uri="mongodb://localhost:27017", database="app", collection="events")
shared = bt.read.delta_sharing("config.share#share.schema.table")
grids = bt.read.zarr("s3://bucket/array.zarr")
```

Partitioned Parquet needs only local files, so it runs end to end. `parquet_dataset`
recovers the partition columns from the Hive directory layout and prunes whole
partitions when a filter matches the partition key:

```python
import os
import tempfile
import pyarrow.parquet as pq

root = os.path.join(tempfile.mkdtemp(), "events")
pq.write_to_dataset(
    pa.table({"value": [1, 2], "day": ["mon", "tue"]}),
    root,
    partition_cols=["day"],
)
ds = bt.read.parquet_dataset(root)
print(ds.select("value").sort("value").to_pydict())
# {'value': [1, 2]}
```

## What you get back

Every constructor returns a lazy `Dataset`. Inspect the column names with the
`columns` property; nothing is read until a terminal operation runs.

```python
people = bt.from_pydict({"id": [1, 2], "name": ["alice", "bob"]})
print(people.columns)
# ['id', 'name']
```

The reads above run on the compiled Rust data plane. `engine_version` reports which
engine build is loaded, distinct from the Python package version:

```python
print(isinstance(bt.engine_version(), str))
# True
```

## Next steps

- [Transformations](transformations.md): reshape and derive columns.
- [Filtering](filtering.md): select rows and remove duplicates.
- [Lakehouse tables](lakehouse.md): read Delta/Iceberg tables and time-travel.
- [Data quality](data-quality.md): validate inputs as they arrive.
- [IO API](../api/io.md): the full `bt.read` reader reference.
