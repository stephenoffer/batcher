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

## What you get back

Every constructor returns a lazy `Dataset`. Inspect the column names with the
`columns` property; nothing is read until a terminal operation runs.

```python
people = bt.from_pydict({"id": [1, 2], "name": ["alice", "bob"]})
print(people.columns)
# ['id', 'name']
```

## Next steps

- [Transformations](transformations.md): reshape and derive columns.
- [Filtering](filtering.md): select rows and remove duplicates.
