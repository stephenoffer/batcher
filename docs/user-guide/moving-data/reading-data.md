# Reading data

A pipeline starts by building a {py:class}`Dataset <batcher.Dataset>` from a source. Sources come in two groups:
in-memory constructors, which wrap data already in the process, and path readers,
which load from disk or object storage. Both are lazy.

## In-memory constructors

These wrap data the process already holds, so they need no files and no credentials. They
are what the rest of the documentation uses for its runnable examples.

### From a column dict

{py:func}`from_pydict <batcher.from_pydict>` takes a column-oriented dictionary. This is the constructor used
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

### From arrow

{py:func}`from_arrow <batcher.from_arrow>` wraps a `pyarrow.Table`, a `RecordBatch`, or a list of batches with
no copy of the underlying buffers.

```python
import pyarrow as pa

table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
ds = bt.from_arrow(table)
print(ds.to_pydict())
# {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
```

### From a streaming factory

{py:func}`from_batches <batcher.from_batches>` builds a streaming source from a callable that returns a fresh
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

{py:func}`from_items <batcher.from_items>` builds a `Dataset` from a Python list, one row per item.
A dict item expands to columns, and a scalar becomes a single `item` column. `date_range`
generates a calendar dimension, the date-typed sibling of `range`.

```python
print(bt.from_items([1, 2, 3]).to_pydict())
# {'item': [1, 2, 3]}
print(bt.date_range("2024-01-01", "2024-01-03").count())
# 3
```

### Python values Arrow cannot type

Every column crossing into the engine has to be an Arrow type. Numbers, strings, bytes,
lists, dicts, dates, timestamps, decimals, and NumPy arrays all convert. A few everyday
Python types do not, and each has a one-line answer:

| Value | Pass instead |
| --- | --- |
| `uuid.UUID` | `str(u)` for a text column, or `u.bytes` for 16-byte binary |
| An `enum.Enum` member | its `.value` |
| `pathlib.Path` | `str(path)` |
| A PIL `Image` | `np.asarray(img)`, or keep the encoded bytes |
| A torch `Tensor` | `tensor.cpu().numpy()` |

Anything else raises a {py:class}`PlanError <batcher.PlanError>` naming the column and what
it holds. Nothing is silently pickled into an object column: an opaque Python object costs
10 to 100 times more on every transfer downstream, and a failure you can read beats a
slowdown you have to go looking for.

```python
import uuid

try:
    bt.from_pydict({"id": [uuid.uuid4()]})
except bt.PlanError as err:
    print("id" in str(err))
# True
```

### From other frameworks

Adapters convert a frame from another library into a `Dataset`:
{py:func}`from_pandas <batcher.from_pandas>`, {py:func}`from_polars <batcher.from_polars>`, {py:func}`from_numpy <batcher.from_numpy>`, {py:func}`from_spark <batcher.from_spark>`, {py:func}`from_dask <batcher.from_dask>`,
{py:func}`from_huggingface <batcher.from_huggingface>`, {py:func}`from_torch <batcher.from_torch>`, and {py:func}`from_tf <batcher.from_tf>`. They require the corresponding
library to be installed.

```python
# docs: skip
import pandas as pd

ds = bt.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
```

## File and path readers

File readers take a local path, a glob pattern, or an object-store URL. They need
real files, so the examples below are shown but not executed here.

{py:obj}`bt.read(path, format=None, **opts) <batcher.read>` detects the format from the path when
`format` is omitted. The format-specific helpers `read.parquet`, `read.csv`, `read.json`,
and `read.table` accept the same path and option style.

```python
# docs: skip
ds = bt.read("data/events.parquet")          # format inferred from extension
ds = bt.read("data/*.parquet")               # glob across many files
ds = bt.read("output/events/")               # a directory: inferred from the files in it
ds = bt.read("s3://bucket/events.parquet")   # object storage (needs [cloud])
```

A directory has no extension of its own, so the format comes from the files inside it.
That is what lets a sharded or partitioned output read back without naming a format you
never chose on the way in. A directory holding two data formats is not one relation, so
detection declines there and asks for `format=` rather than reading half the data.

Pass a **list** when the inputs share no useful glob. Each entry may be a file, a
directory, or a glob, so unioning two output directories is one call, and a file matched
by more than one entry is still read once.

```python
# docs: skip
ds = bt.read.parquet(["runs/2024-01/", "runs/2024-02/"])   # two outputs, one relation
ds = bt.read.parquet(["a/events.parquet", "b/*.parquet"])  # mixed spellings
```

```python
# docs: skip
ds = bt.read.parquet("data/events.parquet")
ds = bt.read.csv("data/events.csv")
ds = bt.read.json("data/events.jsonl")
```

Many more readers cover the columnar and table formats, and the multimodal ones:
`read.orc`, `read.arrow`, `read.avro`, `read.fasta`, `read.fastq`, `read.bed`, `read.gff`, `read.vcf`, `read.lance`, `read.delta`, `read.iceberg`,
`read.hudi`, `read.sql`, `read.snowflake`, `read.bigquery`, `read.kafka`,
`read.images`, `read.audio`, and `read.video`. Each takes a path or connection plus
format-specific options.

```python
# docs: skip
ds = bt.read.delta("s3://lake/events")
frames = bt.read.images("s3://bucket/photos/*.jpg")
```

## Messy input

Real corpora contain members that will not read. Batcher separates two failures that look
alike and have opposite fixes, so reaching for the wrong flag cannot quietly delete data.

An **unreadable file** is one whose bytes the format cannot parse at all: a truncated
upload, a zero-byte object, a JPEG whose trailer never arrived. `on_error="skip"` drops the
file and reads the rest, and `corrupt_files()` lists what went. Use it when the input is a
corpus you do not control.

```python
# docs: skip
ds = bt.read.parquet("s3://bucket/events/", on_error="skip")
ds.to_pydict()
print(ds.source.corrupt_files())
```

A **third** failure is neither of those: bytes that are readable but not in the encoding
you asked for. A text corpus assembled from scrapes, exports and legacy systems is a
mixture, and a single stray byte is not a reason to lose a file. `read.text` replaces what
it cannot decode with U+FFFD by default, in both `mode="line"` and `mode="file"`, and
`errors="strict"` turns it into a per-file failure that `on_error="skip"` will then drop:

```python
import os
import tempfile

import batcher as bt

d = tempfile.mkdtemp()
with open(os.path.join(d, "legacy.txt"), "wb") as f:
    _ = f.write("caf\xe9\n".encode("cp1252"))

print(bt.read.text(d).to_pydict()["text"])
print(bt.read.text(d, encoding="cp1252").to_pydict()["text"])
```

Replacement is a fallback, not an answer: if you know what the bytes are, naming the
`encoding` is the fix.

A **malformed row** is one record inside a file that is otherwise fine: a CSV row carrying
a field the header does not have, or an NDJSON line that is not JSON at all. The file is
readable, so `on_error` is the wrong answer for it — dropping the file discards every good
row to be rid of one bad line. Pass `on_bad_lines` instead, which drops the record.

```python
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "events.csv")
with open(path, "w") as f:
    f.write("id,amount\n1,10\n2,20,stray\n3,30\n")

print(bt.read.csv(path, on_bad_lines="skip").to_pydict())
# {'id': [1, 3], 'amount': [10, 30]}
```

`read.json` takes the same flag, for the same reason and with the same three values.

```python
jsonl = os.path.join(tempfile.mkdtemp(), "events.jsonl")
with open(jsonl, "w") as f:
    f.write('{"id": 1}\n<html>gateway timeout</html>\n{"id": 3}\n')

print(bt.read.json(jsonl, on_bad_lines="skip").to_pydict())
# {'id': [1, 3]}
```

`on_bad_lines` takes `"error"` (the default, which refuses the read), `"warn"` (drop the
row and log it with the offending text), or `"skip"` (drop it silently). Dropped rows are
counted on the metrics export as `malformed_rows_total`, separately from the
`skipped_total` that counts whole files, because a total mixing rows with files answers
neither question.

Coming from another engine, the spellings map as follows.

| Their option | Batcher |
|---|---|
| Spark `mode="FAILFAST"` | `on_bad_lines="error"` (the default) |
| Spark `mode="DROPMALFORMED"` | `on_bad_lines="skip"` |
| Spark `mode="PERMISSIVE"` | no equivalent; Batcher has no corrupt-record column |
| pandas `on_bad_lines=` | the same name and the same three values |
| Polars `ignore_errors=True` | `on_bad_lines="skip"`, plus `schema=` if what you want is an unconvertible value to survive as text |

A value that will not convert to its column's type is a third thing again, and neither flag
touches it. The schema comes from the file's first block, so a column that is integral for
a million rows and then holds `"N/A"` is inference having been shown too little. Declare the
type with `schema=` rather than tolerating the row. `on_bad_lines` deliberately refuses to
delete such a record: dropping it would remove the very rows that were about to tell you
the inferred type is wrong.

## Databases, warehouses, and specialized formats

The same `bt.read` namespace reaches databases, warehouses and the scientific container
formats. They are covered on their own page, because there are enough of them to be a
reference rather than a walkthrough: {doc}`/user-guide/moving-data/reading-databases`.

## What you get back

Every constructor hands back a lazy `Dataset`. Inspect the column names with the
`columns` property. Nothing is read until a terminal operation runs.

```python
people = bt.from_pydict({"id": [1, 2], "name": ["alice", "bob"]})
print(people.columns)
# ['id', 'name']
```

The reads above run on the compiled Rust data plane. {py:func}`engine_version <batcher.engine_version>` reports which
engine build is loaded, distinct from the Python package version:

```python
print(isinstance(bt.engine_version(), str))
# True
```

## See also

- {doc}`Transformations </user-guide/transform/rows/transformations>`: reshape and derive columns.
- {doc}`Filtering </user-guide/transform/rows/filtering>`: select rows, drop duplicates.
- {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`: read Delta and Iceberg tables, and travel back
  through their versions.
- {doc}`Data quality </user-guide/trust/data-quality>`: validate inputs as they arrive.
- {doc}`IO API </api/relational/io>`: the full `bt.read` reader reference.
- {doc}`Agent skills </agents>`: `read-and-write-data` covers picking a reader or
  sink, cloud paths, globs, schema evolution, and error tolerance.
- {doc}`/cookbook/io/index`: 6 runnable recipes for readers, writers, and the registries.
