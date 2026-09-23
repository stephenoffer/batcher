# Reading data

This page covers building a {py:class}`Dataset <batcher.Dataset>` from a source, the first step of every pipeline. Sources come in two groups:
in-memory constructors, which wrap data already in the process, and path readers,
which load from disk or object storage. Both are lazy.

The first question is where the data sits right now, and the answer picks the family. The rows under each family then name the constructor or reader for the input you hold.

![Choosing a reader starts from where the data is. Data already in the Python process goes through the bt.from_* constructors: a column dict to from_pydict, an Arrow table or batches to from_arrow, a NumPy array to from_numpy, a pandas or Polars frame to from_pandas or from_polars, a list of Python items to from_items, and a factory that yields batches to from_batches. None of them needs files or credentials. Data at a path goes through bt.read: bt.read(path) when the extension names the format, bt.read on a directory holding one format, read.parquet or read.csv when you name the format yourself, read.delta or read.iceberg for a lakehouse table, read.images or read.video for media, and read.sql or read.snowflake for a database or warehouse. A directory holding two formats needs format= passed explicitly. Every answer returns a lazy Dataset, so nothing is read until a terminal operation such as collect runs.](/_static/diagrams/reader_choice.svg)

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
it holds. Nothing is silently pickled into an object column, because a failure you can read beats a
slowdown you have to go looking for.

```python
import uuid

try:
    bt.from_pydict({"id": [uuid.uuid4()]})
except bt.PlanError as err:
    print("id" in str(err))
# True
```

### From NumPy arrays

{py:func}`from_numpy <batcher.from_numpy>` reads an array's **leading axis as the row axis**, and the rank decides the
column type. A `{name: array}` mapping builds one column per array, and each one follows the same
rules, so an embedding table is one call.

```python
import numpy as np

ds = bt.from_numpy({"id": np.arange(3), "emb": np.zeros((3, 4))})
print(ds.schema.names, ds.count())
# ['id', 'emb'] 3
```

The rules, in the order they apply:

| Array | Column |
| --- | --- |
| 1-D | a scalar column of that dtype |
| `(n, dim)` | a fixed-size-list column of width `dim`, the embedding convention |
| `(n, *shape)`, rank 3 or more | a fixed-shape-tensor column keeping the per-row shape |
| Structured (a compound dtype) | one column per field, each following the rules above |

A structured array is NumPy's own table, so it becomes a table. This is what `np.genfromtxt`,
`np.rec.array`, and an h5py compound dataset produce, and it is the reading `pandas.DataFrame`
gives them too.

```python
rows = np.array([(1, 2.5), (3, 4.5)], dtype=[("id", "i8"), ("score", "f8")])
print(bt.from_numpy(rows).to_pydict())
# {'id': [1, 3], 'score': [2.5, 4.5]}
```

A masked array keeps its mask: a masked value becomes a null, not the fill sitting under it.

```python
print(bt.from_numpy(np.ma.array([1, 2, 3], mask=[False, True, False])).to_pydict())
# {'data': [1, None, 3]}
```

Two shapes have no Arrow column form and are refused rather than approximated. A complex array
has no Arrow type, so split it into two real columns (`{"re": a.real, "im": a.imag}`). A 0-d array
has no row axis at all, so give it one with `np.atleast_1d`.

### From other frameworks

Adapters convert a frame from another library into a `Dataset`:
{py:func}`from_pandas <batcher.from_pandas>`, {py:func}`from_polars <batcher.from_polars>`, {py:func}`from_spark <batcher.from_spark>`, {py:func}`from_daft <batcher.from_daft>`, {py:func}`from_dask <batcher.from_dask>`,
{py:func}`from_huggingface <batcher.from_huggingface>`, {py:func}`from_torch <batcher.from_torch>`, and {py:func}`from_tf <batcher.from_tf>`. They require the corresponding
library to be installed.

Three of them stream rather than collect, so the other engine's result never sits in memory as one table. A Polars `LazyFrame` runs under Polars' `collect_batches` and arrives chunk by chunk. A Daft `DataFrame` arrives through `to_arrow_iter`. A Spark `DataFrame` on PySpark 4.1 or later arrives one partition at a time through its Arrow stream, and on earlier releases it's collected through `toArrow` or `toPandas`. Each of these re-runs the other engine's query whenever the `Dataset` executes. Polars strings arrive as `large_string` rather than `string_view`, which Batcher has no kernels for.

The return legs are {py:meth}`ds.to_polars() <batcher.Dataset.to_polars>`, {py:meth}`ds.to_daft() <batcher.Dataset.to_daft>`, and {py:meth}`ds.to_spark(spark) <batcher.Dataset.to_spark>`. `to_spark` takes the session explicitly. It hands a result of up to 64 MiB to `spark.createDataFrame` as one Arrow table, and stages a larger one as Parquet under `staging_path` for `spark.read.parquet`. On a cluster, point `staging_path` at storage every executor can read.

```python
# docs: skip
import pandas as pd

ds = bt.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
```

When you don't want to name the library, {py:func}`from_any <batcher.from_any>` dispatches on the object's type. It also accepts an object that exports only the DataFrame interchange protocol (`__dataframe__`), the one Polars' `from_dataframe` consumes, and converts it through `pyarrow.interchange`.

{py:func}`from_torch <batcher.from_torch>` takes a tensor, a `{name: tensor}` mapping, a tuple of tensors, or a map-style
`Dataset`, and applies the same rank rules as `from_numpy`. See {doc}`PyTorch </integrations/compute/pytorch>` for
the loader on the way back out, and for how `bfloat16` and the `float8` dtypes are handled.

## File and path readers

File readers take a local path, a glob pattern, or an object-store URL. They need
real files, so the examples below are shown but not executed here.

{py:obj}`bt.read(path, format=None, **opts) <batcher.read>` detects the format from the path when
`format` is omitted. The format-specific helpers `read.parquet`, `read.csv`, `read.json`,
and `read.table` accept the same path and option style.

```python
# docs: skip
ds = bt.read("data/events.parquet")  # format inferred from extension
ds = bt.read("data/*.parquet")  # glob across many files
ds = bt.read("output/events/")  # a directory: inferred from the files in it
ds = bt.read("s3://bucket/events.parquet")  # object storage (needs [cloud])
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
ds = bt.read.parquet(["runs/2024-01/", "runs/2024-02/"])  # two outputs, one relation
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

`read.arrow` reads both Arrow IPC layouts. A file with the IPC file footer splits by record
batch, and a footer-less IPC stream, such as one from Polars' `write_ipc_stream`, is read front
to back as one split.

`read.binary` doubles as a file inventory. A scan that projects only `uri` and `size` answers
from the file listing and never opens a file, which is the Batcher spelling of Daft's
`from_glob_path`:

```python
# docs: skip
inventory = bt.read.binary("s3://bucket/raw/*.jpg").select("uri", "size")
large = inventory.filter(bt.col("size") > 10_000_000)
```

`read.text` keeps blank lines by default. `skip_blank_lines=True` drops lines that are empty
or hold only whitespace, the way Daft's `read_text` does, and the kept rows keep their original
`line_number`.

## Files whose schemas differ

A directory written over months drifts: a column is added, a type widens, a column is dropped. Every file reader takes `schema_mode=`, which decides what one read of those files returns. The following table lists the three modes.

| `schema_mode` | Columns | Types | A file that disagrees |
|---|---|---|---|
| `"strict"` (the default) | the first file's | the first file's | An extra column is dropped, with a warning naming those the last file adds. A missing column, or a value that would change when cast to the first file's type, raises `bt.SchemaError` naming the file. |
| `"union"` | every file's, in first-seen order | the narrowest type that holds every file's values | A missing column reads as null. |
| `"latest"` | the last file's, in its order | the last file's | Older files are cast to it. A value that would change raises. |

"First" and "last" are path order, the order the reader lists the files in, not modification time. A strict cast has to leave every value unchanged: an `int32` file under an `int64` first file reads, while `2.5` under `int64`, `5` under `bool`, a timestamp with a time of day under `date32`, and a UTC timestamp under a naive one all raise. So does a null in a column the first file declares non-nullable.

```python
import os
import tempfile

import batcher as bt

drift = tempfile.mkdtemp()
bt.from_pydict({"id": [1, 2], "amount": [10, 20]}).write.parquet(os.path.join(drift, "p0.parquet"))
bt.from_pydict({"id": [3], "amount": [2.5], "channel": ["web"]}).write.parquet(
    os.path.join(drift, "p1.parquet")
)

try:
    bt.read.parquet(drift).collect(distributed=False)
except bt.SchemaError as error:
    print("p1.parquet" in str(error), "'amount' as double" in str(error))
# True True

print(bt.read.parquet(drift, schema_mode="union").sort("id").to_pydict())
# {'id': [1, 2, 3], 'amount': [10.0, 20.0, 2.5], 'channel': [None, None, 'web']}
```

`union` widens a column only where no value can change: integers to `int64`, a date into a timestamp, a timestamp to the finer unit, a struct to the union of its fields. The one exception is an integer column beside a floating one, which becomes `float64` as DuckDB's `union_by_name` does. A pair with no such type raises instead of guessing: an integer and a string, a naive timestamp and a timezone-aware one, binary and string. Names are case-sensitive, so `A` and `a` are two columns where DuckDB folds them into one. A `uint64` column beside an `int64` one reads as `int64`, the type the engine holds every integer in, and a value above `2**63 - 1` raises naming its file. DuckDB widens that pair to a 128-bit integer. {doc}`/cookbook/data-engineering/modeling/schema-evolution` has the full type table.

A distributed read answers exactly as a single-node one does in every mode: the same rows and column types, or the same `bt.SchemaError` about a file that breaks the contract. When several files do, which one is named depends on which is read first.

## CSV options

`read.csv` takes the pandas and Polars spellings of its options. The following table lists them by their canonical name, with the other spellings each one accepts.

| Option | Also spelled | Meaning |
|---|---|---|
| `delimiter` | `sep`, `separator` | The field separator. One character. |
| `quote_char` | `quotechar` | The quote character, `'"'` by default. One character, or `False` for no quoting. |
| `escape_char` | `escapechar` | A character that escapes the next one inside a quoted field. |
| `has_header` | `header` | `None` or `False`: the file has no header row. An integer `n`: line `n` is the header and the lines before it are skipped. |
| `column_names` | `names`, `new_columns` | Column names to use. With a header row, add `skip_rows=1` so the header is not read as data. |
| `skip_rows` | `skiprows` | Lines to skip before the header. |
| `skip_rows_after_header` | `skip_rows_after_names` | Lines to skip after the header. |
| `null_values` | `na_values` | A token, or a list of them, read as null in addition to Arrow's defaults (`""`, `NA`, `NULL`, `null`, `NaN` and others). A quoted value is never a null. |
| `true_values`, `false_values` | | Extra tokens read as `true` and `false`. |
| `decimal_point` | | The decimal separator, for `1,5`-style numbers. |
| `try_parse_dates` | `parse_dates` | `True` widens date inference. A list of columns types those columns as timestamps. |
| `schema` | `dtype`, `dtypes`, `schema_overrides` | A `pa.Schema` declares every column. A `{column: type}` dict overrides the inferred type of some. |
| `encoding` | | The text encoding, `"utf8"` by default. |
| `on_bad_lines` | `on_bad_rows` | What to do with a line whose field count is wrong. See {ref}`messy input <reading-messy-input>`. |

The writer takes `delimiter`, `header` and `null_value` (also `na_rep`). With `null_value`, a null is written as the bare token and a string equal to the token is quoted, so reading the file back with `null_values=` returns the same nulls and the same column types:

```python
table = bt.from_pydict({"k": [1, None], "note": ["ok", "NULL"]})
out = os.path.join(tempfile.mkdtemp(), "nulls.csv")
table.write.csv(out, null_value="NULL")
print(bt.read.csv(out, null_values="NULL").to_pydict())
# {'k': [1, None], 'note': ['ok', 'NULL']}
```

A large local CSV is read in parallel as byte ranges, each starting where a record starts. A newline inside a quoted field does not start one, so a range never cuts a quoted field in two. A file in which the `escape_char` occurs, a remote file, and a compressed file are each read whole, because Batcher cannot prove where their records start without reading all of them first.

(reading-messy-input)=

## Messy input

Real corpora contain members that will not read. Batcher separates three failures that look
alike and have different fixes, so reaching for the wrong flag cannot quietly delete data.

An **unreadable file** is one whose bytes the format cannot parse at all: a truncated
upload, a zero-byte object, a JPEG whose trailer never arrived. `on_error="skip"` drops the
file and reads the rest. Use it when the input is a corpus you do not control.

The source object behind the read keeps the audit trail. `corrupt_files()` names every
path it dropped, so a short result is explainable rather than mysterious:

```python
import os
import tempfile

import batcher as bt
from batcher.io import ParquetSource

corpus = tempfile.mkdtemp()
bt.from_pydict({"id": [1, 2]}).write.parquet(os.path.join(corpus, "good.parquet"))
with open(os.path.join(corpus, "zbad.parquet"), "wb") as f:
    _ = f.write(b"not a parquet file")

print(bt.read.parquet(corpus, on_error="skip").count())
# 2

source = ParquetSource(corpus, on_error="skip")
_ = source.read()
print([os.path.basename(path) for path in source.corrupt_files()])
# ['zbad.parquet']
```

A **malformed row** is one record inside a file that is otherwise fine: a CSV row carrying
a field the header does not have, or an NDJSON line that is not JSON at all. The file is
readable, so `on_error` is the wrong answer for it. Dropping the file would discard every
good row to be rid of one bad line. Pass `on_bad_lines` instead, which drops the record.

```python
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

The third failure is **a wrong encoding**: bytes that are readable, but not in the encoding
you asked for. A text corpus assembled from scrapes, exports and legacy systems is a
mixture, and a single stray byte is not a reason to lose a file. `read.text` replaces what
it cannot decode with U+FFFD by default, in both `mode="line"` and `mode="file"`.
`errors="strict"` turns it into a per-file failure that `on_error="skip"` will then drop:

```python
d = tempfile.mkdtemp()
with open(os.path.join(d, "legacy.txt"), "wb") as f:
    _ = f.write("caf\xe9\n".encode("cp1252"))

print(bt.read.text(d).to_pydict()["text"])
print(bt.read.text(d, encoding="cp1252").to_pydict()["text"])
```

Replacement is a fallback, not an answer. If you know what the bytes are, naming the
`encoding` is the fix.

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

The same {py:obj}`bt.read <batcher.read>` namespace reaches everything else, on two further pages.
{doc}`/user-guide/moving-data/specialized-formats` covers the scientific and container formats:
Zarr, HDF5, WARC, PDF, LiDAR, and robot logs. {doc}`/integrations/databases/databases` covers a SQL
database or warehouse, where the interesting part is not the call but the connection: which backend
serves your scheme, where the credentials come from, and how to split one extract into parallel
queries.

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
- {doc}`/user-guide/moving-data/streaming/index`: the same readers over unbounded sources.
- {doc}`IO API </api/relational/io>`: the full {py:obj}`bt.read <batcher.read>` reader reference.
- {doc}`Agent skills </agents>`: `read-and-write-data` covers picking a reader or
  sink, cloud paths, globs, schema evolution, and error tolerance.
- {doc}`/cookbook/io/index`: 6 runnable recipes for readers, writers, and the registries.
