# Reading, writing, and interop

This page covers Batcher's readers, writers, constructors, and exporters, and where the reader and writer you already use in another engine maps onto them. Start here when the first thing a ported script does is load a file or hand a frame to Batcher.

## Reading and writing

Batcher gives you one callable namespace per direction. {py:obj}`bt.read(path) <batcher.read>` infers the format, and typed methods such as {py:meth}`bt.read.parquet <batcher.api.io_namespace.reader.Reader.parquet>` and {py:meth}`bt.read.delta <batcher.api.io_namespace.reader.Reader.delta>` name it explicitly. {py:obj}`ds.write <batcher.Dataset.write>` mirrors it.

```python
import batcher as bt

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC"], "amount": [10, 20, 30]})
ds.write.parquet("/tmp/sales")  # ds.write("/tmp/sales") also infers parquet
back = bt.read.parquet("/tmp/sales")
print(sorted(back.to_pydict()["amount"]))
# [10, 20, 30]
```

Every reader and writer in PySpark, Polars, Daft, and Ray Data has a row in the generated reference, with its Batcher spelling and what differs, such as a default save mode. PySpark's readers and writers are on {doc}`spark/io`. The Polars, Daft, and Ray Data readers are module functions, on {doc}`polars/io`, {doc}`daft/module`, and {doc}`ray-data/io`. Their writers are frame methods, on {doc}`polars/dataframe`, {doc}`daft/dataframe`, and {doc}`ray-data/dataset`.

Polars splits reading into eager `read_*` and lazy `scan_*`. Batcher doesn't need the split. Every `bt.read.*` returns a {py:class}`Dataset <batcher.Dataset>` plan and does no I/O until a terminal operation, with the projection and predicate pushdown `scan_*` gives you. There's one spelling per format, and it's the lazy one.

If your fingers already type `pd.read_csv` or `pl.read_parquet`, the Batcher spelling moves the format behind a dot. There is no top-level `bt.read_csv`, and every common format has one typed reader:

| Reads | Batcher |
|-------|---------|
| CSV files, directories, globs | {py:meth}`bt.read.csv(p) <batcher.api.io_namespace.reader.Reader.csv>` |
| Parquet | {py:meth}`bt.read.parquet(p) <batcher.api.io_namespace.reader.Reader.parquet>` |
| newline-delimited JSON | {py:meth}`bt.read.json(p) <batcher.api.io_namespace.reader.Reader.json>` |
| Arrow IPC / Feather | {py:meth}`bt.read.arrow(p) <batcher.api.io_namespace.reader.Reader.arrow>` |
| ORC | {py:meth}`bt.read.orc(p) <batcher.api.io_namespace.reader.Reader.orc>` |
| Avro | {py:meth}`bt.read.avro(p) <batcher.api.io_namespace.reader.Reader.avro>` |
| Excel workbooks | {py:meth}`bt.read.excel(p) <batcher.api.io_namespace.reader.Reader.excel>` |
| Delta Lake tables | {py:meth}`bt.read.delta(p) <batcher.api.io_namespace.reader.Reader.delta>` |
| Iceberg tables | {py:meth}`bt.read.iceberg(t) <batcher.api.io_namespace.reader.Reader.iceberg>` |
| any SQL database | {py:meth}`bt.read.sql(q, uri=...) <batcher.api.io_namespace.reader.Reader.sql>` |

For a source that isn't a file at all, {py:meth}`bt.read.table(name, ...) <batcher.api.io_namespace.reader.Reader.table>` constructs any
registered connector by name, which is the escape hatch behind the typed readers.

## Getting data in from another library

Whatever object you're holding, there's a constructor for it:

| You have | Call |
|----------|------|
| A `{column: values}` dict | {py:func}`bt.from_pydict(d) <batcher.from_pydict>` |
| A list of row dicts | {py:func}`bt.from_pylist(rows) <batcher.from_pylist>` |
| A list of row tuples | {py:func}`bt.from_records(rows, columns=[...]) <batcher.from_records>` |
| A generator or any iterable | {py:func}`bt.from_iter(gen) <batcher.from_iter>` |
| A pandas or Polars frame | {py:func}`bt.from_pandas(df) <batcher.from_pandas>` / {py:func}`bt.from_polars(df) <batcher.from_polars>` |
| A DuckDB relation or connection | {py:func}`bt.from_duckdb(rel) <batcher.from_duckdb>` |
| An Arrow table, or anything Arrow-exporting | {py:func}`bt.from_arrow(t) <batcher.from_arrow>` |
| Something whose type you don't know | {py:func}`bt.from_any(obj) <batcher.from_any>` |

Reach for `bt.from_any` in migration code and glue. It dispatches on the type and routes to the right constructor, so a script that accepts "a frame" from a caller doesn't have to branch. {py:func}`bt.sql <batcher.sql>` uses it for every bound table, which is why you can pass a pandas frame or a plain dict straight into a query:

```python
import batcher as bt

print(bt.sql("SELECT x * 2 AS y FROM t", t={"x": [1, 2, 3]}).to_pydict())
# {'y': [2, 4, 6]}
```

## Concatenating and generating

{py:func}`bt.concat <batcher.concat>` means frame concatenation, exactly as `pd.concat` and `pl.concat` do, and
takes Polars' `how` vocabulary. The string-building `concat` keeps its own explicit
name, {py:func}`bt.concat_str <batcher.concat_str>`:

```python
import batcher as bt

a = bt.from_pydict({"x": [1, 2]})
b = bt.from_pydict({"x": [3, 4]})
print(bt.concat([a, b]).to_pydict())
# {'x': [1, 2, 3, 4]}

wide = bt.concat([bt.from_pydict({"x": [1]}), bt.from_pydict({"y": ["a"]})], how="diagonal")
print(wide.to_pydict())
# {'x': [1, None], 'y': [None, 'a']}

ds = bt.from_pydict({"first": ["ada"], "last": ["lovelace"]})
print(ds.select(name=bt.concat_str(bt.col("first"), bt.lit(" "), bt.col("last"))).to_pydict())
# {'name': ['ada lovelace']}
```

`how` is `"vertical"` (the default), `"vertical_relaxed"` to deduplicate,
`"diagonal"` to stack over the union of the columns, or `"horizontal"` to place frames
side by side by row position.

{py:func}`bt.range <batcher.range>` follows `builtins.range`, single-argument form included, and {py:func}`bt.date_range <batcher.date_range>`
follows `pandas.date_range` and `polars.date_range`:

```python
import batcher as bt

print(bt.range(5).to_pydict())
# {'value': [0, 1, 2, 3, 4]}
print(bt.date_range("2024-01-01", periods=3, interval="1mo").count())
# 3
```

Pass `end=` or `periods=`, the stride as `interval=` (Polars) or `freq=` (pandas), and
`closed=` to drop an endpoint the way pandas' `inclusive=` does.

## Moving data in and out

Most `from_*` constructors have a matching `to_*` exporter, so Batcher slots into an existing pipeline without a copy where the framework's Arrow bridge allows it.

```python
# docs: skip
import pandas as pd
import batcher as bt

ds = bt.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))  # pandas  -> Batcher
pdf = ds.filter(bt.col("a") > 1).to_pandas()  # Batcher -> pandas
pl_df = ds.to_polars()  # Batcher -> Polars
table = ds.to_arrow()  # Batcher -> pyarrow.Table
```

Each row pairs a source system with its constructor and, where one exists, its exporter.

| Source | In | Out |
|--------|----|----|
| Arrow | {py:func}`bt.from_arrow(t) <batcher.from_arrow>` | {py:meth}`ds.to_arrow() <batcher.Dataset.to_arrow>` |
| Python dict / rows | {py:func}`bt.from_pydict(d) <batcher.from_pydict>` / {py:func}`bt.from_pylist(rows) <batcher.from_pylist>` | {py:meth}`ds.to_pydict() <batcher.Dataset.to_pydict>` / {py:meth}`ds.to_pylist() <batcher.Dataset.to_pylist>` |
| pandas | {py:func}`bt.from_pandas(df) <batcher.from_pandas>` | {py:meth}`ds.to_pandas() <batcher.Dataset.to_pandas>` |
| Polars | {py:func}`bt.from_polars(df) <batcher.from_polars>` | {py:meth}`ds.to_polars() <batcher.Dataset.to_polars>` |
| NumPy | {py:func}`bt.from_numpy(arr) <batcher.from_numpy>` | {py:meth}`ds.to_numpy() <batcher.Dataset.to_numpy>` |
| Ray Data | {py:func}`bt.from_ray_dataset(ds) <batcher.from_ray_dataset>` | {py:meth}`ds.to_ray_dataset() <batcher.Dataset.to_ray_dataset>` |
| Spark | {py:func}`bt.from_spark(df) <batcher.from_spark>` | n/a |
| Dask | {py:func}`bt.from_dask(ddf) <batcher.from_dask>` | n/a |
| HuggingFace | {py:func}`bt.from_huggingface(ds) <batcher.from_huggingface>` | n/a |
| PyTorch | {py:func}`bt.from_torch(ds) <batcher.from_torch>` | {py:meth}`ds.to_torch() <batcher.Dataset.to_torch>` / {py:meth}`ds.ml.to_torch_dataloader() <batcher.api.dataset.ml.DatasetML.to_torch_dataloader>` |
| TensorFlow | {py:func}`bt.from_tf(ds) <batcher.from_tf>` | {py:meth}`ds.ml.to_tf() <batcher.api.dataset.ml.DatasetML.to_tf>` |

The `to_torch` and `ml.to_tf` exporters yield a re-iterable dataset of per-batch tensor
dicts, so a multi-epoch training loop streams the query in bounded memory.

## See also

- {doc}`/getting-started/migration/transforming`: the verbs that run between the read and the write.
- {doc}`/user-guide/moving-data/reading-data`: the full reader reference, with cloud paths and globs.
- {doc}`/user-guide/moving-data/writing-data`: save modes, partitioning, and atomic writes.
- {doc}`/getting-started/migration/differences`: the exporters that don't exist, and how to verify a port.
