# Reading, writing, and interop

This page maps the reader, writer, constructor, and exporter you already use in pandas,
Polars, or PySpark onto the Batcher equivalent. Start here when the first thing a ported
script does is load a file or hand a frame to Batcher.

Every reader is lazy. `bt.read.parquet(p)` returns a plan and does no I/O until a
terminal operation, so there is no eager/lazy pair to choose between.

## Reading and writing

Batcher gives you one callable namespace per direction. `bt.read(path)` infers the
format, and the typed methods such as `bt.read.parquet` and `bt.read.delta` are
explicit and discoverable. `ds.write` mirrors it.

```python
import batcher as bt

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC"], "amount": [10, 20, 30]})
ds.write.parquet("/tmp/sales")          # ds.write("/tmp/sales") also infers parquet
back = bt.read.parquet("/tmp/sales")
print(sorted(back.to_pydict()["amount"]))
# [10, 20, 30]
```

The reader and writer spellings map across as follows.

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Read Parquet | `pd.read_parquet(p)` | `pl.read_parquet(p)` | `spark.read.parquet(p)` | `bt.read.parquet(p)` |
| Scan Parquet (lazy) | n/a | `pl.scan_parquet(p)` | n/a | `bt.read.parquet(p)` |
| Read CSV | `pd.read_csv(p)` | `pl.read_csv(p)` | `spark.read.csv(p)` | `bt.read.csv(p)` |
| Read Delta | n/a | `pl.read_delta(p)` | `spark.read.format("delta").load(p)` | `bt.read.delta(p)` |
| Autodetect | n/a | n/a | `spark.read.load(p)` | `bt.read(p)` |
| Write Parquet | `df.to_parquet(p)` | `df.write_parquet(p)` | `df.write.parquet(p)` | `ds.write.parquet(p)` |
| Write Delta | n/a | `df.write_delta(p)` | `df.write.format("delta").save(p)` | `ds.write.delta(p)` |

Polars splits reading into eager `read_*` and lazy `scan_*`. Batcher doesn't need the
split, because every `bt.read.*` is already lazy. It returns a `Dataset` plan and does
no I/O until a terminal op, with the projection and predicate pushdown `scan_*` gives
you. There is one spelling per format, and it's the lazy one.

If your fingers already type `pd.read_csv`, keep typing it. Every common format also
has a top-level shorthand under the ecosystem-standard name, and each one is the same
lazy reader as its `bt.read.*` twin:

| Shorthand | Same as | Reads |
|-----------|---------|-------|
| `bt.read_csv(p)` | `bt.read.csv(p)` | CSV files, directories, globs |
| `bt.read_parquet(p)` | `bt.read.parquet(p)` | Parquet |
| `bt.read_json(p)` / `bt.read_ndjson(p)` | `bt.read.json(p)` | newline-delimited JSON |
| `bt.read_ipc(p)` | `bt.read.arrow(p)` | Arrow IPC / Feather |
| `bt.read_orc(p)` | `bt.read.orc(p)` | ORC |
| `bt.read_avro(p)` | `bt.read.avro(p)` | Avro |
| `bt.read_excel(p)` | `bt.read.excel(p)` | Excel workbooks |
| `bt.read_delta(p)` | `bt.read.delta(p)` | Delta Lake tables |
| `bt.read_iceberg(t)` | `bt.read.iceberg(t)` | Iceberg tables |
| `bt.read_database(q, uri=...)` | `bt.read.sql(q, uri=...)` | any SQL database |

For a source that isn't a file at all, `bt.read_table(name, ...)` constructs any
registered connector by name, which is the escape hatch behind all of the above.

## Getting data in from another library

Whatever object you're holding, there's a constructor for it. The names follow pandas
and Polars, so `from_dict` and `from_dicts` mean what they mean there:

| You have | Call |
|----------|------|
| A `{column: values}` dict | `bt.from_pydict(d)`, or `bt.from_dict(d)` |
| A list of row dicts | `bt.from_pylist(rows)`, or `bt.from_dicts(rows)` |
| A list of row tuples | `bt.from_records(rows, columns=[...])` |
| A generator or any iterable | `bt.from_iter(gen)` |
| A pandas or Polars frame | `bt.from_pandas(df)` / `bt.from_polars(df)` |
| A DuckDB relation or connection | `bt.from_duckdb(rel)` |
| An Arrow table, or anything Arrow-exporting | `bt.from_arrow(t)` |
| Something whose type you don't know | `bt.from_any(obj)` |

`bt.from_any` is the one to reach for in migration code and glue: it dispatches on the
type and routes to the right constructor, so a script that accepts "a frame" from a
caller doesn't have to branch. `bt.sql` uses it for every bound table, which is why
you can pass a pandas frame or a plain dict straight into a query:

```python
import batcher as bt

print(bt.sql("SELECT x * 2 AS y FROM t", t={"x": [1, 2, 3]}).to_pydict())
# {'y': [2, 4, 6]}
```

## Concatenating and generating

`bt.concat` means frame concatenation, exactly as `pd.concat` and `pl.concat` do, and
takes Polars' `how` vocabulary. The string-building `concat` keeps its own explicit
name, `bt.concat_str`:

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

`bt.range` follows `builtins.range`, single-argument form included, and `bt.date_range`
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

Every `from_*` constructor has a symmetric `to_*` exporter, so Batcher slots into an
existing pipeline without a copy where the framework's Arrow bridge allows it.

```python
# docs: skip
import pandas as pd
import batcher as bt

ds = bt.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))   # pandas  -> Batcher
pdf = ds.filter(bt.col("a") > 1).to_pandas()          # Batcher -> pandas
pl_df = ds.to_polars()                                # Batcher -> Polars
table = ds.to_arrow()                                 # Batcher -> pyarrow.Table
```

Each row pairs a source system with its constructor and, where one exists, its exporter.

| Source | In | Out |
|--------|----|----|
| Arrow | `bt.from_arrow(t)` | `ds.to_arrow()` |
| Python dict / rows | `bt.from_pydict(d)` / `bt.from_pylist(rows)` | `ds.to_pydict()` / `ds.to_pylist()` |
| pandas | `bt.from_pandas(df)` | `ds.to_pandas()` |
| Polars | `bt.from_polars(df)` | `ds.to_polars()` |
| NumPy | `bt.from_numpy(arr)` | n/a |
| Ray Data | `bt.from_ray_dataset(ds)` | n/a |
| Spark | `bt.from_spark(df)` | n/a |
| Dask | `bt.from_dask(ddf)` | n/a |
| HuggingFace | `bt.from_huggingface(ds)` | n/a |
| PyTorch | `bt.from_torch(ds)` | `ds.to_torch()` / `ds.to_torch_dataloader()` |
| TensorFlow | `bt.from_tf(ds)` | `ds.to_tf()` |

The `to_torch` and `to_tf` exporters yield a re-iterable dataset of per-batch tensor
dicts, so a multi-epoch training loop streams the query in bounded memory.

## See also

- {doc}`/getting-started/migration/transforming`: the verbs that run between the read and the write.
- {doc}`/user-guide/moving-data/reading-data`: the full reader reference, with cloud paths and globs.
- {doc}`/user-guide/moving-data/writing-data`: save modes, partitioning, and atomic writes.
