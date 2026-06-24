# Dataset

A `Dataset` is a lazy, immutable handle to a query plan. Every transformation
returns a new `Dataset` and runs no work. Execution happens only when you call a
terminal operation such as `collect`, `to_pydict`, or `write.parquet`.

This page is the full reference: how to construct a dataset, every transformation
method, every terminal method, and the `GroupBy` builder.

## Construction

The most direct entry point is {py:obj}`bt.from_pydict <batcher.from_pydict>`, which builds a dataset from a
column-oriented dict. It is used throughout this page because it needs no files.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a", "c"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "qty": [1, 2, 3, 4, 5, 6],
    }
)
print(ds.columns)
# ['category', 'price', 'qty']
```

| Entry point | Source |
| --- | --- |
| {py:obj}`bt.from_pydict(mapping) <batcher.from_pydict>` | A column-oriented dict (`{name: [values]}`). |
| {py:obj}`bt.from_arrow(table_or_batches) <batcher.from_arrow>` | A pyarrow `Table`, `RecordBatch`, or list of batches. |
| {py:obj}`bt.from_batches(factory, schema) <batcher.from_batches>` | A reusable factory that yields Arrow batches (streaming source). |
| {py:obj}`bt.from_pandas(df) <batcher.from_pandas>` | A pandas `DataFrame`. |
| {py:obj}`bt.from_polars(df) <batcher.from_polars>` | A Polars `DataFrame`. |
| {py:obj}`bt.from_numpy(...) <batcher.from_numpy>` | NumPy arrays. |
| {py:obj}`bt.from_spark <batcher.from_spark>`, {py:obj}`bt.from_dask <batcher.from_dask>`, {py:obj}`bt.from_huggingface <batcher.from_huggingface>`, {py:obj}`bt.from_torch <batcher.from_torch>`, {py:obj}`bt.from_tf <batcher.from_tf>` | Framework adapters. |

File and object-store readers share the same surface; only the source changes.

```python
# docs: skip
ds = bt.read("s3://bucket/events.parquet")
ds = bt.read.parquet("data/events.parquet")
ds = bt.read.csv("data/events.csv")
```

{py:obj}`bt.read(path, format=None, **opts) <batcher.read>` dispatches on the path or an explicit
`format`. Dedicated readers exist for parquet, csv, json, orc, arrow, avro,
lance, delta, iceberg, hudi, images, audio, video, and SQL/warehouse sources
(`read.snowflake`, `read.bigquery`, `read.kafka`, and more).

## Transformations

Each method returns a new `Dataset`. They chain.

| Method | Effect |
| --- | --- |
| `.filter(predicate)` | Keep rows where the boolean expression is true. |
| `.select(*names, **derived)` | Choose existing columns by name and derive new ones as keywords. |
| `.with_columns(**named)` | Add or replace columns, keeping the rest. |
| `.with_column(name, expr)` | Add or replace a single column. |
| `.drop(*names)` | Remove columns. |
| `.rename(mapping)` | Rename columns via `{"old": "new"}`. |
| `.sort(*by, descending=False, nulls_first=False)` | Order rows. `by` is a name or expression. |
| `.limit(n, offset=0)` | Take `n` rows after skipping `offset`. |
| `.head(n=5)` | Take the first `n` rows. |
| `.sample(fraction=None, *, n=None, seed=None)` | Sample a `fraction` of rows or a fixed count `n`. Deterministic and partition-independent (a stable seeded content hash), so identical single-node or distributed. |
| `.distinct()` | Drop duplicate rows. |
| `.union(*others, distinct=False)` | Concatenate datasets; set `distinct=True` to dedupe. |
| `.intersect(other)` | Rows present in both. |
| `.except_(other)` | Rows in this dataset but not the other. |
| `.join(other, ...)` | Relational join (see below). |
| `.window(...)` | Per-row windowed columns (see below). |
| `.group_by(*keys, **derived)` | Start a grouped aggregation (returns `GroupBy`). |
| `.map_batches(fn, ...)` | Apply a Python function to whole Arrow batches. |
| `.repartition(num_files=None, *, by=None, target_size_mb=None)` | Set how the next `write` lays out files (data unchanged). |

### filter

```python
print(ds.filter(bt.col("price") >= 30.0).to_pydict())
# {'category': ['a', 'b', 'a', 'c'], 'price': [30.0, 40.0, 50.0, 60.0], 'qty': [3, 4, 5, 6]}
```

### select and with_columns

`select` chooses the full output: positional arguments must be existing column
names, and keyword arguments derive new named columns. `with_columns` adds or
replaces columns and keeps everything else.

```python
print(ds.select("category", total=bt.col("price") * bt.col("qty")).to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a', 'c'],
#  'total': [10.0, 40.0, 90.0, 160.0, 250.0, 360.0]}

print(ds.with_columns(total=bt.col("price") * bt.col("qty")).columns)
# ['category', 'price', 'qty', 'total']

print(ds.with_column("price_plus_one", bt.col("price") + 1.0).columns)
# ['category', 'price', 'qty', 'price_plus_one']
```

### drop and rename

```python
print(ds.drop("qty").columns)
# ['category', 'price']

print(ds.rename({"price": "unit_price"}).columns)
# ['category', 'unit_price', 'qty']
```

### sort

`by` may be a column name or an expression. `descending` and `nulls_first` may be
a single bool or a list aligned with `by`.

```python
print(ds.sort("price", descending=True).select("price").to_pydict())
# {'price': [60.0, 50.0, 40.0, 30.0, 20.0, 10.0]}

print(ds.sort("category", "price", descending=[False, True]).select("category", "price").to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b', 'c'], 'price': [50.0, 30.0, 10.0, 40.0, 20.0, 60.0]}
```

### limit, head, distinct

```python
print(ds.limit(2, offset=1).select("category").to_pydict())
# {'category': ['b', 'a']}

print(ds.head(2).select("category").to_pydict())
# {'category': ['a', 'b']}

print(ds.select("category").distinct().sort("category").to_pydict())
# {'category': ['a', 'b', 'c']}
```

### Set operations

`union` concatenates; `intersect` and `except_` are set semantics.

```python
left = ds.select("category")
right = bt.from_pydict({"category": ["a", "b"]})

print(left.union(right, distinct=True).sort("category").to_pydict())
# {'category': ['a', 'b', 'c']}

print(left.intersect(right).sort("category").to_pydict())
# {'category': ['a', 'b']}

print(left.except_(right).to_pydict())
# {'category': ['c']}
```

### join

`join(other, on=None, left_on=None, right_on=None, how="inner", suffix="_right")`.
Use `on` when both sides share a key name, or `left_on`/`right_on` when they
differ. `how` is one of `inner`, `left`, `right`, `full`, `outer`, `semi`,
`anti`. Columns that collide get `suffix`.

```python
dim = bt.from_pydict({"category": ["a", "b"], "region": ["west", "east"]})
joined = ds.join(dim, on="category", how="inner").sort("price")
print(joined.to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a'], 'price': [10.0, 20.0, 30.0, 40.0, 50.0],
#  'qty': [1, 2, 3, 4, 5], 'region': ['west', 'east', 'west', 'east', 'west']}
```

### window

`window(partition_by=(), order_by=(), functions={...}, frame=None)` adds columns
without collapsing rows. `functions` maps an output name to a spec:

- Ranking (needs `order_by`): `"row_number"`, `"rank"`, `"dense_rank"`.
- Aggregates: `("sum"|"avg"|"min"|"max"|"count", "column")`, optionally with a frame.
- Value: `("first_value"|"last_value"|"lag"|"lead", "column"[, offset])`.

`order_by` entries are a column name, `("col", descending_bool)`, or an
expression. `frame=(start, end)` gives ROWS offsets where negative is preceding,
0 is current, positive is following, and `None` is unbounded.

```python
ranked = ds.window(
    partition_by=["category"],
    order_by=[("price", True)],
    functions={"rnk": "row_number"},
).sort("category", "price")
print(ranked.select("category", "price", "rnk").to_pydict())
# {'category': ['a', 'a', 'a', 'b', 'b', 'c'], 'price': [10.0, 30.0, 50.0, 20.0, 40.0, 60.0],
#  'rnk': [3, 2, 1, 2, 1, 1]}
```

### map_batches

`map_batches(fn, batch_size=None, output_columns=None, num_workers=1, num_gpus=0.0, concurrency=None)`
applies a Python function to whole Arrow `RecordBatch`es, never per row. It is the
escape hatch for logic that has no expression form. When the function changes the
schema, pass `output_columns` so later operations know the new columns. The `.ml`
accessor exposes the same call with ML defaults; see [the ML accessor](ml.md).

```python
import pyarrow.compute as pc


def add_total(batch):
    total = pc.multiply(batch.column("price"), batch.column("qty"))
    return batch.append_column("total", total)


with_total = ds.map_batches(add_total, output_columns=["category", "price", "qty", "total"])
print(with_total.select("category", "total").to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a', 'c'],
#  'total': [10.0, 40.0, 90.0, 160.0, 250.0, 360.0]}
```

### repartition

`repartition` changes only the file layout the next `write` produces, not the data.
Pass exactly one sizing option: `num_files` (split into that many files),
`target_size_mb` (coalesce into ~that-size files — the small-files fix), or neither
with only `by` to Hive-partition by column(s). `by` may combine with a sizing
option. For in-place use against an existing path, see {py:obj}`bt.compact <batcher.compact>`.

```python
# docs: skip
ds.repartition(target_size_mb=128).write("out/")
ds.repartition(by="dt").write("out/")
```

## GroupBy

`group_by(*keys, **derived)` returns a `GroupBy`. Finalize it with
`agg(**named_aggregates)`, where each keyword names an output column and its value
is an aggregate expression. {py:obj}`bt.count() <batcher.count>` is `COUNT(*)`; column aggregates such as
`.sum()` and `.mean()` are methods on an expression. There is no `.alias` on an
aggregate; the keyword is the name.

```python
summary = (
    ds.with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("category")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)
print(summary.to_pydict())
# {'category': ['c', 'a', 'b'], 'revenue': [360.0, 350.0, 200.0], 'orders': [1, 3, 2]}
```

Call `group_by()` with no keys for a global aggregate, and pass several keys to
group by each unique combination. Derived keys are allowed as keyword expressions
or via `with_columns`. See [Aggregations](../user-guide/aggregations.md) for the
full aggregate function set.

## Terminal operations

A terminal operation executes the plan.

| Method | Returns |
| --- | --- |
| `.collect(distributed=False, num_workers=None, spill=False, num_partitions=16, adaptive=False, transport="disk")` | A pyarrow `Table`. |
| `.to_pydict()` | A `dict[str, list]`. |
| `.to_pylist()` | A `list[dict]`, one dict per row. |
| `.count()` | Row count as an `int`. |
| `.iter_batches(batch_size=None)` | An iterator of pyarrow `RecordBatch`es. |
| `.explain()` | The plan as a `str`. |
| `.show(limit=10)` | Prints a preview; returns `None`. |
| `.write(path, fmt=None, partition_by=None, distributed=False, num_workers=None, **kw)` | A `WriteManifest`. |
| `.write.parquet(path, compression="zstd", **kw)` | A `WriteManifest`. |
| `.write.csv(path, **kw)`, `.write.json(path, **kw)` | A `WriteManifest`. |

```python
table = ds.collect()
print(table.num_rows)
# 6

print(ds.to_pylist()[0])
# {'category': 'a', 'price': 10.0, 'qty': 1}

print(ds.count())
# 6
```

`iter_batches` streams results, choosing the execution mode automatically: a
breaker-free pipeline is consumed as batches are produced (bounded memory), while
plans that must materialize do so first.

```python
total_rows = sum(batch.num_rows for batch in ds.iter_batches())
print(total_rows)
# 6
```

`explain` returns the plan for inspection.

```python
print(ds.explain().splitlines()[0])
# Scan  (≈6 rows, exact)
```

Writers persist results; they need a real path, so they are not run here.

```python
# docs: skip
ds.write.parquet("output/data.parquet")
ds.write("output/", fmt="parquet", partition_by=["category"])
```

## Introspection

`.columns` is a property listing the output column names. There is no `.schema`
property and no `.to_pandas` / `.to_arrow` on a `Dataset`; `collect()` already
returns a pyarrow `Table`, so call `.to_pandas()` on that if you need pandas.

```python
print(ds.columns)
# ['category', 'price', 'qty']
```

## Reshaping

`explode(col)` turns a list column into one row per element (SQL `UNNEST`).
`unnest(col)` is the struct counterpart — it expands a struct column's fields into
top-level columns in place (Polars `unnest`; Spark `select("s.*")`):

```python
import pyarrow as pa

s = pa.StructArray.from_arrays([pa.array([1, 2]), pa.array(["a", "b"])], names=["n", "t"])
ds = bt.from_arrow(pa.table({"id": [10, 20], "s": s}))
print(ds.unnest("s").columns)
# ['id', 'n', 't']
```

## Descriptive statistics

`describe()` returns a small summary `Dataset` (pandas/Polars-style): a `statistic`
label column and one column per input column. Numeric columns report count /
null_count / mean / std / min / quartiles / max; non-numeric columns report count
and null_count only. It **executes** the query (the summary is the result). Pass
`percentiles=` to choose the quantile rows. `null_count()` is the lazy per-column
null tally (it lowers to one aggregate, so nothing runs until a terminal op).

```python
ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})
ds.describe().show()
print(ds.null_count().to_pydict())
# {'g': [0], 'x': [0]}
```

`profile()` is the quick "what does this column look like" check before a load: it
**executes** and returns one row per column with `count`, `null_count`,
`null_fraction`, and `approx_distinct` (HyperLogLog cardinality).

## Data quality and dimension upserts

| Accessor | Purpose |
| --- | --- |
| `.dq` | Data-quality expectations. Constraint methods accumulate (returning a new `DatasetDQ`); a terminal method (`fail` / `drop` / `quarantine` / `validate`) applies them. |
| `.scd` | Slowly-changing-dimension upserts. The dataset is the incoming dimension snapshot (natural keys + attributes). |

## Next steps

- [Expressions](expressions.md): the column expressions used above.
- [SQL](sql.md): run SQL against a dataset.
- [The ML accessor](ml.md): batch inference and embeddings.
