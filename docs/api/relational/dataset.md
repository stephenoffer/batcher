# Dataset

A {py:class}`Dataset <batcher.Dataset>` is a lazy, immutable handle to a query plan. Every transformation
returns a new `Dataset` and runs no work. Execution happens only when you call a
terminal operation such as `collect`, `to_pydict`, or `write.parquet`.

This page is the full reference: how to construct a dataset, every transformation
method, every terminal method, and the {py:class}`GroupBy <batcher.GroupBy>` builder.

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

The following table lists the in-memory and framework entry points, ordered from the most commonly used:

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
| {py:meth}`.select(*names, **derived) <batcher.Dataset.select>` | Choose existing columns by name and derive new ones as keywords. |
| {py:meth}`.with_columns(**named) <batcher.Dataset.with_columns>` | Add or replace columns, keeping the rest. |
| {py:meth}`.with_column(name, expr) <batcher.Dataset.with_column>` | Add or replace a single column. |
| `.drop(*names)` | Remove columns. |
| `.rename(mapping)` | Rename columns via `{"old": "new"}`. |
| `.sort(*by, descending=False, nulls_first=False)` | Order rows. `by` is a name or expression. |
| {py:meth}`.limit(n, offset=0) <batcher.Dataset.limit>` | Take `n` rows after skipping `offset`. |
| `.head(n=5)` | Take the first `n` rows. |
| `.tail(n=5)` | Take the last `n` rows (executes a `count` first). |
| {py:meth}`.sample(fraction=None, *, n=None, seed=None) <batcher.Dataset.sample>` | Sample a `fraction` of rows or a fixed count `n`. Deterministic and partition-independent (a stable seeded content hash), so identical single-node or distributed. |
| {py:meth}`.split_at_indices(indices) <batcher.Dataset.split_at_indices>` | Cut into consecutive row ranges at the given positions (Ray Data's spelling). Every part stays lazy. |
| {py:meth}`.split_proportionately(proportions) <batcher.Dataset.split_proportionately>` | Cut into parts holding the given row fractions, with exact sizes (executes a `count` first). |
| {py:meth}`.distinct() <batcher.Dataset.distinct>` | Drop duplicate rows. |
| `.union(*others, distinct=False)` | Concatenate datasets; set `distinct=True` to dedupe. |
| `.intersect(other)` | Rows present in both. |
| {py:meth}`.except_(other) <batcher.Dataset.except_>` | Rows in this dataset but not the other. |
| `.join(other, ...)` | Relational join (see below). |
| {py:meth}`.window(...) <batcher.Dataset.window>` | Per-row windowed columns (see below). |
| {py:meth}`.group_by(*keys, **derived) <batcher.Dataset.group_by>` | Start a grouped aggregation (returns `GroupBy`). |
| `.rollup(*keys)` | Aggregate at every prefix of `keys` plus the grand total (SQL `ROLLUP`). |
| `.cube(*keys)` | Aggregate at every subset of `keys` (SQL `CUBE`). |
| {py:meth}`.grouping_sets(*sets) <batcher.Dataset.grouping_sets>` | Aggregate at exactly the levels given (SQL `GROUPING SETS`). |
| `.map_batches(fn, ...)` | Apply a Python function to whole Arrow batches. |
| {py:meth}`.offload_blobs(column="bytes", ...) <batcher.Dataset.offload_blobs>` | Move a large-payload column to a content-addressed store, leaving URI handles ({doc}`blob-by-reference </ml/preparing/multimodal/index>`). |
| {py:meth}`.materialize_blobs(...) <batcher.Dataset.materialize_blobs>` | Read offloaded payloads back from their handles (inverse of {py:meth}`offload_blobs <batcher.Dataset.offload_blobs>`). |
| `.repartition(num_files=None, *, by=None, target_size_mb=None)` | Set how the next `write` lays out files (data unchanged). |

### filter

```python
print(ds.filter(bt.col("price") >= 30.0).to_pydict())
# {'category': ['a', 'b', 'a', 'c'], 'price': [30.0, 40.0, 50.0, 60.0], 'qty': [3, 4, 5, 6]}
```

### select and with_columns

`select` chooses the full output: positional arguments must be existing column
names, and keyword arguments derive new named columns. {py:meth}`with_columns <batcher.Dataset.with_columns>` adds or
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

{py:meth}`window(partition_by=(), order_by=(), functions={...}, frame=None) <batcher.Dataset.window>` adds columns
without collapsing rows. `functions` maps an output name to a spec:

- Ranking (needs `order_by`): `"row_number"`, `"rank"`, `"dense_rank"`.
- Aggregates: `("sum"|"avg"|"min"|"max"|"count", "column")`, optionally with a frame.
- Value: `("first_value"|"last_value"|"lag"|"lead", "column"[, offset])`.

`order_by` entries are a column name, `("col", descending_bool)`, or an expression. `frame=(start, end)` gives ROWS offsets where negative is preceding, 0 is current, positive is following, and `None` is unbounded. A third element selects the frame unit. `frame=(start, end, "groups")` counts peer groups, meaning ties in the order key, instead of physical rows. `frame=(None, 0, "range")` is the value-based peer frame, equivalent to `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.

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
accessor exposes the same call with ML defaults; see {doc}`the ML accessor </api/models/ml>`.

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

`repartition` changes only the file layout the next `write` produces, not the data. Pass exactly one sizing option. `num_files` splits into that many files. `target_size_mb` coalesces into files of roughly that size, which is the fix for the small-files problem. Pass neither and give only `by` to Hive-partition by one or more columns. `by` may combine with a sizing option. For in-place use against an existing path, see {py:obj}`bt.compact <batcher.compact>`.

```python
# docs: skip
ds.repartition(target_size_mb=128).write("out/")
ds.repartition(by="dt").write("out/")
```

## GroupBy

{py:meth}`group_by(*keys, **derived) <batcher.Dataset.group_by>` returns a `GroupBy`. Finalize it with
`agg(**named_aggregates)`, where each keyword names an output column and its value
is an aggregate expression. {py:obj}`bt.count() <batcher.count>` is `COUNT(*)`; column aggregates such as
`.sum()` and `.mean()` are methods on an expression. There is no `.alias` on an
aggregate; the keyword is the name.

For reducing every value column the same way, `GroupBy` also has the shortcut
methods `sum`, `mean`, `min`, `max`, `median`, `quantile(q)`, `n_unique`, `std`,
`var`, `count` for non-null values per column, and `len` for the per-group row count. Each reduces all non-key columns by default, or the column names or selector you pass. `agg` also accepts a bare positional aggregate such as `agg(col("x").sum())`, which keeps its source column name.

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
or via `with_columns`. See {doc}`Aggregations </user-guide/analyze/aggregations>` for the
full aggregate function set.

### Subtotals: rollup, cube and grouping sets

A subtotal report needs the same aggregate at several grouping levels at once.
{py:meth}`rollup(*keys) <batcher.Dataset.rollup>` aggregates at every prefix of the keys and then the grand total,
{py:meth}`cube(*keys) <batcher.Dataset.cube>` at every subset, and {py:meth}`grouping_sets(*sets) <batcher.Dataset.grouping_sets>` at exactly the levels you
name (`[]` is the grand total). Each returns a builder you finish with `.agg(...)`,
and each is the DataFrame spelling of the SQL clause of the same name. The two
front-ends produce identical rows. A key that is not part of a level reads as NULL,
which is how a subtotal row is told apart from a detail row:

```python
report = ds.rollup("category").agg(revenue=bt.col("price").sum())
print(report.sort("category").to_pydict())
# {'category': ['a', 'b', 'c', None], 'revenue': [90.0, 60.0, 60.0, 210.0]}
```

## Terminal operations

A terminal operation executes the plan.

| Method | Returns |
| --- | --- |
| {py:meth}`.collect(distributed=False, num_workers=None, spill=False, num_partitions=16, adaptive=False, transport="disk") <batcher.Dataset.collect>` | A pyarrow `Table`. |
| {py:meth}`.to_pydict() <batcher.Dataset.to_pydict>` | A `dict[str, list]`. |
| {py:meth}`.to_pylist() <batcher.Dataset.to_pylist>` | A `list[dict]`, one dict per row. |
| `.count()` | Row count as an `int`. |
| `.min(column)`, `.max(column)`, `.sum(column)`, `.mean(column)`, `.std(column)`, `.var(column)`, `.n_unique(column)` | A single-column reduction as a scalar (nulls ignored). |
| `.median(column)` / `.quantile(column, q)` | The exact median / `q`-quantile as a scalar. |
| {py:meth}`.product(column) <batcher.Dataset.product>` / {py:meth}`.mode(column) <batcher.Dataset.mode>` | The product of the values / the most frequent value. |
| {py:meth}`.skewness(column) <batcher.Dataset.skewness>` / {py:meth}`.kurtosis(column) <batcher.Dataset.kurtosis>` / {py:meth}`.mad(column) <batcher.Dataset.mad>` | Shape and spread: lopsidedness, tail weight, and the outlier-tolerant mean absolute deviation. |
| {py:meth}`.any(column) <batcher.Dataset.any>` / {py:meth}`.all(column) <batcher.Dataset.all>` | Reduce a boolean column (SQL `BOOL_OR` / `BOOL_AND`); an empty column is `None`, not `False`/`True`. |
| `.corr(x, y)` / `.cov(x, y, ddof=1)` | Pearson correlation / covariance of two columns. |
| {py:meth}`.iter_batches(batch_size=None) <batcher.Dataset.iter_batches>` | An iterator of pyarrow `RecordBatch`es. |
| `.explain()` | The plan as a `str`. |
| {py:meth}`.show(limit=10) <batcher.Dataset.show>` | Prints a preview; returns `None`. |
| {py:obj}`.write(path, fmt=None, partition_by=None, distributed=False, num_workers=None, **kw) <batcher.Dataset.write>` | A {py:class}`WriteManifest <batcher.io.WriteManifest>`. |
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

{py:meth}`iter_batches <batcher.Dataset.iter_batches>` streams results, choosing the execution mode automatically: a
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
# scan                            est≈6 (exact)
```

Writers persist results; they need a real path, so they are not run here.

```python
# docs: skip
ds.write.parquet("output/data.parquet")
ds.write("output/", fmt="parquet", partition_by=["category"])
```

## Introspection

`.columns` lists the output column names and `.schema` gives the pyarrow `Schema`. Neither executes the plan.

```python
print(ds.columns)
# ['category', 'price', 'qty']
```

## Interoperability

A `Dataset` implements the Arrow **PyCapsule stream interface**
(`__arrow_c_stream__`), so any library that speaks the Arrow C Data Interface consumes one directly. There's no {py:meth}`to_arrow() <batcher.Dataset.to_arrow>` call, no copy, and no conversion through Python objects.

```python
# docs: skip
import duckdb
import polars as pl
import pyarrow as pa

pl.DataFrame(ds)                  # Polars
duckdb.sql("SELECT * FROM ds")    # DuckDB, by variable name
pa.table(ds)                      # pyarrow
```

The stream is **lazy**: batches are pulled from the plan as the consumer reads them, so
a result larger than memory streams into DuckDB rather than landing in it first. Because
the consumer's iteration is what drives execution, this is a terminal operation.

{py:meth}`collect() <batcher.Dataset.collect>` returns a pyarrow `Table` when you want the whole result in hand, and
{py:meth}`to_pandas() <batcher.Dataset.to_pandas>` / {py:meth}`to_arrow() <batcher.Dataset.to_arrow>` are there for the direct conversions.

## Reshaping

{py:meth}`explode(col) <batcher.Dataset.explode>` turns a list column into one row per element (SQL `UNNEST`).
`unnest(col)` is the struct counterpart. It expands a struct column's fields into top-level columns in place, matching Polars `unnest` and Spark `select("s.*")`:

```python
import pyarrow as pa

s = pa.StructArray.from_arrays([pa.array([1, 2]), pa.array(["a", "b"])], names=["n", "t"])
ds = bt.from_arrow(pa.table({"id": [10, 20], "s": s}))
print(ds.unnest("s").columns)
# ['id', 'n', 't']
```

## Descriptive statistics

{py:meth}`describe() <batcher.Dataset.describe>` returns a small summary `Dataset` (pandas/Polars-style): a `statistic`
label column and one column per input column. Numeric columns report count /
null_count / mean / std / min / quartiles / max; non-numeric columns report count
and null_count only. It **executes** the query (the summary is the result). Pass
`percentiles=` to choose the quantile rows. {py:meth}`null_count() <batcher.Dataset.null_count>` is the lazy per-column
null tally (it lowers to one aggregate, so nothing runs until a terminal op).

```python
ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})
ds.describe().show()
print(ds.null_count().to_pydict())
# {'g': [0], 'x': [0]}
```

{py:meth}`profile() <batcher.Dataset.profile>` is the quick "what does this column look like" check before a load: it
**executes** and returns one row per column with `count`, `null_count`,
`null_fraction`, and `approx_distinct` (HyperLogLog cardinality).

## Data quality and dimension upserts

Two accessors hang off a `Dataset` for validation and dimension maintenance:

| Accessor | Purpose |
| --- | --- |
| `.dq` | Data-quality expectations. Constraint methods accumulate (returning a new {py:class}`DatasetDQ <batcher.api.dataset.dq.DatasetDQ>`); a terminal method (`fail` / `drop` / `quarantine` / `validate`) applies them. |
| `.scd` | Dimension maintenance. `type1` / `type2` / `type3` take an incoming snapshot (natural keys + attributes); `apply_changes` takes a CDC change feed, with deletes, redeliveries, and out-of-order rows. |

## See also

- {doc}`Expressions </api/relational/expressions>`: the column expressions used above.
- {doc}`SQL </api/relational/sql>`: run SQL against a dataset.
- {doc}`The ML accessor </api/models/ml>`: batch inference and embeddings.
- {doc}`/cookbook/dataset/index`: 14 runnable recipes for the verbs on this page.
