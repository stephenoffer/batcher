# Transforming and collecting

This page maps the transformation verbs and terminal operations you already know onto their Batcher spellings, and lists the familiar names Batcher keeps.

## Transforming

Each transformation returns a new lazy {py:class}`Dataset <batcher.Dataset>`, so a pipeline reads as one chained expression:

```python
import batcher as bt
from batcher import col

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC"], "amount": [10, 20, 30]})
out = (
    ds.filter(col("amount") > 10)
    .with_columns(tax=col("amount") * 0.1)
    .group_by("city")
    .agg(total=col("amount").sum(), n=bt.count())
)
print(out.to_pydict())
# {'city': ['LA', 'NYC'], 'total': [20, 30], 'n': [1, 1]}
```

The tables below map pandas. Polars, PySpark, Daft, and Ray Data have a generated name-by-name reference: {doc}`polars/dataframe`, {doc}`spark/dataframe`, {doc}`daft/dataframe`, and {doc}`ray-data/dataset`.

:::{dropdown} The pandas verb table
:open:

| Task | pandas | Batcher |
|------|--------|---------|
| Select / project | `df[["a", "b"]]` | {py:meth}`ds.select("a", "b") <batcher.Dataset.select>` |
| Derive column | `df.assign(c=...)` | {py:meth}`ds.with_columns(c=...) <batcher.Dataset.with_columns>` |
| Filter rows | `df[df.a > 1]` | {py:meth}`ds.filter(col("a") > 1) <batcher.Dataset.filter>` |
| Group + aggregate | `df.groupby("k").agg(...)` | {py:meth}`ds.group_by("k").agg(...) <batcher.Dataset.group_by>` |
| Group + sum all | `df.groupby("k").sum()` | `ds.group_by("k").sum()` |
| Group + Python function | `df.groupby("k").apply(fn)` | `ds.group_by("k").map_groups(fn)` |
| Mean aggregate | `df.a.mean()` | `col("a").mean()` |
| Sort | `df.sort_values("a")` | {py:meth}`ds.sort("a") <batcher.Dataset.sort>` |
| Join | `df.merge(o, on="k")` | {py:meth}`ds.join(o, on="k") <batcher.Dataset.join>` |
| ASOF join | `pd.merge_asof(...)` | {py:meth}`ds.join_asof(o, on=..., by=...) <batcher.Dataset.join_asof>` |
| Distinct | `df.drop_duplicates()` | {py:meth}`ds.distinct() <batcher.Dataset.distinct>` |
| Limit | `df.head(n)` | {py:meth}`ds.limit(n) <batcher.Dataset.limit>` |
| Collect list | `df.groupby(k)[c].agg(list)` | `col(c).array_agg()` |
| First / last | `df.groupby(k).first()` | `col(c).first(order_by=..)` |
| Column ref | `df["a"]` | `ds["a"]` |
| Row slice | `df[:n]` | `ds[:n]` |
| Fill nulls | `df.fillna(0)` | {py:meth}`ds.fill_null(0) <batcher.Dataset.fill_null>` |
| Drop nulls | `df.dropna()` | {py:meth}`ds.drop_nulls() <batcher.Dataset.drop_nulls>` |
| Cast | `df.astype({...})` | {py:meth}`ds.cast({...}) <batcher.Dataset.cast>` |
| Global agg | `df.sum()` | {py:meth}`ds.agg(...) <batcher.Dataset.agg>` |
| Explode list | `df.explode("c")` | {py:meth}`ds.explode("c") <batcher.Dataset.explode>` |
| Unpivot / melt | `df.melt(...)` | {py:meth}`ds.unpivot(index=..., on=...) <batcher.Dataset.unpivot>` |
| Sample rows | `df.sample(frac=f)` | {py:meth}`ds.sample(f, seed=...) <batcher.Dataset.sample>` |
| Pivot / wide | `df.pivot_table(...)` | {py:meth}`ds.pivot(index=..., on=..., values=...) <batcher.Dataset.pivot>` |

:::

The verbs you reach for most, in action:

```python
users = bt.from_pydict({"id": [1, 2], "name": ["ann", "bo"]})
orders = bt.from_pydict({"uid": [1, 1, 2], "amt": [5, 7, 9]})
print(orders.join(users, left_on="uid", right_on="id").sort("amt").to_pydict())
# {'uid': [1, 1, 2], 'amt': [5, 7, 9], 'name': ['ann', 'ann', 'bo']}
```

```python
tags = bt.from_pydict({"k": ["a", "b"], "tags": [["x", "y"], ["z"]]})
print(tags.explode("tags").to_pydict())
# {'k': ['a', 'a', 'b'], 'tags': ['x', 'y', 'z']}
```

```python
wide = bt.from_pydict({"id": [1, 2], "q1": [10, 20], "q2": [30, 40]})
long = wide.unpivot(index="id", on=["q1", "q2"])
print(long.sort(["id", "variable"]).to_pydict())
# {'id': [1, 1, 2, 2], 'variable': ['q1', 'q2', 'q1', 'q2'], 'value': [10, 30, 20, 40]}
print(long.pivot(index="id", on="variable", values="value").sort("id").to_pydict())
# {'id': [1, 2], 'q1': [10, 20], 'q2': [30, 40]}
```

pandas' rolling and ranking idioms become a window. Any aggregate or ranking expression takes {py:meth}`.over(...) <batcher.AggExpr.over>`, and {py:meth}`ds.window(...) <batcher.Dataset.window>` is the table-shaped form:

```python
print(
    ds.with_columns(rank=bt.rank().over(partition_by="city", order_by="amount"))
    .sort("amount")
    .to_pydict()
)
# {'city': ['NYC', 'LA', 'NYC'], 'amount': [10, 20, 30], 'rank': [1, 1, 2]}
```

## Terminal operations

A terminal operation makes the plan run:

| Task | pandas | Batcher |
|------|--------|---------|
| Materialize | (eager) | {py:meth}`ds.collect() <batcher.Dataset.collect>` / {py:meth}`ds.to_arrow() <batcher.Dataset.to_arrow>` |
| Row count | `len(df)` | {py:meth}`ds.count() <batcher.Dataset.count>` |
| Preview | `df.head()` | {py:meth}`ds.show() <batcher.Dataset.show>` |
| Summary stats | `df.describe()` | {py:meth}`ds.describe() <batcher.Dataset.describe>` |
| Null counts | `df.isnull().sum()` | {py:meth}`ds.null_count() <batcher.Dataset.null_count>` |

```python
print(ds.count(), ds.limit(1).to_pylist())
# 3 [{'city': 'NYC', 'amount': 10}]
print(ds.null_count().to_pydict())
# {'city': [0], 'amount': [0]}
```

{py:meth}`ds.iter_batches() <batcher.Dataset.iter_batches>` streams Arrow batches, {py:meth}`ds.ml.iter_torch_batches() <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` streams tensors, {py:meth}`ds.explain() <batcher.Dataset.explain>` returns the plan, and {py:meth}`ds.stats() <batcher.Dataset.stats>` reports measured per-operator statistics.

{py:obj}`ds.write(path, mode=...) <batcher.Dataset.write>` takes the Spark save modes: `overwrite` (the default), `error`, `ignore`, and `append` for lakehouse sinks. {py:meth}`ds.write.delta(uri, merge_on=["id"]) <batcher.api.io_namespace.writer.Writer.delta>` runs a transactional `MERGE INTO` upsert in one call.

## Names that carry over, and names that don't

Batcher keeps one spelling per operation. A familiar name it spells differently, such as `groupby`, `merge`, `fillna`, or `drop_duplicates`, raises `AttributeError` naming the replacement:

```python
try:
    ds.fillna(0)
except AttributeError as exc:
    print("use `.fill_null`" in str(exc))
# True
```

A few familiar names are real methods:

| You type | What it does |
|---|---|
| {py:meth}`ds.filter("x > 2") <batcher.Dataset.filter>` | the same filter as `ds.filter(bt.col("x") > 2)`, from a SQL predicate string, as pandas `query` and Daft `filter` take it |
| {py:meth}`ds.first() <batcher.Dataset.first>` / {py:meth}`ds.last() <batcher.Dataset.last>` / {py:meth}`ds.item() <batcher.Dataset.item>` | terminal row accessors |
| {py:obj}`ds.width <batcher.Dataset.width>` | `len(ds.columns)` |
| {py:meth}`ds.info() <batcher.Dataset.info>` / {py:meth}`ds.glimpse() <batcher.Dataset.glimpse>` / {py:meth}`ds.memory_usage() <batcher.Dataset.memory_usage>` | schema-and-count summaries |
| {py:meth}`ds.iter_rows() <batcher.Dataset.iter_rows>` / {py:meth}`ds.iter_slices() <batcher.Dataset.iter_slices>` | row and slice iterators, alongside {py:meth}`ds.iter_batches() <batcher.Dataset.iter_batches>` |

Argument names follow the Batcher spelling too. `ds.sort()` takes `descending=` and `nulls_first=`, not `by=`, `ascending=` or `na_position=`. `ds.sample()` reads a positional `int` as a row count and a `float` as a fraction, and takes `seed=`. {py:meth}`ds.unpivot() <batcher.Dataset.unpivot>` takes `index=`, `on=`, `variable_name=` and `value_name=`. {py:meth}`ds.select_dtypes() <batcher.Dataset.select_dtypes>` accepts a Python type, a dtype name, or a list of either, as `include` or as `exclude=`. {py:meth}`ds.rename() <batcher.Dataset.rename>` accepts a function applied to every column name.

A list of columns works wherever a verb takes several, as in Polars, PySpark, and Ray Data:

```python
import batcher as bt

sales = bt.from_pydict({"region": ["e", "w", "e"], "city": ["a", "b", "c"], "v": [1, 2, 3]})
print(sales.select(["region", "v"]).sort(["region", "v"]).to_pydict())
# {'region': ['e', 'e', 'w'], 'v': [1, 3, 2]}
```

An aggregate can name its output with {py:meth}`.alias() <batcher.AggExpr.alias>`, the Polars and PySpark spelling:

```python
print(
    sales.group_by("region")
    .agg(bt.col("v").sum().alias("total"), bt.count().alias("n"))
    .sort("region")
    .to_pydict()
)
# {'region': ['e', 'w'], 'total': [4, 2], 'n': [2, 1]}
```

`filter` ANDs several predicates, treats a keyword as an equality test, and accepts a SQL string:

```python
import batcher as bt

ds = bt.from_pydict({"status": ["paid", "open", "paid"], "amount": [10, 20, 30]})
print(ds.filter(bt.col("amount") > 5, status="paid").to_pydict())
# {'status': ['paid', 'paid'], 'amount': [10, 30]}
print(ds.filter("amount > 15").to_pydict())
# {'status': ['open', 'paid'], 'amount': [20, 30]}
```

`agg` also takes the pandas dict spec:

```python
print(ds.group_by("status").agg({"amount": ["min", "max"]}).sort("status").to_pydict())
# {'status': ['open', 'paid'], 'amount_min': [20, 10], 'amount_max': [20, 30]}
```

## See also

- {doc}`/getting-started/migration/differences`: the APIs Batcher deliberately does not have, and what to use instead.
- {doc}`/getting-started/migration/polars/index` and {doc}`/getting-started/migration/spark/index`: every Polars and PySpark name, with its Batcher spelling and status.
- {doc}`/user-guide/transform/rows/transformations`: the same verbs taught rather than tabulated.
- {doc}`/user-guide/transform/columns/expressions`: the expression language the table above assumes.
- {doc}`/getting-started/migration/reading-and-writing`: the readers, writers, and framework bridges on either side of these verbs.
