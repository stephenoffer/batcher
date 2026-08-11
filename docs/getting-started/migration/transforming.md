# Transforming and collecting

This page maps the transformation verbs and terminal operations you already know onto
their Batcher spellings, and lists the names Batcher accepts unchanged from pandas and
Polars.

Transformations are lazy and return a new {py:class}`Dataset <batcher.Dataset>`. A terminal operation is what makes
the plan run.

## Transforming

Transformations chain off a `Dataset` and return a new one, so a whole pipeline reads
as a single expression:

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

The transformation verbs map across as follows, ordered roughly by how often you reach
for them.

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Select / project | `df[["a", "b"]]` | `df.select("a", "b")` | `df.select("a", "b")` | {py:meth}`ds.select("a", "b") <batcher.Dataset.select>` |
| Derive column | `df.assign(c=...)` | `df.with_columns(c=...)` | `df.withColumn("c", ...)` | {py:meth}`ds.with_columns(c=...) <batcher.Dataset.with_columns>` |
| Filter rows | `df[df.a > 1]` | `df.filter(pl.col("a") > 1)` | `df.filter(df.a > 1)` | {py:meth}`ds.filter(col("a") > 1) <batcher.Dataset.filter>` |
| Group + aggregate | `df.groupby("k").agg(...)` | `df.group_by("k").agg(...)` | `df.groupBy("k").agg(...)` | {py:meth}`ds.group_by("k").agg(...) <batcher.Dataset.group_by>` |
| Group + sum all | `df.groupby("k").sum()` | `df.group_by("k").sum()` | n/a | `ds.group_by("k").sum()` |
| Group + Python function | `df.groupby("k").apply(fn)` | `df.group_by("k").map_groups(fn)` | `df.groupBy("k").applyInPandas(fn, s)` | `ds.group_by("k").map_groups(fn)` |
| Mean aggregate | `df.a.mean()` | `pl.col("a").mean()` | `F.avg("a")` | `col("a").mean()` |
| Sort | `df.sort_values("a")` | `df.sort("a")` | `df.orderBy("a")` | {py:meth}`ds.sort("a") <batcher.Dataset.sort>` |
| Join | `df.merge(o, on="k")` | `df.join(o, on="k")` | `df.join(o, "k")` | {py:meth}`ds.join(o, on="k") <batcher.Dataset.join>` |
| ASOF join | `pd.merge_asof(...)` | `df.join_asof(...)` | `ASOF JOIN` | {py:meth}`ds.join_asof(o, on=..., by=...) <batcher.Dataset.join_asof>` |
| Distinct | `df.drop_duplicates()` | `df.unique()` | `df.distinct()` | {py:meth}`ds.distinct() <batcher.Dataset.distinct>` |
| Limit | `df.head(n)` | `df.head(n)` | `df.limit(n)` | {py:meth}`ds.limit(n) <batcher.Dataset.limit>` |
| Window rank | n/a | `pl.col(..).rank().over(..)` | `F.rank().over(Window...)` | `rank().over(partition_by=.., order_by=..)` |
| Window | n/a | {py:meth}`.over(...) <batcher.AggExpr.over>` | `F....over(Window...)` | {py:meth}`ds.window(partition_by=..., functions=...) <batcher.Dataset.window>` |
| Collect list | `df.groupby(k)[c].agg(list)` | `pl.col(c).implode()` | `F.collect_list(c)` | `col(c).array_agg()` |
| First / last | `df.groupby(k).first()` | `pl.col(c).first()` | `F.first(c)` | `col(c).first(order_by=..)` |
| Column ref | `df["a"]` | `df["a"]` | `df["a"]` | `ds["a"]` |
| Row slice | `df[:n]` | `df[:n]` | n/a | `ds[:n]` |
| Fill nulls | `df.fillna(0)` | `df.fill_null(0)` | `df.fillna(0)` | {py:meth}`ds.fill_null(0) <batcher.Dataset.fill_null>` |
| Drop nulls | `df.dropna()` | `df.drop_nulls()` | `df.dropna()` | {py:meth}`ds.drop_nulls() <batcher.Dataset.drop_nulls>` |
| Cast | `df.astype({...})` | `df.cast({...})` | `df.withColumn(...)` | {py:meth}`ds.cast({...}) <batcher.Dataset.cast>` |
| Global agg | `df.sum()` | `df.select(...sum())` | `df.agg(...)` | {py:meth}`ds.agg(...) <batcher.Dataset.agg>` |
| Explode list | `df.explode("c")` | `df.explode("c")` | `df.select(explode(...))` | {py:meth}`ds.explode("c") <batcher.Dataset.explode>` |
| Unpivot / melt | `df.melt(...)` | `df.unpivot(...)` | `df.unpivot(...)` | {py:meth}`ds.unpivot(index=..., on=...) <batcher.Dataset.unpivot>` |
| Sample rows | `df.sample(frac=f)` | `df.sample(fraction=f)` | `df.sample(fraction=f)` | {py:meth}`ds.sample(f, seed=...) <batcher.Dataset.sample>` |
| Pivot / wide | `df.pivot_table(...)` | `df.pivot(...)` | `df.groupBy(i).pivot(c)` | {py:meth}`ds.pivot(index=..., on=..., values=...) <batcher.Dataset.pivot>` |
| Window expr | n/a | `e.over(...)` | `e.over(Window...)` | `agg.over(partition_by=...)` |

In aggregates and the `window()` function table, Batcher uses `mean` as the canonical
name, matching pandas and Polars. `avg` is accepted as a synonym, so SQL muscle memory
still works.

## Terminal operations

A terminal operation is what triggers the plan to run. These are the equivalents.

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Materialize | (eager) | `df.collect()` | `df.collect()` | {py:meth}`ds.collect() <batcher.Dataset.collect>` / {py:meth}`ds.to_arrow() <batcher.Dataset.to_arrow>` |
| Row count | `len(df)` | `df.height` | `df.count()` | {py:meth}`ds.count() <batcher.Dataset.count>` |
| Preview | `df.head()` | `df.head()` | `df.show()` | {py:meth}`ds.show() <batcher.Dataset.show>` |
| Summary stats | `df.describe()` | `df.describe()` | `df.summary()` | {py:meth}`ds.describe() <batcher.Dataset.describe>` |
| Null counts | `df.isnull().sum()` | `df.null_count()` | n/a | {py:meth}`ds.null_count() <batcher.Dataset.null_count>` |
| Stream batches | n/a | n/a | `df.toLocalIterator()` | {py:meth}`ds.iter_batches() <batcher.Dataset.iter_batches>` |
| Stream as NumPy / tensors | n/a | n/a | n/a | {py:meth}`ds.ml.to_numpy_batches() <batcher.api.dataset.ml.DatasetML.to_numpy_batches>`, {py:meth}`ds.ml.iter_torch_batches() <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` |
| Explain plan | n/a | `df.explain()` | `df.explain()` | {py:meth}`ds.explain() <batcher.Dataset.explain>` |
| Measured per-op stats | n/a | n/a | n/a | {py:meth}`ds.stats() <batcher.Dataset.stats>` |

{py:obj}`ds.write(path, mode=...) <batcher.Dataset.write>` takes the Spark save modes: `overwrite`, the default,
`error`, `ignore`, and `append`, which lakehouse sinks accept. For Delta upserts,
{py:meth}`ds.write.delta(uri, merge_on=["id"]) <batcher.api.io_namespace.writer.Writer.delta>` runs a transactional `MERGE INTO` that updates
matched rows and inserts new ones. That's the Spark and Delta `MERGE` in one call.

## Names and arguments that carry over unchanged

Batcher accepts the spelling you already type for a long list of operations, so a
ported script usually needs fewer edits than the table above suggests. Each alias is a
real method that delegates to the Batcher primary, not a shim, so it returns the same
plan and the same result.

| You type | Batcher primary |
|---|---|
| {py:meth}`ds.to_dicts() <batcher.Dataset.to_dicts>` | {py:meth}`ds.to_pylist() <batcher.Dataset.to_pylist>` |
| {py:meth}`ds.to_dict() <batcher.Dataset.to_dict>` | {py:meth}`ds.to_pydict() <batcher.Dataset.to_pydict>` |
| {py:meth}`ds.drop_duplicates() <batcher.Dataset.drop_duplicates>` | {py:meth}`ds.distinct() <batcher.Dataset.distinct>` |
| {py:meth}`ds.with_row_count() <batcher.Dataset.with_row_count>` | {py:meth}`ds.with_row_index() <batcher.Dataset.with_row_index>` |
| {py:meth}`ds.vstack(other) <batcher.Dataset.vstack>` / {py:meth}`ds.append(other) <batcher.Dataset.append>` | {py:meth}`ds.union(other) <batcher.Dataset.union>` |
| {py:meth}`ds.difference(other) <batcher.Dataset.difference>` | {py:meth}`ds.except_(other) <batcher.Dataset.except_>` |
| {py:meth}`ds.persist() <batcher.Dataset.persist>` | {py:meth}`ds.cache() <batcher.Dataset.cache>` |
| {py:meth}`ds.coalesce(n) <batcher.Dataset.coalesce>` | {py:meth}`ds.repartition(n) <batcher.Dataset.repartition>` |
| {py:meth}`ds.transform(fn) <batcher.Dataset.transform>` | {py:meth}`ds.pipe(fn) <batcher.Dataset.pipe>` |
| {py:meth}`ds.fillna(v) <batcher.Dataset.fillna>` / {py:meth}`ds.dropna() <batcher.Dataset.dropna>` | {py:meth}`ds.fill_null(v) <batcher.Dataset.fill_null>` / {py:meth}`ds.drop_nulls() <batcher.Dataset.drop_nulls>` |
| {py:meth}`ds.groupby(...) <batcher.Dataset.groupby>` / {py:meth}`ds.merge(...) <batcher.Dataset.merge>` | {py:meth}`ds.group_by(...) <batcher.Dataset.group_by>` / {py:meth}`ds.join(...) <batcher.Dataset.join>` |
| {py:meth}`ds.sort_values(...) <batcher.Dataset.sort_values>` / {py:meth}`ds.nlargest(...) <batcher.Dataset.nlargest>` | {py:meth}`ds.sort(...) <batcher.Dataset.sort>` / {py:meth}`ds.top_k(...) <batcher.Dataset.top_k>` |
| `gb.nunique()` / `gb.size()` | `gb.n_unique()` / `gb.len()` |
| {py:meth}`ds.query("x > 2") <batcher.Dataset.query>` | {py:meth}`ds.filter(bt.col("x") > 2) <batcher.Dataset.filter>` |
| {py:meth}`ds.to_parquet(p) <batcher.Dataset.to_parquet>` / {py:meth}`ds.to_csv(p) <batcher.Dataset.to_csv>` / {py:meth}`ds.to_json(p) <batcher.Dataset.to_json>` | {py:meth}`ds.write.parquet(p) <batcher.api.io_namespace.writer.Writer.parquet>` and friends |
| {py:meth}`ds.first() <batcher.Dataset.first>` / {py:meth}`ds.last() <batcher.Dataset.last>` / {py:meth}`ds.item() <batcher.Dataset.item>` | terminal row accessors |
| {py:obj}`ds.width <batcher.Dataset.width>` / {py:obj}`ds.height <batcher.Dataset.height>` / {py:obj}`ds.empty <batcher.Dataset.empty>` | `len(ds.columns)` / {py:meth}`ds.count() <batcher.Dataset.count>` / {py:meth}`ds.is_empty() <batcher.Dataset.is_empty>` |
| {py:meth}`ds.info() <batcher.Dataset.info>` / {py:meth}`ds.glimpse() <batcher.Dataset.glimpse>` / {py:meth}`ds.memory_usage() <batcher.Dataset.memory_usage>` | schema-and-count summaries |
| {py:meth}`ds.iter_rows() <batcher.Dataset.iter_rows>` / {py:meth}`ds.iter_slices() <batcher.Dataset.iter_slices>` | {py:meth}`ds.iter_batches() <batcher.Dataset.iter_batches>` |
| {py:meth}`ds.lazy() <batcher.Dataset.lazy>` / {py:meth}`ds.copy() <batcher.Dataset.copy>` | identity, because a `Dataset` is already lazy and immutable |

Argument names carry over too. `ds.sort()` takes `by=` and `ascending=` alongside
`descending=`, and `na_position=` alongside `nulls_first=`. `ds.sample()` reads a
positional `int` as a row count and a `float` as a fraction, and accepts `frac=` and
`random_state=`. {py:meth}`ds.melt() <batcher.Dataset.melt>` takes `id_vars=`, `value_vars=`, and `var_name=`.
{py:meth}`ds.select_dtypes() <batcher.Dataset.select_dtypes>` accepts a Python type, a dtype name, or a list of either, and an
`exclude=` argument. {py:meth}`ds.rename() <batcher.Dataset.rename>` accepts a function applied to every column name.

Two shorthands have no pandas equivalent but save the parenthesizing that `&`
otherwise needs. Several predicates are ANDed, and a keyword is an equality test:

```python
import batcher as bt

ds = bt.from_pydict({"status": ["paid", "open", "paid"], "amount": [10, 20, 30]})
print(ds.filter(bt.col("amount") > 5, status="paid").to_pydict())
# {'status': ['paid', 'paid'], 'amount': [10, 30]}
```

`ds.group_by(...).agg()` also takes the pandas dict spec, where a list of reducers
suffixes the output names the way pandas does when it flattens:

```python
print(ds.group_by("status").agg({"amount": ["min", "max"]}).sort("status").to_pydict())
# {'status': ['open', 'paid'], 'amount_min': [20, 10], 'amount_max': [20, 30]}
```

## See also

- {doc}`/getting-started/migration/differences`: the APIs Batcher deliberately does not have, and what to use instead.
- {doc}`/user-guide/transform/rows/transformations`: the same verbs taught rather than tabulated.
- {doc}`/user-guide/transform/columns/expressions`: the expression language the table above assumes.
