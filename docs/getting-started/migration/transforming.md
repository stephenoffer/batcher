# Transforming and collecting

This page maps the transformation verbs and terminal operations you already know onto their Batcher spellings, and lists the familiar names Batcher keeps.

## Transforming

Transformations are lazy. Each one chains off a {py:class}`Dataset <batcher.Dataset>` and returns a new one, so a whole pipeline reads as a single expression, and only a terminal operation runs it:

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

This page's tables map pandas, which has no generated reference. For Polars, PySpark, Daft, and Ray Data, every name has a row in the generated reference, with its status and what differs: {doc}`polars/dataframe`, {doc}`spark/dataframe`, {doc}`daft/dataframe`, and {doc}`ray-data/dataset`, and their sibling pages for expressions and functions.

The pandas transformation verbs map across as follows, ordered roughly by how often you reach for them.

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

pandas has no window expression, so its rolling and ranking idioms become one of two Batcher forms. An aggregate or ranking expression takes {py:meth}`.over(...) <batcher.AggExpr.over>`, such as `bt.rank().over(partition_by=.., order_by=..)`, and {py:meth}`ds.window(partition_by=..., functions=...) <batcher.Dataset.window>` is the table-shaped form.

`mean` is the expression method for an average. The `window()` function table also accepts `"avg"` as a function name.

## Terminal operations

A terminal operation is the call that makes the plan run. The following table lists the pandas equivalents:

| Task | pandas | Batcher |
|------|--------|---------|
| Materialize | (eager) | {py:meth}`ds.collect() <batcher.Dataset.collect>` / {py:meth}`ds.to_arrow() <batcher.Dataset.to_arrow>` |
| Row count | `len(df)` | {py:meth}`ds.count() <batcher.Dataset.count>` |
| Preview | `df.head()` | {py:meth}`ds.show() <batcher.Dataset.show>` |
| Summary stats | `df.describe()` | {py:meth}`ds.describe() <batcher.Dataset.describe>` |
| Null counts | `df.isnull().sum()` | {py:meth}`ds.null_count() <batcher.Dataset.null_count>` |

Several terminals have no pandas counterpart. {py:meth}`ds.iter_batches() <batcher.Dataset.iter_batches>` streams Arrow batches, and {py:meth}`ds.ml.to_numpy_batches() <batcher.api.dataset.ml.DatasetML.to_numpy_batches>` and {py:meth}`ds.ml.iter_torch_batches() <batcher.api.dataset.ml.DatasetML.iter_torch_batches>` stream NumPy arrays or tensors. {py:meth}`ds.explain() <batcher.Dataset.explain>` returns the plan, and {py:meth}`ds.stats() <batcher.Dataset.stats>` runs the query and reports measured per-operator statistics.

{py:obj}`ds.write(path, mode=...) <batcher.Dataset.write>` takes the Spark save modes: `overwrite`, the default, `error`, `ignore`, and `append`, which lakehouse sinks accept. For Delta upserts, {py:meth}`ds.write.delta(uri, merge_on=["id"]) <batcher.api.io_namespace.writer.Writer.delta>` runs a transactional `MERGE INTO` that updates matched rows and inserts new ones. That's the Spark and Delta `MERGE` in one call.

## Names that carry over, and names that don't

Batcher keeps one spelling per operation. A familiar name that Batcher spells differently, such as `groupby`, `merge`, `fillna`, `drop_duplicates` or `orderBy`, raises `AttributeError`, and the message names the Batcher spelling to use. `python -m batcher.migrate <paths>` prints the rewrite for a script that still uses a removed Batcher spelling, and `--write` applies it. The generated reference pages list every foreign name with its Batcher spelling.

A few familiar names are real methods, listed here with what they do:

| You type | What it does |
|---|---|
| {py:meth}`ds.query("x > 2") <batcher.Dataset.query>` | the same filter as {py:meth}`ds.filter(bt.col("x") > 2) <batcher.Dataset.filter>` |
| {py:meth}`ds.first() <batcher.Dataset.first>` / {py:meth}`ds.last() <batcher.Dataset.last>` / {py:meth}`ds.item() <batcher.Dataset.item>` | terminal row accessors |
| {py:obj}`ds.width <batcher.Dataset.width>` | `len(ds.columns)` |
| {py:meth}`ds.info() <batcher.Dataset.info>` / {py:meth}`ds.glimpse() <batcher.Dataset.glimpse>` / {py:meth}`ds.memory_usage() <batcher.Dataset.memory_usage>` | schema-and-count summaries |
| {py:meth}`ds.iter_rows() <batcher.Dataset.iter_rows>` / {py:meth}`ds.iter_slices() <batcher.Dataset.iter_slices>` | row and slice iterators, alongside {py:meth}`ds.iter_batches() <batcher.Dataset.iter_batches>` |

Argument names follow the Batcher spelling too. `ds.sort()` takes `descending=` and `nulls_first=`, not `by=`, `ascending=` or `na_position=`. `ds.sample()` reads a positional `int` as a row count and a `float` as a fraction, and takes `seed=`. {py:meth}`ds.unpivot() <batcher.Dataset.unpivot>` takes `index=`, `on=`, `variable_name=` and `value_name=`.
{py:meth}`ds.select_dtypes() <batcher.Dataset.select_dtypes>` accepts a Python type, a dtype name, or a list of either, as `include` or as `exclude=`. {py:meth}`ds.rename() <batcher.Dataset.rename>` accepts a function applied to every column name.

A list of columns works wherever a verb takes several, which is how Polars, PySpark,
and Ray Data all spell it. `ds.select(["a", "b"])`, `ds.sort(["a", "b"])` and
`ds.group_by(["region"])` need no rewrite to positional arguments, and a list mixes with
bare names in the same call. The verbs that read a list this way are `select`,
`with_columns`, `filter`, `sort`, `group_by`, `rollup`, `cube`, `agg`, `drop`, `unnest`
and `union`.

```python
import batcher as bt

sales = bt.from_pydict({"region": ["e", "w", "e"], "city": ["a", "b", "c"], "v": [1, 2, 3]})
print(sales.select(["region", "v"]).sort(["region", "v"]).to_pydict())
# {'region': ['e', 'e', 'w'], 'v': [1, 3, 2]}
```

The exception is {py:meth}`ds.grouping_sets() <batcher.Dataset.grouping_sets>`, where each argument *is* a list: one
grouping level per argument. There the lists carry the meaning, so they are left alone.

An aggregate names its own output with {py:meth}`.alias() <batcher.AggExpr.alias>`, the Polars and PySpark
spelling, as an alternative to the keyword form. Only that spelling can name a
`bt.count()`, which has no input column to be named after:

```python
print(
    sales.group_by("region")
    .agg(bt.col("v").sum().alias("total"), bt.count().alias("n"))
    .sort("region")
    .to_pydict()
)
# {'region': ['e', 'w'], 'total': [4, 2], 'n': [2, 1]}
```

Two `filter` shorthands have no pandas equivalent, and both save the parentheses `&` otherwise needs. Several predicates are ANDed, and a keyword is an equality test:

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
- {doc}`/getting-started/migration/polars/index` and {doc}`/getting-started/migration/spark/index`: every Polars and PySpark name, with its Batcher spelling and status.
- {doc}`/user-guide/transform/rows/transformations`: the same verbs taught rather than tabulated.
- {doc}`/user-guide/transform/columns/expressions`: the expression language the table above assumes.
- {doc}`/getting-started/migration/reading-and-writing`: the readers, writers, and framework bridges on either side of these verbs.
