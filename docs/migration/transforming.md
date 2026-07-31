# Transforming and collecting

This page maps the transformation verbs and terminal operations you already know onto
their Batcher spellings, and lists the names Batcher accepts unchanged from pandas and
Polars.

Transformations are lazy and return a new `Dataset`. A terminal operation is what makes
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
| Select / project | `df[["a", "b"]]` | `df.select("a", "b")` | `df.select("a", "b")` | `ds.select("a", "b")` |
| Derive column | `df.assign(c=...)` | `df.with_columns(c=...)` | `df.withColumn("c", ...)` | `ds.with_columns(c=...)` |
| Filter rows | `df[df.a > 1]` | `df.filter(pl.col("a") > 1)` | `df.filter(df.a > 1)` | `ds.filter(col("a") > 1)` |
| Group + aggregate | `df.groupby("k").agg(...)` | `df.group_by("k").agg(...)` | `df.groupBy("k").agg(...)` | `ds.group_by("k").agg(...)` |
| Group + sum all | `df.groupby("k").sum()` | `df.group_by("k").sum()` | n/a | `ds.group_by("k").sum()` |
| Group + Python function | `df.groupby("k").apply(fn)` | `df.group_by("k").map_groups(fn)` | `df.groupBy("k").applyInPandas(fn, s)` | `ds.group_by("k").map_groups(fn)` |
| Mean aggregate | `df.a.mean()` | `pl.col("a").mean()` | `F.avg("a")` | `col("a").mean()` |
| Sort | `df.sort_values("a")` | `df.sort("a")` | `df.orderBy("a")` | `ds.sort("a")` |
| Join | `df.merge(o, on="k")` | `df.join(o, on="k")` | `df.join(o, "k")` | `ds.join(o, on="k")` |
| ASOF join | `pd.merge_asof(...)` | `df.join_asof(...)` | `ASOF JOIN` | `ds.join_asof(o, on=..., by=...)` |
| Distinct | `df.drop_duplicates()` | `df.unique()` | `df.distinct()` | `ds.distinct()` |
| Limit | `df.head(n)` | `df.head(n)` | `df.limit(n)` | `ds.limit(n)` |
| Window rank | n/a | `pl.col(..).rank().over(..)` | `F.rank().over(Window...)` | `rank().over(partition_by=.., order_by=..)` |
| Window | n/a | `.over(...)` | `F....over(Window...)` | `ds.window(partition_by=..., functions=...)` |
| Collect list | `df.groupby(k)[c].agg(list)` | `pl.col(c).implode()` | `F.collect_list(c)` | `col(c).array_agg()` |
| First / last | `df.groupby(k).first()` | `pl.col(c).first()` | `F.first(c)` | `col(c).first(order_by=..)` |
| Column ref | `df["a"]` | `df["a"]` | `df["a"]` | `ds["a"]` |
| Row slice | `df[:n]` | `df[:n]` | n/a | `ds[:n]` |
| Fill nulls | `df.fillna(0)` | `df.fill_null(0)` | `df.fillna(0)` | `ds.fill_null(0)` |
| Drop nulls | `df.dropna()` | `df.drop_nulls()` | `df.dropna()` | `ds.drop_nulls()` |
| Cast | `df.astype({...})` | `df.cast({...})` | `df.withColumn(...)` | `ds.cast({...})` |
| Global agg | `df.sum()` | `df.select(...sum())` | `df.agg(...)` | `ds.agg(...)` |
| Explode list | `df.explode("c")` | `df.explode("c")` | `df.select(explode(...))` | `ds.explode("c")` |
| Unpivot / melt | `df.melt(...)` | `df.unpivot(...)` | `df.unpivot(...)` | `ds.unpivot(index=..., on=...)` |
| Sample rows | `df.sample(frac=f)` | `df.sample(fraction=f)` | `df.sample(fraction=f)` | `ds.sample(f, seed=...)` |
| Pivot / wide | `df.pivot_table(...)` | `df.pivot(...)` | `df.groupBy(i).pivot(c)` | `ds.pivot(index=..., on=..., values=...)` |
| Window expr | n/a | `e.over(...)` | `e.over(Window...)` | `agg.over(partition_by=...)` |

In aggregates and the `window()` function table, Batcher uses `mean` as the canonical
name, matching pandas and Polars. `avg` is accepted as a synonym, so SQL muscle memory
still works.

## Terminal operations

A terminal operation is what triggers the plan to run. These are the equivalents.

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Materialize | (eager) | `df.collect()` | `df.collect()` | `ds.collect()` / `ds.to_arrow()` |
| Row count | `len(df)` | `df.height` | `df.count()` | `ds.count()` |
| Preview | `df.head()` | `df.head()` | `df.show()` | `ds.show()` |
| Summary stats | `df.describe()` | `df.describe()` | `df.summary()` | `ds.describe()` |
| Null counts | `df.isnull().sum()` | `df.null_count()` | n/a | `ds.null_count()` |
| Stream batches | n/a | n/a | `df.toLocalIterator()` | `ds.iter_batches()` |
| Stream as NumPy / tensors | n/a | n/a | n/a | `ds.ml.to_numpy_batches()`, `ds.ml.iter_torch_batches()` |
| Explain plan | n/a | `df.explain()` | `df.explain()` | `ds.explain()` |
| Measured per-op stats | n/a | n/a | n/a | `ds.stats()` |

`ds.write(path, mode=...)` takes the Spark save modes: `overwrite`, the default,
`error`, `ignore`, and `append`, which lakehouse sinks accept. For Delta upserts,
`ds.write.delta(uri, merge_on=["id"])` runs a transactional `MERGE INTO` that updates
matched rows and inserts new ones. That's the Spark and Delta `MERGE` in one call.

## Names and arguments that carry over unchanged

Batcher accepts the spelling you already type for a long list of operations, so a
ported script usually needs fewer edits than the table above suggests. Each alias is a
real method that delegates to the Batcher primary, not a shim, so it returns the same
plan and the same result.

| You type | Batcher primary |
|---|---|
| `ds.to_dicts()` | `ds.to_pylist()` |
| `ds.to_dict()` | `ds.to_pydict()` |
| `ds.drop_duplicates()` | `ds.distinct()` |
| `ds.with_row_count()` | `ds.with_row_index()` |
| `ds.vstack(other)` / `ds.append(other)` | `ds.union(other)` |
| `ds.difference(other)` | `ds.except_(other)` |
| `ds.persist()` | `ds.cache()` |
| `ds.coalesce(n)` | `ds.repartition(n)` |
| `ds.transform(fn)` | `ds.pipe(fn)` |
| `ds.fillna(v)` / `ds.dropna()` | `ds.fill_null(v)` / `ds.drop_nulls()` |
| `ds.groupby(...)` / `ds.merge(...)` | `ds.group_by(...)` / `ds.join(...)` |
| `ds.sort_values(...)` / `ds.nlargest(...)` | `ds.sort(...)` / `ds.top_k(...)` |
| `gb.nunique()` / `gb.size()` | `gb.n_unique()` / `gb.len()` |
| `ds.query("x > 2")` | `ds.filter(bt.col("x") > 2)` |
| `ds.to_parquet(p)` / `ds.to_csv(p)` / `ds.to_json(p)` | `ds.write.parquet(p)` and friends |
| `ds.first()` / `ds.last()` / `ds.item()` | terminal row accessors |
| `ds.width` / `ds.height` / `ds.empty` | `len(ds.columns)` / `ds.count()` / `ds.is_empty()` |
| `ds.info()` / `ds.glimpse()` / `ds.memory_usage()` | schema-and-count summaries |
| `ds.iter_rows()` / `ds.iter_slices()` | `ds.iter_batches()` |
| `ds.lazy()` / `ds.copy()` | identity, because a `Dataset` is already lazy and immutable |

Argument names carry over too. `ds.sort()` takes `by=` and `ascending=` alongside
`descending=`, and `na_position=` alongside `nulls_first=`. `ds.sample()` reads a
positional `int` as a row count and a `float` as a fraction, and accepts `frac=` and
`random_state=`. `ds.melt()` takes `id_vars=`, `value_vars=`, and `var_name=`.
`ds.select_dtypes()` accepts a Python type, a dtype name, or a list of either, and an
`exclude=` argument. `ds.rename()` accepts a function applied to every column name.

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

- {doc}`differences`: the APIs Batcher deliberately does not have, and what to use instead.
- {doc}`/user-guide/transform/transformations`: the same verbs taught rather than tabulated.
- {doc}`/user-guide/transform/expressions`: the expression language the table above assumes.
