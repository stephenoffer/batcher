# Aggregations

Aggregations reduce many rows to summary values, either over the whole dataset or
per group. Group with `group_by`, then finalize with `agg`. Each aggregate is a
keyword whose value is an aggregate expression, so the keyword names the output
column.

## Setup

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "qty": [1, 2, 3, 4, 5],
    }
)
```

## group_by and agg

`group_by` takes the grouping keys; `agg` takes the output aggregates. Pass them as
keywords to name the output, or positionally to keep the source column's name.
{py:obj}`bt.count() <batcher.count>` is `COUNT(*)`; the column aggregates are methods
on an expression (`.sum()`, `.mean()`, …) or the top-level shorthands
{py:obj}`bt.sum("x") <batcher.sum>`, `bt.mean`, `bt.min`, `bt.max`, `bt.median`,
`bt.std`, `bt.var`, `bt.n_unique`. `bt.sum("x")` reads as `col("x").sum()`, the
Polars `pl.sum` convention.

```python
# Positional shorthands keep the column name; keywords rename.
by_user = ds.group_by("category").agg(bt.sum("price"), bt.mean("qty")).sort("category")
print(by_user.to_pydict())
# {'category': ['a', 'b'], 'price': [90.0, 60.0], 'qty': [3.0, 3.0]}
```

```python
out = (
    ds.with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("category")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)
print(out.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'orders': [3, 2]}
```

## Shortcut reductions

When you reduce *every* value column the same way, a shortcut method is shorter
than spelling out `agg`. The set is `sum`, `mean`, `min`, `max`, `median`,
`quantile(q)`, `n_unique`, `std`, `var`, `count` (non-null values per column), and
`len` (the per-group row count). With no arguments they reduce every non-key
column, keeping its name; pass column names or a
[selector](transformations.md) to reduce a subset. The arithmetic reductions
(`sum`, `mean`, `median`, `quantile`, `std`, `var`) default to numeric columns
only, like pandas' ``numeric_only``.

```python
print(ds.group_by("category").sum().sort("category").to_pydict())
# {'category': ['a', 'b'], 'price': [90.0, 60.0], 'qty': [9, 6]}

print(ds.group_by("category").mean("price").sort("category").to_pydict())
# {'category': ['a', 'b'], 'price': [30.0, 30.0]}

print(ds.group_by("category").len().sort("category").to_pydict())
# {'category': ['a', 'b'], 'len': [3, 2]}
```

`len` counts rows; `count` counts non-null values of each column (they differ only
when a column has nulls). Reach for `agg` when the reductions differ per column,
when you want custom output names, or when you need a windowed aggregate or a
two-column statistic.

## Aggregate functions

The aggregate methods available inside `agg` are `sum`, `min`, `max`, `mean`,
`var`, `std`, `median`, `quantile(q)`, `count`, and `n_unique` (also spelled
`count_distinct`). {py:obj}`bt.count() <batcher.count>` counts rows. Each of these
builds an {py:class}`AggExpr <batcher.AggExpr>`, the aggregate type that `agg(...)`
consumes and that `.over(...)` lifts into a [window function](window-functions.md).
You rarely name it directly.

```python
stats = ds.group_by("category").agg(
    total=bt.col("price").sum(),
    avg=bt.col("price").mean(),
    lo=bt.col("price").min(),
    hi=bt.col("price").max(),
    med=bt.col("price").median(),
    p90=bt.col("price").quantile(0.9),
    distinct_qty=bt.col("qty").n_unique(),
).sort("category")
print(stats.to_pydict())
# {'category': ['a', 'b'], 'total': [90.0, 60.0], 'avg': [30.0, 30.0], 'lo': [10.0, 20.0],
#  'hi': [50.0, 40.0], 'med': [30.0, 30.0], 'p90': [46.0, 38.0], 'distinct_qty': [3, 2]}
```

## Advanced aggregates

Beyond the basics, `agg` supports `mode`, `first`/`last`, `arg_min`/`arg_max` (the
value of one column at the row that minimizes/maximizes another), the boolean
reductions `bool_and`/`bool_or`, and `array_agg` (collect a group's values into a
list).

```python
adv = ds.group_by("category").agg(
    any_big=(bt.col("price") > 35).bool_or(),
    all_big=(bt.col("price") > 35).bool_and(),
    costliest=bt.col("price").arg_max(bt.col("price")),
).sort("category")
print(adv.to_pydict())
# {'category': ['a', 'b'], 'any_big': [True, True], 'all_big': [False, False],
#  'costliest': [50.0, 40.0]}
```

## Bivariate aggregates

The two-column statistical aggregates summarize how a pair of columns move
together within each group. {py:obj}`bt.corr(x, y) <batcher.corr>` is the Pearson
correlation coefficient in `[-1, 1]` (SQL `CORR`); {py:obj}`bt.covar_pop(x, y)
<batcher.covar_pop>` and {py:obj}`bt.covar_samp(x, y) <batcher.covar_samp>` are the
population and sample covariance (SQL `COVAR_POP` / `COVAR_SAMP`, dividing by `n`
and `n - 1` respectively). Reach for `corr` to score the strength and sign of a
relationship: whether ad spend tracks revenue per region, say.

```python
market = bt.from_pydict(
    {
        "region": ["west", "west", "west", "east", "east", "east"],
        "spend": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "revenue": [10.0, 20.0, 30.0, 30.0, 20.0, 10.0],
    }
)
bivariate = market.group_by("region").agg(
    r=bt.corr(bt.col("spend"), bt.col("revenue")),
    cov_p=bt.covar_pop(bt.col("spend"), bt.col("revenue")),
    cov_s=bt.covar_samp(bt.col("spend"), bt.col("revenue")),
).sort("region")
print(bivariate.to_pydict())
# {'region': ['east', 'west'], 'r': [-1.0, 1.0],
#  'cov_p': [-6.666666666666667, 6.666666666666667], 'cov_s': [-10.0, 10.0]}
```

## Approximate aggregates

Exact distinct counts and quantiles get expensive on large inputs. The
sketch-backed aggregates trade a little accuracy for bounded memory and
mergeability: `approx_n_unique` (HyperLogLog), `approx_quantile(q)` and
`approx_median` (KLL). They merge exactly across partitions, so the estimate is
identical single-node or distributed. On small inputs it typically matches the
exact count.

```python
approx = ds.group_by("category").agg(
    exact=bt.col("qty").n_unique(),
    approx=bt.col("qty").approx_n_unique(),
).sort("category")
print(approx.to_pydict())
# {'category': ['a', 'b'], 'exact': [3, 2], 'approx': [3, 2]}
```

## Multiple grouping keys

Pass several keys to `group_by` to group by each unique combination.

```python
sales = bt.from_pydict(
    {
        "category": ["a", "a", "b", "b"],
        "region": ["west", "east", "west", "east"],
        "amount": [10.0, 20.0, 30.0, 40.0],
    }
)
by_pair = sales.group_by("category", "region").agg(
    total=bt.col("amount").sum(), n=bt.count()
).sort("category", "region")
print(by_pair.to_pydict())
# {'category': ['a', 'a', 'b', 'b'], 'region': ['east', 'west', 'east', 'west'],
#  'total': [20.0, 10.0, 40.0, 30.0], 'n': [1, 1, 1, 1]}
```

## Global aggregates

Call `group_by()` with no keys to aggregate the whole dataset into one row.

```python
totals = ds.group_by().agg(
    total=bt.col("price").sum(), rows=bt.count()
)
print(totals.to_pydict())
# {'total': [150.0], 'rows': [5]}
```

For a plain row count, the `count()` terminal is shorter:

```python
print(ds.count())
# 5
```

## Derived grouping keys

`group_by` accepts derived expressions, not just column names. Define the key in
`with_columns` (or pass an expression) and group on it.

```python
buckets = (
    ds.with_columns(tier=bt.when(bt.col("price") >= 30.0).then(bt.lit("high")).otherwise(bt.lit("low")))
    .group_by("tier")
    .agg(n=bt.count(), revenue=bt.col("price").sum())
    .sort("tier")
)
print(buckets.to_pydict())
# {'tier': ['high', 'low'], 'n': [3, 2], 'revenue': [120.0, 30.0]}
```

## Next steps

- [Joins](joins.md): combine grouped results with other datasets.
- [Window functions](window-functions.md): per-row aggregates that keep the rows.
- [Performance and memory](performance.md): cache a rollup you reuse, and spill the
  aggregations too big for memory.
- [Expressions API](../api/expressions.md): every aggregate and approximate-aggregate
  method in one place.
