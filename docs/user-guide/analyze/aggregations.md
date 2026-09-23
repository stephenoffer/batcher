# Aggregations

This page covers aggregation in Batcher: reducing many rows to summary values, over the whole dataset or per group. You group with `group_by` and finish with `agg`, and everything from a plain sum to a per-group regression fits that one shape. Every aggregate on this page is built from mergeable parts, so the same code runs on one core or across a cluster.

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

## Group and aggregate

A grouped aggregate makes two moves. `group_by` gathers the rows that share a key, and `agg` reduces each group to one output row.

![Five input rows of category and price, a 10.0, b 20.0, a 30.0, b 40.0 and a 50.0, flow through group_by into two groups: a holds 10.0, 30.0 and 50.0, and b holds 20.0 and 40.0. agg then reduces each group to one row, a with total 90.0 and rows 3, and b with total 60.0 and rows 2, from ds.group_by("category").agg(total=bt.col("price").sum(), rows=bt.count()). Below, a group_by with no keys treats the whole dataset as one group and returns one row, total 150.0 and rows 5.](/_static/diagrams/group_by_flow.svg)

`group_by` takes the grouping keys and `agg` takes the output aggregates. Pass an aggregate as a keyword to name the output, or positionally to keep the source column's name. {py:obj}`bt.count() <batcher.count>` is `COUNT(*)`. The column aggregates are methods on an expression (`.sum()`, `.mean()`, and so on) or the top-level shorthands {py:obj}`bt.sum("x") <batcher.sum>`, {py:func}`bt.mean <batcher.mean>`, {py:func}`bt.min <batcher.min>`, {py:func}`bt.max <batcher.max>`, {py:func}`bt.median <batcher.median>`, {py:func}`bt.std <batcher.std>`, {py:func}`bt.var <batcher.var>`, {py:func}`bt.count_distinct <batcher.count_distinct>`. {py:obj}`bt.sum("x") <batcher.sum>` reads as `col("x").sum()`, the Polars `pl.sum` convention.

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

When you reduce *every* value column the same way, a shortcut method is shorter than spelling out `agg`. The set is `sum`, `mean`, `min`, `max`, `median`, `quantile(q)`, `count_distinct`, `std`, `var`, `count` (non-null values per column), and `len` (the per-group row count). With no arguments they reduce every non-key column, keeping its name. Pass column names or a {doc}`selector </user-guide/transform/rows/transformations>` to reduce a subset. The arithmetic reductions (`sum`, `mean`, `median`, `quantile`, `std`, `var`) default to numeric columns only, matching pandas' `numeric_only`.

```python
print(ds.group_by("category").sum().sort("category").to_pydict())
# {'category': ['a', 'b'], 'price': [90.0, 60.0], 'qty': [9, 6]}

print(ds.group_by("category").mean("price").sort("category").to_pydict())
# {'category': ['a', 'b'], 'price': [30.0, 30.0]}

print(ds.group_by("category").len().sort("category").to_pydict())
# {'category': ['a', 'b'], 'len': [3, 2]}
```

`len` counts rows and `count` counts the non-null values of each column, so the two differ only when a column has nulls. Reach for `agg` when the reductions differ per column, when you want custom output names, or when you need a windowed aggregate or a two-column statistic.

## Aggregate functions

The aggregate methods available inside `agg` are `sum`, `min`, `max`, `mean`, `var`, `std`, `median`, `quantile(q)`, `count`, and {py:meth}`count_distinct <batcher.plan.expr_ir.core.Expr.count_distinct>`. {py:obj}`bt.count() <batcher.count>` counts rows. Each of these builds an {py:class}`AggExpr <batcher.AggExpr>`, the aggregate type that `agg(...)` consumes and that {py:meth}`.over(...) <batcher.AggExpr.over>` lifts into a {doc}`window function </user-guide/analyze/window-functions>`. You rarely name it directly.

```python
stats = (
    ds.group_by("category")
    .agg(
        total=bt.col("price").sum(),
        avg=bt.col("price").mean(),
        lo=bt.col("price").min(),
        hi=bt.col("price").max(),
        med=bt.col("price").median(),
        p90=bt.col("price").quantile(0.9),
        distinct_qty=bt.col("qty").count_distinct(),
    )
    .sort("category")
)
print(stats.to_pydict())
# {'category': ['a', 'b'], 'total': [90.0, 60.0], 'avg': [30.0, 30.0], 'lo': [10.0, 20.0],
#  'hi': [50.0, 40.0], 'med': [30.0, 30.0], 'p90': [46.0, 38.0], 'distinct_qty': [3, 2]}
```

## Advanced aggregates

Beyond the basics, `agg` supports `mode`, `first`/`last`, `min_by`/`max_by` (the value of one column at the row that minimizes/maximizes another), `arg_min`/`arg_max` (the position of a column's own extreme), `top_k` and `mode_top_k` (a group's largest and most frequent values as a list), the boolean reductions `bool_and`/`bool_or`, and `array_agg` (collect a group's values into a list).

```python
adv = (
    ds.group_by("category")
    .agg(
        any_big=(bt.col("price") > 35).bool_or(),
        all_big=(bt.col("price") > 35).bool_and(),
        costliest=bt.col("price").max_by(bt.col("price")),
    )
    .sort("category")
)
print(adv.to_pydict())
# {'category': ['a', 'b'], 'any_big': [True, True], 'all_big': [False, False],
#  'costliest': [50.0, 40.0]}
```

Each of these also has a top-level SQL-style spelling that reads `bt.<agg>("col")`, the same shorthand {py:obj}`bt.sum("x") <batcher.sum>` is for `col("x").sum()`: {py:obj}`bt.product(x) <batcher.product>`, {py:obj}`bt.mode(x) <batcher.mode>`, {py:obj}`bt.skew(x) <batcher.skew>` / {py:obj}`bt.kurtosis(x) <batcher.kurtosis>`, {py:obj}`bt.bool_and(x) <batcher.bool_and>` / {py:obj}`bt.bool_or(x) <batcher.bool_or>`, {py:obj}`bt.bit_and(x) <batcher.bit_and>` / {py:obj}`bt.bit_or(x) <batcher.bit_or>` / {py:obj}`bt.bit_xor(x) <batcher.bit_xor>`, and {py:obj}`bt.array_agg(x) <batcher.array_agg>`.

```python
shorthand = (
    ds.group_by("category")
    .agg(
        prod=bt.product("price"),
        values=bt.array_agg("price"),
    )
    .sort("category")
)
print(shorthand.to_pydict())
# {'category': ['a', 'b'], 'prod': [15000.0, 800.0], 'values': [[10.0, 30.0, 50.0], [20.0, 40.0]]}
```

## Bivariate aggregates

The two-column statistical aggregates summarize how a pair of columns move together within each group. {py:obj}`bt.corr(x, y) <batcher.corr>` is the Pearson correlation coefficient in `[-1, 1]` (SQL `CORR`). {py:obj}`bt.covar_pop(x, y) <batcher.covar_pop>` and {py:obj}`bt.covar_samp(x, y) <batcher.covar_samp>` are the population and sample covariance (SQL `COVAR_POP` / `COVAR_SAMP`, dividing by `n` and `n - 1` respectively). Reach for `corr` to score the strength and sign of a relationship: whether ad spend tracks revenue per region, say.

```python
market = bt.from_pydict(
    {
        "region": ["west", "west", "west", "east", "east", "east"],
        "spend": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "revenue": [10.0, 20.0, 30.0, 30.0, 20.0, 10.0],
    }
)
bivariate = (
    market.group_by("region")
    .agg(
        r=bt.corr(bt.col("spend"), bt.col("revenue")),
        cov_p=bt.covar_pop(bt.col("spend"), bt.col("revenue")),
        cov_s=bt.covar_samp(bt.col("spend"), bt.col("revenue")),
    )
    .sort("region")
)
print(bivariate.to_pydict())
# {'region': ['east', 'west'], 'r': [-0.9999999999999998, 0.9999999999999998],
#  'cov_p': [-6.666666666666667, 6.666666666666667], 'cov_s': [-10.0, 10.0]}
```

A perfect correlation prints as `0.9999999999999998` rather than `1.0`: the coefficient is a ratio of floating-point sums, so the last bits are the arithmetic's, not the data's.

## Expressions over aggregates

An `agg` keyword takes not just a single aggregate but a whole expression *over* aggregates: a ratio, a difference, any arithmetic. `col("price").sum() / bt.count()` is an average priced as one aggregate pass; `col("price").max() - col("price").min()` is the per-group spread. The engine computes each distinct aggregate once and evaluates the surrounding arithmetic in a projection, so the result is identical single-node and distributed. Aggregates cannot be nested.

```python
derived = (
    ds.group_by("category")
    .agg(
        revenue=(bt.col("price") * bt.col("qty")).sum(),
        avg_price=bt.col("price").sum() / bt.count(),
        spread=bt.col("price").max() - bt.col("price").min(),
    )
    .sort("category")
)
print(derived.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0], 'avg_price': [30.0, 30.0], 'spread': [40.0, 20.0]}
```

## Linear regression

The `regr_*` family is built on expressions over aggregates. It fits a least-squares line of a dependent column `y` on an independent column `x` per group, matching the SQL / DuckDB / PostgreSQL functions. {py:obj}`bt.regr_slope(y, x) <batcher.regr_slope>` and {py:obj}`bt.regr_intercept(y, x) <batcher.regr_intercept>` give the line, {py:obj}`bt.regr_r2(y, x) <batcher.regr_r2>` its fit, and {py:obj}`bt.regr_count(y, x) <batcher.regr_count>`, {py:obj}`bt.regr_avgx(y, x) <batcher.regr_avgx>` / {py:obj}`bt.regr_avgy(y, x) <batcher.regr_avgy>`, and {py:obj}`bt.regr_sxx(y, x) <batcher.regr_sxx>` / {py:obj}`bt.regr_syy(y, x) <batcher.regr_syy>` / {py:obj}`bt.regr_sxy(y, x) <batcher.regr_sxy>` the underlying moments. Every function uses only rows where both columns are non-null. Because each result is an expression, you can round or combine it further.

```python
market = bt.from_pydict(
    {
        "region": ["west", "west", "west", "east", "east", "east"],
        "spend": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "revenue": [15.0, 25.0, 35.0, 35.0, 25.0, 15.0],
    }
)
fit = (
    market.group_by("region")
    .agg(
        slope=bt.regr_slope(bt.col("revenue"), bt.col("spend")).round(2),
        intercept=bt.regr_intercept(bt.col("revenue"), bt.col("spend")).round(2),
        r2=bt.regr_r2(bt.col("revenue"), bt.col("spend")).round(4),
        n=bt.regr_count(bt.col("revenue"), bt.col("spend")),
    )
    .sort("region")
)
print(fit.to_pydict())
# {'region': ['east', 'west'], 'slope': [-10.0, 10.0], 'intercept': [45.0, 5.0], 'r2': [1.0, 1.0], 'n': [3, 3]}
```

## Derived statistics

Because an aggregate result is itself an expression, a family of standard statistics that the base aggregates don't name directly comes for free. Each is a small formula over the mergeable primitives, so it stays identical single-node and distributed. {py:obj}`bt.var_pop(x) <batcher.var_pop>` / {py:obj}`bt.stddev_pop(x) <batcher.stddev_pop>` are the *population* variance and standard deviation, where Batcher's `var` and `std` are the sample forms. {py:obj}`bt.geometric_mean(x) <batcher.geometric_mean>`, {py:obj}`bt.harmonic_mean(x) <batcher.harmonic_mean>`, and {py:obj}`bt.rms(x) <batcher.rms>` are the geometric, harmonic, and quadratic means. {py:obj}`bt.cv(x) <batcher.cv>`, {py:obj}`bt.sem(x) <batcher.sem>`, and {py:obj}`bt.midrange(x) <batcher.midrange>` give the coefficient of variation, the standard error of the mean, and the midrange. {py:obj}`bt.weighted_mean(value, weight) <batcher.weighted_mean>` averages one column in proportion to another. You can also apply a math function to any aggregate yourself, such as `col("x").sum().sqrt()` or `col("x").mean().round(2)`.

```python
stats = (
    ds.group_by("category")
    .agg(
        pop_std=bt.stddev_pop("price").round(3),
        geo=bt.geometric_mean("price").round(3),
        rms=bt.rms("price").round(3),
        cv=bt.cv("price").round(3),
    )
    .sort("category")
)
print(stats.to_pydict())
# {'category': ['a', 'b'], 'pop_std': [16.33, 10.0], 'geo': [24.662, 28.284], 'rms': [34.157, 31.623], 'cv': [0.667, 0.471]}
```

## Approximate aggregates

Exact distinct counts and quantiles get expensive on large inputs. The sketch-backed aggregates trade a little accuracy for bounded memory and mergeability: `approx_count_distinct` uses HyperLogLog, and `approx_quantile(q)` and `approx_median` use DDSketch, with roughly 1% relative error. Both merge to the same state in any order: HyperLogLog takes the register-wise max and DDSketch sums fixed logarithmic buckets. So each estimate is identical single-node or distributed, and on small inputs `approx_count_distinct` typically matches the exact count. Each also has a top-level spelling: {py:obj}`bt.approx_count_distinct(x) <batcher.approx_count_distinct>`, {py:obj}`bt.approx_quantile(x, q) <batcher.approx_quantile>`, and {py:obj}`bt.approx_median(x) <batcher.approx_median>`. They sit alongside the exact {py:obj}`bt.quantile(x, q) <batcher.quantile>` and the value-tally {py:obj}`bt.histogram(x) <batcher.histogram>`.

```python
approx = (
    ds.group_by("category")
    .agg(
        exact=bt.col("qty").count_distinct(),
        approx=bt.col("qty").approx_count_distinct(),
    )
    .sort("category")
)
print(approx.to_pydict())
# {'category': ['a', 'b'], 'exact': [3, 2], 'approx': [3, 2]}
```

## Grouping keys

A key is a column name or an expression, and there can be any number of them. Pass several names to {py:meth}`group_by <batcher.Dataset.group_by>` to group by each unique combination.

```python
sales = bt.from_pydict(
    {
        "category": ["a", "a", "b", "b"],
        "region": ["west", "east", "west", "east"],
        "amount": [10.0, 20.0, 30.0, 40.0],
    }
)
by_pair = (
    sales.group_by("category", "region")
    .agg(total=bt.col("amount").sum(), n=bt.count())
    .sort("category", "region")
)
print(by_pair.to_pydict())
# {'category': ['a', 'a', 'b', 'b'], 'region': ['east', 'west', 'east', 'west'],
#  'total': [20.0, 10.0, 40.0, 30.0], 'n': [1, 1, 1, 1]}
```

A derived key works the same way. Define it in {py:meth}`with_columns <batcher.Dataset.with_columns>` (or pass the expression straight to `group_by`) and group on the result.

```python
buckets = (
    ds.with_columns(
        tier=bt.when(bt.col("price") >= 30.0).then(bt.lit("high")).otherwise(bt.lit("low"))
    )
    .group_by("tier")
    .agg(n=bt.count(), revenue=bt.col("price").sum())
    .sort("tier")
)
print(buckets.to_pydict())
# {'tier': ['high', 'low'], 'n': [3, 2], 'revenue': [120.0, 30.0]}
```

## Filtering groups

{py:meth}`having <batcher.GroupBy.having>` keeps only the groups for which a predicate over their aggregates holds, which is SQL's `HAVING`. It goes between `group_by` and the reduction that finishes it, and the predicate's aggregates run in the same pass as the outputs.

```python
big = ds.group_by("category").having(bt.col("price").sum() > 70).agg(n=bt.count())
print(big.to_pydict())
# {'category': ['a'], 'n': [3]}
```

A filter before `group_by` removes rows, and `having` removes whole groups after they are summarized. A group whose predicate is null is dropped, as in SQL.

## Group order

A grouped result has no defined row order, which is why the examples above end in `sort`. Pass `maintain_order=True` to `group_by` to get the groups in the order each one first appears in the input, as Polars does. The order is computed rather than observed: Batcher numbers the input rows, keeps each group's smallest number, and sorts on it, so the order is the same under `collect`, spilling, `iter_batches` and `distributed=True`. The price is one sort over the groups.

```python
visits = bt.from_pydict({"city": ["oslo", "lima", "oslo", "baku"], "n": [1, 2, 3, 4]})
print(visits.group_by("city", maintain_order=True).agg(total=bt.col("n").sum()).to_pydict())
# {'city': ['oslo', 'lima', 'baku'], 'total': [4, 2, 4]}
```

It orders the groups an aggregation emits, so `head`, `tail` and `map_groups`, which return rows, raise rather than ignore it. A streaming input has no first appearance to sort by and raises too.

## Global aggregates

Call `group_by()` with no keys to aggregate the whole dataset into one row.

```python
totals = ds.group_by().agg(total=bt.col("price").sum(), rows=bt.count())
print(totals.to_pydict())
# {'total': [150.0], 'rows': [5]}
```

For a plain row count, the `count()` terminal is shorter:

```python
print(ds.count())
# 5
```

Each single-column reduction also has a **scalar terminal** that skips the one-row frame and hands back the value itself. `min`, `max`, `sum`, `mean`, `median`, `quantile`, `std`, `var` and `count_distinct` are joined by the distribution's shape and by the boolean reductions:

```python
print(ds.product("price"), ds.mode("category"))
# 12000000.0 a

print(ds.skew("price"), ds.kurtosis("price"), ds.mad("price"))
# 0.0 -1.2000000000000004 10.0
```

{py:meth}`mad <batcher.Dataset.mad>` is the mean absolute deviation. It does not square the deviations the way the standard deviation does, so one far-out value moves it far less. Prefer it when outliers are expected rather than exceptional.

{py:meth}`any <batcher.Dataset.any>` and {py:meth}`all <batcher.Dataset.all>` reduce a boolean column. Both return `None` for an empty or all-null column rather than `False` or `True`, so a vacuous answer never passes for a checked one:

```python
flags = bt.from_pydict({"ok": [True, True, False]})
print(flags.any("ok"), flags.all("ok"))
# True False
```

## See also

- {doc}`Joins </user-guide/analyze/joins>`: combine grouped results with other datasets.
- {doc}`Window functions </user-guide/analyze/window-functions>`: per-row aggregates that keep the rows.
- {doc}`Performance and memory </user-guide/operate/tuning/performance>`: cache a rollup you reuse, and spill the aggregations too big for memory.
- {doc}`Expressions API </api/relational/expressions>`: every aggregate and approximate-aggregate method in one place.
- {doc}`/cookbook/dataset/verbs/grouping`: a runnable script for `agg`, multi-key rollups, and the cube/rollup variants.
