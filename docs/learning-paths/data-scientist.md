# Data scientist learning path

This path is for interactive analysis. You shape data with expressions, ask questions
in SQL or through the DataFrame API, and summarize the answers with aggregations.
Nothing runs while you compose: the API is lazy and immutable, and a terminal operation
is what materializes the result.

## Reading order

1. [Getting started](../getting-started/index.md): install and run a first query.
2. [Concepts](../getting-started/concepts/index.md): datasets, laziness, expressions.
3. [Expressions](../user-guide/expressions.md): column math, conditionals, string
   and date accessors.
4. [Filtering](../user-guide/filtering.md): predicates and `is_in` / `between`.
5. [Aggregations](../user-guide/aggregations.md): `group_by`, `.agg`, quantiles.
6. [SQL](../user-guide/sql.md): query a dataset with {py:obj}`bt.sql <batcher.sql>`.
7. [Window functions](../user-guide/window-functions.md): ranking and rolling
   aggregates.
8. [Expression API reference](../api/expressions.md) and
   [SQL API reference](../api/sql.md).

## Example: derive and summarize

```python
import batcher as bt

sales = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a", "c"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }
)

summary = (
    sales.with_columns(bucket=bt.when(bt.col("price") > 35.0).then(bt.lit("high")).otherwise(bt.lit("low")))
    .group_by("bucket")
    .agg(avg_price=bt.col("price").mean(), n=bt.count())
    .sort("bucket")
)
print(summary.to_pydict())
# {'bucket': ['high', 'low'], 'avg_price': [50.0, 20.0], 'n': [3, 3]}
```

## Example: ask the same question in SQL

{py:obj}`bt.sql <batcher.sql>` binds a dataset to a table name, runs the query, and hands
back a new dataset.

```python
counts = bt.sql(
    "SELECT category, COUNT(*) AS n FROM t GROUP BY category ORDER BY category",
    t=sales,
)
print(counts.to_pydict())
# {'category': ['a', 'b', 'c'], 'n': [3, 2, 1]}
```

## Runnable examples

Run any of these directly with `python examples/<name>.py`:

- `feature_engineering.py` scales columns, buckets them, encodes categories, imputes
  what is missing, all with expressions.
- `preprocessors.py` builds the same features from fit/transform preprocessor objects
  and `Chain`.
- `timeseries.py` covers date-part extraction and resampling, plus period-over-period
  change.
- `window_functions.py` ranks rows and computes rolling aggregates with `.over(...)`.
- `sql.py` asks the same questions in SQL, composed with the DataFrame API.


## Recipes

The [analytics cookbook](../examples/analytics/index.md) works through the queries you
actually write, and the trap in each one: the cohort query that puts one user in three
cohorts, the 3-sigma rule that never fires because the outlier inflates its own sigma, the
funnel self-join that cross-products inside each user.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Cohort analysis
:link: ../examples/analytics/cohort-analysis
:link-type: doc
Assign the cohort once, not per row.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Funnel analysis
:link: ../examples/analytics/funnel-analysis
:link-type: doc
Ordering matters, and the naive join explodes.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Sessionization
:link: ../examples/analytics/sessionization
:link-type: doc
A gap, a flag, a cumulative sum.
:::

:::{grid-item-card} {octicon}`check;1.1em` A/B testing
:link: ../examples/analytics/ab-testing
:link-type: doc
Per-event and per-user disagree, and one of them is wrong.
:::
::::

:::{seealso}
- [SQL to DataFrame](../tutorials/sql-to-dataframe.md) — the same query, both ways.
- [Window functions](../user-guide/window-functions.md) and [pivoting](../user-guide/pivoting.md).
- [Explain plans](../user-guide/explain-plans.md) — why your query did what it did.
:::
