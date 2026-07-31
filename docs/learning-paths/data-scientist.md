# Data scientist learning path

This path is for interactive analysis. You shape data with expressions, ask questions
in SQL or through the DataFrame API, and summarize the answers with aggregations.
Nothing runs while you compose: the API is lazy and immutable, and a terminal operation
is what materializes the result.

## Reading order

1. {doc}`Getting started <../getting-started/index>`: install and run a first query.
1. {doc}`Concepts <../getting-started/concepts/index>`: datasets, laziness, expressions.
1. {doc}`Expressions </user-guide/transform/expressions>`: column math, conditionals, string
   and date accessors.
1. {doc}`Filtering </user-guide/transform/filtering>`: predicates and `is_in` / `between`.
1. {doc}`Aggregations </user-guide/analyze/aggregations>`: `group_by`, `.agg`, quantiles.
1. {doc}`SQL </user-guide/analyze/sql>`: query a dataset with {py:obj}`bt.sql <batcher.sql>`.
1. {doc}`Window functions </user-guide/analyze/window-functions>`: ranking and rolling
   aggregates.
1. {doc}`Expression API reference </api/relational/expressions>` and
   {doc}`SQL API reference </api/relational/sql>`.

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

The {doc}`analytics cookbook </cookbook/analytics/index>` works through the queries you
actually write, and the trap in each one: the cohort query that puts one user in three
cohorts, the 3-sigma rule that never fires because the outlier inflates its own sigma, the
funnel self-join that cross-products inside each user.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Cohort analysis
:link: /cookbook/analytics/cohort-analysis
:link-type: doc
Assign the cohort once, not per row.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Funnel analysis
:link: /cookbook/analytics/funnel-analysis
:link-type: doc
Ordering matters, and the naive join explodes.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Sessionization
:link: /cookbook/analytics/sessionization
:link-type: doc
A gap, a flag, a cumulative sum.
:::

:::{grid-item-card} {octicon}`check;1.1em` A/B testing
:link: /cookbook/analytics/ab-testing
:link-type: doc
Per-event and per-user disagree, and one of them is wrong.
:::
::::

:::{seealso}
- {doc}`SQL to DataFrame <../tutorials/sql-to-dataframe>`: the same query, both ways.
- {doc}`Window functions </user-guide/analyze/window-functions>` and {doc}`pivoting </user-guide/analyze/pivoting>`.
- {doc}`Explain plans </user-guide/operate/explain-plans>`: why your query did what it did.
:::


## See also

- {doc}`ml-engineer`: the path onward, once a model needs to run in production.
- {doc}`../cookbook/statistics/index`: short runnable recipes for the analysis steps.
- {doc}`/cookbook/analytics/index`: worked analytics problems end to end.
