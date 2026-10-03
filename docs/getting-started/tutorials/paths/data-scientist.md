# Data scientist learning path

Interactive analysis: shape data with expressions, ask questions in SQL or the DataFrame API, and summarize with aggregations. Nothing runs while you compose; a terminal operation materializes the result.

## Reading order

1. {doc}`Getting started </getting-started/index>`: install and run a first query.
1. {doc}`Concepts </getting-started/concepts/index>`: datasets, laziness, expressions.
1. {doc}`Expressions </user-guide/transform/columns/expressions>`: column math, conditionals, string and date accessors.
1. {doc}`Filtering </user-guide/transform/rows/filtering>`: predicates and `is_in` / `between`.
1. {doc}`Aggregations </user-guide/analyze/aggregations>`: `group_by`, `.agg`, quantiles.
1. {doc}`SQL </user-guide/analyze/sql>`: query a dataset with {py:obj}`bt.sql <batcher.sql>`.
1. {doc}`Window functions </user-guide/analyze/window-functions>`: ranking and rolling aggregates.
1. {doc}`Expression API reference </api/relational/expressions>` and {doc}`SQL API reference </api/relational/sql>`.

## Example: derive and summarize

Bucket each price as high or low, then average each bucket:

```python
import batcher as bt

sales = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a", "c"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }
)

summary = (
    sales.with_columns(
        bucket=bt.when(bt.col("price") > 35.0).then(bt.lit("high")).otherwise(bt.lit("low"))
    )
    .group_by("bucket")
    .agg(avg_price=bt.col("price").mean(), n=bt.count())
    .sort("bucket")
)
print(summary.to_pydict())
# {'bucket': ['high', 'low'], 'avg_price': [50.0, 20.0], 'n': [3, 3]}
```

## Example: ask the same question in SQL

{py:obj}`bt.sql <batcher.sql>` binds a dataset to a table name, runs the query, and hands back a new dataset.

```python
counts = bt.sql(
    "SELECT category, COUNT(*) AS n FROM t GROUP BY category ORDER BY category",
    t=sales,
)
print(counts.to_pydict())
# {'category': ['a', 'b', 'c'], 'n': [3, 2, 1]}
```

## Example: share of category total

A window aggregate compares each row to its group without collapsing rows:

```python
share = sales.with_columns(
    share=(bt.col("price") / bt.col("price").sum().over("category")).round(2)
)
print(share.select("category", "share").to_pydict())
# {'category': ['a', 'b', 'a', 'b', 'a', 'c'], 'share': [0.11, 0.33, 0.33, 0.67, 0.56, 1.0]}
```

## Example: medians and filters

```python
print(sales.filter(bt.col("category").is_in(["a", "b"]))
      .group_by("category").agg(p50=bt.col("price").median())
      .sort("category").to_pydict())
# {'category': ['a', 'b'], 'p50': [30.0, 30.0]}
```

## Runnable examples

Run any of these directly with `python examples/<name>.py`:

- `feature_engineering.py` scales columns, buckets them, encodes categories, and imputes what is missing, all with expressions.
- `preprocessors.py` builds the same features from fit/transform preprocessor objects and {py:class}`Chain <batcher.ml.preprocessors.Chain>`.
- `timeseries.py` covers date-part extraction and resampling, plus period-over-period change.
- `window_functions.py` ranks rows and computes rolling aggregates with {py:meth}`.over(...) <batcher.AggExpr.over>`.
- `sql.py` asks the same questions in SQL, composed with the DataFrame API.

## Recipes

The {doc}`analytics cookbook </cookbook/analytics/index>` works through the analyses you write every week: cohorts, funnels, sessions, and experiments.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Cohort analysis
:link: /cookbook/analytics/behavior/cohort-analysis
:link-type: doc
Assign the cohort once, not per row.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Funnel analysis
:link: /cookbook/analytics/behavior/funnel-analysis
:link-type: doc
Ordered steps per user, without a self-join.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Sessionization
:link: /cookbook/analytics/behavior/sessionization
:link-type: doc
A gap, a flag, a cumulative sum.
:::

:::{grid-item-card} {octicon}`check;1.1em` A/B testing
:link: /cookbook/analytics/inference/ab-testing
:link-type: doc
Compare variants at the right unit of analysis.
:::
::::

## See also

- {doc}`SQL to DataFrame </getting-started/tutorials/foundations/sql-to-dataframe>`: the same query, both ways.
- {doc}`Window functions </user-guide/analyze/window-functions>` and {doc}`pivoting </user-guide/analyze/pivoting>`: ranking, rolling aggregates, and reshaping.
- {doc}`Explain plans </user-guide/operate/tuning/explain-plans>`: why your query did what it did.
- {doc}`/getting-started/tutorials/paths/ml-engineer`: the path onward, once a model needs to run in production.
- {doc}`/cookbook/metrics/statistics/index`: short runnable recipes for the analysis steps.
- {doc}`/cookbook/analytics/index`: worked analytics problems end to end.
