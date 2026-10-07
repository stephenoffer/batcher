# Analyze

This section covers turning rows into answers with Batcher. It starts with looking at what a dataset holds. From there it covers summarizing and combining tables, and ends with domain work such as time series, geospatial queries, robotics logs and graphs.

Nothing here needs a separate engine. Grouping, joins, windows and SQL all lower into one plan, which the optimizer prunes and pushes down before a row is read. That plan runs on a laptop. It spills when it outgrows memory, and it scales out on a Ray cluster without a rewrite.

```python
import batcher as bt

sales = bt.from_pydict(
    {"region": ["west", "east", "west", "east"], "rep": ["ann", "bo", "cy", "bo"], "amount": [120.0, 80.0, 45.0, 300.0]}
)
print(sales.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict())
# {'region': ['east', 'west'], 'total': [380.0, 165.0]}
```

Joins, windows and SQL look much the same:

```python
reps = bt.from_pydict({"rep": ["ann", "bo", "cy"], "team": ["a", "b", "a"]})
print(sales.join(reps, on="rep").select("rep", "team", "amount").sort("amount").to_pydict())
# {'rep': ['cy', 'bo', 'ann', 'bo'], 'team': ['a', 'b', 'a', 'b'], 'amount': [45.0, 80.0, 120.0, 300.0]}

ranked = sales.with_columns(rank=bt.col("amount").rank(descending=True).over("region"))
print(ranked.sort("region", "rank").select("region", "rep", "rank").to_pydict())
# {'region': ['east', 'east', 'west', 'west'], 'rep': ['bo', 'bo', 'ann', 'cy'], 'rank': [1, 2, 1, 2]}

print(bt.sql("SELECT region, MAX(amount) AS top FROM sales GROUP BY region ORDER BY region", sales=sales).to_pydict())
# {'region': ['east', 'west'], 'top': [300.0, 120.0]}
```

The domain pages reuse those operators. `ST_*` geometry runs natively in Rust over WKB. Graph algorithms are joins over an edge table. And when the answer already sits in a Parquet footer or a table manifest, {py:obj}`ds.meta <batcher.Dataset.meta>` reads it from there instead of scanning.

## Pick your analysis

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`search;1.1em` Inspect a dataset
:link: /user-guide/analyze/inspecting-data
:link-type: doc
Schema, previews and `describe`, up to correlation matrices and approximate quantiles, with what each costs.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Aggregations
:link: /user-guide/analyze/aggregations
:link-type: doc
Group and summarize, from a plain sum to per-group regression and approximate sketches.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Joins
:link: /user-guide/analyze/joins
:link-type: doc
Every join type plus set operations, as-of matching and lookups against a key-value store.
:::

:::{grid-item-card} {octicon}`clock;1.1em` Time series
:link: /user-guide/analyze/time-series
:link-type: doc
Bucketing and gap filling, smoothing, as-of alignment.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Window functions
:link: /user-guide/analyze/window-functions
:link-type: doc
Ranking, running totals, lag and lead.
:::

:::{grid-item-card} {octicon}`table;1.1em` Pivoting
:link: /user-guide/analyze/pivoting
:link-type: doc
Long to wide and back, from Python or SQL.
:::

:::{grid-item-card} {octicon}`globe;1.1em` Domain analytics
:link: /user-guide/analyze/domains/index
:link-type: doc
Geometry, robot coordinate frames and graph algorithms, built on the same operators.
:::

:::{grid-item-card} {octicon}`database;1.1em` SQL
:link: /user-guide/analyze/sql
:link-type: doc
SQL that builds the same plan as the DataFrame API, with sessions and Python functions.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` SQL parameters, scripts, and errors
:link: /user-guide/analyze/sql-parameters
:link-type: doc
Bind values with `params=`, check a query before it runs, and read SQL errors.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Model and AI functions in SQL
:link: /user-guide/analyze/sql-model-functions
:link-type: doc
`ML_PREDICT`, `AI_GENERATE`, and `AI_EXTRACT`.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Metadata shortcuts
:link: /user-guide/analyze/metadata-shortcuts
:link-type: doc
Answer from the footer instead of the data, with {py:obj}`ds.meta <batcher.Dataset.meta>`.
:::
::::

## See also

Where to go before, beside and after this section.

- {doc}`/user-guide/transform/index`: shape the rows before you analyze them.
- {doc}`/api/relational/dataset`: the reference for every method in this section.
- {doc}`/cookbook/index`: whole analytics pipelines you can run.
- {doc}`/examples/analytics`: the same analyses, statistics through robotics, as standalone scripts.

```{toctree}
:hidden:

inspecting-data
aggregations
joins
time-series
window-functions
pivoting
sql
sql-parameters
sql-model-functions
metadata-shortcuts
domains/index
```
