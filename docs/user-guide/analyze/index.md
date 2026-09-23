# Analyze

This section covers turning rows into answers with Batcher: inspecting a dataset to see what it holds, grouping and summarizing, joining, ranking and windowing, and the domain analytics built on the same operators, from time series and geospatial to robotics logs and graphs.

Every one of them is an ordinary relational operation, so none of them needs a separate engine. A grouped aggregate, a join, and a window all lower into one plan that the optimizer reorders, prunes, and pushes down before any row is read, and the Rust data plane runs it over Arrow batches. The stateful operators are built from mergeable parts, so the query you write on a laptop runs unchanged across a Ray cluster and spills to disk when a group or a join side outgrows memory. Even a window with no `PARTITION BY` scales out for most functions, split along its ordering. SQL lowers to the same plan, so a query can start in SQL and finish in Python, or the other way round.

The domain pages hold to the same standard. `ST_*` geometry runs natively in Rust over WKB, with no Python and no conversion step, and on 2 million real map points it measured 1.35x to 12.9x faster than DuckDB's spatial extension over the same Arrow table. Graph algorithms are joins over an edge table stored wherever a table can live. And when the answer is already in a Parquet footer or a table manifest, Batcher reads it from there instead of scanning.

## Pick your analysis

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`search;1.1em` Inspect a dataset
:link: /user-guide/analyze/inspecting-data
:link-type: doc
Schema, previews, `describe`, correlation matrices, approximate quantiles, and what each one costs.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Aggregations
:link: /user-guide/analyze/aggregations
:link-type: doc
Group and summarize, from a plain sum to per-group regression and approximate sketches.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Joins
:link: /user-guide/analyze/joins
:link-type: doc
Inner, outer, semi, anti, set operations, as-of matching, and lookups against a key-value store.
:::

:::{grid-item-card} {octicon}`clock;1.1em` Time series
:link: /user-guide/analyze/time-series
:link-type: doc
Bucketing, gap filling, smoothing, as-of alignment.
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
Geospatial geometry, robotics coordinate frames, and graph algorithms, all on the same operators.
:::

:::{grid-item-card} {octicon}`database;1.1em` SQL
:link: /user-guide/analyze/sql
:link-type: doc
SQL that builds the same plan as the DataFrame API, with sessions and Python functions.
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

- {doc}`/user-guide/transform/index`: select, filter, and shape the rows before you analyze them.
- {doc}`/api/relational/dataset`: the reference for every method in this section.
- {doc}`/cookbook/index`: complete analytics pipelines as runnable recipes.
- {doc}`/examples/analytics`: statistics, time series, geospatial, graph, and robotics as standalone scripts.

```{toctree}
:hidden:

inspecting-data
aggregations
joins
time-series
window-functions
pivoting
sql
sql-model-functions
metadata-shortcuts
domains/index
```
