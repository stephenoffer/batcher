# User guide

Task-oriented guides for the Dataset API, grouped by what you're doing. Each page
covers one capability with runnable examples.

## Transform

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`pencil;1.1em` Transformations
:link: transformations
:link-type: doc
Select, derive, reshape, and explode columns.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Filtering
:link: filtering
:link-type: doc
Predicates, null handling, and sampling.
:::

:::{grid-item-card} {octicon}`code;1.1em` Expressions
:link: expressions
:link-type: doc
The composable column language and its accessors.
:::
::::

## Analyze

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Aggregations
:link: aggregations
:link-type: doc
Group, summarize, pivot, and roll up.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Joins
:link: joins
:link-type: doc
Inner, outer, semi, anti, and as-of joins.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Window functions
:link: window-functions
:link-type: doc
Ranking, running totals, lag and lead.
:::

:::{grid-item-card} {octicon}`database;1.1em` SQL
:link: sql
:link-type: doc
Full SQL that lowers to the same engine.
:::
::::

## Move data

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`download;1.1em` Reading data
:link: reading-data
:link-type: doc
Files, object storage, databases, streams.
:::

:::{grid-item-card} {octicon}`upload;1.1em` Writing data
:link: writing-data
:link-type: doc
Files, lakehouse tables, and sinks.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` Cloud storage
:link: cloud-storage
:link-type: doc
S3, GCS, Azure, and on-prem.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: streaming
:link-type: doc
Windows, watermarks, exactly-once.
:::
::::

## Operate

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`light-bulb;1.1em` Best practices
:link: best-practices
:link-type: doc
Patterns for fast, reliable pipelines.
:::

:::{grid-item-card} {octicon}`bug;1.1em` Troubleshooting
:link: troubleshooting
:link-type: doc
Diagnose and fix common issues.
:::
::::

```{toctree}
:hidden:
:caption: Transform

transformations
filtering
expressions
```

```{toctree}
:hidden:
:caption: Analyze

aggregations
joins
window-functions
sql
```

```{toctree}
:hidden:
:caption: Move data

reading-data
writing-data
cloud-storage
streaming
```

```{toctree}
:hidden:
:caption: Operate

best-practices
troubleshooting
```
