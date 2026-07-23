# User guide

Task-oriented guides for the Dataset API, grouped by what you're doing. One page per
capability, every example runnable.

## Transform

Reshape rows and columns: the expression language, and the operators that select, filter, order, and deduplicate.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`pencil;1.1em` Transformations
:link: transformations
:link-type: doc
Select and derive columns; reshape and explode them.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Filtering
:link: filtering
:link-type: doc
Predicates, null handling, sampling.
:::

:::{grid-item-card} {octicon}`code;1.1em` Expressions
:link: expressions
:link-type: doc
The composable column language and its accessors.
:::

:::{grid-item-card} {octicon}`sort-desc;1.1em` Sorting
:link: sorting
:link-type: doc
Order rows; nulls, NaN, ties, top-n.
:::

:::{grid-item-card} {octicon}`duplicate;1.1em` Distinct and dedup
:link: distinct-and-dedup
:link-type: doc
Exact, keyed, and near-duplicate removal.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Sampling
:link: sampling
:link-type: doc
Reproducible samples and train/test splits.
:::

:::{grid-item-card} {octicon}`code-square;1.1em` UDFs
:link: udfs
:link-type: doc
Your Python over whole Arrow batches.
:::

:::{grid-item-card} {octicon}`typography;1.1em` Type system
:link: type-system
:link-type: doc
Arrow types, boundary widening, casts, nulls.
:::
::::

## Analyze

Reduce many rows to an answer: grouping, joining, ranking, and the SQL front-end over the same engine.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Aggregations
:link: aggregations
:link-type: doc
Group and summarize; pivot; roll up.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Joins
:link: joins
:link-type: doc
Inner, outer, semi, anti, as-of.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Window functions
:link: window-functions
:link-type: doc
Ranking, running totals, lag and lead.
:::

:::{grid-item-card} {octicon}`table;1.1em` Pivoting
:link: pivoting
:link-type: doc
Long to wide and back.
:::

:::{grid-item-card} {octicon}`database;1.1em` SQL
:link: sql
:link-type: doc
Full SQL that lowers to the same engine.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Metadata shortcuts
:link: metadata-shortcuts
:link-type: doc
Answer from the footer instead of the data, with `ds.meta`.
:::
::::

## Move data

Get data in and out: the readers and writers, the storage layer underneath them, and unbounded sources.

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
Files, lakehouse tables, sinks.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Custom connectors
:link: custom-connectors
:link-type: doc
Plug in your own source or sink format.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` Cloud storage
:link: cloud-storage
:link-type: doc
S3, GCS, Azure, on-prem.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Lakehouse
:link: lakehouse
:link-type: doc
Delta, Iceberg, and Hudi tables.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: streaming
:link-type: doc
Windows, watermarks, exactly-once.
:::
::::

## Trust

Decide what counts as a valid row, and who may read which rows and columns.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`checklist;1.1em` Data quality
:link: data-quality
:link-type: doc
Expectations, and the fail/drop/quarantine choice.
:::

:::{grid-item-card} {octicon}`shield-lock;1.1em` Governance and security
:link: governance
:link-type: doc
Column masks, row-level security, lineage, audit.
:::
::::

## Operate

Run the pipeline and understand what it did: caching, plans, progress, and the fixes for a slow or failing job.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.1em` Caching
:link: caching
:link-type: doc
Reuse a result instead of recomputing it.
:::

:::{grid-item-card} {octicon}`telescope;1.1em` Explain plans
:link: explain-plans
:link-type: doc
Read the plan and the measured profile.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Observability
:link: observability
:link-type: doc
Progress, structured logs, and the web dashboard.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Best practices
:link: best-practices
:link-type: doc
Patterns for pipelines that stay fast.
:::

:::{grid-item-card} {octicon}`bug;1.1em` Troubleshooting
:link: troubleshooting
:link-type: doc
Diagnose and fix common issues.
:::
::::

## See also

:::{seealso}
- {doc}`../api/index`: the reference behind every method these guides use.
- {doc}`../examples/index`: the same operations assembled into complete pipelines.
- {doc}`../ml/index`: the model half of the pipeline, once the relational half is in place.
- {doc}`../configuration/index`: the tunables the performance and memory guides refer to.
- {doc}`../deep-dives/index`: why an operator behaves the way these pages describe.
- {doc}`../integrations/index`: connecting a specific source or sink.
:::

```{toctree}
:hidden:
:caption: Transform

transformations
filtering
expressions
sorting
distinct-and-dedup
sampling
udfs
type-system
```

```{toctree}
:hidden:
:caption: Analyze

aggregations
joins
window-functions
pivoting
sql
metadata-shortcuts
```

```{toctree}
:hidden:
:caption: Move data

reading-data
writing-data
custom-connectors
cloud-storage
lakehouse
streaming
```

```{toctree}
:hidden:
:caption: Trust

data-quality
governance
```

```{toctree}
:hidden:
:caption: Operate

performance
caching
explain-plans
observability
best-practices
troubleshooting
```
