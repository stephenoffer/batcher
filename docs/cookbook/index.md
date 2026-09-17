# Cookbook

The cookbook is 145 runnable recipes, each one a working answer you can copy into a pipeline. They're grouped by domain, and the larger domains carry two kinds. A focused recipe demonstrates one API surface, such as the `.str` accessor or the join types. A complete pipeline solves a whole problem, such as sessionizing a click stream or applying a change feed to a table. Came looking for a method name? Browse the focused recipes. Came with a problem? Start from the pipeline at the top of the domain and work down.

The figure maps the ten domains onto the four groups this page is organized by, with each domain's recipe count.

![The 145 recipes in four groups. The relational core, 59 recipes: Dataset with 14 (joins, grouping, reshaping), Expressions with 39 (strings, dates, nested types), and I/O with 6 (Parquet, text formats, Arrow). Building and running pipelines, 29 recipes: Data engineering with 11 (ingest, reconcile, repair), Analytics with 11 (cohorts, funnels, sessions), and Streaming with 7 (time and restarts). Models and measurement, 47 recipes: ML with 27 (preprocessing to inference) and Metrics with 20 (metrics and statistics). Running it safely, 10 recipes: Governance with 3 (masks, row filters, lineage) and Operations with 7 (configuration, plans, memory).](/_static/diagrams/cookbook_map.svg)

Every recipe is a complete script you can run unchanged. Each builds its own in-memory data and asserts on its own output, so there are no fixtures to download. The test suite runs all of it on every pass, `tests/docs/test_doc_examples.py` for the code written into a page and `tests/docs/test_examples.py` for the scripts a page embeds from `examples/`. A recipe that stops matching the engine fails the build rather than going stale.

If you're not sure where to begin, start with any of the following:

- {doc}`/cookbook/data-engineering/ingest/etl-pipeline`: raw records in, deduplicated and rolled up, Parquet out.
- {doc}`/cookbook/analytics/aggregates/analytics-query`: aggregate, join, and window over one orders table, as SQL and as DataFrame code.
- {doc}`/cookbook/dataset/inspecting/profiling`: the first measurements to take on a table you've just been handed.

## The relational core

The verbs, the column language, and the boundary data crosses.

| Domain | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/dataset/index` | 14 | Joins, grouping, reshaping, deduplication, sampling, and the `meta` accessor |
| {doc}`/cookbook/expressions/index` | 39 | The expression API: the scalar algebra, strings, dates and times, and nested types |
| {doc}`/cookbook/io/index` | 6 | Parquet, text formats, Arrow interop, save modes, and the source and sink registries |

## Building and running pipelines

Whole workloads rather than single calls. Each of these opens with a complete pipeline before the focused problems.

| Domain | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/data-engineering/index` | 11 | Ingest, reconcile, and repair tables, starting from a complete ETL pipeline |
| {doc}`/cookbook/analytics/index` | 11 | Cohorts, funnels, sessions, and rankings, starting from one worked query |
| {doc}`/cookbook/streaming/index` | 7 | Unbounded sources, and what time and restarts do to them |

## Models and measurement

Everything that fits a model or scores one, all of it as aggregates and operators inside the engine.

| Domain | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/ml/index` | 27 | Preprocessors, estimators, validation, inference, and complete ML pipelines |
| {doc}`/cookbook/metrics/index` | 20 | Metrics and statistics as aggregates, so both describing a column and scoring a model are a `select` |

## Running it safely

The policy and operations surfaces, for a pipeline other people depend on.

| Domain | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/governance/index` | 3 | Column masking, row filters, PII transforms, and lineage, as plan rewrites |
| {doc}`/cookbook/operations/index` | 7 | Configuration, plan inspection, memory, observability, and error handling |

## Where this sits

Two sections teach by code, and they differ in what they hold constant:

| Section | One page is | Pick it when |
|---|---|---|
| {doc}`Tutorials </getting-started/tutorials/index>` | One pipeline, built step by step | You are learning the API |
| Cookbook (this section) | One surface or one problem, demonstrated | You know roughly what you need |

## See also

- {doc}`/user-guide/index`: the task-oriented guide behind every recipe here.
- {doc}`/api/index`: the reference, when you want the signature rather than a worked call.
- {doc}`/getting-started/tutorials/paths/index`: these pages sequenced by the job you do.

```{toctree}
:hidden:
:caption: The relational core

dataset/index
expressions/index
io/index
```

```{toctree}
:hidden:
:caption: Building and running pipelines

data-engineering/index
analytics/index
streaming/index
```

```{toctree}
:hidden:
:caption: Models and measurement

ml/index
metrics/index
```

```{toctree}
:hidden:
:caption: Running it safely

governance/index
operations/index
```
