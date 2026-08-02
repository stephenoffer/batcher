# Cookbook

This section holds every piece of runnable code on the site, about 150 pages, grouped by domain. It merges what used to be two sections: focused recipes for one API surface, and complete pipelines that solve a whole problem.

Each domain now carries both. If you came looking for the method name, the focused recipes are what you want. If you came with a problem, start from the pipeline at the top of the domain and specialise down.

Every page is a complete script you can run unchanged. Each builds its own in-memory data and asserts on its own output, so there are no fixtures to set up, and `tests/docs/test_examples.py` executes all of them on every test run.

:::{tip}
Nothing here is more than three clicks from this page, and no listing runs past a dozen entries. If a group feels long, it has sub-groups.
:::

## The relational core

The verbs, the column language, and the boundary data crosses.

| Domain | Pages | Covers |
|---|---|---|
| {doc}`/cookbook/dataset/index` | 14 | Joins, grouping, reshaping, deduplication, sampling, and the `meta` accessor |
| {doc}`/cookbook/expressions/index` | 34 | The expression API: the scalar algebra, strings, dates and times, and nested types |
| {doc}`/cookbook/io/index` | 6 | Parquet, text formats, Arrow interop, save modes, and the source and sink registries |

## Building and running pipelines

Whole workloads rather than single calls. Each of these opens with a complete pipeline before the focused problems.

| Domain | Pages | Covers |
|---|---|---|
| {doc}`/cookbook/data-engineering/index` | 11 | Ingest, reconcile, and repair tables, starting from a complete ETL pipeline |
| {doc}`/cookbook/analytics/index` | 11 | Cohorts, funnels, sessions, and rankings, starting from one worked query |
| {doc}`/cookbook/streaming/index` | 6 | Unbounded sources, and what time and restarts do to them |

## Models and measurement

Everything that fits a model or scores one, all of it as aggregates and operators inside the engine.

| Domain | Pages | Covers |
|---|---|---|
| {doc}`/cookbook/ml/index` | 26 | Preprocessors, estimators, validation, inference, and complete ML pipelines |
| {doc}`/cookbook/metrics/index` | 20 | Metrics and statistics as aggregates, so both describing a column and scoring a model are a `select` |

## Running it safely

| Domain | Pages | Covers |
|---|---|---|
| {doc}`/cookbook/governance/index` | 3 | Column masking, row filters, PII transforms, and lineage, as plan rewrites |
| {doc}`/cookbook/operations/index` | 7 | Configuration, plan inspection, memory, observability, and error handling |

## Where this sits

Two sections teach by code, and they differ in what they hold constant:

| Section | One page is | Pick it when |
|---|---|---|
| {doc}`Tutorials </tutorials/index>` | One pipeline, built step by step | You are learning the API |
| Cookbook (this section) | One surface or one problem, demonstrated | You know roughly what you need |

## See also

- {doc}`/user-guide/index`: the task-oriented guide behind every recipe here.
- {doc}`/api/index`: the reference, when you want the signature rather than a worked call.
- {doc}`/tutorials/paths/index`: these pages sequenced by the job you do.

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
