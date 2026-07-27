# API reference

The Batcher API is small and lazy: you build a `Dataset` from a source, transform it
with expression-based operations, and execute it with a terminal operation that
returns Arrow or writes to a sink. Everything reachable from `import batcher as bt`
is documented here. There are three ways in:

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` Quick reference
:link: reference
:link-type: doc
A one-page cheat sheet of the common functions, methods, and patterns.
:::

:::{grid-item-card} {octicon}`book;1.1em` By area
:link: dataset
:link-type: doc
Example-first pages for one surface at a time, starting with the `Dataset`.
:::

:::{grid-item-card} {octicon}`list-unordered;1.1em` Complete reference
:link: complete
:link-type: doc
The backstop index: every symbol without an area page of its own.
:::
::::

(by-area)=

## By area

These are the curated, example-first references, grouped the way you'd look something up. Each one leads with a runnable example and then enumerates the surface. {doc}`complete` is the backstop index for anything without a page of its own, and {doc}`reference` is the cheat sheet to keep open while you work.

- [Dataset](dataset.md): build, transform, aggregate, join, and collect.
- [Expressions](expressions.md): column math, predicates, operators, and window methods.
- [Expression accessors](expression-accessors.md): every `.str`, `.dt`, `.list`, `.struct`, `.json`, `.map`, `.image`, `.audio`, and `.video` method.
- [Functions](functions.md): scalar, horizontal, aggregate, and window functions.
- [Metrics](metrics.md): scoring and statistical aggregates.
- [Reading and writing](io.md): every reader and writer, with the optional extras.
- [SQL](sql.md): the SQL surface and how it lowers to the DataFrame API.
- [ML](ml.md): the `.ml` accessor, plus the LLM, serving, loader, and vector surfaces.
- [Preprocessors](preprocessors.md): the fit/transform estimators and `Chain`.
- [Models and evaluation](ml-models.md): tabular scoring, in-engine estimators, metrics.
- [Statistics and validation](ml-statistics.md): drift, fairness, resampling, cross-validation.
- [Governance](governance.md): row filters, column masks, grants, and lineage.
- [Configuration](configuration.md): the tunables and how they're set.
- [Errors](exceptions.md): the typed exceptions and what raises them.

## See also

:::{seealso}
- {doc}`../user-guide/index`: the task-oriented guides these pages are the reference for.
- {doc}`../getting-started/quickstart`: the shortest path to a running query.
- {doc}`../migration/index`: the equivalent spelling if you know another engine's API.
- {doc}`../configuration/options`: the field-by-field configuration reference.
- {doc}`../agents/index`: the same surface packaged as instructions for a coding agent.
:::

```{toctree}
:hidden:
:caption: Reference

reference
complete
```

```{toctree}
:hidden:
:caption: By area

dataset
expressions
expression-accessors
functions
metrics
io
sql
ml
preprocessors
ml-models
ml-statistics
governance
configuration
exceptions
```
