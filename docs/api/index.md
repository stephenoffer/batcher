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
Every public symbol, with its full signature and docstring.
:::
::::

(by-area)=

## By area

These are the curated, example-first references, grouped the way you'd look something up. Each one leads with a runnable example and then enumerates the surface. {doc}`complete` is the exhaustive backstop behind them, and {doc}`reference` is the cheat sheet to keep open while you work.

- [Dataset](dataset.md): build, transform, aggregate, join, and collect.
- [Expressions](expressions.md): column math, predicates, and the `.str`, `.dt`, `.list`, `.struct`, and `.json` accessors.
- [Reading and writing](io.md): every reader and writer, with the optional extras.
- [SQL](sql.md): the SQL surface and how it lowers to the DataFrame API.
- [ML](ml.md): the `.ml` accessor, plus the LLM, serving, loader, and vector surfaces.
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
io
sql
ml
governance
configuration
exceptions
```
