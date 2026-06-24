# API reference

The Batcher API is small and lazy: you build a `Dataset` from a source, transform it
with expression-based operations, and execute it with a terminal operation that
returns Arrow or writes to a sink. Everything reachable from `import batcher as bt`
is documented here — one reference, three ways in:

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` Quick reference
:link: reference
:link-type: doc
A one-page cheat sheet — the common functions, methods, and patterns at a glance.
:::

:::{grid-item-card} {octicon}`book;1.1em` By area
:link: by-area
:link-type: ref
Example-first pages for one surface at a time: the Dataset, expressions, I/O, SQL,
ML, configuration, and errors.
:::

:::{grid-item-card} {octicon}`list-unordered;1.1em` Complete reference
:link: complete
:link-type: doc
A generated page for every public symbol, with its full signature and docstring.
:::
::::

(by-area)=

## By area

The curated, example-first references, grouped the way you'd look something up. The
[complete reference](complete.md) is the exhaustive generated backstop behind them;
the [quick reference](reference.md) is the cheat sheet.

- [Dataset](dataset.md) — build, transform, aggregate, join, and collect.
- [Expressions](expressions.md) — column math, predicates, and the `.str` / `.dt` /
  `.list` / `.struct` / `.json` accessors.
- [Reading and writing](io.md) — every reader and writer, with the optional extras.
- [SQL](sql.md) — the SQL surface and how it lowers to the DataFrame API.
- [ML](ml.md) — the `.ml` accessor: `map_batches`, `infer`, `embed`.
- [Configuration](configuration.md) — the tunables and how they're set.
- [Errors](exceptions.md) — the typed exceptions and what raises them.

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
configuration
exceptions
```
