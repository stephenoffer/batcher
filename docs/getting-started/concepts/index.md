# Core concepts

Batcher has two halves. Python is the control plane: it builds a query plan and
optimizes it. Rust is the data plane: it runs that plan over Apache Arrow record
batches. Almost everything surprising about the API follows from that split, by way
of the four ideas below.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.1em` Lazy, immutable datasets
:link: lazy
:link-type: doc
A `Dataset` is a handle to a plan; nothing runs until a terminal operation.
:::

:::{grid-item-card} {octicon}`code;1.1em` Expressions run in Rust
:link: expressions
:link-type: doc
You describe column work; Rust evaluates it over whole Arrow batches.
:::

:::{grid-item-card} {octicon}`server;1.1em` One core to a cluster
:link: scaling
:link-type: doc
Mergeable operators give identical results on a laptop or a cluster.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Adaptive re-optimization
:link: adaptive
:link-type: doc
The optimizer re-plans mid-query on measured row counts, not static guesses.
:::
::::

## Where to go next

- [Reading data](../../user-guide/reading-data.md): every way to get a dataset.
- [Transformations](../../user-guide/transformations.md),
  [Aggregations](../../user-guide/aggregations.md),
  [Joins](../../user-guide/joins.md),
  [Window functions](../../user-guide/window-functions.md).

```{toctree}
:hidden:

lazy
expressions
scaling
adaptive
```
