# Core concepts

Batcher splits cleanly into a Python control plane and a Rust data plane: Python
builds and optimizes a query plan, Rust runs it over Apache Arrow record batches.
Four ideas follow from that split and explain how the rest of the API behaves.

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
Column work is described, not looped — evaluated over Arrow batches in Rust.
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

- [Reading data](../../user-guide/reading-data.md): every way to build a dataset.
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
