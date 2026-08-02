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

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>`: every way to get a dataset.
- {doc}`Transformations </user-guide/transform/rows/transformations>`,
  {doc}`Aggregations </user-guide/analyze/aggregations>`,
  {doc}`Joins </user-guide/analyze/joins>`, and
  {doc}`Window functions </user-guide/analyze/window-functions>`: the verbs, once a dataset exists.
- {doc}`../../architecture/index`: the same split, at the level of the whole system.
- {doc}`/architecture/deep-dives/query/query-lifecycle`: what actually happens between `collect()` and
  the Arrow batches coming back.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading the plan these concepts describe.

```{toctree}
:hidden:

lazy
expressions
scaling
adaptive
```
