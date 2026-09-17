# Core concepts

Four ideas explain almost everything Batcher does, and each takes a few minutes to read. They all follow from one split. Python is the *control plane*: it builds a query plan, optimizes it, and decides how much memory the work may use, without touching a row. Rust is the *data plane*: it runs that plan over Apache Arrow batches on every core you have, or across a cluster.

The two planes meet at one boundary, a JSON plan plus Arrow batches:

![Two stacked layers. The top layer, the Python control plane, runs left to right from Dataset and SQL, which are lazy and immutable, to Kyber, which optimizes the plan, to Carbonite, which checks feasibility and allocates, to Core, which executes and measures. An arrow labeled JSON IR plus Arrow batches crosses down to the Rust data plane, which holds bc-py for the zero-copy FFI, bc-interp for the interpreter, parallel, and JIT paths, bc-runtime for mergeable operators, bc-codegen for the Cranelift JIT, bc-sketches for HLL, KLL, and Count-Min sketches, and bc-transport for Arrow Flight.](/_static/diagrams/two_planes.svg)

That split is what lets the optimizer see your whole query before it runs, keeps Python out of the per-row loop, and lets the same plan run on a laptop or a cluster.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.1em` Lazy, immutable datasets
:link: lazy
:link-type: doc
A {py:class}`Dataset <batcher.Dataset>` is a handle to a plan. Nothing runs until a terminal operation.
:::

:::{grid-item-card} {octicon}`code;1.1em` Expressions run in Rust
:link: expressions
:link-type: doc
You describe column work. Rust evaluates it over whole Arrow batches.
:::

:::{grid-item-card} {octicon}`server;1.1em` One core to a cluster
:link: scaling
:link-type: doc
Every stateful operator is written once, so a laptop and a cluster run the same code.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Adaptive re-optimization
:link: adaptive
:link-type: doc
Every run is measured, and the optimizer plans the next one from what it saw.
:::
::::

When a term on these pages or elsewhere in the docs is new to you, the {doc}`glossary` defines it in a sentence or two and links to the page that covers it.

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>`: every way to get a dataset.
- {doc}`Transformations </user-guide/transform/rows/transformations>`, {doc}`Aggregations </user-guide/analyze/aggregations>`, {doc}`Joins </user-guide/analyze/joins>`, and {doc}`Window functions </user-guide/analyze/window-functions>`: the verbs, once a dataset exists.
- {doc}`/architecture/index`: the same split, at the level of the whole system.
- {doc}`/architecture/deep-dives/query/query-lifecycle`: what happens between {py:meth}`collect() <batcher.Dataset.collect>` and the Arrow batches coming back.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading the plan these concepts describe.

```{toctree}
:hidden:

lazy
expressions
scaling
adaptive
glossary
```
