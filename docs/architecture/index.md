# Architecture

This section describes how Batcher is built: a Python control plane that plans every query, and a Rust data plane that executes it over Apache Arrow.

The split is the design. Python builds and optimizes the plan but never touches a row, which keeps the optimizer easy to extend and lets it learn from every run. Rust does all per-row work over Arrow batches at native speed. The two meet at one boundary, a JSON plan plus zero-copy Arrow batches. Every stateful operator is written once as mergeable algebra, which is why a query returns the same rows, column names and column types on one core, many cores, or a cluster.

![Batcher's two planes: a Python control plane hands a JSON IR plus zero-copy Arrow batches to the Rust data plane.](/_static/diagrams/two_planes.svg)

The section describes the engine at three zoom levels. Read them in this order:

| Level | Zoom | Read it when |
|---|---|---|
| The pages below | The shape of the system | You want to know how the pieces fit |
| {doc}`Deep dives </architecture/deep-dives/index>` | One mechanism at a time | You want to know why a query behaved that way |
| {doc}`Internals </architecture/internals/index>` | One subsystem's design | You are about to change the engine |

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.1em` Overview
:link: overview
:link-type: doc
The two planes, the crate layout, and how they fit together.
:::

:::{grid-item-card} {octicon}`workflow;1.1em` Execution
:link: execution
:link-type: doc
Morsels, the interpreter and JIT tiers, and the parallel scheduler.
:::

:::{grid-item-card} {octicon}`git-branch;1.1em` Optimization
:link: optimization
:link-type: doc
Kyber's passes, cost-based choices, and adaptive re-optimization.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` Fault tolerance
:link: fault-tolerance
:link-type: doc
Retries, shuffle recompute, epoch fencing, and backpressure.
:::

:::{grid-item-card} {octicon}`milestone;1.1em` What makes Batcher different
:link: differentiators
:link-type: doc
The six design decisions that separate it from DuckDB, Polars, Spark, and Ray Data, and where each one stops.
:::

:::{grid-item-card} {octicon}`telescope;1.1em` Deep dives
:link: /architecture/deep-dives/index
:link-type: doc
Twenty-eight pages, one mechanism each: the query lifecycle, the operators, memory, distribution, and the adaptive loop.
:::

:::{grid-item-card} {octicon}`tools;1.1em` Internals
:link: /architecture/internals/index
:link-type: doc
The design-level record of Kyber, Carbonite, and the execution engine, plus how to extend and test them.
:::
::::

## See also

- {doc}`/getting-started/concepts/glossary`: morsel, pipeline breaker, mergeable algebra, and the rest, one line each.
- {doc}`/getting-started/concepts/index`: the same ideas at the level a user needs them.
- {doc}`/benchmarks/index`: what the design measures out at, against DuckDB, Polars, Spark, and Daft.
- {doc}`/user-guide/operate/tuning/explain-plans`: reading what the optimizer decided for one query.
- {doc}`/api/symbols/index`: the surface the control plane exposes over all of this.

```{toctree}
:hidden:

overview
execution
optimization
fault-tolerance
differentiators
deep-dives/index
internals/index
```
