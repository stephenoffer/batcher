# Architecture

This section describes how a query flows from the Python control plane to the Rust data
plane. Python builds and optimizes the plan but never touches a row. Rust runs every
per-row operation over Apache Arrow. The two planes meet at one typed, zero-copy
boundary, which is also why a result is identical on one core or a hundred.

![Batcher's two planes: a Python control plane hands a JSON IR plus zero-copy Arrow batches to the Rust data plane.](/_static/diagrams/two_planes.svg)

Three sections describe the engine at three zoom levels, and they are meant to be read in
this order:

| Section | Zoom level | Read it when |
|---|---|---|
| Architecture (this section) | The shape of the system | You want to know how the pieces fit |
| {doc}`Deep dives <../deep-dives/index>` | One mechanism at a time | You want to know why a query behaved that way |
| {doc}`Internals <../internals/index>` | One subsystem's design | You are about to change the engine |

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
The six design decisions that separate it from DuckDB, Spark, and Ray Data, and where each one stops.
:::
::::

```{toctree}
:hidden:

overview
execution
optimization
fault-tolerance
differentiators
```
