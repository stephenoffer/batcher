# Architecture

This section describes how Batcher is built: a Python control plane that plans every query, and a Rust data plane that executes it over Apache Arrow.

The split is the design. Python plans and never touches a row, so the optimizer stays easy to extend and learns from every run. Rust does the per-row work over Arrow batches. They meet at one boundary: a JSON plan plus zero-copy Arrow batches.

Every stateful operator is written once as mergeable algebra. That's why a query returns the same rows, column names and column types on one core or a cluster.

![Batcher's two planes: a Python control plane hands a JSON IR plus zero-copy Arrow batches to the Rust data plane.](/_static/diagrams/two_planes.svg)

You can see both halves from Python. The plan is inspectable before anything runs, and the result comes back as Arrow:

```python
# docs: run
import batcher as bt

ds = bt.from_pydict({"city": ["Oslo", "Lima", "Oslo"], "temp": [3, 19, 5]})
warm = ds.filter(bt.col("temp") > 4)
assert "pushed[temp > 4]" in warm.explain()  # the control plane's plan
print(warm.collect().to_pydict())  # the data plane's Arrow result
# {'city': ['Lima', 'Oslo'], 'temp': [19, 5]}
```

The section works at three zoom levels. The pages below give the shape of the system. {doc}`Deep dives </architecture/deep-dives/index>` take one mechanism at a time, for when you want to know why a query behaved the way it did. {doc}`Internals </architecture/internals/index>` hold each subsystem's design, for when you're about to change the engine.

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
Morsels, the JIT, and the parallel scheduler that runs them.
:::

:::{grid-item-card} {octicon}`git-branch;1.1em` Optimization
:link: optimization
:link-type: doc
Kyber's passes and how it re-plans on measured sizes.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` Fault tolerance
:link: fault-tolerance
:link-type: doc
How a distributed query rides out a lost worker.
:::

:::{grid-item-card} {octicon}`milestone;1.1em` What makes Batcher different
:link: differentiators
:link-type: doc
Six design decisions that set it apart from DuckDB, Polars, Spark and Ray Data.
:::

:::{grid-item-card} {octicon}`telescope;1.1em` Deep dives
:link: /architecture/deep-dives/index
:link-type: doc
Twenty-eight pages, one mechanism each, from the query lifecycle to the adaptive loop.
:::

:::{grid-item-card} {octicon}`tools;1.1em` Internals
:link: /architecture/internals/index
:link-type: doc
The design-level record of Kyber, Carbonite, and the execution engine, plus how to extend and test them.
:::
::::

## See also

The {doc}`/getting-started/concepts/glossary` defines morsels, pipeline breakers and mergeable algebra in a line each, and {doc}`/getting-started/concepts/index` covers the same ideas at the level a user needs them. To see what the optimizer decided for one query, read {doc}`/user-guide/operate/tuning/explain-plans`. {doc}`/benchmarks/index` has what the design measures out at against DuckDB, Polars, Spark and Daft, and {doc}`/api/symbols/index` lists the surface the control plane exposes.

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
