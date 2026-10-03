# Core concepts

Four ideas explain almost everything Batcher does, and they start from one split. Python is the *control plane*. It builds and optimizes a query plan without touching a row. Rust is the *data plane*, and it runs that plan over Apache Arrow batches on every core or across a cluster.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4]})
plan = ds.filter(bt.col("x") > 1).select(y=bt.col("x") * 10)  # Python builds a plan
print(plan.to_pydict())  # Rust runs it
# {'y': [20, 30, 40]}
```

The two planes meet at one boundary, a JSON plan plus Arrow batches:

![Two stacked layers. The top layer, the Python control plane, runs left to right from Dataset and SQL, which are lazy and immutable, to Kyber, which optimizes the plan, to Carbonite, which checks feasibility and allocates, to Core, which executes and measures. An arrow labeled JSON IR plus Arrow batches crosses down to the Rust data plane, which holds bc-py for the zero-copy FFI, bc-interp for the interpreter, parallel, and JIT paths, bc-runtime for mergeable operators, bc-codegen for the Cranelift JIT, bc-sketches for HLL, KLL, and Misra-Gries sketches, and bc-transport for Arrow Flight.](/_static/diagrams/two_planes.svg)

That split pays off twice. The optimizer sees your whole query before anything runs, while Python never enters the per-row loop. And the plan doesn't care whether it lands on a laptop or a cluster.

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
Large joins re-plan on the row counts they measure, and what each run learns feeds the next plan.
:::
::::

New term? The {doc}`glossary` defines it in a sentence and links to the page that covers it.

## See also

Once you have a dataset, {doc}`Reading data </user-guide/moving-data/reading-data>` shows every way to get one, and the verbs live in {doc}`Transformations </user-guide/transform/rows/transformations>`, {doc}`Aggregations </user-guide/analyze/aggregations>`, {doc}`Joins </user-guide/analyze/joins>` and {doc}`Window functions </user-guide/analyze/window-functions>`.

For the same split at the level of the whole system, read {doc}`/architecture/index`. {doc}`/architecture/deep-dives/query/query-lifecycle` follows a query from {py:meth}`collect() <batcher.Dataset.collect>` until the Arrow batches come back. To read the plan these concepts describe, see {doc}`/user-guide/operate/tuning/explain-plans`.

```{toctree}
:hidden:

lazy
expressions
scaling
adaptive
glossary
```
