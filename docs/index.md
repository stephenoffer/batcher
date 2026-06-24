# Batcher

```{raw} html
<div class="bt-hero">
  <p class="bt-hero-eyebrow">JIT &middot; Adaptive &middot; Arrow-native</p>
  <p class="bt-hero-tagline">One data engine, from your laptop to your cluster.</p>
  <p class="bt-hero-sub">
    Batcher runs SQL, DataFrame, and ML workloads on a JIT-compiling Rust core, and
    re-optimizes the query while it is still running. Sub-second on a laptop,
    bounded-memory at petabyte scale &mdash; the same code either way.
  </p>
  <p class="bt-hero-cta">
    <a class="bt-btn bt-btn-primary" href="getting-started/index.html">Get started</a>
    <a class="bt-btn" href="getting-started/quickstart.html">Quickstart</a>
    <a class="bt-btn" href="https://github.com/stephenoffer/batcher">GitHub</a>
  </p>
</div>
```

Most engines plan a query once, before they have seen a single row, then commit to
that plan whatever the data turns out to be. Batcher measures the data as it flows
and re-plans the rest of the query on real numbers, so a query that starts on a bad
estimate corrects itself mid-flight.

## Why Batcher

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` Adaptive re-optimization
Kyber re-plans at pipeline breakers from measured cardinalities. DuckDB optimizes
once; Spark adapts only at stage boundaries.
:::

:::{grid-item-card} {octicon}`server;1.1em` One algebra, one to many machines
Stateful operators are written once as mergeable partial / combine / finalize
primitives. One core or a thousand run the same code, with bounded memory and spill
to disk.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` JIT-compiled expressions
A Cranelift fast path compiles column expressions once per operator and reuses the
machine code across batches, with a checked fallback to the interpreter.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Lazy, immutable API
A Dataset is a handle to a plan. Each operation returns a new one; nothing runs until
a terminal call such as `collect` or `write.parquet`.
:::
::::

## The same query, two ways

Write it as a DataFrame pipeline or as SQL. Both build the same plan and run on the
same engine.

::::{tab-set}
:::{tab-item} DataFrame
```python
import batcher as bt

sales = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "qty": [1, 2, 3, 4, 5],
    }
)

revenue = (
    sales.with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("category")
    .agg(revenue=bt.col("total").sum())
    .sort("revenue", descending=True)
)
print(revenue.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0]}
```
:::

:::{tab-item} SQL
```python
import batcher as bt

sales = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "qty": [1, 2, 3, 4, 5],
    }
)

revenue = bt.sql(
    "SELECT category, SUM(price * qty) AS revenue "
    "FROM sales GROUP BY category ORDER BY revenue DESC",
    sales=sales,
)
print(revenue.to_pydict())
# {'category': ['a', 'b'], 'revenue': [350.0, 200.0]}
```
:::
::::

Files and object stores use the same API. Only the source changes.

```python
# docs: skip
ds = bt.read("s3://bucket/events.parquet")
ds.filter(bt.col("status") == "active").write.parquet("output/active.parquet")
```

## How it works

The design splits in two. Python builds the plan, optimizes it, and decides how much
it should cost, but never touches a row. Every per-row operation runs in Rust over
Apache Arrow. Between them sits one boundary: a JSON plan going down, zero-copy Arrow
batches coming back.

![Batcher's two planes: a Python control plane hands a JSON IR plus Arrow batches to the Rust data plane.](_static/diagrams/two_planes.png)

## How it compares

| Reach for Batcher when | Because |
| --- | --- |
| You outgrow DuckDB's single node | the same query scales out, and re-optimizes mid-flight rather than planning once |
| Polars is fast but stops at one machine | the mergeable algebra runs the same code on a cluster |
| Spark's overhead dominates small jobs | it runs in-process locally, with no cluster to spin up |

Speed is measured correctness-first: the benchmark harness refuses to time a query
whose result does not match DuckDB, and every operator is differential-tested against
it. The numbers live in [`benchmarks/`](https://github.com/stephenoffer/batcher/tree/main/benchmarks).

## Where to go next

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Getting started
:link: getting-started/index
:link-type: doc
Install, run a first pipeline, and learn the lazy execution model.
:::

:::{grid-item-card} {octicon}`book;1.1em` Tutorials
:link: tutorials/index
:link-type: doc
Worked, end-to-end walkthroughs you can run as written.
:::

:::{grid-item-card} {octicon}`code;1.1em` User guide
:link: user-guide/index
:link-type: doc
Task-oriented guides for every part of the Dataset API.
:::

:::{grid-item-card} {octicon}`list-unordered;1.1em` API reference
:link: api/index
:link-type: doc
Every public class, method, and function.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Machine learning
:link: ml/index
:link-type: doc
Batch inference, embeddings, and training-data loaders.
:::

:::{grid-item-card} {octicon}`gear;1.1em` Configuration
:link: configuration/index
:link-type: doc
Memory, spill, parallelism, and the adaptive knobs.
:::
::::

```{toctree}
:hidden:
:caption: Learn

getting-started/index
tutorials/index
learning-paths/index
```

```{toctree}
:hidden:
:caption: Guides

user-guide/index
ml/index
configuration/index
migration/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
```

```{toctree}
:hidden:
:caption: Design

architecture/index
internals/index
```
