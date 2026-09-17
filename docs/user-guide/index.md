# User guide

This guide covers everything you do with a Batcher `Dataset`, organized by the job in front of you: shape the data, analyze it, move it in and out, trust what it says, and run the pipeline in production.

Every guide teaches the same small model. A `Dataset` is lazy, so each call adds a step to a plan and nothing runs until you ask for a result. The optimizer sees the whole chain at once, pushes filters and column selections down to the source, and hands the plan to a Rust data plane that works over Apache Arrow batches. SQL builds the same plan as the DataFrame API, so you can switch between them mid-pipeline. The same code runs on a laptop core or a Ray cluster, and it spills to disk when the data outgrows memory.

```python
import batcher as bt

orders = bt.from_pydict(
    {"region": ["west", "east", "west", "east"], "amount": [120.0, 80.0, 45.0, 300.0]}
)
large = orders.filter(bt.col("amount") > 50)
print(bt.sql("SELECT region, SUM(amount) AS total FROM large GROUP BY region ORDER BY region", large=large).to_pydict())
# {'region': ['east', 'west'], 'total': [380.0, 120.0]}
```

Every example in this guide runs like that one. The test suite executes each code block on every commit, so what you read is what the engine does today.

## Find your guide

The five sections follow the order a pipeline runs in. Move data brings rows in through `bt.read`, Transform decides which rows and columns you keep, Analyze turns them into answers, and Move data sends the result out through `ds.write`. Trust and Operate aren't steps a row passes through. Checks and policies lower into the same plan as the steps, and operating covers the whole run.

![The five user-guide sections along one pipeline. Move data reads rows in with bt.read and hands a lazy Dataset to Transform, which decides which rows and columns survive. The rows it keeps go to Analyze, for grouping, joins, windows and SQL, and the answers go back out through Move data with ds.write. Above the chain, Trust adds data-quality checks and row and column policies to the same plan the steps run in. Below it, Operate inspects and tunes the whole run: explain() shows the plan before it runs, explain(analyze=True) measures it, tuning and caching make it fast, and progress events and metrics show a run as it happens.](/_static/diagrams/user_guide_areas.svg)

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`code;1.1em` Transform
:link: /user-guide/transform/index
:link-type: doc
Select, derive, filter, sort, sample, and deduplicate, with an expression language that runs in Rust instead of a Python loop.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Analyze
:link: /user-guide/analyze/index
:link-type: doc
Aggregations, joins, windows, time series, geospatial, graphs, and full SQL over the same engine.
:::

:::{grid-item-card} {octicon}`arrow-switch;1.1em` Move data
:link: /user-guide/moving-data/index
:link-type: doc
Read and write files, object storage, databases, lakehouse tables, and unbounded streams.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` Trust
:link: /user-guide/trust/index
:link-type: doc
Data-quality checks and row and column governance, enforced inside the query plan.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Operate
:link: /user-guide/operate/index
:link-type: doc
Explain plans, tuning, caching, and running a pipeline you can see into.
:::
::::

## Where to start

New to Batcher? Read {doc}`/user-guide/transform/columns/expressions` first. Every other guide builds on expressions, and the rest of the section reads quickly once they click. If you already know what you want to compute, jump straight to the guide for that job. Each one opens with a setup block and builds from there.

## See also

- {doc}`../api/index`: the reference behind every method these guides use.
- {doc}`../cookbook/index`: the same operations as runnable recipes and complete pipelines.
- {doc}`../ml/index`: the model half of the pipeline, once the relational half is in place.
- {doc}`../configuration/index`: the settings the performance and memory guides refer to.
- {doc}`/architecture/deep-dives/index`: why an operator behaves the way these pages describe.
- {doc}`../integrations/index`: connecting a specific source or sink.

```{toctree}
:hidden:

transform/index
analyze/index
moving-data/index
trust/index
operate/index
```
