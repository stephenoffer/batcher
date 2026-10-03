# User guide

This guide covers everything you do with a Batcher `Dataset`. It's organized by the job in front of you, from getting rows in to running the pipeline in production.

Every guide teaches one small model. A `Dataset` is lazy. Each call adds a step to a plan, and nothing runs until you ask for a result, at which point the optimizer has the whole chain in view: it pushes filters and column selections down to the source, then hands the plan to a Rust engine working over Apache Arrow batches.

```python
import batcher as bt

orders = bt.from_pydict(
    {"region": ["west", "east", "west", "east"], "amount": [120.0, 80.0, 45.0, 300.0]}
)
large = orders.filter(bt.col("amount") > 50)
print(large.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict())
# {'region': ['east', 'west'], 'total': [380.0, 120.0]}
```

SQL builds the same plan. Switch mid-pipeline if you like:

```python
print(bt.sql("SELECT region, SUM(amount) AS total FROM large GROUP BY region ORDER BY region", large=large).to_pydict())
# {'region': ['east', 'west'], 'total': [380.0, 120.0]}
```

The test suite runs every code block in this guide on every commit. The examples cannot drift from the engine.

## Find your guide

The sections follow a pipeline. Move data reads rows in with {py:obj}`bt.read <batcher.read>`, Transform shapes them, Analyze turns them into answers, and Move data writes the result back out with `ds.write`. Trust and Operate sit outside the chain. Trust adds checks and policies to the plan. Operate watches the run.

![The five user-guide sections along one pipeline. Move data reads rows in with bt.read and hands a lazy Dataset to Transform, which decides which rows and columns survive. The rows it keeps go to Analyze, for grouping, joins, windows and SQL, and the answers go back out through Move data with ds.write. Above the chain, Trust adds data-quality checks and row and column policies to the same plan the steps run in. Below it, Operate inspects and tunes the whole run: explain() shows the plan before it runs, explain(analyze=True) measures it, tuning and caching make it fast, and progress events and metrics show a run as it happens.](/_static/diagrams/user_guide_areas.svg)

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`code;1.1em` Transform
:link: /user-guide/transform/index
:link-type: doc
Pick and derive rows with an expression language that runs in Rust, not a Python loop.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Analyze
:link: /user-guide/analyze/index
:link-type: doc
Profile a table, then aggregate, join, window or query it in SQL. Time series and geospatial work, even graph algorithms, ride the same engine.
:::

:::{grid-item-card} {octicon}`arrow-switch;1.1em` Move data
:link: /user-guide/moving-data/index
:link-type: doc
Files, object storage, databases, lakehouse tables and unbounded streams, in and out.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` Trust
:link: /user-guide/trust/index
:link-type: doc
Data-quality checks and row and column governance, enforced inside the query plan.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Operate
:link: /user-guide/operate/index
:link-type: doc
Read the plan, tune it, and watch the pipeline while it runs.
:::
::::

## Where to start

New to Batcher? Read {doc}`/user-guide/transform/columns/expressions` first, since every other guide leans on it. Handed a table you've never seen? {doc}`/user-guide/analyze/inspecting-data` shows its schema and nulls before you touch it. Otherwise, go straight to the guide for your job.

## See also

The rest of the docs, by what you need next.

- {doc}`/getting-started/tour`: one runnable example of each capability, when you want the shape before the detail.
- {doc}`../api/index`: the reference behind every method these guides use.
- {doc}`../cookbook/index`: the same operations as runnable recipes and complete pipelines.
- {doc}`../ml/index`: the model half of the pipeline, once the relational half is in place.
- {doc}`../configuration/index`: the settings the performance and memory guides refer to.
- {doc}`/architecture/deep-dives/index`: why an operator behaves the way these pages describe.
- {doc}`../integrations/index`: connecting a specific source or sink.
- {doc}`/getting-started/concepts/glossary`: a one-line definition for any term on these pages that is new to you.

```{toctree}
:hidden:

transform/index
analyze/index
moving-data/index
trust/index
operate/index
```
