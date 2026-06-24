# Examples

End-to-end examples organized by workload. Each is a complete, runnable walkthrough —
start from the one closest to what you're building. They use small in-memory data so
they run anywhere; the same code scales to files, object storage, and a cluster.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` ETL pipeline
:link: etl
:link-type: doc
Read, clean, deduplicate, derive, roll up, and write Parquet — the full extract /
transform / load loop.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Analytics query
:link: analytics
:link-type: doc
Aggregate, join, and window over an orders table — the same query as SQL or DataFrame.
:::

:::{grid-item-card} {octicon}`rocket;1.1em` First pipeline
:link: ../tutorials/first-pipeline
:link-type: doc
A guided tour of build → transform → aggregate → collect, then point it at files.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` Batch inference
:link: ../tutorials/batch-inference
:link-type: doc
Run a model over Arrow batches with the `.ml` accessor — load once per worker, scale
across GPUs.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: ../user-guide/streaming
:link-type: doc
Process unbounded sources as micro-batches through the same operators.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Synthetic data
:link: ../tutorials/synthetic-data-generation
:link-type: doc
Build test datasets in memory with `bt.from_pydict` and expressions.
:::
::::

```{toctree}
:hidden:

etl
analytics
```
