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

:::{grid-item-card} {octicon}`shield;1.1em` Data quality
:link: ../user-guide/data-quality
:link-type: doc
Validate, quarantine, and enforce a data contract with the `ds.dq` accessor.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Lakehouse and SCD
:link: ../user-guide/lakehouse
:link-type: doc
Delta read/write/merge, time-travel, and slowly-changing-dimension history.
:::

:::{grid-item-card} {octicon}`pulse;1.1em` Performance and memory
:link: ../user-guide/performance
:link-type: doc
Cache reused results, spill out of core, and read the query plan.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Synthetic data
:link: ../tutorials/synthetic-data-generation
:link-type: doc
Build test datasets in memory with `bt.from_pydict` and expressions.
:::
::::

## Runnable scripts

Every example above maps to a self-contained script in the
[`examples/` directory](https://github.com/batcher/batcher/tree/main/examples) that
builds its own in-memory data and asserts on its output — run any of them with
`python examples/<name>.py`. Highlights by workload:

- **ETL** — `data_quality.py`, `lakehouse_scd.py`, `feature_engineering.py`,
  `timeseries.py`, `window_functions.py`
- **ML** — `ml_inference.py`, `streaming_pipeline.py`
- **Operations** — `performance_caching.py`, `spill.py`, `adaptive_optimization.py`,
  `distributed.py`

Pick the one closest to what you are building, or follow a
[learning path](../learning-paths/index.md) for a role-ordered tour.

```{toctree}
:hidden:

etl
analytics
```
