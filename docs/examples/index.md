# Examples

End-to-end examples, grouped by workload. Each one runs as written, on small in-memory
data, so it works anywhere you paste it. Start from whichever sits closest to what you
are building; the same code scales up to object storage and a cluster.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` ETL pipeline
:link: etl
:link-type: doc
Read, clean, deduplicate, derive, roll up, then write Parquet: the whole
extract/transform/load loop.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Analytics query
:link: analytics
:link-type: doc
Aggregate, join, then window over an orders table, spelled as SQL or as DataFrame code.
:::

:::{grid-item-card} {octicon}`rocket;1.1em` First pipeline
:link: ../tutorials/first-pipeline
:link-type: doc
A guided tour of build → transform → aggregate → collect, then point it at files.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` Batch inference
:link: ../tutorials/batch-inference
:link-type: doc
Run a model over Arrow batches with the `.ml` accessor. Load once per worker, scale
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
Enforce a data contract with the `ds.dq` accessor: validate rows, quarantine the bad
ones.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Lakehouse and SCD
:link: ../user-guide/lakehouse
:link-type: doc
Delta read/write/merge, plus time-travel and slowly-changing-dimension history.
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
[`examples/` directory](https://github.com/batcher/batcher/tree/main/examples). Each
builds its own in-memory data and asserts on its output. Run any of them with
`python examples/<name>.py`.

On the ETL side there are `data_quality.py`, `lakehouse_scd.py`,
`feature_engineering.py`, `timeseries.py`, and `window_functions.py`. The ML scripts
are `ml_inference.py`, `preprocessors.py`, and `streaming_pipeline.py`. For the
operational picture, read `performance_caching.py` and `spill.py` for memory,
`adaptive_optimization.py` for the re-planning loop, and `distributed.py` for a cluster.

Or follow a [learning path](../learning-paths/index.md) for a role-ordered tour.

## Recipe collections

Four cookbooks, grouped by the kind of problem rather than the API used.

- [Data engineering](data-engineering/index.md): ingest, reconcile, and repair tables.
- [Analytics](analytics/index.md): cohorts, funnels, sessions, and rankings.
- [ML](ml/index.md): embeddings, inference, and training data.
- [Streaming](streaming/index.md): unbounded sources, time, and restarts.

```{toctree}
:hidden:

etl
analytics
data-engineering/index
analytics/index
ml/index
streaming/index
```
