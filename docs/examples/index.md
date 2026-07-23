# Examples

Working code, grouped by the problem it solves.

Every example here runs as written, on small in-memory data, so it works anywhere you
paste it. That is deliberate: you should be able to read a recipe, run it, and change one
line to see what happens, without first standing up a bucket or a cluster. The same code
scales to object storage and many nodes without editing the pipeline.

There are two kinds of page. A *worked example* walks one complete pipeline end to end and
explains each decision. A *recipe* answers one focused question, usually with the trap that
makes it harder than it looks.

## Worked examples

Two full pipelines, each covering the whole arc from source to written output.

- {doc}`etl`: read, clean, deduplicate, derive, roll up, then write Parquet.
- {doc}`analytics`: aggregate, join, then window over an orders table, spelled both as SQL
  and as DataFrame code.

If you would rather be guided than read finished code, the tutorials cover the same ground
at a slower pace: {doc}`../tutorials/first-pipeline` builds a pipeline from nothing, and
{doc}`../tutorials/batch-inference` does the same for a model.

## Recipe collections

Four cookbooks, grouped by the kind of problem rather than the API used. Each collection
opens by naming the traps its recipes exist to avoid.

- {doc}`data-engineering/index`: ingest, reconcile, and repair tables.
- {doc}`analytics/index`: cohorts, funnels, sessions, and rankings.
- {doc}`ml/index`: embeddings, inference, and training data.
- {doc}`streaming/index`: unbounded sources, time, and restarts.

## Runnable scripts

Every example above maps to a self-contained script in the
[`examples/` directory](https://github.com/batcher/batcher/tree/main/examples). Each builds
its own in-memory data and asserts on its output, so a script that stops matching the engine
fails rather than drifting. Run any of them with `python examples/<name>.py`.

On the ETL side there are `data_quality.py`, `lakehouse_scd.py`, `feature_engineering.py`,
`timeseries.py`, and `window_functions.py`. The ML scripts are `ml_inference.py`,
`preprocessors.py`, and `streaming_pipeline.py`. For the operational picture, read
`performance_caching.py` and `spill.py` for memory, `adaptive_optimization.py` for the
re-planning loop, and `distributed.py` for a cluster.

## See also

:::{seealso}
- {doc}`../learning-paths/index`: the same material ordered by role, if you want a tour
  rather than a lookup.
- {doc}`../user-guide/index`: the reference behind every operation these examples use.
- {doc}`../user-guide/streaming`: unbounded sources through the same operators.
- {doc}`../user-guide/data-quality`: enforcing a contract with `ds.dq`.
- {doc}`../user-guide/lakehouse`: Delta read, write, and merge, plus time travel and SCD.
- {doc}`../user-guide/performance`: caching, spilling, and reading a plan when one of these
  pipelines meets real data.
:::

```{toctree}
:hidden:
:caption: Worked examples

etl
analytics
```

```{toctree}
:hidden:
:caption: Recipe collections

data-engineering/index
analytics/index
ml/index
streaming/index
```
