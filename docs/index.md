# Batcher

```{raw} html
<div class="bt-hero">
  <p class="bt-hero-eyebrow">Any data &middot; Any AI workload &middot; Batch &amp; streaming</p>
  <p class="bt-hero-tagline">One engine for every kind of data, and every kind of AI.</p>
  <p class="bt-hero-sub">
    Structured tables, unstructured text, images, audio, video. SQL, DataFrames, and
    expressions. Batch jobs and live streams. Batcher runs all of it on a single
    engine &mdash; from a laptop to a cluster &mdash; and tunes itself as the query runs.
  </p>
  <p class="bt-hero-cta">
    <a class="bt-btn bt-btn-primary" href="getting-started/index.html">Get started</a>
    <a class="bt-btn" href="getting-started/quickstart.html">Quickstart</a>
    <a class="bt-btn" href="https://github.com/stephenoffer/batcher">GitHub</a>
  </p>
</div>
```

Data work has splintered into a tool per job: one for SQL, another for DataFrames, a
third for streaming, more for images and models. Each one is another system to run
and another seam to leak. Batcher collapses that stack into a single engine.

![One engine: any source — Parquet, media, Kafka, lakehouse — flows into Batcher and back out to any workload: SQL and ETL, batch inference, embeddings, and training data.](_static/diagrams/hub.png)

## Any data, any workload

The same engine reads a Parquet table, a folder of images, or a Kafka stream, and the
same pipeline can clean it, query it, or feed it to a model.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` Structured
Parquet, CSV, JSON, and the lakehouse formats (Delta, Iceberg, Hudi) — filtered,
joined, and aggregated with SQL or DataFrames.
:::

:::{grid-item-card} {octicon}`file;1.1em` Unstructured
Text, logs, and documents read whole or by the line, then parsed into clean columns
at scale.
:::

:::{grid-item-card} {octicon}`image;1.1em` Multimodal
Images, audio, and video decoded straight into tensors, so one pipeline can clean a
table and feed a model.
:::

:::{grid-item-card} {octicon}`search;1.1em` Vectors & embeddings
First-class list and tensor columns with the vector ops behind embeddings, similarity
search, and RAG.
:::
::::

## Write it your way

Express a transformation as a DataFrame, as SQL, or as composable column expressions.
All three build the same plan and run on the same engine, so you can mix them freely.

::::{tab-set}
:::{tab-item} DataFrame
```python
import batcher as bt

sales = bt.from_pydict({"cat": ["a", "b", "a"], "amt": [10.0, 20.0, 30.0]})
revenue = sales.group_by("cat").agg(total=bt.col("amt").sum())
print(revenue.sort("total", descending=True).to_pydict())
# {'cat': ['a', 'b'], 'total': [40.0, 20.0]}
```
:::

:::{tab-item} SQL
```python
import batcher as bt

sales = bt.from_pydict({"cat": ["a", "b", "a"], "amt": [10.0, 20.0, 30.0]})
revenue = bt.sql("SELECT cat, SUM(amt) AS total FROM sales GROUP BY cat", sales=sales)
print(revenue.sort("total", descending=True).to_pydict())
# {'cat': ['a', 'b'], 'total': [40.0, 20.0]}
```
:::

:::{tab-item} Expressions
```python
import batcher as bt

ds = bt.from_pydict({"price": [10.0, 20.0, 30.0], "qty": [1, 2, 3]})
revenue = bt.col("price") * bt.col("qty")            # a value you build once
tier = bt.when(revenue > 40).then(bt.lit("high")).otherwise(bt.lit("low"))
print(ds.select(revenue=revenue, tier=tier).to_pydict())
# {'revenue': [10.0, 40.0, 90.0], 'tier': ['low', 'low', 'high']}
```
:::
::::

Expressions carry typed accessors for every column kind — `.str`, `.dt`, `.list`,
`.struct` — so the column language is the same whether you reach for it from a
DataFrame or from SQL.

## Run it anywhere, at any cadence

The source is the only thing that changes between a laptop and production:

```python
# docs: skip
bt.read("data/*.parquet")                  # local files
bt.read("s3://bucket/events/*.parquet")    # object storage
bt.read.images("s3://photos/**/*.jpg")     # multimodal
bt.read.kafka(topic="events")              # streams
```

And because batch is just the bounded case of streaming, the same pipeline runs as a
one-shot job, a micro-batch, or a continuous stream — you change one argument:

```python
# docs: skip
pipeline.write.parquet("out/")                                       # batch (default)
pipeline.write.parquet("out/", trigger=bt.Trigger.processing_time("5s"))  # micro-batch
pipeline.write.parquet("out/", trigger=bt.Trigger.continuous("1s"))       # continuous
```

Same operators, same results — only the cadence changes, with watermarks, event-time
windows, and exactly-once checkpoints handled for the streaming cases.

## It tunes itself

You don't size batches, pick join strategies, or guess partition counts. Batcher
measures the data as it flows and re-plans the rest of the query on real numbers, so a
query that starts on a bad estimate corrects itself instead of stalling — the kind of
mid-flight adaptation a plan-once optimizer can't do. The
[architecture guide](architecture/index.md) covers how, if you're curious.

## How it compares

| Reach for Batcher when | Because |
| --- | --- |
| You outgrow DuckDB's single node | the same query scales out, and re-optimizes mid-flight rather than planning once |
| Polars is fast but stops at one machine | the same code runs from one core to a cluster |
| Spark's overhead dominates small jobs | it runs in-process locally, with no cluster to spin up |
| You're gluing a query tool to a loader to a model server | one engine spans all three over the same Arrow data |

Speed is measured correctness-first: the benchmark harness refuses to time a query
whose result doesn't match DuckDB, and every operator is differential-tested against
it. The numbers live in [`benchmarks/`](https://github.com/stephenoffer/batcher/tree/main/benchmarks).

## Where to go next

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Getting started
:link: getting-started/index
:link-type: doc
Install, run a first pipeline, and learn the lazy execution model.
:::

:::{grid-item-card} {octicon}`book;1.1em` Tutorials & examples
:link: examples/index
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
Every public class and function, generated from the docstrings.
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
examples/index
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
