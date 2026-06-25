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

Data work has splintered into a tool per job: one for SQL, another for DataFrames,
a third for streaming, more for images and models. Each new tool is another system
to run and another seam to leak. Batcher's goal is to collapse that stack into one
engine — covering every kind of data and every AI workload, from a quick query on a
laptop to a petabyte job on a cluster, with the same code throughout.

## Any data

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` Structured
Tables from Parquet, CSV, JSON, and the lakehouse formats (Delta, Iceberg, Hudi).
Filter, join, and aggregate them with SQL or DataFrames, and write the results back.
:::

:::{grid-item-card} {octicon}`file;1.1em` Unstructured
Text, logs, and documents. Read whole files or lines, parse and extract fields, and
turn messy input into clean columns at scale.
:::

:::{grid-item-card} {octicon}`image;1.1em` Multimodal
Images, audio, and video, decoded straight into tensors — so the same pipeline that
cleans a table can feed a model, with no separate loader to wire up.
:::

:::{grid-item-card} {octicon}`search;1.1em` Vectors & embeddings
List and tensor columns are first-class, with the vector operations that back
embeddings, similarity search, and RAG feature pipelines.
:::
::::

## Any AI workload

One engine carries a job from raw data to a served model, instead of handing it
between systems:

- **Analytics & ETL** — clean, join, and aggregate, then write to files or a lakehouse.
- **Batch inference** — run a model over Arrow batches with the `.ml` accessor; the
  scheduler places the work on GPUs and across workers.
- **Embeddings & RAG** — embed text or images and build the vector tables search needs.
- **Training data** — shuffle, batch, and stream examples straight into your trainer.

## Three ways to express your logic

Write a transformation as a DataFrame or as SQL — both build the same plan and run on
the same engine, so you can mix them freely.

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

Underneath both is the **expression API**: column logic as composable values you
build once and reuse. Arithmetic, comparisons, conditionals, and typed accessors
(`.str`, `.dt`, `.list`, `.struct`) all build the same `Expr`, evaluated in Rust.

```python
import batcher as bt

ds = bt.from_pydict({"price": [10.0, 20.0, 30.0], "qty": [1, 2, 3]})

revenue = bt.col("price") * bt.col("qty")          # an expression, not a result
tier = bt.when(revenue > 40).then(bt.lit("high")).otherwise(bt.lit("low"))

print(ds.select(revenue=revenue, tier=tier).to_pydict())
# {'revenue': [10.0, 40.0, 90.0], 'tier': ['low', 'low', 'high']}
```

The typed accessors give every column type its own vocabulary:

```python
# docs: skip
clean = bt.col("email").str.lower().str.trim()      # string ops
year = bt.col("signup_ts").dt.year()                # datetime parts
has_tag = bt.col("tags").list.contains("ai")        # list / array ops
```

## Batch, micro-batch, or continuous — change one line

Batch is just the bounded case of streaming, so the pipeline you wrote for a file
runs on a live stream untouched. You choose *how* it runs with one argument:

```python
# docs: skip
pipeline = bt.read.kafka(topic="events").filter(bt.col("status") == "active")

# Batch: process what's there now, then stop (the default).
pipeline.write.parquet("out/")

# Micro-batch: a new batch every 5 seconds.
pipeline.write.parquet("out/", trigger=bt.Trigger.processing_time("5s"))

# Continuous: low-latency, always on.
pipeline.write.parquet("out/", trigger=bt.Trigger.continuous("1s"))
```

Same operators, same results — only the cadence changes. Watermarks, event-time
windows, and exactly-once checkpoints come along for the streaming cases.

## Reads from anywhere

The source is the only thing that changes between a laptop and production:

```python
# docs: skip
bt.read("data/*.parquet")                  # local files
bt.read("s3://bucket/events/*.parquet")    # object storage
bt.read.images("s3://photos/**/*.jpg")     # multimodal
bt.read.delta("s3://lake/events")          # lakehouse tables
bt.read.kafka(topic="events")              # streams
```

## It tunes itself

You don't size batches, pick join strategies, or guess partition counts. Batcher
measures the data as it flows and re-plans the rest of the query on real numbers, so
a query that starts on a bad estimate corrects itself instead of stalling — the kind
of mid-flight adaptation a plan-once optimizer can't do. The
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
