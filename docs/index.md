# Batcher

```{raw} html
<div class="bt-hero">
  <img class="bt-hero-logo" src="_static/logo.png" alt="The Batcher logo: a letter B drawn as five horizontal bars sweeping from cyan through electric blue to magenta.">
  <p class="bt-hero-eyebrow">Any data &middot; Any workload &middot; Batch &amp; streaming</p>
  <p class="bt-hero-tagline">One engine for every kind of data, from SQL to models.</p>
  <p class="bt-hero-sub">
    Tables, text, images, audio, video. SQL, DataFrames, and expressions. Batch jobs and
    live streams, analytics and inference. Batcher runs all of it on one engine, from a
    laptop to a cluster, and tunes itself as the query runs.
  </p>
  <p class="bt-hero-cta">
    <a class="bt-btn bt-btn-primary" href="getting-started/index.html">Get started</a>
    <a class="bt-btn" href="getting-started/quickstart.html">Quickstart</a>
    <a class="bt-btn" href="benchmarks/index.html">See the numbers</a>
    <a class="bt-btn" href="https://github.com/stephenoffer/batcher">GitHub</a>
  </p>
</div>

<div class="bt-stats">
  <div class="bt-stat">
    <span class="bt-stat-value">4.0&times;</span>
    <span class="bt-stat-label">faster than DuckDB on the same Arrow</span>
    <span class="bt-stat-src">TPC-H sf1, 22 queries, 48 cores</span>
  </div>
  <div class="bt-stat">
    <span class="bt-stat-value">6.3&times;</span>
    <span class="bt-stat-label">faster than DuckDB on the same Arrow</span>
    <span class="bt-stat-src">ClickBench, 43 queries, 48 cores</span>
  </div>
  <div class="bt-stat">
    <span class="bt-stat-value">6 / 6</span>
    <span class="bt-stat-label">benchmark suites faster than Polars</span>
    <span class="bt-stat-src">TPC-H, ClickBench, H2O, JSON, operators</span>
  </div>
  <div class="bt-stat">
    <span class="bt-stat-value">2.4&times;</span>
    <span class="bt-stat-label">faster than Ray Data on GPU inference</span>
    <span class="bt-stat-src">100,000 images, six T4 nodes, same checksum</span>
  </div>
  <div class="bt-stat">
    <span class="bt-stat-value">81%</span>
    <span class="bt-stat-label">sustained GPU utilization</span>
    <span class="bt-stat-src">ResNet-50 batch inference, 8&times;T4, 2,504 img/s</span>
  </div>
</div>
```

Every figure on this site is correctness-gated: the harness runs the query on each engine,
compares the results, and refuses to record a timing when they disagree. A missing number
means a wrong answer, not a slow one.

## What Batcher is

Data work has splintered into a tool per job. One for SQL, another for DataFrames, a third
for streaming, more again for images and models. Every one of them is a system to run and a
seam to leak. Batcher collapses that stack into a single engine: a Python control plane over
a Rust data plane on Apache Arrow.

![One engine: any source, whether Parquet, media, Kafka, or a lakehouse table, flows into Batcher and back out to any workload: SQL and ETL, batch inference, embeddings, and training data.](_static/diagrams/hub.svg)

One decision buys most of that. Every stateful operator exists once, as a mergeable
`partial -> combine -> finalize` triple in Rust over Arrow. One core, ninety-six cores, and a
cluster differ only in how that triple is scheduled, so scaling out is a scheduling decision
rather than a port. The same triple is the incremental form, so batch is the bounded case of
streaming rather than a second execution model. Because the operator is identical everywhere,
a measurement taken anywhere is valid everywhere, which is what lets the optimizer plan from
evidence instead of vendor constants. And decode, embedding, vector search, and inference are
expressions in that same algebra, so a predicate pushes beneath a JPEG decode and a tensor
never leaves the engine.

## Start from your job

Pick the path that matches the work in front of you. Each one is an ordered reading list through the tutorials and guides.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Data engineer
:link: /getting-started/tutorials/paths/data-engineer
:link-type: doc
Read, reshape, join, and write: pipelines, lakehouse tables, and data quality.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Data scientist
:link: /getting-started/tutorials/paths/data-scientist
:link-type: doc
Expressions, aggregations, SQL, and window functions over a dataset.
:::

:::{grid-item-card} {octicon}`cpu;1.1em` ML engineer
:link: /getting-started/tutorials/paths/ml-engineer
:link-type: doc
Batch inference, embeddings, and GPUs through `.ml`.
:::

:::{grid-item-card} {octicon}`server;1.1em` Platform engineer
:link: /getting-started/tutorials/paths/platform-engineer
:link-type: doc
Configuration, environment defaults, memory limits, and object storage.
:::
::::

Coming from another engine? {doc}`The migration guides </getting-started/migration/index>` translate Spark, pandas, Polars, DuckDB, Ray Data, and Daft code into Batcher.

## Write it your way

Express a transformation as a DataFrame, as SQL, or as composable expressions, then run it as
a batch job or a live stream. Every form builds the same plan and runs on the same engine, so
you can mix them freely.

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
revenue = bt.col("price") * bt.col("qty")  # a value you build once
tier = bt.when(revenue > 40).then(bt.lit("high")).otherwise(bt.lit("low"))
print(ds.select(revenue=revenue, tier=tier).to_pydict())
# {'revenue': [10.0, 40.0, 90.0], 'tier': ['low', 'low', 'high']}
```
:::

:::{tab-item} Streaming
```python
# docs: skip
import batcher as bt

# the same group-by, now over an unbounded source
clicks = bt.read.kafka(topic="clicks")
counts = clicks.group_by("page").agg(n=bt.count())

# batch (default), micro-batch, or continuous: change one argument
counts.write.parquet("out/", trigger=bt.Trigger.processing_time("10s"))
```
:::
::::

Expressions carry typed accessors for every column kind ({py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`),
so the column language stays the same whether you reach for it from a DataFrame, from SQL, or
inside a stream.

## What it does

Each card is one capability family, linked to the guide that covers it.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` Read anything
:link: /user-guide/moving-data/reading-data
:link-type: doc
Parquet, CSV, JSON, Arrow, ORC, Avro. Text, logs, and documents. Images, audio, and video.
Databases and warehouses over JDBC. Kafka, Kinesis, Pulsar, and Pub/Sub.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Query and transform
:link: /user-guide/index
:link-type: doc
Filter, project, join, aggregate, window, pivot, sort, sample, and explode, in SQL or
DataFrame form. Typed accessors for strings, dates, lists, structs, and JSON.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Lakehouse tables
:link: /user-guide/moving-data/lakehouse
:link-type: doc
Delta, Iceberg, and Hudi with transactional writes, `MERGE INTO` upserts, change feeds,
time travel, schema evolution, and compaction.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: /user-guide/moving-data/streaming
:link-type: doc
Unbounded sources, triggers, watermarks and late data, windowed and stateful aggregation,
stream joins, checkpointing, and exactly-once delivery into a transactional sink.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Models and inference
:link: /ml/index
:link-type: doc
Batch inference on GPU, LLM scoring, embeddings and vector search, RAG, tabular models,
preprocessors, and zero-copy loaders for PyTorch training.
:::

:::{grid-item-card} {octicon}`image;1.1em` Multimodal and vectors
:link: /ml/preparing/multimodal/index
:link-type: doc
Images, audio, and video decoded straight into tensor columns, with first-class list and
tensor types and the vector ops behind similarity search.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` Quality and governance
:link: /user-guide/trust/data-quality
:link-type: doc
Data-quality contracts that fail, drop, or quarantine bad rows. Column masking and
row-level security applied as a plan rewrite, plus column-level lineage.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Scale and operate
:link: /user-guide/operate/tuning/performance
:link-type: doc
Out-of-core spill, caching, a Ray-backed distributed path, explain plans, a live progress
UI, and metrics. The same code from a laptop to a cluster.
:::
::::

## It tunes itself

You don't size batches, pick join strategies, or guess partition counts. Batcher re-optimizes
at stage boundaries on measured cardinalities, the same mechanism and the same granularity as
Spark AQE, but available single-node too. It engages only on a joined query big enough to pay
for the re-planning, which is 5M rows or roughly 320 MB for each pipeline breaker the loop
would cut at, so most small queries never reach it.

The half with no equivalent in DuckDB or Spark is what happens *between* runs. A sketch-backed
learned-stats and bandit loop records what each query actually did, so the plan improves the
more often you run it.

![The loop that outlives one query. In run N, Kyber plans on whatever it knows and Core executes and measures. Core writes measured cardinalities, operator wall times, column sketches, fitted cost coefficients and bandit arm rewards to the MetadataHub, keyed by plan signature and, for anything in machine units, by hardware fingerprint. Run N plus one reads that before planning, then measures and records again. The query ends and the hub does not, which is the difference from Spark AQE.](_static/diagrams/cross_run_learning.svg)

{doc}`architecture/differentiators` covers both halves, and where each one stops.

## The numbers

Every figure below is correctness-gated, and DuckDB is measured two ways. *Same Arrow* is DuckDB executing over the identical zero-copy input Batcher runs on, which isolates the two execution engines. *Native store* is DuckDB over its own compressed, dictionary-encoded, zone-mapped format, ingested before the clock starts: a storage engine plus an execution engine, against Batcher's execution engine alone.

The suite results come from one sweep on a 48-core box on 2026-09-13, best of five, one process per case. They're speedups, so bigger is better: 4.0x means Batcher finishes in a quarter of the other engine's time. The tables under {doc}`benchmarks/index` report the inverse, a `batcher / other` time ratio where lower is better.

| Suite | vs DuckDB, same Arrow | vs DuckDB, native store | vs Polars | Cases where Batcher is fastest |
|---|---|---|---|---|
| TPC-H sf1, 22 queries | **4.0x** | **1.4x** | **1.9x** | 16 of 22 |
| ClickBench, 43 queries | **6.3x** | **1.5x** | **2.7x** | 28 of 43 |
| Semi-structured JSON, 5 queries | **3.1x** | **2.9x** | **over 60x** | 5 of 5 |
| H2O.ai `join`, 5 queries | **1.7x** | **1.6x** | **2.0x** | 5 of 5 |
| Operator mix, 46 kernels | **2.1x** | **1.3x** | **6.3x** | 33 of 46 |
| H2O.ai `groupby`, 10 queries | **1.2x** | 0.95x | **1.9x** | 4 of 10 |

Batcher is faster than Polars and than DuckDB on the same Arrow in all six suites, and faster than DuckDB's native store in five. The last column counts a case only when Batcher beats every engine in the sweep. Nearly half of the remaining cases are storage wins for DuckDB's compressed format rather than execution gaps.

| Other workloads | Result |
|---|---|
| GPU batch inference, 100,000 images on six T4 nodes | **2.4x** Ray Data and **5.4x** Daft, identical checksums |
| ResNet-50 batch inference, 8xT4 | **2,504 img/s** at 81% GPU utilization |
| Text embeddings, MiniLM, 8xT4 | **33,611 text/s** |
| Image decode to tensor, one 96-core node | **5,693 img/s**, 2.4x Daft |
| TPC-H sf10 q6, cluster against cluster | **2.4x** Daft on equal hardware, and Daft's answer is wrong |

![Bar chart of the TPC-H scale-factor-10 suite on the same Arrow input, from the 2026-08-28 sweep on 92 cores. Batcher is 3.03x faster than DuckDB reading the same Arrow and 2.86x faster than Polars.](_static/diagrams/tpch_sf10.svg)

Those rows were not all measured on the same machine, because the workload families were
not. A figure is meaningful within its row. {doc}`benchmarks/index` carries the full grid,
the hardware per family, and the reproduction commands.

## How it compares

Each tool stops somewhere. Batcher aims at the whole range on one engine. This is a capability
view rather than a benchmark; for timings, read {doc}`benchmarks/index`.

```{raw} html
<table class="bt-matrix">
<thead><tr><th>Capability</th>
<th>Batcher</th>
<th>DuckDB</th>
<th>Polars</th>
<th>Spark</th>
</tr></thead><tbody>
<tr><td>Runs in-process, no cluster</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="n">—</span></td></tr>
<tr><td>Sub-second small queries</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="n">—</span></td></tr>
<tr><td>Scales to a cluster</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="y">✓</span></td></tr>
<tr><td>Same code, laptop to cluster</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td></tr>
<tr><td>SQL</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td></tr>
<tr><td>DataFrame API</td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td></tr>
<tr><td>Composable expression API</td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td></tr>
<tr><td>Cost-based optimizer</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td></tr>
<tr><td>Stage-boundary re-optimization, single-node</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td></tr>
<tr><td>Cross-query learned statistics</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td></tr>
<tr><td>Streaming</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td></tr>
<tr><td>ML / batch inference</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td></tr>
<tr><td>Multimodal (images, audio, video)</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td></tr>
<tr><td>Out-of-core spill</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td></tr>
</tbody></table>
<p class="bt-matrix-legend"><span class="y">✓</span> built-in &nbsp; <span class="p">~</span> partial or via an add-on &nbsp; <span class="n">—</span> not supported.</p>
```

## Find your way around

The site has ten sections, and they branch by what you are doing rather than by which part of
the engine you are touching.

| Section | What is in it |
| --- | --- |
| {doc}`Getting started </getting-started/index>` | Install, a first query, the core concepts, and translations from Spark, pandas, Polars, DuckDB, and Daft |
| {doc}`Tutorials </getting-started/tutorials/index>` | End-to-end walkthroughs, and a reading path ordered by the job you do |
| {doc}`User guide </user-guide/index>` | One page per capability: moving data, transforming, analyzing, trusting, and operating it |
| {doc}`ML and inference </ml/index>` | Preparing data for models, batch inference, retrieval and generation, evaluation, and training loaders |
| {doc}`Integrations </integrations/index>` | Kafka, Snowflake, BigQuery, Delta, Iceberg, Hudi, MongoDB, Elasticsearch, Ray, PyTorch, Hugging Face |
| {doc}`Cookbook </cookbook/index>` | 145 runnable pages, from a one-method recipe to a complete pipeline, each executed on every test run |
| {doc}`API reference </api/index>` | Every public name, a one-page quick reference, and the full signature listing |
| {doc}`Configuration </configuration/index>` | Profiles, options, environment variables, accelerators, and fault tolerance |
| {doc}`Benchmarks </benchmarks/index>` | The full grid against DuckDB, Polars, Spark, and Daft, with the methodology and the losses |
| {doc}`Architecture </architecture/index>` | How the engine works at three zoom levels, from the shape of the system down to one mechanism |

Writing Batcher with an AI agent? {doc}`The skill catalog <agents>` holds task-scoped recipes
for driving the engine correctly.

```{toctree}
:hidden:
:caption: Start here

getting-started/index
```

```{toctree}
:hidden:
:caption: Guides

user-guide/index
ml/index
integrations/index
agents
```

```{toctree}
:hidden:
:caption: Recipes

cookbook/index
examples/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
configuration/index
benchmarks/index
```

```{toctree}
:hidden:
:caption: How it works

architecture/index
```
