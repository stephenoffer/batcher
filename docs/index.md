# Batcher

```{raw} html
<div class="bt-hero">
  <p class="bt-hero-eyebrow">Any data &middot; Any workload &middot; Batch &amp; streaming</p>
  <p class="bt-hero-tagline">One engine for every kind of data, from SQL to models.</p>
  <p class="bt-hero-sub">
    Structured tables, unstructured text, images, audio, video. SQL, DataFrames, and
    expressions. Batch jobs and live streams, analytics and inference. Batcher runs all
    of it on a single engine, from a laptop to a cluster, and tunes itself as the query
    runs.
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
    <span class="bt-stat-value">1.89&times;</span>
    <span class="bt-stat-label">faster than DuckDB on the same Arrow</span>
    <span class="bt-stat-src">TPC-H sf10, 96 cores, 21 of 22 queries won</span>
  </div>
  <div class="bt-stat">
    <span class="bt-stat-value">43 / 43</span>
    <span class="bt-stat-label">ClickBench queries won</span>
    <span class="bt-stat-src">vs DuckDB on the same Arrow, 43 of 43 correct</span>
  </div>
  <div class="bt-stat">
    <span class="bt-stat-value">81%</span>
    <span class="bt-stat-label">sustained GPU utilization</span>
    <span class="bt-stat-src">ResNet-50 batch inference, 8&times;T4, 2,504 img/s</span>
  </div>
  <div class="bt-stat">
    <span class="bt-stat-value">0.05 ms</span>
    <span class="bt-stat-label">to answer <code>count()</code> after a transform chain</span>
    <span class="bt-stat-src">read from footer statistics, no scan</span>
  </div>
</div>
```

Every figure above is correctness-gated: the harness refuses to record a timing for a query
whose result does not match the oracle. {doc}`benchmarks/index` has the methodology and the
hardware per family, plus the two comparisons that run the other way.

## What Batcher is

Data work has splintered into a tool per job. One for SQL, another for DataFrames, a third
for streaming, more again for images and models. Every one of them is a system to run and a
seam to leak. Batcher collapses that stack into a single engine: a Python control plane over
a Rust data plane on Apache Arrow.

![One engine: any source, whether Parquet, media, Kafka, or a lakehouse table, flows into Batcher and back out to any workload: SQL and ETL, batch inference, embeddings, and training data.](_static/diagrams/hub.svg)

The stack most teams run was designed when a big server had 16 cores, data landed overnight,
and machine learning happened somewhere else. All of that changed, and each change left a
seam:

| The seam | What it costs |
|---|---|
| A fast single-node engine hits a ceiling | Scaling out means porting the pipeline to a system with different semantics |
| Batch and streaming are two execution models | Any pipeline that crosses between them is a rewrite |
| Catalog statistics assume a schema that holds still | No catalog holds the selectivity of a regex or the cost of a model call, and both now sit mid-plan |
| Most data is documents, images, audio, and video | An engine whose types stop at scalars can only hand it somewhere else |
| A pipeline is I/O, then decode, then inference, then a join | Run that in one worker and the expensive device idles through every JPEG |
| The natural way to write a transform is a per-row callback | The optimizer cannot see into it and cannot reorder it |

Batcher answers all six, and mostly with one decision. Every stateful operator exists once, as
a mergeable `partial → combine → finalize` triple in Rust over Arrow. One core, ninety-six cores, and a
cluster differ only in how that triple is scheduled. The same triple is the incremental form,
so batch is the bounded case of streaming. And because the operator is identical everywhere, a
measurement taken anywhere is valid everywhere, which is what lets the optimizer plan from
evidence instead of vendor constants. Decode, embedding, vector search, and inference are
expressions in the same algebra, so a predicate pushes beneath a JPEG decode and a tensor never
leaves the engine.

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
:link: user-guide/index
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
stream joins, checkpointing, and exactly-once sinks.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Models and inference
:link: ml/index
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
:link: /user-guide/operate/performance
:link-type: doc
Out-of-core spill, caching, a Ray-backed distributed path, explain plans, a live progress
UI, and metrics. The same code from a laptop to a cluster.
:::
::::

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
revenue = bt.col("price") * bt.col("qty")            # a value you build once
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

# batch (default) → micro-batch → continuous: change one argument
counts.write.parquet("out/", trigger=bt.Trigger.processing_time("10s"))
```
:::
::::

Expressions carry typed accessors for every column kind (`.str`, `.dt`, `.list`, `.struct`),
so the column language stays the same whether you reach for it from a DataFrame, from SQL, or
inside a stream.

## It tunes itself

You don't size batches, pick join strategies, or guess partition counts. Batcher re-optimizes
at stage boundaries on measured cardinalities, the same mechanism and the same granularity as
Spark AQE, but available single-node too. It engages only on a joined query whose scan input
clears 20M rows or roughly 1.3 GB, so most small queries never reach it.

The half with no equivalent in DuckDB or Spark is what happens *between* runs. A sketch-backed
learned-stats and bandit loop records what each query actually did, so the plan improves the
more often you run it. {doc}`architecture/differentiators` covers both halves, and where each
one stops.

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

## The headline numbers

Every engine reads the identical zero-copy Arrow input, so these compare execution rather than
storage formats. {doc}`benchmarks/index` carries the full grid, the hardware per family, the
reproduction commands, and the shapes where DuckDB and Daft lead.

| Suite | Result |
|---|---|
| TPC-H sf10, 22 queries, 96 cores | won 21 of 22 vs DuckDB, 1.89x on the suite total |
| TPC-H sf1, 22 queries, 16 cores | won 22 of 22 vs DuckDB, 1.1x to 7.1x faster |
| ClickBench, 43 queries | won 43 of 43 vs DuckDB, and 43 of 43 correct |
| Semi-structured JSON, 5 queries | won 5 of 5 vs DuckDB, 3.6x to 12.5x faster |
| ResNet-50 batch inference, 8xT4 | 2,504 img/s at 81% GPU utilization |
| Text embeddings, MiniLM, 8xT4 | 33,611 text/s |
| TPC-H sf10 q6, cluster vs cluster | 2.4x Daft on equal hardware, and Daft's answer is wrong |

![Diverging bar chart of the TPC-H scale-factor-10 suite ratio. Batcher is 1.89x faster than DuckDB reading the same Arrow, winning 21 of 22 queries, and 2.26x faster than Polars, winning 17 of 22. Batcher is 2.08x behind DuckDB on its own native store, winning 4 of 22.](_static/diagrams/tpch_sf10.svg)

## Find your way around

The docs branch by what you are doing, not by which part of the engine you are touching.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Start here
:link: getting-started/index
:link-type: doc
Install Batcher, run a first query, then the core concepts the rest of the API rests on.
:::

:::{grid-item-card} {octicon}`arrow-switch;1.1em` Coming from another tool
:link: migration/index
:link-type: doc
Spark, pandas, Polars, DuckDB, and Daft translated verb by verb, ending in a check that the
port returns the same rows.
:::

:::{grid-item-card} {octicon}`book;1.1em` Learn by doing
:link: tutorials/index
:link-type: doc
End-to-end tutorials, plus a {doc}`cookbook <cookbook/index>` of about 150 runnable
pages, from one-method recipes to complete pipelines.
:::

:::{grid-item-card} {octicon}`checklist;1.1em` Follow a path
:link: learning-paths/index
:link-type: doc
An ordered reading list for a data engineer, data scientist, ML engineer, or platform
engineer.
:::

:::{grid-item-card} {octicon}`repo;1.1em` Build something
:link: user-guide/index
:link-type: doc
One page per capability, plus {doc}`ml/index` for models and {doc}`configuration/index` for
the knobs.
:::

:::{grid-item-card} {octicon}`code-square;1.1em` Look something up
:link: api/index
:link-type: doc
The API reference, a one-page quick reference, and the full signature listing.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Connect your stack
:link: integrations/index
:link-type: doc
Kafka, Snowflake, BigQuery, Delta, Iceberg, Hudi, MongoDB, Elasticsearch, Ray, PyTorch, and
Hugging Face.
:::

:::{grid-item-card} {octicon}`telescope;1.1em` Understand the engine
:link: architecture/index
:link-type: doc
The architecture, {doc}`deep dives <deep-dives/index>` one mechanism at a time, and the
{doc}`subsystem internals <internals/index>` for contributors.
:::

:::{grid-item-card} {octicon}`dependabot;1.1em` Working with an agent
:link: agents/index
:link-type: doc
The skill catalog: task-scoped recipes an AI agent can follow to write a Batcher pipeline,
then debug and scale it.
:::
::::

```{toctree}
:hidden:
:caption: Start here

getting-started/index
tutorials/index
learning-paths/index
migration/index
```

```{toctree}
:hidden:
:caption: Guides

user-guide/index
ml/index
integrations/index
agents/index
```

```{toctree}
:hidden:
:caption: Recipes

cookbook/index
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
deep-dives/index
internals/index
```
