# Batcher

```{raw} html
<div class="bt-hero">
  <p class="bt-hero-eyebrow">Any data &middot; Any workload &middot; Batch &amp; streaming</p>
  <p class="bt-hero-tagline">One engine for every kind of data, from SQL to models.</p>
  <p class="bt-hero-sub">
    Structured tables, unstructured text, images, audio, video. SQL, DataFrames, and
    expressions. Batch jobs and live streams, analytics and inference. Batcher runs all
    of it on a single engine &mdash; from a laptop to a cluster &mdash; and tunes itself
    as the query runs.
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
    <span class="bt-stat-value">42 / 43</span>
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
whose result does not match the oracle. {doc}`benchmarks/index` has the methodology, the
hardware, and the comparisons that run the other way.

Data work has splintered into a tool per job. One for SQL, another for DataFrames, a
third for streaming, more again for images and models. Every one of them is a system to
run and a seam to leak. Batcher collapses that stack into a single engine.

![One engine: any source, whether Parquet, media, Kafka, or a lakehouse table, flows into Batcher and back out to any workload: SQL and ETL, batch inference, embeddings, and training data.](_static/diagrams/hub.svg)

## Why Batcher

Every tool stops somewhere, and the gaps between them are where the time goes.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`git-branch;1.1em` Outgrow it, rewrite it
A fast single-node engine hits a ceiling. Scaling out means porting the pipeline to a
different system with different semantics.
:::

:::{grid-item-card} {octicon}`stack;1.1em` A tool per job
SQL in one engine, DataFrames in another, separate loaders and servers for ML. Every
hand-off between them is a place for data and effort to leak.
:::

:::{grid-item-card} {octicon}`gear;1.1em` Tuned by hand
Batch sizes, partition counts, join order. Guess wrong and the job stalls or runs out
of memory, often only once it's big enough to matter.
:::
::::

Batcher answers all three at once. The same code runs from a laptop to a cluster, one
engine covers SQL and DataFrames and ML, and the plan re-tunes itself while it runs. You
build the pipeline once, and it keeps working as the data grows.

## Everything it does

One engine, one API, one plan. These are the capability families, each linked to the guide
that covers it.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` Read anything
:link: user-guide/reading-data
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
:link: user-guide/lakehouse
:link-type: doc
Delta, Iceberg, and Hudi with transactional writes, `MERGE INTO` upserts, change feeds,
time travel, schema evolution, and compaction.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: user-guide/streaming
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
:link: ml/multimodal
:link-type: doc
Images, audio, and video decoded straight into tensor columns, with first-class list and
tensor types and the vector ops behind similarity search.
:::

:::{grid-item-card} {octicon}`shield-check;1.1em` Quality and governance
:link: user-guide/data-quality
:link-type: doc
Data-quality contracts that fail, drop, or quarantine bad rows. Column masking and
row-level security applied as a plan rewrite, plus column-level lineage.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Scale and operate
:link: user-guide/performance
:link-type: doc
Out-of-core spill, caching, a Ray-backed distributed path, explain plans, a live progress
UI, and metrics. The same code from a laptop to a cluster.
:::
::::

## Write it your way

Express a transformation as a DataFrame, as SQL, or as composable expressions, then run
it as a batch job or a live stream. Every form builds the same plan and runs on the same
engine, so you can mix them freely.

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

Expressions carry typed accessors for every column kind (`.str`, `.dt`, `.list`,
`.struct`), so the column language stays the same whether you reach for it from a
DataFrame, from SQL, or inside a stream.

## It tunes itself

You don't size batches, pick join strategies, or guess partition counts. Batcher
re-optimizes at stage boundaries on measured cardinalities, the same mechanism and the same
granularity as Spark AQE, but available single-node too. It engages only on a joined query
whose scan input clears 20M rows or roughly 1.3 GB, so most small queries never reach it.

![A capability matrix comparing DuckDB, Spark AQE, and Batcher on three properties: re-planning inside one query, running on a single node, and carrying what was learned into the next run. DuckDB optimizes once and keeps no cross-run state. Spark AQE re-plans at stage boundaries but needs shuffle stages and keeps no cross-run state. Batcher re-plans at the same stage-boundary granularity, runs the same loop on a single node, and carries sketches, calibrated costs, and a bandit into the next run.](_static/diagrams/adaptive_positioning.svg)

The half that has no equivalent in DuckDB or Spark is what happens *between* runs. A
sketch-backed learned-stats and bandit loop records what each query actually did, so the
plan improves the more often you run it. {doc}`architecture/differentiators` covers both
halves, and where each one stops.

## How it compares

Each tool stops somewhere. Batcher aims at the whole range on one engine.

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
<p class="bt-matrix-legend"><span class="y">✓</span> built-in &nbsp; <span class="p">~</span> partial or via an add-on &nbsp; <span class="n">—</span> not supported. A capability view, not a benchmark.</p>
```

Speed is measured correctness-first: the benchmark harness refuses to time a query
whose result doesn't match DuckDB, and every operator is differential-tested against
it.

## Benchmarks

Numbers, not adjectives, and every one is **correctness-gated**: the harness runs each query
on every engine, checks they return the identical result, and only then trusts the timing. A
fast wrong answer is a bug, not a win. The gate earns its keep. On TPC-H q6 both Daft and
Polars compute the wrong revenue, and the harness refuses to time them.

These are the headline results. The full picture, including the methodology behind every
figure, is in {doc}`benchmarks/index`.

### Analytics: four suites

Every engine reads the **identical zero-copy Arrow input**, so this compares execution rather
than storage formats. The suites below were measured on a single node, at the scale and date
each row names.

| suite | vs DuckDB on the same Arrow |
|---|---|
| **TPC-H sf10**, all 22 queries, 96 cores | **won 21 of 22**, **1.89x** on the suite total |
| **TPC-H sf1**, all 22 queries, 16 cores | **won 22 of 22**, 1.1x to 7.1x faster |
| **ClickBench**, 43 queries | **won 42 of 43**, and 43/43 correct |
| **Semi-structured JSON**, 5 queries | **won 5 of 5**, 3.6x to 12.5x faster |
| **Operator mix**, 11 kernels | **won 10 of 11** |

![Diverging bar chart of the TPC-H scale-factor-10 suite ratio. Batcher is 1.89x faster than DuckDB reading the same Arrow, winning 21 of 22 queries, and 2.26x faster than Polars, winning 17 of 22. Batcher is 2.08x behind DuckDB on its own native store, winning 4 of 22.](_static/diagrams/tpch_sf10.svg)

At sf10 Batcher is also **2.26x** faster than Polars over the suite, winning 17 of 22. On the
JSON suite it is **11x to 100x** faster than Polars, whose SQL front-end cannot express most of
TPC-H at all (multi-table `FROM`, `EXISTS`, non-equi joins).

Run TPC-H against DuckDB's own compressed store instead, where it decompresses as it scans and
never pays an Arrow ingest, and DuckDB leads: **2.08x** on the sf10 suite, with Batcher winning
4 of 22. That comparison is not like-for-like, and both columns are published per query in
{doc}`benchmarks/tpch`.

Seven ClickBench queries return in about 0.2 ms because Kyber answers them from **metadata**,
meaning footer statistics and sketches, rather than scanning at all. Those are excluded from
the ranges above, so the headline reflects execution rather than planning.

The same lazy control plane answers `count()` in **0.05 ms** after a chain of transformations,
and reading 20M rows across 64 Parquet files and summing a column takes **72 ms**.

### AI and multimodal

Models are one workload family the engine runs, alongside the SQL, streaming, and lakehouse
work above, and they run on the same operators and the same plan. Ten GPU workload families
on 8xT4, real models, every run gated on prediction agreement. On every family where device
utilization was sampled, the GPU holds at or above the 80% target:

![Horizontal bar chart of sustained GPU utilization by workload family on 8xT4 with real models and 100 percent output agreement. Compute-bound ResNet-50 FP16 inference holds 100 percent at 4,707 images per second, a decode-heavy JPEG to ResNet pipeline 93.4 percent at 3,860, fractional GPU packing of EfficientNet-B0 89 percent at 6,764, zero-config inference with no batch size given 82 percent at 2,451, ResNet-50 batch inference 81 percent at 2,504, and image embeddings 80 percent at 2,502. A dashed line marks the 80 percent target.](_static/diagrams/gpu_utilization.svg)

Throughput on the model workloads runs from **33,611 text/s** embedding with MiniLM and
**38,546 clip/s** on audio feature extraction down to **169 img/s** on a diffusion model, and
`iter_torch_batches` feeds a training loop at **1.06 M rows/s** zero-copy. Decoding JPEGs into
tensors runs at 5,693 img/s, 2.4x Daft. `map_batches(Model, num_gpus=1)` with no batch size
given picks a VRAM-safe default and lands within 2% of the hand-tuned path.

Stage-overlapped streaming is what produces those utilization figures. The CPU decode of the
next morsel runs while the GPU forward of the current one is still in flight.

![Two panels comparing a two-stage ResNet-50 pipeline before and after stage overlap, with the same result and the same order. Throughput rises from 942 to 2,504 images per second. GPU utilization rises from 30 percent to 81 percent of the device kept busy.](_static/diagrams/stage_overlap.svg)

### Cluster against cluster

The mergeable algebra means the *same* operators run distributed. TPC-H sf10 q6 on an 8-node,
128-CPU cluster, with **both engines distributed** and reading the same S3 parquet:

| engine | time | correct? |
|---|---:|---|
| **Batcher** | **224 ms** | ✅ |
| Daft | 536 ms | ❌ wrong answer |
| DuckDB (single-node, its best) | 457 ms | ✅ |

**2.4x faster than Daft on equal hardware, and correct where Daft is not.**

**{doc}`Full benchmarks and methodology <benchmarks/index>`**

### Why the wins happen

None of this is tuning. Each result traces to a design choice you can read about in the
{doc}`architecture guide <architecture/index>`.

Batcher runs in-process and native over Arrow, with no task-scheduler or object-store hop per
operation, so a small query pays almost no fixed cost before it starts doing real work. On the
same input, that is what lets an execution engine win a suite outright rather than query by
query.

On the AI side, GPU inference loads a model once per session and overlaps CPU prep with the
GPU forward pass, which is what holds the device at or above 80% utilization wherever it was
sampled. And plans re-tune on measured cardinalities mid-query, so a bad estimate corrects
itself rather than stalling or running out of memory.

## Find your way around

The docs branch by what you are doing, not by which part of the engine you are touching.
Pick the row that matches you.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Start here
:link: getting-started/index
:link-type: doc
Install Batcher, run a first query, then the {doc}`core concepts <getting-started/concepts/index>`
the rest of the API rests on.
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
End-to-end {doc}`tutorials <tutorials/index>`, task-sized {doc}`examples <examples/index>`, and
a {doc}`cookbook <cookbook/index>` of runnable single-purpose scripts.
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
The {doc}`API reference <api/index>`, a one-page {doc}`quick reference <api/reference>`, and
the {doc}`full signature listing <api/complete>`.
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
The {doc}`architecture <architecture/index>`, {doc}`what makes it different
<architecture/differentiators>`, and {doc}`deep dives <deep-dives/index>` one mechanism at a
time.
:::
::::

```{toctree}
:hidden:
:caption: Learn

getting-started/index
tutorials/index
examples/index
cookbook/index
learning-paths/index
```

```{toctree}
:hidden:
:caption: Guides

user-guide/index
ml/index
integrations/index
configuration/index
migration/index
agents/index
```

```{toctree}
:hidden:
:caption: Reference

api/index
benchmarks/index
```

```{toctree}
:hidden:
:caption: How it works

architecture/index
deep-dives/index
internals/index
```
