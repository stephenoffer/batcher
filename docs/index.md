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

Data work has splintered into a tool per job. One for SQL, another for DataFrames, a
third for streaming, more again for images and models. Every one of them is a system to
run and a seam to leak. Batcher collapses that stack into a single engine.

![One engine: any source — Parquet, media, Kafka, lakehouse — flows into Batcher and back out to any workload: SQL and ETL, batch inference, embeddings, and training data.](_static/diagrams/hub.png)

## Why Batcher

The tools we reach for each stop somewhere, and the gaps between them are where the
time goes:

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

## Any data, any workload

The same engine reads a Parquet table, a folder of images, or a Kafka stream, and the
same pipeline can clean it, query it, or feed it to a model.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` Structured
Parquet, CSV, JSON, and the lakehouse formats (Delta, Iceberg, Hudi), filtered and
joined and aggregated with SQL or DataFrames.
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

## Explore the capabilities

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.1em` Getting started
:link: getting-started/index
:link-type: doc
Install and run your first pipeline.
:::

:::{grid-item-card} {octicon}`download;1.1em` Reading data
:link: user-guide/reading-data
:link-type: doc
Files, object storage, databases, and streams.
:::

:::{grid-item-card} {octicon}`pencil;1.1em` Transformations
:link: user-guide/transformations
:link-type: doc
Select, derive, reshape, and explode columns.
:::

:::{grid-item-card} {octicon}`filter;1.1em` Filtering
:link: user-guide/filtering
:link-type: doc
Predicates, null handling, and sampling.
:::

:::{grid-item-card} {octicon}`graph;1.1em` Aggregations
:link: user-guide/aggregations
:link-type: doc
Group, summarize, pivot, and roll up.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Joins
:link: user-guide/joins
:link-type: doc
Inner, outer, semi, anti, and as-of joins.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Window functions
:link: user-guide/window-functions
:link-type: doc
Ranking, running totals, lag and lead.
:::

:::{grid-item-card} {octicon}`code;1.1em` Expressions
:link: user-guide/expressions
:link-type: doc
The composable column language and its accessors.
:::

:::{grid-item-card} {octicon}`database;1.1em` SQL
:link: user-guide/sql
:link-type: doc
Full SQL that lowers to the same engine.
:::

:::{grid-item-card} {octicon}`broadcast;1.1em` Streaming
:link: user-guide/streaming
:link-type: doc
Watermarks, windows, and exactly-once output.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Machine learning
:link: ml/index
:link-type: doc
Batch inference, embeddings, and training data.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` Cloud & lakehouse
:link: user-guide/cloud-storage
:link-type: doc
S3, GCS, Azure, and Delta / Iceberg / Hudi.
:::

:::{grid-item-card} {octicon}`arrow-switch;1.1em` Coming from pandas / Polars / Spark
:link: migration/index
:link-type: doc
Translate the API you already know, side by side.
:::

:::{grid-item-card} {octicon}`meter;1.1em` Benchmarks
:link: benchmarks/index
:link-type: doc
Correctness-gated numbers across every workload, including where we still lose.
:::

:::{grid-item-card} {octicon}`book;1.1em` Recipes
:link: examples/index
:link-type: doc
Four cookbooks: data engineering, analytics, ML, and streaming.
:::

:::{grid-item-card} {octicon}`plug;1.1em` Integrations
:link: integrations/index
:link-type: doc
Kafka, Snowflake, BigQuery, Delta, Iceberg, Ray, PyTorch, and the rest of your stack.
:::

:::{grid-item-card} {octicon}`telescope;1.1em` Deep dives
:link: deep-dives/index
:link-type: doc
How the engine actually works, one mechanism at a time.
:::
::::

## It tunes itself

You don't size batches, pick join strategies, or guess partition counts. Batcher measures
the data as it flows and re-plans the rest of the query on real numbers, so a query that
starts on a bad estimate corrects itself instead of stalling. A plan-once optimizer cannot
do that, because by the time it learns it was wrong the query is already over. The
[architecture guide](architecture/index.md) covers how, if you're curious.

## How it compares

Each tool stops somewhere; Batcher's aim is the whole range on one engine.

```{raw} html
<table class="bt-matrix">
<thead><tr><th>Capability</th>
<th>Batcher</th>
<th>DuckDB</th>
<th>Polars</th>
<th>Spark</th>
<th>Ray&nbsp;Data</th>
</tr></thead><tbody>
<tr><td>Runs in-process, no cluster</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td></tr>
<tr><td>Sub-second small queries</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td></tr>
<tr><td>Scales to a cluster</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td></tr>
<tr><td>Same code, laptop to cluster</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td><td><span class="p">~</span></td></tr>
<tr><td>SQL</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td><td><span class="n">—</span></td></tr>
<tr><td>DataFrame API</td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td></tr>
<tr><td>Composable expression API</td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="n">—</span></td></tr>
<tr><td>Cost-based optimizer</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td><td><span class="n">—</span></td></tr>
<tr><td>Adaptive re-optimization mid-query</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td><td><span class="n">—</span></td></tr>
<tr><td>Streaming</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td><td><span class="p">~</span></td></tr>
<tr><td>ML / batch inference</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td></tr>
<tr><td>Multimodal (images, audio, video)</td><td><span class="y">✓</span></td><td><span class="n">—</span></td><td><span class="n">—</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td></tr>
<tr><td>Out-of-core spill</td><td><span class="y">✓</span></td><td><span class="y">✓</span></td><td><span class="p">~</span></td><td><span class="y">✓</span></td><td><span class="y">✓</span></td></tr>
</tbody></table>
<p class="bt-matrix-legend"><span class="y">✓</span> built-in &nbsp; <span class="p">~</span> partial or via an add-on &nbsp; <span class="n">—</span> not supported. A capability view, not a benchmark.</p>
```

Speed is measured correctness-first: the benchmark harness refuses to time a query
whose result doesn't match DuckDB, and every operator is differential-tested against
it.

## Benchmarks

Numbers, not adjectives. Every one is **correctness-gated**: the harness runs each query
on every engine, checks they return the identical result, and only then trusts the timing. A
fast wrong answer is a bug, not a win. (The gate earns its keep: on TPC-H q6 both Daft and
Polars compute the wrong revenue, and the harness refuses to time them.)

### AI and multimodal: the widest margin

Ten GPU workload families on 8×T4, real models, every one at least 2× Ray Data:

| workload | vs Ray Data |
|---|---:|
| text embeddings (MiniLM) | **47×** |
| audio feature extraction | **12.5×** |
| LLM batch inference (gpt2) | **11.1×** |
| image generation (diffusion) | **8.6×** |
| training ingest (`iter_torch_batches`) | **3.0×** |
| batch inference (ResNet-50) | **2.05×** |
| zero-config `map_batches(Model, num_gpus=1)` | Ray Data hard-errors |

Image decode → tensor beats **both** competitors: 2.4× Daft and 6.1× Ray Data. Stage-overlapped
streaming keeps the device fed. The CPU decode of the next morsel runs while the GPU forward of
the current one is still in flight, which took a two-stage ResNet-50 pipeline from 942 to
**2,504 img/s** and utilization from ~30% to **81%**.

### Analytics and I/O

Single node, 16 cores, TPC-H at scale 1. Ratios are `batcher / competitor`, so below 1.0 means
Batcher is faster:

| operator | vs DuckDB | vs Polars |
|---|---:|---:|
| filter → count | **0.20×** | **0.07×** |
| global sum | **0.19×** | **0.27×** |
| group-by sum | **0.76×** | **0.44×** |
| sort → top-N | 1.06× | **0.02×** |
| window `rank()` | 1.66× | **0.22×** |

Reading data is where the gap to Ray Data is structural rather than incidental: Parquet read →
sum is **20.8×**, CSV **14.3×**, `count()` roughly **1,400×** (it comes from metadata, not a scan).

### Where Batcher loses

On join-heavy TPC-H, DuckDB is still ahead. It wins 16 of 21 comparable queries, a geometric
mean of **~1.4× in its favor**, and Daft leads on per-batch Python UDFs by ~2×. The cause is understood
(single-node parallelism reaches ~1.7–3.8× on 16 cores; more CPU work per query) and is a
runtime-efficiency effort, not a tuning knob. Correctness is not in question: Batcher matches
DuckDB on all 22 queries.

**[Full benchmarks, methodology, and reproduction commands →](benchmarks/index.md)**

### Why the wins happen

None of this is tuning. Each speedup traces to a design choice you can read about in the
[architecture guide](architecture/index.md).

Batcher runs in-process and native over Arrow, with no task-scheduler or object-store hop
per operation, so the fixed cost that dominates Ray Data's small and medium queries is
never paid at all. That alone is most of the 50–450×.

On the AI side, GPU inference loads a model once per session and overlaps CPU prep with the
GPU forward pass, which is where the 2–47× comes from. And plans re-tune on measured
cardinalities mid-query, so a bad estimate corrects itself rather than stalling or running
out of memory.

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
:caption: Performance

benchmarks/index
```

```{toctree}
:hidden:
:caption: Integrate

integrations/index
```

```{toctree}
:hidden:
:caption: Understand

deep-dives/index
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
