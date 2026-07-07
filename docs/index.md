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
Batch sizes, partition counts, join order — guess wrong and the job stalls or runs out
of memory, often only at scale.
:::
::::

Batcher answers all three at once: the same code from a laptop to a cluster, one
engine across SQL, DataFrames, and ML, and a plan that re-tunes itself as it runs — so
you build the pipeline once and it keeps working as the data grows.

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

Express a transformation as a DataFrame, as SQL, or as composable expressions — and
run it as a batch job or a live stream. Every form builds the same plan and runs on
the same engine, so you mix them freely.

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

Expressions carry typed accessors for every column kind — `.str`, `.dt`, `.list`,
`.struct` — so the column language is the same whether you reach for it from a
DataFrame, from SQL, or in a stream.

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
::::

## It tunes itself

You don't size batches, pick join strategies, or guess partition counts. Batcher
measures the data as it flows and re-plans the rest of the query on real numbers, so a
query that starts on a bad estimate corrects itself instead of stalling — the kind of
mid-flight adaptation a plan-once optimizer can't do. The
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

Numbers, not adjectives — and every one is **correctness-gated**: the harness runs each query
on every engine, checks they return the *identical* result (a sorted row multiset within float
tolerance), and only *then* trusts the timing. A fast wrong answer is a bug, not a win. Setup:
TPC-H `lineitem` (6M rows at scale 1, 60M at scale 10), read once into Arrow and shared
byte-identically across engines; a 9-node / 128-CPU cluster; 8×T4 GPUs for the ML runs. Full
methodology and per-scale tables: [performance guide](user-guide/performance.md).

### Analytical SQL, single-node (vs DuckDB / Polars)

Each cell is `batcher / fastest-competitor` wall time — **below 1.0 means Batcher is faster**
(`0.40×` = 2.5× faster). Batcher wins the operator core, and the margin *holds or grows* from
6M to 60M rows — it scales, it doesn't just start fast:

| operator | sf1 (6M) | sf10 (60M) |
|--------------------------------|:-------:|:--------:|
| group-by sum, one key          | 0.45×   | 0.64×    |
| group-by, two keys             | 0.53×   | 0.89×    |
| filter → count                 | 0.32×   | **0.12×** |
| sort → top-N (`LIMIT`)         | 0.69×   | 0.76×    |
| window `rank()`                | 0.56×   | **0.40×** |
| window running `sum()`         | 0.36×   | **0.32×** |
| window `lag()`                 | 0.54×   | 0.50×    |

At 60M rows `rank() OVER (PARTITION BY …)` is **~2.5× faster than DuckDB** and **~13× faster
than Polars**. Under a tight memory budget where *both* engines spill to disk, Batcher stays
alive and competitive — high-cardinality `DISTINCT` even flips to a **1.4× win** out-of-core.

### Distributed data plane (vs Ray Data)

In-process and native, Batcher pays none of Ray Data's per-operation task-scheduling and
block/pandas-bridge cost (~300–4500 ms fixed, even on the cluster). Same query, same data,
best-of-N wall time:

| operation | batcher | Ray Data | speedup |
|-----------------------------|--------:|---------:|:-------:|
| group-by sum | 14 ms | 1,824 ms | **127×** |
| global sum | 4 ms | 1,804 ms | **440×** |
| filter → count | 7 ms | 310 ms | **46×** |
| sort → top-20 (`LIMIT`) | 15 ms | 4,569 ms | **306×** |

And on Ray Data's *own* streaming `map_batches` home turf — CPU inference, ETL, file I/O — where
it should be strongest, Batcher still leads (warm shared process pool, zero-copy shared-memory
input, GIL-releasing threads for NumPy/torch): `map_batches` transform **2.3×**, row-exploding
`flat_map` **3.5×**, chained multi-stage map **3.2×**, Parquet read **21×**, `iter_torch_batches`
training-data ingest **3.0×**.

### GPU batch inference & ML (8×T4, vs Ray Data)

Stage-overlapped streaming keeps the device fed (a CPU decode stage runs while the GPU forward
of the previous morsel is still in flight), and session-warm pools load the model **once per
session** instead of once per job:

| GPU workload | batcher | Ray Data | vs Ray |
|--------------|--------:|---------:|:------:|
| **LLM batch inference** (gpt2 generate, 2048 prompts) | 814 prompt/s | 73 prompt/s | **11.1×** |
| batch inference (ResNet-50, iterative) | 2576 img/s @ 78% util | 1257 @ 41% | **2.05×** |
| batch embeddings (2048-d vectors) | 2502 img/s @ 80% util | 1267 @ 41% | **1.98×** |
| zero-config `map_batches(Model, num_gpus=1)` | 2451 img/s @ 82% util | *hard-errors* | Ray refuses |

Stage-overlap alone lifted a two-stage decode → ResNet-50 pipeline from **942 → 2504 img/s** and
GPU utilization from **~30% → 81%** — same result, the device just stops idling through the CPU
decode. Batcher reaches **≥80% sustained GPU utilization out of the box** and runs the
zero-`batch_size` call Ray Data rejects outright. On a single maximally-large compute-bound job
both saturate the same GPUs at the same FLOPs (≈ parity) — the honest ceiling.

### Why the wins happen

The speedups are structural, not tuning — each traces to a design choice you can read in the
[architecture guide](architecture/index.md):

- **In-process, native, over Arrow.** No task-scheduler or object-store hop per operation, so the
  fixed cost that dominates Ray Data's small/medium queries (50–450×) simply isn't paid.
- **Composite-key hashing + specialized kernels.** Two-key aggregation and `DISTINCT` hash their
  composite keys directly instead of through a row encoder, so the win *grows* with row count.
- **Warm model pools + stage-overlapped streaming.** GPU inference loads the model once per
  session and overlaps CPU prep with the GPU forward — the 2–11× on real batch-inference shapes.
- **Adaptive re-optimization.** Plans re-tune on measured cardinalities mid-query, so a bad
  estimate corrects itself instead of stalling or OOMing — the thing a plan-once engine can't do.

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
