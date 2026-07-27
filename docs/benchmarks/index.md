# Benchmarks

This page summarizes Batcher's measured results across analytics, I/O, and AI workloads, and links to the per-engine and per-workload detail.

Numbers, not adjectives. Every figure here comes from a run that was correctness-gated first. The harness executes the query on every engine, checks they return the identical result as a sorted row multiset within float tolerance, and only then records a time. A fast wrong answer is a bug, not a win. A benchmark that disagrees with the oracle reports `FAILED` and produces no number at all.

:::{important}
That gate is not decoration. On TPC-H q6 it caught two other engines returning 75,207,768.19 where the official TPC-H answer is 123,141,078.23. They fold the predicate bound `0.06 + 0.01` in IEEE double, getting `0.06999999999999999`, and so drop every `l_discount = 0.07` row. Batcher returns the official answer exactly, and the harness declines to time a wrong result. Read every table on this site knowing that a missing number means a wrong answer, not a slow one.
:::

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` AI & GPU workloads
:link: ai-and-gpu
:link-type: doc
Ten workload families on 8xT4, real models, correctness-gated. 33,611 text/s embedding, 2,504 img/s at 81% GPU.
:::

:::{grid-item-card} {octicon}`database;1.1em` Analytics & I/O
:link: analytics
:link-type: doc
Operators, TPC-H, ClickBench, and the connectors, against DuckDB and Polars on the same Arrow.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Methodology
:link: methodology
:link-type: doc
Hardware, correctness gating, and the commands to reproduce every number.
:::
::::

## The short version

Batcher sweeps the classical analytics suites against DuckDB reading the same Arrow, winning 22 of 22 TPC-H, 42 of 43 ClickBench, and 5 of 5 JSON. The same engine carries the modern half of the workload: AI, multimodal, and last-mile training ingest, where it runs real models on 8xT4 with the GPU held above 80% utilization on every family measured.

| Workload | Measured |
|---|---|
| **TPC-H**, all 22 queries | vs DuckDB on the same Arrow: **won 22 of 22** |
| **ClickBench**, 43 queries | vs DuckDB on the same Arrow: **won 42 of 43**, 43/43 correct |
| **Semi-structured JSON**, 5 queries | **3.6x to 12.5x** DuckDB, **11x to 100x** Polars |
| **Operator mix**, 11 kernels | vs DuckDB on the same Arrow: **won 10 of 11** |
| **Sort → top-N, window functions** | **5x to 50x** Polars |
| **Image decode → tensor** | 5,693 img/s, **2.4x** Daft |
| **TPC-H sf10 q6, cluster against cluster** | **2.4x** Daft, and Daft's answer is wrong |
| **Text embeddings** (MiniLM, 8xT4) | **33,611 text/s** |
| **Batch inference** (ResNet-50, 8xT4) | **2,504 img/s at 81% GPU utilization** |
| **Training ingest** (`iter_torch_batches`) | **1.06 M rows/s**, zero-copy DLPack |
| **Parquet read → aggregate** | 20M rows across 64 files in **72 ms** |
| **`count()` after a transform chain** | **0.05 ms**, answered from metadata |

:::{note}
Those rows were not all measured on the same machine, because the workload families were not. The DuckDB and Polars comparisons ran on a 16-core node, the ingest work on a 96-core node, and the model work on an 8xT4 cluster. A figure is meaningful within its row. A number lifted out of one row and set against a number from another is not. {doc}`methodology` lists the hardware per family.
:::

## Where the wins come from

**Execution over the same bytes.** DuckDB reading the identical zero-copy Arrow input is the like-for-like execution comparison, and Batcher wins every TPC-H query on it. A filtered count is 5x DuckDB because it fuses to a `count_if` over the one column the predicate touches and never materializes the rest.

**A control plane that answers what it can without scanning.** `count()` after a transform chain returns in 0.05 ms from Parquet footer statistics and plan-level reasoning. Seven ClickBench queries return in about 0.2 ms for the same reason. Those are excluded from the ranges above, so the headline reflects execution rather than planning.

**A device that stays fed.** Stage-overlapped streaming runs the CPU decode of morsel *k+1* while the GPU forward of morsel *k* is still in flight, which lifted a two-stage ResNet-50 pipeline from 942 to 2,504 img/s and GPU utilization from about 30% to 81%. Session-warm pools then load a model once per session rather than once per job, which is worth about 2x on iterative inference and far more when the model is large.

**Native, in-process I/O.** Reading 20M rows across 64 Parquet files and summing a column takes 72 ms, CSV 98 ms, JSON 302 ms. Files decode concurrently in-process, and Parquet, CSV, and JSON decode all release the GIL.

## The measured trade-offs

Publishing only the wins would make this page marketing rather than measurement. Two comparisons run the other way, both understood and both tracked.

**DuckDB's native store on join-heavy SQL.** At scale factor 1 on 16 cores, against DuckDB's own compressed format rather than shared Arrow, DuckDB is faster on 15 of 22 queries with a geometric mean of about 1.40x. DuckDB decompresses its own format as it scans and never pays an Arrow ingest, which is an advantage Batcher's Arrow-only contract trades away deliberately: the same operators that read that Arrow also run distributed, stream, and carry tensors. On the like-for-like Arrow comparison those same queries are 2x to 5x wins.

**Daft on join-heavy queries and tight per-batch UDFs.** Daft leads on the multi-join TPC-H shapes and by roughly 2x on a per-batch Python UDF. The cause is single-node parallelism that plateaus after about 8 of 16 cores, and it is a runtime-parallelism and kernel-efficiency effort rather than a tuning knob. It is the top open lever in `benchmarks/BENCHMARK_RESULTS.md`.

Correctness is not part of either trade: Batcher matches DuckDB on all 22 TPC-H queries.

## Scaling out

The same mergeable operators run distributed, so scale-out is a scheduling decision rather than a rewrite, and the distributed result is bit-identical to the single-node one. That also means distribution is cheap to decline. At TPC-H scale factor 1, where the network shuffle costs more than it saves, the distributed path stays within about 7% of the single node rather than falling off a cliff:

| Path | Time |
|---|---:|
| Batcher, single node | 86 ms |
| Batcher, distributed (4 workers) | 92 ms |

At scale the picture inverts. On an 8-node, 128-CPU cluster reading TPC-H parquet from S3, Batcher takes the join by 1.7x to 2.2x over Daft's Ray runner and answers a metadata count 162x to 250x faster. {doc}`scaling` has the full grid, including the shape where Daft leads.

## Reproduce it

Every number here is regenerated by the harness in `benchmarks/`. The full run log lives in `benchmarks/BENCHMARK_RESULTS.md`, failures and regressions included. See {doc}`methodology` for the hardware each family was measured on and the exact commands.

```bash
python benchmarks/run.py --benchmark tpch --tier single    # vs DuckDB / Polars
python benchmarks/run.py --benchmark operators --tier multi # the data-plane lineup
python benchmarks/scenarios/image_decode.py                 # multimodal ingest
```

## Head to head

One page per engine, each with the trade-offs stated as plainly as the wins.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` DuckDB
:link: vs-duckdb
:link-type: doc
Batcher takes the operators and the shared-Arrow suite. DuckDB takes join-heavy SQL on its native store.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Polars
:link: vs-polars
:link-type: doc
50x on top-N. Polars takes high-cardinality hashing by 2x.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Daft
:link: vs-daft
:link-type: doc
2.4x on image decode. Daft takes the multi-join queries.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Spark
:link: vs-spark
:link-type: doc
Architecture only. No head-to-head numbers are published yet.
:::
::::

And one page per workload family.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` TPC-H
:link: tpch
:link-type: doc
All 22 queries, both comparisons.
:::

:::{grid-item-card} {octicon}`image;1.1em` Multimodal ingest
:link: multimodal-ingest
:link-type: doc
Images, point clouds, audio, video.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` Scaling out
:link: scaling
:link-type: doc
The distributed runs, and a measured negative result.
:::
::::

## See also

- {doc}`../user-guide/performance` for making your own query faster, with the levers these numbers come from.
- {doc}`../deep-dives/morsel-parallelism` and {doc}`../deep-dives/jit-compilation` for the two mechanisms behind most of the operator wins.
- {doc}`../deep-dives/mergeable-algebra` for why the distributed result is bit-identical to the single-node one.
- {doc}`../deep-dives/adaptive-reoptimization` for the stage-boundary re-optimization and the cross-query learned-stats loop, which no number on this page captures.
- {doc}`../tutorials/optimizing-a-slow-query` for the diagnosis loop.

```{toctree}
:hidden:

ai-and-gpu
analytics
vs-duckdb
vs-polars
vs-daft
vs-spark
tpch
multimodal-ingest
scaling
methodology
```
