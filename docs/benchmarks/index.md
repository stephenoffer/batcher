# Benchmarks

This page summarizes Batcher's measured results across analytics, I/O, and AI workloads, and links to the per-engine and per-workload detail.

Numbers, not adjectives. Every figure here comes from a run that was correctness-gated first. The harness executes the query on every engine, checks they return the identical result as a sorted row multiset within float tolerance, and only then records a time. A fast wrong answer is a bug, not a win. A benchmark that disagrees with the oracle reports `FAILED` and produces no number at all.

:::{important}
That gate is not decoration. On TPC-H q6 it caught two other engines returning 75,207,768.19 where the official TPC-H answer is 123,141,078.23. They fold the predicate bound `0.06 + 0.01` in IEEE double, getting `0.06999999999999999`, and so drop every `l_discount = 0.07` row. Batcher returns the official answer exactly, and the harness declines to time a wrong result. Read every table on this site knowing that a missing number means a wrong answer, not a slow one.
:::

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.1em` AI & GPU workloads
:link: /benchmarks/results/ai-and-gpu
:link-type: doc
Ten workload families on 8xT4, real models, correctness-gated. 33,611 text/s embedding, 2,504 img/s at 81% GPU.
:::

:::{grid-item-card} {octicon}`database;1.1em` Analytics & I/O
:link: /benchmarks/results/analytics
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

Batcher leads the classical analytics suites against DuckDB reading the same Arrow — 22 of 22 TPC-H at scale factor 1, 43 of 43 ClickBench, 5 of 5 JSON — and, since 2026-08-15, against DuckDB's own native compressed store as well on TPC-H, ClickBench, JSON and the operator mix, with TPC-DS at parity. That second bar is the harder one and the one worth arguing about: it puts DuckDB's storage engine *and* its execution engine against Batcher's execution engine alone. Where Batcher still loses is stated in the table below rather than omitted from it.

Model and multimodal work is one more workload family on that same engine rather than a separate system, and it is measured the same way: real models on 8xT4, with the GPU held above 80% utilization on every family sampled.

Coverage is **346 benchmarks across ten suites**, including the full 99-query TPC-DS set, all 113 Join Order Benchmark queries against the real IMDb dataset, and the H2O.ai db-benchmark group-by and join sweeps. {doc}`methodology` lists them.

Suite geometric means at scale factor 1 on 96 cores / 184 GiB, measured 2026-08-15. `duckdb`
is DuckDB on its native compressed store (the harder bar); `duckdb_arrow` is DuckDB over the
same zero-copy Arrow Batcher runs on (the like-for-like one). Lower is better and **below
1.0x means Batcher is faster**:

| Suite | vs `duckdb` | vs `duckdb_arrow` | vs Polars |
|---|---:|---:|---:|
| **Semi-structured JSON** (5) | **0.23x**, 5 of 5 | **0.04x** | **0.01x** |
| **ClickBench** (43) | **0.62x**, 30 of 43 | **0.07x**, 43 of 43 | **0.33x** |
| **Operator mix** (19) | **0.66x**, 10 of 19 | **0.36x** | **0.12x** |
| **TPC-H sf1** (22) | **0.77x**, 17 of 22 | **0.27x**, 22 of 22 | **0.43x** |
| **H2O.ai `join`** (5) | **0.89x**, 3 of 5 | **0.24x** | **0.63x** |
| **TPC-DS sf1** (99) | ~1.00x, 39-42 of 99 | — | — |
| **H2O.ai `groupby`** (10) | 1.23x, 4 of 10 | **0.11x**, 10 of 10 | **0.49x** |
| **Join Order Benchmark** (113) | 1.49x, 31 of 109 | — | — |
| **TPC-H sf10** (22) | 1.27x, 6 of 22 | — | — |

| Other workloads | Measured |
|---|---|
| **Sort → top-N, window functions** | **5x to 50x** Polars |
| **Image decode → tensor** | 5,693 img/s, **2.4x** Daft |
| **TPC-H sf10 q6, cluster against cluster** | **2.4x** Daft, and Daft's answer is wrong |
| **Text embeddings** (MiniLM, 8xT4) | **33,611 text/s** |
| **Batch inference** (ResNet-50, 8xT4) | **2,504 img/s at 81% GPU utilization** |
| **Training ingest** ({py:meth}`iter_torch_batches <batcher.api.dataset.ml.DatasetML.iter_torch_batches>`) | **1.06 M rows/s**, zero-copy DLPack |
| **Parquet read → aggregate** | 20M rows across 64 files in **72 ms** |
| **`count()` after a transform chain** | **0.05 ms**, answered from metadata |

:::{warning}
That last row is honest about *what it measures* and the same caveat applies inside the suite
table: five of the 43 ClickBench queries and two of the 19 operator cases are **answered from
recorded column statistics rather than executed**. An unfiltered `SUM`, `AVG` or
`COUNT(DISTINCT)` over an immutable in-memory relation is served from a statistic the first
run computed. The answers are exact — they match DuckDB — but the timing is a memo lookup, not
a scan. Excluding those cases, ClickBench is **0.77x over 38 queries** and the operator mix
**0.77x over 17**. Use those figures when the claim is about execution speed.
:::

:::{note}
Those rows were not all measured on the same machine, because the workload families were not. The suite table above and the ingest work ran on a 96-core / 184 GiB node, some older per-operator figures on a 16-core node, and the model work on an 8xT4 cluster. A figure is meaningful within its row. A number lifted out of one row and set against a number from another is not. {doc}`methodology` lists the hardware per family.
:::

## Where the wins come from

**Execution over the same bytes.** DuckDB reading the identical zero-copy Arrow input is the like-for-like execution comparison, and Batcher wins 22 of 22 TPC-H queries on it at sf1 and 21 of 22 at sf10. A filtered count is 5x DuckDB because it fuses to a {py:func}`count_if <batcher.count_if>` over the one column the predicate touches and never materializes the rest.

**A control plane that answers what it can without scanning.** `count()` after a transform chain returns in 0.05 ms from Parquet footer statistics and plan-level reasoning. Seven ClickBench queries return in about 0.2 ms for the same reason. Those are excluded from the ranges above, so the headline reflects execution rather than planning.

**Stage overlap on the GPU path.** Stage-overlapped streaming runs the CPU decode of morsel *k+1* while the GPU forward of morsel *k* is still in flight, which lifted a two-stage ResNet-50 pipeline from 942 to 2,504 img/s and GPU utilization from about 30% to 81%. Session-warm pools then load a model once per session rather than once per job, which is worth about 2x on iterative inference and far more when the model is large.

**Native, in-process I/O.** Reading 20M rows across 64 Parquet files and summing a column takes 72 ms, CSV 98 ms, JSON 302 ms. Files decode concurrently in-process, and Parquet, CSV, and JSON decode all release the GIL.

## Reading the comparisons

Every table on this site is a like-for-like execution comparison: the same Arrow buffers, the same queries, and a correctness gate before any timing is recorded.

One comparison on these pages is deliberately not like-for-like, and it is published anyway because it is what you get from `duckdb` at a prompt. Measured against DuckDB's own compressed format rather than shared Arrow, DuckDB decompresses its own layout as it scans and never pays an Arrow ingest, so it measures a storage engine plus an execution engine against an execution engine alone. Batcher trades that storage format away on purpose, because the same operators that read that Arrow also run distributed, stream, and carry tensors. Give both engines the same Arrow buffers and the same queries are 2x to 5x Batcher wins.

Correctness is not part of that trade: Batcher matches DuckDB on all 22 TPC-H queries.

## Scaling out

The same mergeable operators run distributed, so scale-out is a scheduling decision rather than a rewrite, and the distributed result is identical to the single-node one: the same row multiset, the same column names, the same column types. A floating-point reduction is identical up to reassociation, because `combine` is associative in exact arithmetic and IEEE addition is not, so the partition count moves the last bits. That also means distribution is cheap to decline. At TPC-H scale factor 1, where the network shuffle costs more than it saves, the distributed path stays within about 7% of the single node rather than falling off a cliff:

| Path | Time |
|---|---:|
| Batcher, single node | 86 ms |
| Batcher, distributed (4 workers) | 92 ms |

At scale the picture inverts. On an 8-node, 128-CPU cluster reading TPC-H parquet from S3, Batcher takes the join by 1.7x to 2.2x over Daft's Ray runner and answers a metadata count 162x to 250x faster. {doc}`/benchmarks/results/scaling` has the full grid.

## Reproduce it

Every number here is regenerated by the harness in `benchmarks/`, and the complete engineering record lives in `benchmarks/BENCHMARK_RESULTS.md`. See {doc}`methodology` for the hardware each family was measured on and the exact commands.

```bash
python benchmarks/run.py --benchmark tpch --tier single     # vs DuckDB / Polars
python benchmarks/run.py --benchmark tpcds                  # all 99 queries
python benchmarks/run.py --benchmark job                    # all 113 queries, real IMDb
python benchmarks/run.py --benchmark operators --tier multi # the data-plane lineup
python benchmarks/scenarios/image_decode.py                 # multimodal ingest
```

## Head to head

{doc}`One page per engine </benchmarks/comparisons/index>`, each measured on the same Arrow input.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` DuckDB
:link: /benchmarks/comparisons/vs-duckdb
:link-type: doc
Batcher takes the operators and the shared-Arrow suite, all 22 TPC-H queries at sf1.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Polars
:link: /benchmarks/comparisons/vs-polars
:link-type: doc
50x on top-N, 1.26x on the TPC-H suite.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Daft
:link: /benchmarks/comparisons/vs-daft
:link-type: doc
2.4x on image decode, 1.7x to 2.2x on the distributed join.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Spark
:link: /benchmarks/comparisons/vs-spark
:link-type: doc
Architecture and design comparison.
:::
::::

And {doc}`one page per workload family </benchmarks/results/index>`.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` TPC-H
:link: /benchmarks/results/tpch
:link-type: doc
All 22 queries, both comparisons.
:::

:::{grid-item-card} {octicon}`image;1.1em` Multimodal ingest
:link: /benchmarks/results/multimodal-ingest
:link-type: doc
Images, point clouds, audio, video.
:::

:::{grid-item-card} {octicon}`cloud;1.1em` Scaling out
:link: /benchmarks/results/scaling
:link-type: doc
The distributed runs, and the full scale-out grid.
:::
::::

## See also

- {doc}`/user-guide/operate/tuning/performance` for making your own query faster, with the levers these numbers come from.
- {doc}`/architecture/deep-dives/operators/morsel-parallelism` and {doc}`/architecture/deep-dives/query/jit-compilation` for the two mechanisms behind most of the operator wins.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra` for why the distributed result matches the single-node one, and for the one place floating-point reassociation shows through.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization` for the stage-boundary re-optimization and the cross-query learned-stats loop, which no number on this page captures.
- {doc}`/tutorials/foundations/optimizing-a-slow-query` for the diagnosis loop.

```{toctree}
:hidden:

results/index
comparisons/index
methodology
```
