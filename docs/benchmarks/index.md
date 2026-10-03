# Benchmarks

This section publishes Batcher's measured performance: analytics against DuckDB, Polars, Daft, Spark and PyArrow, plus GPU and multimodal pipelines against Ray Data and Daft. No figure was timed until its answer was right.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` Analytics and I/O
:link: /benchmarks/results/analytics
:link-type: doc
Operators, TPC-H, ClickBench and the connectors, on identical Arrow input.
:::

:::{grid-item-card} {octicon}`zap;1.1em` AI and GPU workloads
:link: /benchmarks/results/ai-and-gpu
:link-type: doc
Ten model families on 8xT4 with real models: 33,611 text/s embedding, 2,504 img/s inference.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` Methodology
:link: methodology
:link-type: doc
How every number was gated, and the commands that reproduce it.
:::
::::

## The single-node board

Lower is better. Each ratio is the suite's geometric mean of `batcher_ms / engine_ms`, so anything below 1.00 is a Batcher win. The sweep ran 2026-09-13 on a 48-core (24 physical plus SMT), 92 GiB box, best of five, one process per case.

| Suite | DuckDB, native store | DuckDB, same Arrow | Polars |
|---|---:|---:|---:|
| Semi-structured JSON | 0.35 | 0.32 | 0.01 |
| H2O.ai `join` | 0.63 | 0.58 | 0.51 |
| ClickBench | 0.65 | 0.16 | 0.37 |
| TPC-H sf1 | 0.72 | 0.25 | 0.54 |
| Operator mix | 0.75 | 0.47 | 0.16 |
| H2O.ai `groupby` | 1.05 | 0.83 | 0.53 |

The two DuckDB columns ask different questions. *DuckDB on the same Arrow* reads the identical zero-copy buffers Batcher reads, so it compares two execution engines. *DuckDB on its native store* first ingests the data into its compressed format. That pits DuckDB's storage engine and execution engine together against Batcher's execution engine alone, the harder bar.

## Larger data, harder planning, other engines

Three more suites stress what a scale-factor-1 board can't. All three ran against DuckDB on its native store:

| Suite | Result | Measured |
|---|---|---|
| TPC-H sf10 (60M-row `lineitem`) | 0.963x, suite total down from 2,938 ms to 2,323 ms | 2026-08-25, 96 cores, 184 GiB |
| TPC-DS sf1 (99) | 0.98x, all 99 correctness checks passing | 2026-09-11, 48 cores, 92 GiB |
| Join Order Benchmark (113) | Total 8,131 ms against DuckDB's 8,885 ms | 2026-08-25, 96 cores, 184 GiB |

Against the other engines the margins are wider. Each row below comes from its own sweep, so compare within a row:

| Engine | Result | Measured |
|---|---|---|
| Daft | TPC-H sf1 0.21x, sf10 0.17x, ClickBench 0.11x, operators 0.07x | 2026-08-28, 92-core box |
| Spark | TPC-H sf1 20x to 50x faster than local-mode Spark | 2026-08-15, 96-core box |
| PyArrow | Operator mix 0.03x | 2026-08-28, 92-core box |
| Ray Data and Daft | 100,000 images on six single-T4 nodes: 18.72 s against 44.19 s and 101.10 s | 2026-09-06 |

## AI, GPU and multimodal

Model work runs on the same engine as the SQL. These runs used real models, and every engine had to return identical predictions. The cluster was 8xT4 unless the row says otherwise:

| Workload | Batcher |
|---|---:|
| Text embeddings, sentence-transformers MiniLM | 33,611 text/s |
| Audio features, torchaudio mel plus ResNet-18 | 38,546 clip/s |
| Fractional-GPU packing, EfficientNet-B0 two per GPU | 6,764 img/s at 89% GPU |
| ResNet-50 batch inference | 2,504 img/s at 81% GPU |
| Image decode and resize, one 96-core node | 5,693 img/s, 2.4x Daft |

Overlap does most of it. While the GPU runs the forward pass on one morsel, the CPU decodes the next, and that alone took a two-stage ResNet-50 pipeline from 942 to 2,504 img/s. {doc}`/benchmarks/results/ai-and-gpu` has all ten families.

## Every number passed a correctness gate

A fast wrong answer gets no ratio. Each query runs on every engine first, and the results are compared as a sorted row multiset within float tolerance, with ordered results also checked for order. Only the engines that agree are timed. {doc}`methodology` has the details.

:::{dropdown} Reproduce these boards
`python benchmarks/run.py --list` prints every registered benchmark, and [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md) records each run.

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,duckdb,duckdb_arrow,polars --isolate
python benchmarks/run.py --benchmark tpch --scale 10 --engines batcher,duckdb
python benchmarks/run.py --benchmark tpcds --engines batcher,duckdb
python benchmarks/run.py --benchmark job --engines batcher,duckdb
python benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py
python benchmarks/scenarios/image_decode.py
```
:::

:::{dropdown} Scope of these numbers
- Timings are steady state: a query's repeats, not its first run, which also fills Batcher's plan cache and learned statistics.
- Results are single node unless a row names a cluster.
- An unfiltered `SUM`, `AVG` or `COUNT(DISTINCT)` over an in-memory table is answered exactly from statistics recorded on an earlier run.
:::

## Head to head

Pick the engine you run today. Its page has the full standing and the architectural reason behind it:

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`database;1.1em` DuckDB
:link: /benchmarks/comparisons/vs-duckdb
:link-type: doc
Faster on every suite over the same Arrow, and on five of six against DuckDB's own storage.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Polars
:link: /benchmarks/comparisons/vs-polars
:link-type: doc
Faster on every suite on the current board, and 50x on top-N.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Daft
:link: /benchmarks/comparisons/vs-daft
:link-type: doc
2.4x on image decode, 1.7x to 2.2x on the distributed join.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Spark
:link: /benchmarks/comparisons/vs-spark
:link-type: doc
20x to 50x on TPC-H sf1, and the architecture behind it.
:::

:::{grid-item-card} {octicon}`table;1.1em` PyArrow
:link: /benchmarks/comparisons/vs-pyarrow
:link-type: doc
The same Arrow kernels underneath, with a scheduler and a planner on top.
:::
::::

To read by workload instead, {doc}`/benchmarks/results/index` has TPC-H query by query, the engine matrix, multimodal ingest and scaling out.

## See also

These pages explain where the numbers come from.

- {doc}`/user-guide/operate/tuning/performance` for making your own query faster, with the levers these numbers come from.
- {doc}`/architecture/deep-dives/operators/morsel-parallelism` and {doc}`/architecture/deep-dives/query/jit-compilation` for the two mechanisms behind most of the operator wins.
- {doc}`/architecture/deep-dives/operators/mergeable-algebra` for why a distributed result matches the single-node one.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization` for stage-boundary re-optimization and the cross-query learned-stats loop.
- {doc}`/getting-started/tutorials/foundations/optimizing-a-slow-query` for the diagnosis loop.

```{toctree}
:hidden:

results/index
comparisons/index
methodology
```
