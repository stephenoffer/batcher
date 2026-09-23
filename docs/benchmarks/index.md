# Benchmarks

This section publishes Batcher's measured performance: analytics against DuckDB, Polars, Daft, Spark and PyArrow, and GPU and multimodal pipelines against Ray Data and Daft. Every figure passed a correctness gate before it was timed, and each one names the machine, the date and the script that reproduces it.

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
The correctness gate, the hardware, and the commands behind every number.
:::
::::

## The single-node board

Batcher is faster than Polars on every suite, faster than DuckDB reading the same Arrow on every suite, and faster than DuckDB on its own compressed storage on five suites of six. That is the most recent full sweep, taken 2026-09-13 on a quiet 48-core (24 physical plus SMT), 92 GiB box with four engines, best of five, one process per case.

Each ratio is the suite's geometric mean of `batcher_ms / engine_ms`, so lower is better and anything below 1.00 is a Batcher win. The last column counts the cases where any engine, DuckDB's native store included, beat Batcher's time:

| Suite | DuckDB, native store | DuckDB, same Arrow | Polars | Cases slower than the fastest engine |
|---|---:|---:|---:|---:|
| Semi-structured JSON | **0.35** | **0.32** | **0.01** | **0 of 5** |
| H2O.ai `join` | **0.63** | **0.58** | **0.51** | **0 of 5** |
| ClickBench | **0.65** | **0.16** | **0.37** | 15 of 43 |
| TPC-H sf1 | **0.72** | **0.25** | **0.54** | 6 of 22 |
| Operator mix | **0.75** | **0.47** | **0.16** | 13 of 46 |
| H2O.ai `groupby` | 1.05 | **0.83** | **0.53** | 6 of 10 |

Across the board that is 40 slower cases of 131, and 19 of the 40 are storage rather than execution: both Arrow-native engines trail Batcher and only DuckDB reading its own compressed store is ahead, by one to four milliseconds. [`benchmarks/results/LOSS_BACKLOG.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/results/LOSS_BACKLOG.md) lists all forty.

The two DuckDB columns answer different questions. *DuckDB on the same Arrow* runs over the identical zero-copy buffers Batcher reads, so it compares two execution engines. *DuckDB on its native store* ingests the data into its compressed, dictionary-encoded format first, so it puts DuckDB's storage engine and execution engine together against Batcher's execution engine alone. That is the harder bar, and Batcher leads it on five suites.

## Larger data and harder planning

Three suites stress what a scale-factor-1 board can't: ten times the rows, the full TPC-DS set, and the many-way joins of the Join Order Benchmark over the real IMDb dataset. All three are measured against DuckDB on its native store:

| Suite | Result | Measured |
|---|---|---|
| TPC-H sf10 (60M-row `lineitem`) | **0.963x**, suite total down from 2,938 ms to 2,323 ms | 2026-08-25, 96 cores, 184 GiB |
| TPC-DS sf1 (99) | **0.98x**, all 99 correctness checks passing | 2026-09-11, 48 cores, 92 GiB |
| Join Order Benchmark (113) | Total **8,131 ms against DuckDB's 8,885 ms**, geomean 1.11x | 2026-08-25, 96 cores, 184 GiB |

The Join Order Benchmark's two statistics point in different directions, and both are worth quoting. Batcher wins the large queries by wide margins, such as q17f at 75 ms and q10c at 46 ms, and still loses many small ones. Its total is lower and its geomean sits above 1.

## Against the rest of the field

The margins over the other analytical engines are wider. Each row below comes from a separate sweep, so compare within a row rather than across rows:

| Engine | Result | Measured |
|---|---|---|
| Daft | TPC-H sf1 **0.21x**, sf10 **0.17x**, ClickBench **0.11x**, operators **0.07x** | 2026-08-28, 92-core box |
| Spark | TPC-H sf1 **20x to 50x** faster than local-mode Spark | 2026-08-15, 96-core box |
| PyArrow | Operator mix **0.03x** | 2026-08-28, 92-core box |

GPU inference is measured end to end on a cluster. Scoring 100,000 images on six single-T4 nodes, Batcher finished in **18.72 s against Ray Data's 44.19 s and Daft's 101.10 s**, which is 2.36x and 5.40x faster, with all three engines agreeing on the checksum (2026-09-06).

## AI, GPU and multimodal

Model work runs on the same engine as the SQL and is measured the same way: real models, identical predictions across engines, and a throughput figure only after they agree. The following highlights ran on an 8xT4 cluster unless the row says otherwise:

| Workload | Batcher |
|---|---:|
| Text embeddings, sentence-transformers MiniLM | **33,611 text/s** |
| Audio features, torchaudio mel plus ResNet-18 | **38,546 clip/s** |
| Fractional-GPU packing, EfficientNet-B0 two per GPU | **6,764 img/s at 89% GPU** |
| ResNet-50 batch inference | **2,504 img/s at 81% GPU** |
| Image decode and resize, one 96-core node | **5,693 img/s, 2.4x Daft** |

Stage overlap does most of the work. The CPU decode of the next morsel runs while the GPU forward of the current one is still in flight, which took a two-stage ResNet-50 pipeline from 942 to 2,504 img/s and its GPU utilization from about 30% to 81%. Session-warm model pools help most on short jobs: on a compute-bound pipeline of 125,000 rows, Batcher finished in 1.2 s against Ray Data's 10.5 s (2026-09-11). {doc}`/benchmarks/results/ai-and-gpu` has all ten families.

## A number means the answer was right

The harness runs each query on every engine, compares the results as a sorted row multiset within float tolerance, and times only the engines that agree. An ordered result is also checked for order. A fast wrong answer gets no ratio.

The gate has caught other engines more than once. On TPC-H q6, Daft and the Polars SQL frontend fold the bound `0.06 + 0.01` in IEEE double to `0.06999999999999999`, drop every `l_discount = 0.07` row, and return 75,207,768 where the official sf1 answer is 123,141,078.23. Batcher returns the official answer. {doc}`methodology` describes the gate in full, including how the suite handles a case where the oracle itself is wrong.

## Requirements and limitations

These boards report steady-state timings of queries run on a single node unless a row says otherwise. The following results are where Batcher does not lead, or where a figure needs its context:

- **H2O.ai `groupby` against DuckDB's native store** reads 1.05x. Its remaining losses are low-cardinality string keys that DuckDB holds dictionary-encoded and Batcher reads as full Arrow strings. On the same Arrow the suite is a win at 0.83x.
- **TPC-H at sf100** (600M rows) is still recorded as a loss to DuckDB on a single node.
- **A shuffle-free GPU pipeline at scale** favors Ray Data past about 2 million rows, reaching 0.76x at 4 million, because Batcher's throughput plateaus near 134,000 rows/s with the devices at 48%.
- **A query seen for the first time** costs Batcher about 2.6x its steady state on TPC-H sf1, against 1.15x for DuckDB, until its plan cache and learned statistics fill. The boards time a query's repeats, not its first run.
- **An unfiltered `SUM`, `AVG` or `COUNT(DISTINCT)` over an in-memory table** is answered from statistics Batcher recorded on an earlier run. The answer is exact, and on the ClickBench and operator-mix cases of that shape the timing measures a lookup rather than a scan.

## Reproduce

Every number here is regenerated by the harness in `benchmarks/`, and [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md) carries the full record of each run. `python benchmarks/run.py --list` prints all 373 registered benchmarks. The following commands reproduce the boards above, one suite per invocation:

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,duckdb,duckdb_arrow,polars --isolate
python benchmarks/run.py --benchmark tpch --scale 10 --engines batcher,duckdb
python benchmarks/run.py --benchmark tpcds --engines batcher,duckdb
python benchmarks/run.py --benchmark job --engines batcher,duckdb
python benchmarks/gpu_backend/vs_ray_daft_gpu_inference.py
python benchmarks/scenarios/image_decode.py
```

## Head to head

Each engine has a page with the full standing and the architectural reason behind it:

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

{doc}`/benchmarks/results/index` arranges the same measurements by workload instead: TPC-H query by query, the engine matrix, multimodal ingest, and scaling out.

## See also

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
