# vs DuckDB

This page compares Batcher with DuckDB on single-node analytics: the suites, the operators, lakehouse reads, and the architecture behind the results.

DuckDB is the single-node analytical engine to beat. On identical Arrow input Batcher is faster on every suite measured. Against DuckDB's own compressed store, the harder bar, Batcher leads TPC-H at sf1 and sf10, TPC-DS, ClickBench, JSON, the H2O.ai join task and the operator mix.

:::{important}
Every number below passed the correctness gate first. The harness compares the two engines' results as a sorted row multiset within float tolerance, checks the order of every `ORDER BY` result, and refuses to record a ratio when they disagree. Batcher matches DuckDB on all 22 TPC-H queries.
:::

## Two bars, both published

DuckDB can be measured two ways, and the difference between them is not a detail.

`duckdb_arrow`
: DuckDB executing over the same zero-copy Arrow Batcher runs on. This compares two execution engines on identical bytes.

`duckdb`
: DuckDB over its own native store, ingested before the clock starts: compressed, dictionary-encoded and zone-mapped. This measures DuckDB's storage engine plus its execution engine against Batcher's execution engine alone. It is DuckDB at its best, and the harder bar.

Batcher keeps Arrow as its only columnar format on purpose, because the same operators that read it also run distributed, stream and carry tensors. Both bars are reported, because quoting only the first would be choosing the flattering one.

## The suites

The current board was swept 2026-09-13 on a quiet 48-core (24 physical plus SMT), 92 GiB box, best of five, one process per case. Each cell is a suite geomean of `batcher_ms / duckdb_ms`, so **below 1.00 means Batcher is faster**:

| Suite | vs `duckdb` (native store) | vs `duckdb_arrow` (same Arrow) |
|---|---:|---:|
| Semi-structured JSON (5) | **0.35** | **0.32** |
| H2O.ai `join` (5) | **0.63** | **0.58** |
| ClickBench (43) | **0.65** | **0.16** |
| TPC-H sf1 (22) | **0.72** | **0.25** |
| Operator mix (46) | **0.75** | **0.47** |
| H2O.ai `groupby` (10) | 1.05 | **0.83** |

On identical input Batcher's execution engine is 4x DuckDB's on TPC-H and 6x on ClickBench. Against the native store the margins narrow, because that comparison adds DuckDB's storage engine to its side.

Three larger suites were measured against the native store in separate sweeps:

| Suite | vs `duckdb` | Measured |
|---|---|---|
| TPC-H sf10 (60M-row `lineitem`) | **0.963x**, suite total 2,323 ms | 2026-08-25, 96 cores, 184 GiB |
| TPC-DS sf1 (99) | **0.98x** | 2026-09-11, 48 cores, 92 GiB |
| Join Order Benchmark (113) | Total **8,131 ms against 8,885 ms**, geomean 1.11x | 2026-08-25, 96 cores, 184 GiB |

The Join Order Benchmark's two statistics point in different directions. Batcher wins the large many-way joins, such as q17f in 75 ms and q10c in 46 ms, and still loses many of the small ones, so its total is lower while its geomean sits above 1.

`duckdb_arrow` has no TPC-DS figure because DuckDB over registered Arrow is killed on q64 at 132 GB resident on a scale-factor-1 dataset. Batcher returns that query in 3.2 ms, and DuckDB on its native store in 58.4 ms.

## Operators

The operator mix times single kernels over TPC-H `lineitem` at sf1 (6,001,215 rows) held in Arrow and shared byte-identically. The following table is the complete run of 2026-07-13 on a 16-core, 30 GB node, from before the mix grew to 46 cases. The ratio is `batcher / duckdb`, so **below 1.0 means Batcher is faster**:

| Operator | Batcher | DuckDB | vs DuckDB |
|---|---:|---:|---:|
| Global sum | 0.5 ms | 2.7 ms | **0.19x** |
| Filter then count | 0.6 ms | 2.7 ms | **0.20x** |
| Group-by, two keys | 11.6 ms | 16.9 ms | **0.68x** |
| Window running `sum()` | 171 ms | 240 ms | **0.71x** |
| Group-by sum, one key | 7.6 ms | 10.0 ms | **0.76x** |
| Window `sum()` over partition | 92.7 ms | 99.9 ms | **0.93x** |
| Sort then top-N | 14.1 ms | 13.3 ms | 1.06x |
| Filter then project | 13.9 ms | 12.9 ms | 1.08x |
| Join then aggregate | 98.3 ms | 85.6 ms | 1.15x |
| Window `lag()` | 179.7 ms | 151.4 ms | 1.19x |
| Window `rank()` | 220.7 ms | 132.7 ms | 1.66x |

The filtered count is the widest margin, and it comes from the plan. `.count()` over a filter compiles to a `COUNT(*)` aggregate, so projection pushdown prunes the scan to the one column the predicate touches and the count fuses into a single {py:func}`count_if <batcher.count_if>` pass. Nothing else is read, and no matching row is materialized.

The 46-case mix added string functions, set operations, scalar expressions and six join shapes. It found large wins that nothing had measured, a semi-join at 0.28x and an anti-join at 0.27x, and a set of string and temporal kernels where DuckDB does less work per row. [`benchmarks/results/LOSS_BACKLOG.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/results/LOSS_BACKLOG.md) tracks every case where Batcher is behind.

:::{tip}
The same margins are reachable from your own query. {py:meth}`ds.explain() <batcher.Dataset.explain>` shows whether the predicate reached the scan and which columns survived pruning, and {py:meth}`ds.stats() <batcher.Dataset.stats>` reports what each operator cost. {doc}`/getting-started/tutorials/foundations/optimizing-a-slow-query` walks the loop.
:::

## Lakehouse reads

A selective predicate on a Delta table should open one data file, not all of them. The transaction log records each file's column bounds, and Batcher reads it at plan time. The benchmark counts rows matching one day across 10M rows in 200 Delta data files, one `day` per file, on a single node, measured 2026-07-13 with [`benchmarks/scenarios/lakehouse_bench.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/scenarios/lakehouse_bench.py):

| `count(*) WHERE day = 42` | Time | Files opened |
|---|---:|---:|
| **Batcher** | **7.4 ms** | **1** |
| Batcher, before file skipping | 98.8 ms | 200 |
| DuckDB `delta_scan` | 21.8 ms | not reported |

Batcher is 2.9x faster than `delta_scan` here, and it opens the one file the predicate selects. An unfiltered `count(*)` takes 0.85 ms and opens no data file at all, because the log answers it. The record doesn't name the machine for this run, so read the ratio rather than the absolute times.

## What a static optimizer can't do

The largest difference doesn't show up as a number. DuckDB is single-node and its optimizer is static: it commits to a plan before the first row is read.

Batcher re-optimizes at stage boundaries on measured cardinalities. That is the same granularity Spark AQE works at, available on a single node too, and it adds a sketch-backed learned-statistics loop that carries across queries, so a recurring query plans better each time it runs. The same mergeable operators then run across a cluster and return the same rows, with floating-point reductions agreeing to the last bits because the partition count sets the summation order.

The within-query loop isn't always on. It engages on a query with a join once the input clears 5M rows, or about 320 MB, for each pipeline breaker it would cut at, so the simplest joined shape qualifies at about 10M rows and a query with no join doesn't qualify single-node at any size.

## Requirements and limitations

The following results are where DuckDB leads or where a figure needs its context:

- **H2O.ai `groupby` against the native store** reads 1.05x. The remaining losses are low-cardinality string keys that DuckDB holds dictionary-encoded and Batcher reads as full Arrow strings, a storage difference: on the same Arrow the suite is 0.83x. Composite string keys no longer pay for being strings, since a two-string-key group-by over 10M rows runs as fast as the same query on two `int64` keys.
- **Strings in general.** Batcher has no `StringView` and decodes dictionaries at the scan, and 19 of the 40 cases on the current board where it trails are that kind of storage gap.
- **TPC-H at sf100** (600M rows) is still recorded as a loss on a single node.
- **A first-seen query** costs Batcher about 2.6x its steady state on TPC-H sf1 against DuckDB's 1.15x. The boards time repeats.
- **An unfiltered `SUM`, `AVG` or `COUNT(DISTINCT)` over an in-memory table** is answered exactly from statistics recorded on an earlier run, so on the ClickBench and operator cases of that shape the timing is a lookup rather than a scan.

## Reproduce

The following commands rerun each result:

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,duckdb,duckdb_arrow --isolate
python benchmarks/run.py --benchmark clickbench --engines batcher,duckdb,duckdb_arrow --isolate
python benchmarks/run.py --benchmark tpch --scale 10 --engines batcher,duckdb
python benchmarks/run.py --benchmark job --engines batcher,duckdb
python benchmarks/run.py --benchmark operators --tier single
python benchmarks/scenarios/lakehouse_bench.py
```

## See also

- {doc}`/benchmarks/results/tpch` for the per-query breakdown.
- {doc}`/benchmarks/results/analytics` for the other suites.
- {doc}`/benchmarks/comparisons/vs-polars` and {doc}`/benchmarks/comparisons/vs-daft` for the other single-node engines.
- {doc}`/architecture/deep-dives/operators/join-algorithms` for the join strategies behind the TPC-H results.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization` for stage-boundary re-optimization and its size floor.
- {doc}`/benchmarks/methodology` for hardware, gating, and why cross-hardware comparison means nothing.
