# Analytics and I/O

This page reports Batcher's results on the classical side of the engine: TPC-H, ClickBench, the operator mix, JSON and scans, measured against DuckDB, Polars and Daft on identical input.

:::{important}
Every timing on this page passed the correctness gate first. The harness runs the query on each engine, compares the results as a sorted row multiset within float tolerance, and refuses to record a ratio when they disagree. On TPC-H q6 that gate withholds a number from two other engines, because both return the wrong revenue.
:::

## The suites

The current board was swept 2026-09-13 on a quiet 48-core (24 physical plus SMT), 92 GiB box, four engines, best of five, one process per case. Each cell is a suite geomean of `batcher_ms / engine_ms`, so **below 1.00 means Batcher is faster**:

| Suite | DuckDB, native store | DuckDB, same Arrow | Polars |
|---|---:|---:|---:|
| Semi-structured JSON (5) | **0.35** | **0.32** | **0.01** |
| H2O.ai `join` (5) | **0.63** | **0.58** | **0.51** |
| ClickBench (43) | **0.65** | **0.16** | **0.37** |
| TPC-H sf1 (22) | **0.72** | **0.25** | **0.54** |
| Operator mix (46) | **0.75** | **0.47** | **0.16** |
| H2O.ai `groupby` (10) | 1.05 | **0.83** | **0.53** |

Against DuckDB reading the same Arrow, the like-for-like execution comparison, Batcher is 4x faster on TPC-H and 6x on ClickBench. Against DuckDB's native compressed store, which pairs DuckDB's storage engine with its execution engine, Batcher leads five suites of six.

Two sweeps add what that board doesn't carry. The scan suite's 27 cases, which read Parquet, CSV and JSON across several file layouts, measured **0.57x DuckDB and 0.24x Polars** (2026-09-11, same 48-core box, pairwise lineups). TPC-DS sf1 measured **0.98x** against DuckDB's native store on the same day. {doc}`/benchmarks/results/tpch` carries TPC-H query by query, and {doc}`/benchmarks/results/engine-matrix` sets every engine side by side.

## Operators

The operator mix times single data-plane kernels over TPC-H `lineitem` at sf1 (6,001,215 rows) held in Arrow and shared byte-identically across engines. The following table is the complete 11-case run of 2026-07-13 on a 16-core, 30 GB node, before the mix grew to 46 cases. The ratios are `batcher / engine`, so **below 1.0 means Batcher is faster**:

| Operator | Batcher | DuckDB | Polars | vs DuckDB | vs Polars |
|---|---:|---:|---:|---:|---:|
| Global sum | 0.5 ms | 2.7 ms | 1.8 ms | **0.19x** | **0.27x** |
| Filter then count | 0.6 ms | 2.7 ms | 8.4 ms | **0.20x** | **0.07x** |
| Group-by, two keys | 11.6 ms | 16.9 ms | 28.8 ms | **0.68x** | **0.40x** |
| Window running `sum()` | 171 ms | 240 ms | 786 ms | **0.71x** | **0.22x** |
| Group-by sum, one key | 7.6 ms | 10.0 ms | 17.1 ms | **0.76x** | **0.44x** |
| Window `sum()` over partition | 92.7 ms | 99.9 ms | 73.8 ms | **0.93x** | 1.26x |
| Sort then top-N | 14.1 ms | 13.3 ms | 600.7 ms | 1.06x | **0.02x** |
| Filter then project | 13.9 ms | 12.9 ms | 9.2 ms | 1.08x | 1.51x |
| Join then aggregate | 98.3 ms | 85.6 ms | 86.9 ms | 1.15x | 1.13x |
| Window {py:func}`lag() <batcher.lag>` | 179.7 ms | 151.4 ms | 3,216.9 ms | 1.19x | **0.06x** |
| Window `rank()` | 220.7 ms | 132.7 ms | 988.8 ms | 1.66x | **0.22x** |

A filtered count is the widest DuckDB margin, and it comes from the plan rather than the kernel. `.count()` over a filter compiles to a `COUNT(*)` aggregate, projection pushdown prunes the scan to the one column the predicate reads, and the count fuses into a single {py:func}`count_if <batcher.count_if>` pass. Against Polars, top-N is 50x because a fused top-N heap keeps only the running best rows and never sorts the relation.

The 46-case mix added string functions, set operations, scalar expressions and six more join shapes. Some of the new cases are large wins, such as a semi-join at 0.28x DuckDB and an anti-join at 0.27x. Others are the losses listed in `benchmarks/results/LOSS_BACKLOG.md`, which is where work on them is tracked.

## Against Daft

Daft is a fast multi-core Rust engine. On a five-engine board taken 2026-08-28 on a 92-core box, best of five, Batcher's geomean against it was 0.21x on TPC-H sf1, 0.17x at sf10, 0.11x on ClickBench, 0.07x on the operators and 0.04x on JSON. The Daft geomeans cover only the queries Daft answers correctly. It returns wrong results on TPC-H q6 and q15, and it can't plan q21 or q22. {doc}`/benchmarks/comparisons/vs-daft` has the full comparison, including multimodal and distributed work.

## Requirements and limitations

These results are single-node and steady state. The following caveats apply:

- **H2O.ai `groupby`** is the one suite where DuckDB's native store is ahead, at 1.05x, on low-cardinality string keys it keeps dictionary-encoded.
- **Several ClickBench cases and the global-sum operator** are an unfiltered `SUM`, `AVG` or `COUNT(DISTINCT)` over an in-memory table. Batcher answers them exactly from statistics it recorded on an earlier run, so their timing is a lookup rather than a scan.
- **The operator table** above is from a smaller, older machine than the suite board. Compare ratios within it, not its absolute times against another table.

## Reproduce

The following commands rerun each board on this page:

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,duckdb,duckdb_arrow,polars --isolate
python benchmarks/run.py --benchmark clickbench --engines batcher,duckdb,duckdb_arrow,polars --isolate
python benchmarks/run.py --benchmark operators --tier single
python benchmarks/run.py --benchmark scan --engines batcher,duckdb
python benchmarks/run.py --benchmark tpch --engines batcher,daft
```

## See also

- {doc}`/benchmarks/results/tpch`: the per-query detail behind the TPC-H rows.
- {doc}`/benchmarks/comparisons/vs-duckdb`, {doc}`/benchmarks/comparisons/vs-polars` and {doc}`/benchmarks/comparisons/vs-daft`: the same numbers arranged one engine at a time.
- {doc}`/benchmarks/results/ai-and-gpu`: the other half of the measurement.
- {doc}`/architecture/deep-dives/query/expression-evaluation`: how expressions evaluate over a morsel.
- {doc}`/architecture/deep-dives/operators/aggregation-internals`: the radix combine behind the group-by numbers.
- {doc}`/user-guide/operate/tuning/performance`: making your own query faster.
- {doc}`/benchmarks/methodology`: hardware, gating, and the reproduce commands.
