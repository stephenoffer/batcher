# Analytics and I/O

The classical side of the engine: relational operators, TPC-H, ClickBench, and the connectors, measured against DuckDB and Polars on identical input.

:::{important}
Every timing on this page passed the correctness gate first. The harness runs the query on each engine, compares the results as a sorted row multiset within float tolerance, and refuses to record a time when they disagree. It is not a formality: on TPC-H q6 two other engines return the wrong revenue, so neither gets a number there.
:::

:::{note}
Not every table below was measured on the same machine, because the workload families were not. Figures hold *within* a table; a number from one table set against a number from another says nothing. {doc}`methodology` lists the hardware per family.
:::

## Operators

Single node, 16 cores, TPC-H `lineitem` at scale factor 1 (6M rows) held in Arrow and shared byte-identically across engines. Ratios are `batcher / competitor`, so **below 1.0 means Batcher is faster**. All correctness checks passed.

| Operator | Batcher | DuckDB | Polars | vs DuckDB | vs Polars |
|---|---:|---:|---:|---:|---:|
| global sum | 0.5 ms | 2.7 | 1.8 | **0.19x** | **0.27x** |
| filter → count | 0.6 ms | 2.7 | 8.4 | **0.20x** | **0.07x** |
| group-by sum, one key | 7.6 ms | 10.0 | 17.1 | **0.76x** | **0.44x** |
| group-by, two keys | 11.6 ms | 16.9 | 28.8 | **0.68x** | **0.40x** |
| window running `sum()` | 171 ms | 240 | 786 | **0.71x** | **0.22x** |
| sort → top-N (`LIMIT`) | 14.1 ms | 13.3 | 601 | 1.06x | **0.02x** |
| window `sum()` over partition | 92.7 ms | 99.9 | 73.8 | **0.93x** | 1.26x |
| filter → project | 13.9 ms | 12.9 | 9.2 | 1.08x | 1.51x |
| join → aggregate | 98.3 ms | 85.6 | 86.9 | 1.15x | 1.13x |
| window `lag()` | 180 ms | 151 | 3,217 | 1.19x | **0.06x** |
| window `rank()` | 221 ms | 133 | 989 | 1.66x | **0.22x** |

Batcher takes the scan-and-aggregate core: 6 of these 11 kernels against DuckDB and 8 against Polars. A filtered count is 5x DuckDB and 14x Polars, because it fuses to a `count_if` over the one column the predicate touches and never materializes the rest. Against Polars the sort and window gaps are the widest on the page: top-N is 50x and `lag()` is 17x, because a fused top-N heap beats a full sort.

DuckDB leads `rank()`, `lag()`, and the join-heavy shapes in this run.

:::{note}
Two of those rows have moved since. A later sweep in `benchmarks/BENCHMARK_RESULTS.md` puts the mix at **9 of 11 against DuckDB and 9 of 11 against Polars**, and a fix to the hash-join probe (which was allocating and zeroing a null mask per morsel for a key that is never null) took `join → aggregate` from 1.25x behind DuckDB to **0.90x to 0.97x**, a parity-to-win. The table above is kept as the last full, published sweep rather than being edited row by row.
:::

## TPC-H and ClickBench

All 22 TPC-H queries at scale factor 1 on 16 cores, plus the 43-query ClickBench suite. **Batcher matches DuckDB's result on every query in both.**

Against DuckDB reading the same zero-copy Arrow, which is the like-for-like execution comparison that Batcher's Arrow-only contract makes fair:

| Suite | Result |
|---|---|
| TPC-H, 22 comparable queries | **won 22 of 22** |
| ClickBench, 43 queries | **won 42 of 43**, 43 of 43 correct |
| Semi-structured JSON, 5 queries | **won 5 of 5**: 3.6x to 12.5x DuckDB, 11x to 100x Polars |

Against DuckDB on its own native compressed store, DuckDB is faster on 15 of 22 TPC-H queries, geometric mean about 1.40x. That comparison is not like-for-like: DuckDB decompresses its own format as it scans and never pays an Arrow ingest, which is exactly what an engine gives up to keep Arrow as its only columnar contract. Batcher wins the scan-and-aggregate-dominated queries there too (q15 0.46x, q12 0.74x, q11 0.80x, q1 and q9 0.88x) and trails on the join- and subquery-heavy ones. {doc}`tpch` has both columns per query.

Two notes on the other engines' surfaces, in both directions:

- Batcher answers all 22 queries, correlated subqueries included.
- Polars cannot parse most of the TPC-H suite through its SQL frontend (multi-table `FROM`, `EXISTS`, non-equi joins), and on q6 it returns the wrong revenue. Its column in the harness is mostly `ERR`, which is a statement about its SQL surface rather than its speed.

### Where the join gap comes from

It is not a tuning knob. Single-node parallelism plateaus after about 8 of 16 cores, and Batcher performs about 2x more CPU work per query on these shapes. Closing it is a runtime-parallelism and kernel-efficiency effort, and it is the top open lever in `benchmarks/BENCHMARK_RESULTS.md`.

## Connectors

20M rows across 64 files, single node, 16 cores. Writes produce a directory of shards.

| Operation | Batcher | Effective rate |
|---|---:|---:|
| `read_parquet` + sum | **72 ms** | ~278 M rows/s |
| `read_csv` + sum | **98 ms** | ~204 M rows/s |
| `read_json` + sum | **302 ms** | ~66 M rows/s |
| `write_parquet` (dir) | **317 ms** | ~63 M rows/s |
| `write_csv` (dir) | **326 ms** | ~61 M rows/s |
| `write_json` (dir) | **1,016 ms** | ~20 M rows/s |

Reads are fast because files decode concurrently in-process, with no per-file task scheduling and no object-store round trip, and because Parquet, CSV, and JSON decode all release the GIL.

The JSON writer used to be the outlier in this table. A `to_pylist()` plus a per-row `json.dumps` took over 65 seconds for a single file. Encoding across processes and streaming the result took that to 1.0 s.

## The lazy control plane

A metadata question should not execute a query. Batcher answers these from Parquet metadata or from the plan itself, even after a chain of transformations:

| Operation | Batcher |
|---|---:|
| `count()` | **0.05 ms** |
| `head(10)` | **~0 ms** |
| `filter(pred).count()` | **47 ms** |
| `limit(100).collect()` | **71 ms** |

`filter(...).count()` took 2,187 ms until `.count()` was compiled to a `COUNT(*)` aggregate. Projection pushdown then prunes the scan to just the predicate's column and fuses it into `count_if`, a 46x improvement from one planner change.

## Against Daft

Daft is a fast, mature multi-core Rust engine, roughly DuckDB-class. The split at TPC-H scale factor 1:

| Shape | Result |
|---|---|
| Top-N and sort-limit | **Batcher, 8x to 10x**, on the fused top-N heap |
| Global aggregation, group-by, single-stage expression ETL | Parity |
| Join-heavy queries | Daft, up to 2x |
| Per-batch Python UDFs | Daft, about 2x |

Batcher is faster on 11 of the 18 queries Daft answers correctly, and the spread is wider in Batcher's favor (up to 7x) than against it (up to 2x). Daft computes TPC-H q6 incorrectly, folding `0.06 + 0.01` in IEEE double to `0.06999999999999999` and dropping every `l_discount = 0.07` row, and cannot parse the `SUBSTRING(x FROM a FOR b)` in q22. The gap to Daft is speed on two shapes, never correctness.

## See also

- {doc}`tpch`: the per-query detail behind both comparisons.
- {doc}`vs-duckdb`, {doc}`vs-polars`, {doc}`vs-daft`: the same numbers arranged one engine at a time.
- {doc}`ai-and-gpu`: the other half of the measurement.
- {doc}`../deep-dives/expression-evaluation` and {doc}`../deep-dives/jit-compilation`: where the filtered count's 5x comes from.
- {doc}`../deep-dives/aggregation-internals`: the radix combine behind the group-by numbers.
- {doc}`../user-guide/performance`: making *your* query faster.
- {doc}`methodology`: hardware, gating, and the reproduce commands.
