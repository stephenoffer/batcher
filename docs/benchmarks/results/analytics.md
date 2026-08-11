# Analytics and I/O

The classical side of the engine: relational operators, TPC-H, ClickBench, and the connectors, measured against DuckDB and Polars on identical input.

:::{important}
Every timing on this page passed the correctness gate first. The harness runs the query on each engine, compares the results as a sorted row multiset within float tolerance, and refuses to record a time when they disagree. It is not a formality: on TPC-H q6 two other engines return the wrong revenue, so neither gets a number there.
:::

:::{note}
Not every table below was measured on the same machine, because the workload families were not. Figures hold *within* a table; a number from one table set against a number from another says nothing. {doc}`/benchmarks/methodology` lists the hardware per family.
:::

## Operators

Single node, 16 cores, TPC-H `lineitem` at scale factor 1 (6M rows) held in Arrow and shared byte-identically across engines. Ratios are `batcher / competitor`, so **below 1.0 means Batcher is faster**. All correctness checks passed.

Against DuckDB:

| Operator | Batcher | DuckDB | Ratio |
|---|---:|---:|---:|
| global sum | 0.5 ms | 2.7 ms | **0.19x** |
| filter → count | 0.6 ms | 2.7 ms | **0.20x** |
| group-by, two keys | 11.6 ms | 16.9 ms | **0.68x** |
| window running `sum()` | 171 ms | 240 ms | **0.71x** |
| group-by sum, one key | 7.6 ms | 10.0 ms | **0.76x** |
| window `sum()` over partition | 92.7 ms | 99.9 ms | **0.93x** |

Against Polars:

| Operator | Batcher | Polars | Ratio |
|---|---:|---:|---:|
| sort → top-N (`LIMIT`) | 14.1 ms | 601 ms | **0.02x** |
| window {py:func}`lag() <batcher.lag>` | 180 ms | 3,217 ms | **0.06x** |
| filter → count | 0.6 ms | 8.4 ms | **0.07x** |
| window running `sum()` | 171 ms | 786 ms | **0.22x** |
| window `rank()` | 221 ms | 989 ms | **0.22x** |
| global sum | 0.5 ms | 1.8 ms | **0.27x** |
| group-by, two keys | 11.6 ms | 28.8 ms | **0.40x** |
| group-by sum, one key | 7.6 ms | 17.1 ms | **0.44x** |

Batcher takes the scan-and-aggregate core against both engines. A filtered count is 5x DuckDB and 14x Polars, because it fuses to a {py:func}`count_if <batcher.count_if>` over the one column the predicate touches and never materializes the rest. Against Polars the sort and window margins are the widest on the page: top-N is 50x and `lag()` is 17x, because a fused top-N heap beats a full sort.

:::{note}
Two of those rows have moved since. A later sweep in `benchmarks/BENCHMARK_RESULTS.md` puts the mix at **9 of 11 against DuckDB and 9 of 11 against Polars**, and a fix to the hash-join probe (which was allocating and zeroing a null mask per morsel for a key that is never null) took `join → aggregate` to **0.90x to 0.97x**. The table above is kept as the last full, published sweep rather than being edited row by row.
:::

## TPC-H and ClickBench

All 22 TPC-H queries at scale factor 1 on 16 cores, plus the 43-query ClickBench suite. **Batcher matches DuckDB's result on every query in both.**

Against DuckDB reading the same zero-copy Arrow, which is the like-for-like execution comparison that Batcher's Arrow-only contract makes fair:

| Suite | Result |
|---|---|
| TPC-H, 22 comparable queries | **won 22 of 22** |
| ClickBench, 43 queries | **won 43 of 43**, 43 of 43 correct |
| Semi-structured JSON, 5 queries | **won 5 of 5**: 3.6x to 12.5x DuckDB, 11x to 100x Polars |

Against DuckDB on its own native compressed store, Batcher leads on 13 of 22 queries as of the 2026-07-26 sweep, taking the scan-and-aggregate-dominated ones outright (q15 0.46x, q12 0.74x, q11 0.80x, q1 and q9 0.88x). That comparison is not like-for-like: DuckDB decompresses its own format as it scans and never pays an Arrow ingest, which is exactly what an engine gives up to keep Arrow as its only columnar contract. {doc}`/benchmarks/results/tpch` has the per-query detail.

Two notes on the other engines' surfaces:

- Batcher answers all 22 queries, correlated subqueries included.
- Polars cannot parse most of the TPC-H suite through its SQL frontend (multi-table `FROM`, `EXISTS`, non-equi joins), and on q6 it returns the wrong revenue. Its column in the harness is mostly `ERR`, which is a statement about its SQL surface rather than its speed.

## Connectors

20M rows across 64 files, single node, 16 cores. Writes produce a directory of shards.

| Operation | Batcher | Effective rate |
|---|---:|---:|
| {py:func}`read_parquet <batcher.read_parquet>` + sum | **72 ms** | ~278 M rows/s |
| {py:func}`read_csv <batcher.read_csv>` + sum | **98 ms** | ~204 M rows/s |
| {py:func}`read_json <batcher.read_json>` + sum | **302 ms** | ~66 M rows/s |
| `write_parquet` (dir) | **317 ms** | ~63 M rows/s |
| `write_csv` (dir) | **326 ms** | ~61 M rows/s |
| `write_json` (dir) | **1,016 ms** | ~20 M rows/s |

Reads are fast because files decode concurrently in-process, with no per-file task scheduling and no object-store round trip, and because Parquet, CSV, and JSON decode all release the GIL.

The JSON writer used to be the outlier in this table. A {py:meth}`to_pylist() <batcher.Dataset.to_pylist>` plus a per-row `json.dumps` took over 65 seconds for a single file. Encoding across processes and streaming the result took that to 1.0 s.

## The lazy control plane

A metadata question should not execute a query. Batcher answers these from Parquet metadata or from the plan itself, even after a chain of transformations:

| Operation | Batcher |
|---|---:|
| `count()` | **0.05 ms** |
| `head(10)` | **~0 ms** |
| `filter(pred).count()` | **47 ms** |
| {py:meth}`limit(100).collect() <batcher.Dataset.limit>` | **71 ms** |

`filter(...).count()` took 2,187 ms until `.count()` was compiled to a `COUNT(*)` aggregate. Projection pushdown then prunes the scan to just the predicate's column and fuses it into `count_if`, a 46x improvement from one planner change.

## Against Daft

Daft is a fast, mature multi-core Rust engine, roughly DuckDB-class. The split at TPC-H scale factor 1:

| Shape | Result |
|---|---|
| Top-N and sort-limit | **Batcher, 8x to 10x**, on the fused top-N heap |
| Global aggregation, group-by, single-stage expression ETL | Parity |

Batcher is faster on 11 of the 18 queries Daft answers correctly, by up to 7x. Daft computes TPC-H q6 incorrectly, folding `0.06 + 0.01` in IEEE double to `0.06999999999999999` and dropping every `l_discount = 0.07` row, and cannot parse the `SUBSTRING(x FROM a FOR b)` in q22.

## See also

- {doc}`/benchmarks/results/tpch`: the per-query detail behind both comparisons.
- {doc}`/benchmarks/comparisons/vs-duckdb`, {doc}`/benchmarks/comparisons/vs-polars`, {doc}`/benchmarks/comparisons/vs-daft`: the same numbers arranged one engine at a time.
- {doc}`/benchmarks/results/ai-and-gpu`: the other half of the measurement.
- {doc}`/architecture/deep-dives/query/expression-evaluation` and {doc}`/architecture/deep-dives/query/jit-compilation`: where the filtered count's 5x comes from.
- {doc}`/architecture/deep-dives/operators/aggregation-internals`: the radix combine behind the group-by numbers.
- {doc}`/user-guide/operate/tuning/performance`: making *your* query faster.
- {doc}`/benchmarks/methodology`: hardware, gating, and the reproduce commands.
