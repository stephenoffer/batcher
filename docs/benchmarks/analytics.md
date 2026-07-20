# Analytics and I/O

The classical side of the engine: relational operators, TPC-H, and the connectors. This is
also where the honest gaps are, so they are stated here rather than left out.

:::{important}
Every timing on this page passed the correctness gate first. The harness runs the query on
each engine, compares the results as a sorted row multiset within float tolerance, and
refuses to record a time when they disagree. It is not a formality: on TPC-H q6, both Daft
and Polars return the wrong revenue, so neither gets a number there.
:::

:::{note}
Not every table below was measured on the same machine, because the workload families were
not. Ratios hold *within* a table; a number from one table set against a number from another
says nothing. {doc}`methodology` lists the hardware per family.
:::

## Operators

Single node, 16 cores, TPC-H `lineitem` at scale factor 1 (6M rows) held in Arrow and
shared byte-identically across engines. Ratios are `batcher / competitor`, so **below 1.0
means Batcher is faster**. All correctness checks passed.

| Operator | Batcher | DuckDB | Polars | vs DuckDB | vs Polars |
|---|---:|---:|---:|---:|---:|
| global sum | 0.5 ms | 2.7 | 1.8 | **0.19×** | **0.27×** |
| filter → count | 0.6 ms | 2.7 | 8.4 | **0.20×** | **0.07×** |
| group-by sum, one key | 7.6 ms | 10.0 | 17.1 | **0.76×** | **0.44×** |
| group-by, two keys | 11.6 ms | 16.9 | 28.8 | **0.68×** | **0.40×** |
| window running `sum()` | 171 ms | 240 | 786 | **0.71×** | **0.22×** |
| sort → top-N (`LIMIT`) | 14.1 ms | 13.3 | 601 | 1.06× | **0.02×** |
| window `sum()` over partition | 92.7 ms | 99.9 | 73.8 | **0.93×** | 1.26× |
| filter → project | 13.9 ms | 12.9 | 9.2 | 1.08× | 1.51× |
| join → aggregate | 98.3 ms | 85.6 | 86.9 | 1.15× | 1.13× |
| window `lag()` | 180 ms | 151 | 3,217 | 1.19× | **0.06×** |
| window `rank()` | 221 ms | 133 | 989 | 1.66× | **0.22×** |

Batcher takes the scan-and-aggregate core. A filtered count is 5× DuckDB and 14× Polars,
because it fuses to a `count_if` over the one column the predicate touches and never
materializes the rest. Against Polars the sort and window gaps are enormous (top-N is 50×,
`lag()` is 17×): a fused top-N heap beats a full sort, and Polars' window path is simply
slow.

DuckDB still wins `rank()`, `lag()`, and join-heavy shapes on this machine.

## TPC-H

All 22 queries, scale factor 1, 16 cores. **Batcher matches DuckDB's result on every
query.**

:::{warning}
On speed, DuckDB is ahead. It **wins 16 of the 21 comparable queries**, with a geometric
mean of about **1.36× in DuckDB's favor**. Batcher wins the scan-heavy queries: q1 (0.80×),
q6 (0.82×), q12 (0.86×), q14 (0.71×), and q16 (0.99×). It trails on the multi-join ones: q5
(2.99×), q8 (2.30×), q17 (2.46×), q7 (2.15×).
:::

Two caveats, in both directions:

- Batcher does not yet support correlated subqueries, so q21 raises rather than returning
  a wrong answer.
- Polars cannot parse most of the TPC-H suite through its SQL frontend (multi-table `FROM`,
  `EXISTS`, non-equi joins), and on q6 it returns the wrong revenue. Its column in the
  harness is mostly `ERR`, which is a statement about its SQL surface rather than its speed.

### Why the join gap exists

It isn't a tuning knob. Single-node parallelism plateaus after about 8 of 16
cores where Daft uses effectively all of them, and Batcher performs about 2× more CPU work
per query. Closing it is a runtime-parallelism and kernel-efficiency effort, and it is the
top open lever in `benchmarks/BENCHMARK_RESULTS.md`.

## Connectors

20M rows across 64 files, single node. Both engines write a *directory* of shards, which is
Ray Data's default output, so the comparison is like-for-like.

| Operation | Batcher | Ray Data | vs Ray |
|---|---:|---:|---:|
| `read_parquet` + sum | 72 ms | 1,502 | **20.8×** |
| `read_csv` + sum | 98 ms | 1,394 | **14.3×** |
| `read_json` + sum | 302 ms | 1,588 | **5.3×** |
| `write_parquet` (dir) | 317 ms | 1,396 | **4.4×** |
| `write_csv` (dir) | 326 ms | 1,430 | **4.4×** |
| `write_json` (dir) | 1,016 ms | 1,709 | **1.68×** |

Reads win because Batcher decodes files concurrently in-process (Parquet, CSV, and JSON
decode all release the GIL), with none of Ray Data's per-file task scheduling or
object-store round trip.

The JSON writer used to be the embarrassment here. A `to_pylist()` plus a per-row
`json.dumps` took over 65 seconds for a single file, which was 7.7× *behind* Ray Data. It now
encodes across processes and streams the result, at 1.0 s.

## The lazy control plane

Metadata questions should not execute a query. Batcher answers them from Parquet metadata
or the plan itself, even after a transform chain:

| Operation | Batcher | Ray Data | vs Ray |
|---|---:|---:|---:|
| `count()` | 0.05 ms | 76 ms | **~1,400×** |
| `head(10)` | ~0 ms | 170 ms | **>100,000×** |
| `filter(pred).count()` | 47 ms | 695 ms | **15×** |
| `limit(100).collect()` | 71 ms | 173 ms | **2.4×** |

`filter(...).count()` was a loss until `.count()` was compiled to a `COUNT(*)` aggregate:
projection pushdown then prunes the scan to just the predicate's column and fuses it into
`count_if`. That took it from 2,187 ms to 47 ms, moving from 3.2x behind Ray Data to 15x ahead.

## vs Daft

Daft is a fast, mature multi-core Rust engine, roughly DuckDB-class. The honest split at
TPC-H scale factor 1:

| Shape | Result |
|---|---|
| Top-N and sort-limit | Batcher, 8x to 10x, on the fused top-N heap |
| Global aggregation, group-by, single-stage expression ETL | Parity |
| Join-heavy queries | **Daft, up to 2x** |
| Per-batch Python UDFs | **Daft, ~2×** |

Daft computes TPC-H q6 incorrectly (it folds `0.06 + 0.01` in IEEE double to `0.06999999999999999`, dropping every `l_discount = 0.07` row and returning 75.2M
instead of 123.1M) and cannot parse the `SUBSTRING(x FROM a FOR b)` in q22. So the gap to
Daft is purely speed, never correctness.

## See also

- {doc}`tpch`: the per-query detail behind the geometric mean.
- {doc}`vs-duckdb`, {doc}`vs-polars`, {doc}`vs-daft`: the same
  numbers arranged one engine at a time.
- {doc}`ai-and-gpu`: the other half of the measurement.
- {doc}`../deep-dives/expression-evaluation` and
  {doc}`../deep-dives/jit-compilation`: where the filtered count's 5× comes
  from.
- {doc}`../deep-dives/aggregation-internals`: the radix combine behind
  the group-by numbers.
- {doc}`../user-guide/performance`: making *your* query faster.
- {doc}`methodology`: hardware, gating, and the reproduce commands.
