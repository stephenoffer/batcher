# vs DuckDB

This page compares Batcher against DuckDB on single-node analytics: the operator shapes Batcher takes, the query shapes DuckDB takes, and why.

DuckDB is the single-node analytical engine to beat, and on join-heavy SQL it's still ahead.

:::{important}
Every number below was produced by a run that had to pass the correctness gate first: the
harness compares the two engines' results as a sorted row multiset within float tolerance
and refuses to record a time when they disagree. Batcher matches DuckDB's result on all 22
TPC-H queries, so nothing here is a correctness argument. It is a speed argument, in both
directions.
:::

## Scorecard

| Shape | Winner |
|---|---|
| Global aggregate, filtered count | Batcher, by 5× |
| Group-by (one or two keys) | Batcher, 1.3x to 1.5x |
| Window running `sum()` | Batcher, 1.4× |
| Window `rank()`, `lag()` | DuckDB |
| `MEDIAN` / `QUANTILE_CONT` per group | Batcher, 1.1× |
| Join → aggregate | DuckDB, 1.15× |
| TPC-H overall (sf1), DuckDB on its native store | DuckDB, geomean ~1.40× |
| TPC-H overall (sf1), DuckDB reading the same Arrow | Batcher, all 22 queries |
| Delta file skipping (`count(*)` with a predicate) | Batcher, 1.42× |

## Operators

Single node, 16 cores, 30 GB. TPC-H `lineitem` at scale factor 1 (6,001,215 rows) held
in Arrow and shared byte-identically between the engines. The ratio is
`batcher / duckdb`, so **below 1.0 means Batcher is faster**.

| Operator | Batcher | DuckDB | vs DuckDB |
|---|---:|---:|---:|
| global sum | 0.5 ms | 2.7 ms | **0.19×** |
| filter → count | 0.6 ms | 2.7 ms | **0.20×** |
| group-by, two keys | 11.6 ms | 16.9 ms | **0.68×** |
| window running `sum()` | 171 ms | 240 ms | **0.71×** |
| group-by sum, one key | 7.6 ms | 10.0 ms | **0.76×** |
| window `sum()` over partition | 92.7 ms | 99.9 ms | **0.93×** |
| sort → top-N (`LIMIT`) | 14.1 ms | 13.3 ms | 1.06× |
| filter → project | 13.9 ms | 12.9 ms | 1.08× |
| join → aggregate | 98.3 ms | 85.6 ms | 1.15× |
| window `lag()` | 180 ms | 151 ms | 1.19× |
| window `rank()` | 221 ms | 133 ms | 1.66× |

The filtered count is the widest margin, and it is not a micro-optimization: `.count()`
over a filter compiles to a `COUNT(*)` aggregate, so projection pushdown prunes the scan
to the one column the predicate touches and the count fuses into a single `count_if` pass.
Nothing else is read, and no matching row is ever materialized.

`rank()` and `lag()` go the other way. DuckDB's window operator is better on the ordered frame kernels, and that gap is still open.

:::{tip}
Both directions are reachable from your own query. `ds.explain()` shows whether the
predicate reached the scan and which columns survived pruning; `ds.stats()` reports what
each operator actually cost. {doc}`../tutorials/optimizing-a-slow-query` walks the loop.
:::

## Aggregates DuckDB used to win

Two exact aggregates were slower than DuckDB and are now faster. Both were algorithmic,
not tuning. 16 cores.

| Query | Before | After | DuckDB |
|---|---:|---:|---:|
| `MEDIAN(x) GROUP BY flag` (5M rows, 3 groups) | 427 ms | **210 ms** | 232 ms |
| `QUANTILE_CONT(x, 0.9) GROUP BY flag` (5M rows) | 406 ms | **208 ms** | 226 ms |
| `COUNT(DISTINCT id) GROUP BY flag` (2M rows) | 287 ms | **163 ms** | 181 ms |

Median and quantile need the value at one rank, not a sorted list, so the finalize does
quickselect instead of a full sort and runs each group's selection on its own core. The
count-distinct is a Kyber rewrite: a lone `COUNT(DISTINCT x) GROUP BY g` becomes a
distinct over `(g, x)` followed by a count, which parallelizes across the distinct values
instead of the handful of groups.

## Where DuckDB wins: TPC-H

:::{warning}
All 22 queries, scale factor 1, 16 cores, release build, measured 2026-07-18. Against DuckDB's native compressed store, **DuckDB is faster on 15 of 22**, with a geometric mean of about **1.40× in DuckDB's favor**. This is the headline loss on the site, and the operator table above doesn't argue it away.
:::

Against DuckDB reading the same Arrow, the result inverts: Batcher wins all 22, by 1.1× to 6.9×. That's the like-for-like execution comparison. The native-store number is what you get from `duckdb` at a prompt, where DuckDB decompresses its own format as it scans and never pays an Arrow ingest. Both are published, with the per-query breakdown, on {doc}`tpch`.

The pattern is clean. Batcher wins the scan-and-aggregate queries and loses the multi-join ones. The cause is single-node parallelism: it plateaus after roughly 8 cores where DuckDB and Daft use effectively all 16, and Batcher does roughly 2× more CPU work per query. `GROUP BY` alone scales 19.2× while the join alone scales only 5.9×, so the join is the ceiling. Closing that is a runtime-parallelism and kernel-efficiency effort tracked as an open lever in `benchmarks/BENCHMARK_RESULTS.md`. It isn't a knob you can turn.

## Lakehouse reads

A selective predicate on a Delta table should open one data file, not all of them. The
transaction log records each file's column bounds, and reading it at plan time is the whole
game. Single node, 10M rows across 200 Delta files, one `day` per file:

| `count(*) WHERE day = 42` | Time | Files opened |
|---|---:|---:|
| Batcher (before the fix) | 98.8 ms | 200 |
| **Batcher** | **13.4 ms** | **1** |
| DuckDB `delta_scan` | 19.0 ms | baseline |

Batcher was 2.7× slower than DuckDB here and is now 1.42× faster. The bug was in the optimizer, not the connector. The `COUNT(*)`-over-`Filter` fusion deleted the
`Filter` node that source-predicate extraction was looking for, so the most ordinary
lakehouse query in existence pushed nothing down and scanned the whole table. Predicates
are now recovered from the user's plan, where a `Filter` on a `Scan` constrains that scan
whatever the optimizer does above it.

## What DuckDB cannot do

The gap that matters most doesn't appear on this page as a number. DuckDB is single-node and its optimizer is static. It commits to a plan before the first row is read and can't change its mind. Batcher re-optimizes at stage boundaries on measured cardinalities, the same granularity Spark AQE works at but available single-node too, and it carries a sketch-backed cross-query learned-stats loop that DuckDB has no equivalent for. The same mergeable operators then run across a cluster with a bit-identical result.

Two honest caveats. Stage-boundary re-optimization is off for queries under 20M input rows, so most small queries never use it. And it's the same mechanism and granularity as Spark AQE, not something finer. See {doc}`scaling`.

## Reproduce

```bash
python benchmarks/run.py --benchmark operators --tier single --scale 1
python benchmarks/run.py --benchmark tpch      --tier single --scale 1
```

## See also

- {doc}`tpch` for the per-query breakdown.
- {doc}`analytics` for operators, connectors, and the lazy control plane.
- {doc}`vs-polars` and {doc}`vs-daft` for the other two single-node engines.
- {doc}`../deep-dives/aggregation-internals` for the quickselect finalize behind the median and quantile wins.
- {doc}`../deep-dives/join-algorithms` for the operator the TPC-H gap lives in.
- {doc}`../deep-dives/adaptive-reoptimization` for what a static optimizer can't do.
- {doc}`methodology` for hardware, gating, and why cross-hardware comparison is meaningless.
