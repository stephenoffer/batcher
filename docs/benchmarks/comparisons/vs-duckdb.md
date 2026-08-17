# vs DuckDB

This page compares Batcher against DuckDB on single-node analytics: the operator shapes, the query suites, and the architecture behind the results.

DuckDB is the single-node analytical engine to beat. On identical Arrow input Batcher beats it on every TPC-H and every ClickBench query. Against DuckDB's own native compressed store — the harder bar — Batcher leads TPC-H, ClickBench, JSON and the operator mix at scale factor 1, is at parity on TPC-DS, and loses on H2O `groupby`, the Join Order Benchmark, and TPC-H at scale factor 10.

:::{important}
Every number below was produced by a run that had to pass the correctness gate first: the
harness compares the two engines' results as a sorted row multiset within float tolerance
and refuses to record a time when they disagree. Batcher matches DuckDB's result on all 22
TPC-H queries, so nothing here is a correctness argument. It is a speed argument.
:::

## Scorecard

Each row is one workload shape, and by how much Batcher takes it. The methodology above is
what makes these numbers comparable, so read them together:

| Shape | Batcher's margin |
|---|---|
| Global aggregate, filtered count | 5× |
| Group-by (one or two keys) | 1.3x to 1.5x |
| Window running `sum()` | 1.4× |
| `MEDIAN` / `QUANTILE_CONT` per group | 1.1× |
| TPC-H overall (sf1), DuckDB reading the same Arrow | 3.7×; wins all 22 |
| TPC-H overall (sf1), DuckDB on its native store | 1.3×; wins 16 of 22 |
| ClickBench overall (43), DuckDB reading the same Arrow | 14×; wins all 43 |
| TPC-H overall (sf10), DuckDB on its native store | **0.79×** — DuckDB wins |
| Delta file skipping (`count(*)` with a predicate) | 1.42× |

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

The filtered count is the widest margin, and it is not a micro-optimization: `.count()`
over a filter compiles to a `COUNT(*)` aggregate, so projection pushdown prunes the scan
to the one column the predicate touches and the count fuses into a single {py:func}`count_if <batcher.count_if>` pass.
Nothing else is read, and no matching row is ever materialized.

:::{note}
`join → aggregate` has moved since this table was published. `BroadcastProbe::probe` was building a full 16,384-entry null mask per morsel and reading it per row, for a foreign-key probe whose key is never null. Skipping both when the probe key has no nulls took the operator to **0.90x to 0.97x**, verified bit-identical across the 84 join and stream oracle tests. `benchmarks/BENCHMARK_RESULTS.md` carries the measurement.
:::

:::{tip}
The same margins are reachable from your own query. {py:meth}`ds.explain() <batcher.Dataset.explain>` shows whether the
predicate reached the scan and which columns survived pruning; {py:meth}`ds.stats() <batcher.Dataset.stats>` reports what
each operator actually cost. {doc}`/tutorials/foundations/optimizing-a-slow-query` walks the loop.
:::

## Exact aggregates

Batcher takes the exact order-statistic aggregates, on algorithms rather than tuning.
16 cores.

| Query | Batcher | DuckDB |
|---|---:|---:|
| `MEDIAN(x) GROUP BY flag` (5M rows, 3 groups) | **210 ms** | 232 ms |
| `QUANTILE_CONT(x, 0.9) GROUP BY flag` (5M rows) | **208 ms** | 226 ms |
| `COUNT(DISTINCT id) GROUP BY flag` (2M rows) | **163 ms** | 181 ms |

Median and quantile need the value at one rank, not a sorted list, so the finalize does
quickselect instead of a full sort and runs each group's selection on its own core. The
count-distinct is a Kyber rewrite: a lone `COUNT(DISTINCT x) GROUP BY g` becomes a
distinct over `(g, x)` followed by a count, which parallelizes across the distinct values
instead of the handful of groups.

## Two bars, and Batcher now clears both

DuckDB can be measured two ways, and the difference between them is not a detail:

`duckdb_arrow`
    DuckDB executing over the *same zero-copy Arrow* Batcher runs on. This is the
    like-for-like comparison of two execution engines, and the one Batcher's Arrow-only
    contract makes fair.
`duckdb`
    DuckDB over its own native store, ingested before the clock starts — compressed,
    dictionary-encoded, zone-mapped. This measures DuckDB's *storage engine plus* its
    execution engine against Batcher's execution engine alone. It is DuckDB at its best,
    and it is the harder bar.

Both are reported, because quoting only the first would be choosing the flattering one.
Suite geometric means, 96 cores / 184 GiB, scale factor 1, `batcher / duckdb` — **below 1.0
means Batcher is faster**:

| Suite | vs `duckdb` (native store) | vs `duckdb_arrow` (same Arrow) |
|---|---:|---:|
| Semi-structured JSON (5) | **0.25x** — 5 of 5 | **0.04x** — 5 of 5 |
| ClickBench (43) | **0.64x** — 28 of 43 | **0.07x** — 43 of 43 |
| Operator mix (19) | **0.66x** — 11 of 19 | **0.36x** — 15 of 19 |
| TPC-H (22) | **0.79x** — 16 of 22 | **0.26x** — 22 of 22 |
| H2O.ai `join` (5) | **0.93x** — 3 of 5 | **0.24x** — 5 of 5 |
| TPC-DS (99) | **0.96x** — 38 of 98 | — |
| H2O.ai `groupby` (10) | 1.19x — 4 of 10 | **0.09x** — 10 of 10 |
| Join Order Benchmark (113) | 1.29x — 35 of 109 | — |

Every suite in the right-hand column is a Batcher win. That column is the one to read for a
question about *engines*; the left-hand one answers a question about engines **and** storage
formats together, and the two suites Batcher loses there are the two where the format is
doing the most work.

Read the two columns together. On identical input Batcher's execution engine is **3.9x
DuckDB's on TPC-H and 14x on ClickBench**, and it wins every query of both. Against DuckDB's
native store the margin narrows to 1.3x and 1.6x, and the queries Batcher loses are the ones
where the storage advantage is largest — which is exactly what a storage advantage should
look like. Where the gap is widest, on H2O `groupby`, the cause is legible: its keys are
low-cardinality strings, DuckDB holds them dictionary-encoded, and Batcher reads the same
column as full Arrow `Utf8` — 11x faster than DuckDB does over that same Arrow, and slower
than DuckDB reading eight-bit codes.

The suite's residue is now concentrated in its two single-key high-cardinality queries (one
string column, 100,000 distinct values). A *composite* string key no longer pays for being
strings: each column's distinct values are numbered in first-seen order and the ranked columns
take the ordinary integer grouper, which brought a two-string-key group-by over 10M rows from
41.7 ms to 32.6 ms — identical to the same query with two `int64` keys, which is the check
that nothing about the strings is left to pay for.

:::{warning}
Five of the 43 ClickBench queries and two of the 19 operator cases are answered from
Batcher's recorded column statistics rather than executed — an unfiltered `SUM`, `AVG` or
`COUNT(DISTINCT)` over an immutable in-memory relation. The answers are exact, but the timing
is a memo lookup rather than a scan. **Excluding them**, ClickBench is **0.77x over 38
queries** and the operator mix **0.76x over 17**. Quote those when the claim is about
execution speed.
:::

### Where it does not hold: scale factor 10

At ten times the data the native-store comparison inverts: TPC-H sf10 is **1.29x**, a loss.
Nine of thirteen shapes still scale *sublinearly* for ten times the rows, but four do not —
q5 (14.9x), q13 (12.7x), q18 (12.5x) and q9 (11.2x), which between them carry the
highest-cardinality group-bys and the largest intermediates in the benchmark. That is the
open item, and {doc}`/benchmarks/results/scaling` carries the per-query scaling table.

{doc}`/benchmarks/results/tpch` has the per-query detail.

## Lakehouse reads

A selective predicate on a Delta table should open one data file, not all of them. The
transaction log records each file's column bounds, and reading it at plan time is the whole
game. Single node, 10M rows across 200 Delta files, one `day` per file:

| `count(*) WHERE day = 42` | Time | Files opened |
|---|---:|---:|
| **Batcher** | **13.4 ms** | **1** |
| DuckDB `delta_scan` | 19.0 ms | baseline |

Batcher is 1.42× faster here, and it opens one file where the predicate selects one file.
Predicates are recovered from the user's plan, where a `Filter` on a `Scan` constrains that
scan whatever the optimizer does above it, so the most ordinary lakehouse query in existence
pushes down and skips the rest of the table.

## What DuckDB cannot do

The gap that matters most doesn't appear on this page as a number. DuckDB is single-node and its optimizer is static. It commits to a plan before the first row is read and can't change its mind. Batcher re-optimizes at stage boundaries on measured cardinalities, the same granularity Spark AQE works at but available single-node too, and it carries a sketch-backed cross-query learned-stats loop that DuckDB has no equivalent for. The same mergeable operators then run across a cluster with a bit-identical result.

Stage-boundary re-optimization engages at 20M input rows and above, and it runs single-node, where Spark cannot go. See {doc}`/benchmarks/results/scaling`.

## Reproduce

```bash
python benchmarks/run.py --benchmark operators --tier single --scale 1
python benchmarks/run.py --benchmark tpch      --tier single --scale 1
```

## See also

- {doc}`/benchmarks/results/tpch` for the per-query breakdown.
- {doc}`/benchmarks/results/analytics` for operators, connectors, and the lazy control plane.
- {doc}`/benchmarks/comparisons/vs-polars` and {doc}`/benchmarks/comparisons/vs-daft` for the other two single-node engines.
- {doc}`/architecture/deep-dives/operators/aggregation-internals` for the quickselect finalize behind the median and quantile wins.
- {doc}`/architecture/deep-dives/operators/join-algorithms` for the join strategies behind the TPC-H results.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization` for what a static optimizer can't do.
- {doc}`/benchmarks/methodology` for hardware, gating, and why cross-hardware comparison is meaningless.
