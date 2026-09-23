# TPC-H

This page reports Batcher's TPC-H results against DuckDB, Polars, Daft and Spark at scale factors 1 and 10, and the planner work behind them.

Against DuckDB reading the same Arrow, Batcher is about four times faster at sf1 and three times faster at sf10. Against DuckDB on its own compressed store, the harder bar, it leads at both scales: 0.72x at sf1 and 0.963x at sf10.

## Correctness first

:::{important}
Batcher matches DuckDB's result on all 22 queries, and matches the official TPC-H answer on q6. That result gates every number on this page. The harness compares each engine's result with DuckDB's as a sorted row multiset within float tolerance, checks the order of every query that ends in `ORDER BY`, and refuses to record a ratio when they disagree.
:::

The gate earns its keep on other engines. The following table lists what it found in the run recorded in [`benchmarks/results/TPCH_SF1_SF10_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/results/TPCH_SF1_SF10_RESULTS.md):

| Engine | Correctness on the suite |
|---|---|
| DuckDB | The reference. |
| Batcher | Matches DuckDB on all 22, and the official answer on q6. |
| Daft | Wrong results on q6 and q15 at both scales, and returns the wrong columns on q18. Can't plan q21 (`Outer reference columns cannot be bound`) or q22 (`SUBSTRING(x FROM a FOR b)`). |
| Polars | Its SQL frontend fails 9 of 22 queries and returns the wrong revenue on q6, so the harness drives Polars through its native `LazyFrame` pipelines instead. |

:::{dropdown} What goes wrong on q6
The predicate is `l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01`. In IEEE double, `0.06 + 0.01` is `0.06999999999999999`, a hair under `0.07`. An engine that folds the bound in floating point drops every `l_discount = 0.07` row and returns 75,207,768 instead of the official sf1 revenue of 123,141,078.2283. TPC-H defines `l_discount` as `DECIMAL`, so the 0.07 rows belong in the answer. Batcher returns the official figure.
:::

## Where the suite stands

Each row below is the most recent measurement against that engine, so the machines and dates differ by row. Every figure is a geometric mean of per-query `batcher_ms / engine_ms`, so **below 1.00 means Batcher is faster**:

| Against | sf1 | sf10 | Measured |
|---|---:|---:|---|
| DuckDB, native store | **0.72** | | 2026-09-13, 48 cores, 92 GiB |
| DuckDB, native store | | **0.963** | 2026-08-25, 96 cores, 184 GiB |
| DuckDB, same Arrow | **0.25** | | 2026-09-13, 48 cores, 92 GiB |
| DuckDB, same Arrow | | **0.33** | 2026-08-28, 92-core box |
| Polars | **0.54** | | 2026-09-13, 48 cores, 92 GiB |
| Polars | | **0.35** | 2026-08-28, 92-core box |
| Daft | **0.21** | **0.17** | 2026-08-28, 92-core box |

The sf10 result against the native store is the one that moved most recently. It read 1.087x on the tree before the changes of 2026-08-25 and 0.963x after them, measured as a same-day A/B on the same node with only Batcher and DuckDB in the lineup. The suite total fell from 2,938 ms to 2,323 ms, carried by individual queries rather than by the mean:

| Query | Before | After |
|---|---:|---:|
| q9 | 456 ms | **233 ms** |
| q13 | 325 ms | **174 ms** |
| q5 | 189 ms | **122 ms** |
| q3 | 116 ms | **87 ms** |
| q4 | 117 ms | **96 ms** |
| q10 | 158 ms | **139 ms** |

Four changes carried it. A probe-side Bloom filter was built once per build shard and merged serially, and it is now sharded like the hash table beside it. Two fitted constants that existed only to compensate for that cost are gone, so a multi-join query keeps every core. An ordered group key now uses the partitioning its layout already provides. And the group-count estimator no longer reads a clustered key as a small domain.

![Bar chart of the TPC-H scale-factor-10 suite on the same Arrow input, from the 2026-08-28 sweep on 92 cores. Batcher is 3.03x faster than DuckDB reading the same Arrow and 2.86x faster than Polars.]](/_static/diagrams/tpch_sf10.svg)

The chart above is the sf10 board of 2026-08-28 on 92 cores, best of three. Against DuckDB's own compressed store that board read 1.10, and the same-day A/B of 2026-08-25 in the table above read 0.963, so the native-store standing at sf10 sits close to parity while the execution comparison is a clear win.

## Per query

The most recent run published query by query is [`benchmarks/results/TPCH_SF1_SF10_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/results/TPCH_SF1_SF10_RESULTS.md), taken 2026-07-28 on a c5d.24xlarge (96 vCPU, 184 GiB). It predates the sf10 gains above, so its native-store column reads 0.963x at sf1 and 1.521x at sf10. Batcher's total at sf1 was 617.5 ms against DuckDB's 649.2 ms on its native store, and 1,693.0 ms for DuckDB on the same Arrow.

:::{dropdown} Per-query ratios at sf1, 2026-07-28
Each cell is `batcher / engine`, so **below 1.00x means Batcher is faster**. Daft's `--` marks a wrong result and `n/a` a query it can't plan.

| Query | DuckDB native | DuckDB same Arrow | Polars | Daft |
|---|--:|--:|--:|--:|
| q1 | 1.11x | 0.83x | 0.29x | 0.55x |
| q2 | 0.71x | 0.17x | 0.70x | 0.25x |
| q3 | 1.00x | 0.37x | 0.82x | 0.62x |
| q4 | 1.13x | 0.49x | 0.40x | 1.38x |
| q5 | 1.40x | 0.19x | 0.99x | 0.85x |
| q6 | 1.79x | 0.32x | 0.26x | -- |
| q7 | 1.03x | 0.39x | 0.23x | 0.61x |
| q8 | 0.74x | 0.19x | 0.72x | 0.26x |
| q9 | 0.77x | 0.42x | 0.84x | 0.61x |
| q10 | 0.66x | 0.32x | 0.52x | 0.14x |
| q11 | 0.94x | 0.19x | 0.37x | 0.21x |
| q12 | 1.31x | 0.55x | 0.21x | 0.12x |
| q13 | 1.00x | 0.84x | 0.35x | 0.85x |
| q14 | 1.10x | 0.42x | 1.13x | 0.42x |
| q15 | 0.68x | 0.25x | 0.37x | -- |
| q16 | 0.62x | 0.24x | 0.51x | 0.34x |
| q17 | 0.65x | 0.19x | 2.40x | 0.24x |
| q18 | 1.05x | 0.42x | 0.58x | 0.87x |
| q19 | 0.98x | 0.68x | 0.32x | 0.58x |
| q20 | 1.07x | 0.32x | 0.45x | 0.91x |
| q21 | 1.06x | 0.39x | 0.92x | n/a |
| q22 | 1.15x | 0.48x | 0.89x | n/a |
:::

## Planner work behind these numbers

Several of the largest moves on this suite came from the optimizer rather than the kernels, and they show where the wins come from.

**A date grid on the wrong number line.** Core measures a quantile grid from raw Arrow values, so a `date32` column's grid counts days since the Unix epoch. Kyber read it with `date.toordinal()`, which counts from year 1 and is 719,163 days out. Every date literal landed outside its column's grid, so `o_orderdate BETWEEN '1995-01-01' AND '1996-12-31'` estimated 0 rows against a true 455,112, and a join with a zero-row side priced as free. Fixing it took q8 from 735.0 ms to 20.7 ms and the sf1 suite total from 1,843 ms to 871 ms (2026-07-31, 16 cores). The defect bit only from a query's second execution, because the first has no measured grid. A benchmark warms up before it times, so every timed run measured the broken plan.

**Build-side selection.** Broadcast eligibility used to be checked only on the right input, so when the small side arrived on the left the join shuffled a 6M-row build instead of broadcasting. It is now decided from `min(left_bytes, right_bytes)`, which took the q5 `orders` to `lineitem` join from 419 ms to 175 ms.

**Cold-start join cardinality.** The estimator's join model divides by the larger distinct count, but it used to read only distinct counts learned on past runs. A cold join fell back to `max(left, right)`, which underestimates a low-NDV many-to-many join badly enough to steer join order into intermediates of 12M to 18M rows. Cold q5 ran 7,115 ms against a warm 300 ms. Distinct counts are now seeded from source statistics, footer and written-file HLL sketches, so the cold plan has real inputs.

**A serial partition loop.** The radix join joined its partitions on one core, so a join too large to broadcast funnelled a parallel build and probe into a serial kernel. Joining partitions concurrently, and concatenating them in partition order so the output is unchanged, took q4 from 115.6 ms to 43.0 ms and q3 from 110.3 ms to 66.3 ms.

## Requirements and limitations

These figures are single-node and steady state. The following limits apply:

- **Scale factor 100** (600M rows) is still recorded as a loss to DuckDB on a single node.
- **Spark** isn't in the standing table. Local-mode Spark ran 20x to 50x behind Batcher on TPC-H sf1 (2026-08-15), and earlier Spark ratios were taken before three handicaps in its benchmark adapter were removed.
- **Rows from different dates** in the standing table describe different builds and machines. Compare within a row.

## Reproduce

The following commands rerun the suite against each lineup. `BENCH_TPCH_BASE` points the loader at a local mirror when S3 is slow:

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,duckdb,duckdb_arrow,polars --isolate
python benchmarks/run.py --benchmark tpch --scale 10 --engines batcher,duckdb
python benchmarks/run.py --benchmark tpch --scale 10 --engines batcher,duckdb,duckdb_arrow,polars,spark,daft
```

Spark needs a JVM as well as the `pyspark` wheel. Without one its adapter reports unavailable and the lineup drops it.

## See also

- {doc}`/benchmarks/comparisons/vs-duckdb` and {doc}`/benchmarks/comparisons/vs-daft` for the engine-by-engine scorecards.
- {doc}`/benchmarks/results/analytics` for the other suites and the operator mix.
- {doc}`/architecture/deep-dives/operators/join-algorithms` for the shuffle and broadcast paths behind these results.
- {doc}`/architecture/deep-dives/adaptive/cardinality-estimation` for the cold-start distinct-count problem.
- {doc}`/architecture/deep-dives/adaptive/cost-model` for how the build side is chosen.
- {doc}`/user-guide/analyze/sql` for the supported SQL surface.
- {doc}`/benchmarks/methodology` for the correctness gate in detail.
- {doc}`/examples/tpch`: all 22 TPC-H queries as standalone scripts.
