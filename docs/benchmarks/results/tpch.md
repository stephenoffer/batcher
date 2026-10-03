# TPC-H

This page reports Batcher's TPC-H results against DuckDB, Polars, Daft and Spark at scale factors 1 and 10, and the planner work behind them.

Against DuckDB reading the same Arrow, Batcher is about four times faster at sf1 and three times faster at sf10. Against DuckDB on its own compressed store, the harder bar, it leads at both scales: 0.72x at sf1 and 0.963x at sf10.

## Correctness first

Batcher matches DuckDB on all 22 queries and the official TPC-H answer on q6. The harness refuses to record a ratio for any engine whose result disagrees.

:::{dropdown} The correctness record of every engine
From the run recorded in [`benchmarks/results/TPCH_SF1_SF10_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/results/TPCH_SF1_SF10_RESULTS.md):

| Engine | Correctness on the suite |
|---|---|
| DuckDB | The reference. |
| Batcher | Matches DuckDB on all 22, and the official answer on q6. |
| Daft | Wrong results on q6 and q15 at both scales, and returns the wrong columns on q18. Can't plan q21 or q22. |
| Polars | Its SQL frontend fails 9 of 22 queries, so the harness drives Polars through its native `LazyFrame` pipelines. |

On q6 the predicate is `l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01`. An engine that folds `0.06 + 0.01` in IEEE double gets `0.06999999999999999`, drops every `l_discount = 0.07` row, and returns 75,207,768 instead of the official sf1 revenue of 123,141,078.2283.
:::

## Where the suite stands

Each row is the most recent measurement against that engine. Every figure is a geometric mean of per-query `batcher_ms / engine_ms`, so **below 1.00 means Batcher is faster**:

| Against | sf1 | sf10 | Measured |
|---|---:|---:|---|
| DuckDB, native store | **0.72** | | 2026-09-13, 48 cores, 92 GiB |
| DuckDB, native store | | **0.963** | 2026-08-25, 96 cores, 184 GiB |
| DuckDB, same Arrow | **0.25** | | 2026-09-13, 48 cores, 92 GiB |
| DuckDB, same Arrow | | **0.33** | 2026-08-28, 92-core box |
| Polars | **0.54** | | 2026-09-13, 48 cores, 92 GiB |
| Polars | | **0.35** | 2026-08-28, 92-core box |
| Daft | **0.21** | **0.17** | 2026-08-28, 92-core box |

At sf10 against the native store, the changes of 2026-08-25 took the suite total from 2,938 ms to 2,323 ms in a same-day A/B on one node:

| Query | Before | After |
|---|---:|---:|
| q9 | 456 ms | **233 ms** |
| q13 | 325 ms | **174 ms** |
| q5 | 189 ms | **122 ms** |
| q3 | 116 ms | **87 ms** |
| q4 | 117 ms | **96 ms** |
| q10 | 158 ms | **139 ms** |

The gains came from a sharded probe-side Bloom filter, a multi-join plan that keeps every core, reuse of an ordered group key's existing partitioning, and a group-count estimator that reads clustered keys correctly.

![Bar chart of the TPC-H scale-factor-10 suite on the same Arrow input, from the 2026-08-28 sweep on 92 cores. Batcher is 3.03x faster than DuckDB reading the same Arrow and 2.86x faster than Polars.](/_static/diagrams/tpch_sf10.svg)

The chart is the sf10 board of 2026-08-28 on 92 cores, best of three, on the same Arrow input.

## Per query

The per-query board is [`benchmarks/results/TPCH_SF1_SF10_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/results/TPCH_SF1_SF10_RESULTS.md), taken 2026-07-28 on a c5d.24xlarge (96 vCPU, 184 GiB), before the sf10 gains above. Batcher's sf1 total was 617.5 ms against DuckDB's 649.2 ms on its native store and 1,693.0 ms on the same Arrow.

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

Several of the largest moves on this suite came from the optimizer rather than the kernels:

| Change | Effect |
|---|---|
| Date literals read on the same number line as the measured `date32` quantile grid | q8 from 735.0 ms to 20.7 ms, sf1 suite total from 1,843 ms to 871 ms (2026-07-31, 16 cores) |
| Broadcast eligibility decided from `min(left_bytes, right_bytes)` | q5 `orders` to `lineitem` join from 419 ms to 175 ms |
| Cold-start distinct counts seeded from source statistics and file HLL sketches | A cold q5 plans on real inputs instead of a `max(left, right)` fallback |
| Radix-join partitions joined concurrently, concatenated in partition order | q4 from 115.6 ms to 43.0 ms, q3 from 110.3 ms to 66.3 ms |

`explain()` shows the row estimates each operator was planned on and the decisions taken from them, such as the join build side:

```python
import batcher as bt

orders = bt.from_pydict({"o_key": [1, 2, 3], "o_cust": [10, 20, 10]})
items = bt.from_pydict({"l_order": [1, 1, 2, 3, 3], "l_price": [5.0, 7.0, 3.0, 2.0, 4.0]})
q = items.join(orders, left_on="l_order", right_on="o_key").group_by("o_cust").agg(rev=bt.col("l_price").sum())
assert "join build side" in q.explain()
print(q.sort("o_cust").to_pydict())
# {'o_cust': [10, 20], 'rev': [18.0, 3.0]}
```

:::{dropdown} Scope of these numbers
- Figures are single node and steady state.
- Rows from different dates in the standing table describe different builds and machines. Compare within a row.
- Local-mode Spark ran 20x to 50x behind Batcher on TPC-H sf1 (2026-08-15). {doc}`/benchmarks/comparisons/vs-spark` has the details.
:::

## Reproduce

`BENCH_TPCH_BASE` points the loader at a local mirror when S3 is slow.

```bash
python benchmarks/run.py --benchmark tpch --engines batcher,duckdb,duckdb_arrow,polars --isolate
python benchmarks/run.py --benchmark tpch --scale 10 --engines batcher,duckdb
python benchmarks/run.py --benchmark tpch --scale 10 --engines batcher,duckdb,duckdb_arrow,polars,spark,daft
```

Spark needs a JVM as well as the `pyspark` wheel.

## See also

- {doc}`/benchmarks/comparisons/vs-duckdb` and {doc}`/benchmarks/comparisons/vs-daft` for the engine-by-engine scorecards.
- {doc}`/benchmarks/results/analytics` for the other suites and the operator mix.
- {doc}`/architecture/deep-dives/operators/join-algorithms` for the shuffle and broadcast paths behind these results.
- {doc}`/architecture/deep-dives/adaptive/cardinality-estimation` for how distinct counts are estimated.
- {doc}`/architecture/deep-dives/adaptive/cost-model` for how the build side is chosen.
- {doc}`/user-guide/analyze/sql` for the supported SQL surface.
- {doc}`/benchmarks/methodology` for the correctness gate in detail.
- {doc}`/examples/tpch`: all 22 TPC-H queries as standalone scripts.
