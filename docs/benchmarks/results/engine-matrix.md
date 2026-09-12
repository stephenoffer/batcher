# The full engine matrix

This page carries one number per suite per engine: every standard suite against every engine
that can run it, measured in one sweep on one machine.

The point of publishing it whole is that a per-engine page can flatter by omission. Here the
gaps are visible, and each is labeled with which kind of gap it is.

## How to read a cell

| Cell | Meaning |
|---|---|
| a ratio | `batcher_ms / engine_ms`, geometric mean over the suite. Below 1.00 means Batcher is faster. |
| `--` | The engine cannot express this suite. Not a loss. |
| `OOM` | The pair could not be held in memory together on this box. |

A ratio is recorded only for cases that passed the correctness gate against an independent
oracle. Cases an engine could not run are excluded from its geomean and the count says how
many remained.

## The board

48-core Xeon Platinum 8275CL, 92 GiB, quiet box, pairwise lineups, best of five, scale
factor 1 where a suite has one. Measured 2026-09-11.

| Suite | DuckDB | Polars | PyArrow | Daft | Spark |
|---|---:|---:|---:|---:|---:|
| JSON (5) | 0.36 | 0.01 | `--` | 0.06 | 0.01 |
| operators (23) | 0.63 | 0.15 | 0.04 | 0.08 | 0.03 |
| scan (27) | 0.63 | 0.25 | 0.07 | 0.13 | 0.21 |
| H2O `join` (5) | 0.63 | 0.49 | `--` | 0.29 | 0.04 |
| TPC-H sf1 (22) | 0.70 | 0.53 | `--` | 0.22 | 0.03 |
| ClickBench (43) | 0.72 | 0.43 | `--` | 0.15 | 0.02 |
| H2O `groupby` (10) | **1.01** | 0.52 | `--` | 0.38 | 0.07 |
| TPC-DS (99) | 0.98 | `--` | `--` | `OOM` | 0.05 |

Batcher is ahead on every cell that carries a number except one.

## The one suite that is not won, and by which statistic

H2O `groupby` against DuckDB is the only cell above 1.00, and which summary you read decides
whether it is a loss at all:

| statistic | value | |
|---|---:|---|
| geometric mean of per-query ratios | **1.01-1.04** | the number this site reports |
| suite total, Batcher against DuckDB | **0.946** | 732.1 ms against 773.9 ms |

Batcher runs the whole suite **5.4% faster** and still reads as a loss, because a geometric
mean weights a 9.5 ms query exactly as much as a 283 ms one. Batcher wins 4 of the 10
questions and they are the expensive ones -- q10, q6 and q5 are the three costliest in the
suite -- winning 115.6 ms in total while losing 73.8 ms across the other six.

**The geomean is still the number quoted**, here and everywhere else on this site, because
switching statistic when it flatters is how a benchmark stops meaning anything. The suite is
recorded as not won. But a reader deciding whether to run this workload should know that the
thing being lost is a per-query average on the cheapest queries, not throughput.

### Where the per-query losses are

H2O `groupby` against DuckDB sits at 1.01, and the split inside it is consistent rather than
noisy. Batcher wins the questions whose cost is group-by *state* and loses the ones whose
cost is a single pass over a few groups:

| Question | Shape | b/duckdb |
|---|---|---:|
| q5 | `sum` of three columns by a 100,000-group key | 0.58 |
| q6 | `median` and `stddev` by a compound key | 0.61 |
| q10 | group by all six key columns at once | 0.87 |
| q4 | three `avg`s by a **100-group** key | 1.79 |
| q8 | top-2 per group over 100,000 groups | 1.42 |
| q2 | `sum` by two string keys | 1.34 |

The gap is in the aggregate itself rather than in overhead, which is worth stating because
the opposite is the natural guess for a 9 ms query. Profiled, `execute_plan_metered` -- the
native call -- is 8.25 ms of q4's 9.76, leaving about 1.5 ms of control plane; DuckDB runs
the whole query in 5.2 ms. Nor is it the `int32` to `int64` widening at the Arrow boundary:
handed the same rows already as `int64`, Batcher measures 9.75 ms against 9.71 ms for the
narrow input, while DuckDB pays 12% more for the 60% more bytes.

What the shape has in common across the losing questions is *few groups*. Measured on 10 M
rows at 100 groups, Batcher is behind on every aggregate (`count(*)` 1.63x, `sum` 2.21x);
at 100,000 groups the same queries invert (`sum` 0.32x, `avg` of three columns 0.60x). The
group-by state is where Batcher's aggregate is strong, and a 100-group reduction is the case
where there is essentially no state to be strong about.

## What the gaps are

**PyArrow has no SQL surface.** Five suites are blank for it. A comparison exists only where
a case can be written against `Table` and `compute` -- a single operator or a short chain of
them -- which is what the operators and scan suites are and what the multi-table SQL suites
are not.

**Polars' SQL rejects comma joins.** `SELECT ... FROM a, b WHERE a.k = b.k` returns
`SQLInterfaceError: multiple tables in FROM clause are not currently supported (found 2);
use explicit JOIN syntax instead`, so Polars runs zero of the 99 TPC-DS queries. Its
DataFrame API is not affected, and the TPC-H column is measured through it.

**Daft and TPC-DS sf1 do not fit together.** The pair was killed by the cgroup at 71.1 GB
resident, reproduced twice.

## Disagreements, and which of them are an engine's fault

Every disagreement below was caught because a third engine was present to act as oracle,
which is the argument for three-engine lineups even though pairwise is cheaper. They are not
all the same kind of thing, and the difference decides who should fix them.

**An engine disagreeing with every other engine.** `SELECT AVG(UserID) FROM hits` returns a
*negative* mean of non-negative identifiers from Daft (-2.66e12 against 1.948e18), where
Batcher, DuckDB and Polars all agree. Spark disagrees on a timezone
(`2013-07-15 19:40:00Z` against `12:40:00`), on a row count, and aliases `count(*)` as
`count1` where the others project `count_star`. The naming ones are cosmetic; all are
excluded from the ratios rather than counted against them, and the case counts say how many
remained.

**The benchmark's fault, not an engine's.** The three `scan-filter_agg-*` cases disagree for
*both* Ray Data and Daft, and the suite's own module docstring predicts exactly this: its
columns are `int64` drawn uniformly from `[0, 2^63)`, so a bare sum overflows 64 bits and
engines disagree there **by design** -- DuckDB widens, others wrap -- "which would report an
engine bug that is really a benchmark bug". The rule it draws from that is that every sum is
taken over a bounded expression, and it was applied to every `SUM` in the suite and missed
the one `AVG`, which sums before it divides. Those three rows are uncomparable rather than
three engines being wrong, and the fix belongs in the benchmark.

The same mechanism is worth keeping in mind when reading the Daft row above: accumulation
width is a type decision rather than a correctness one in isolation. What makes that case an
engine's fault and this one the benchmark's is only that three engines agree there and two
disagree here.

## Findings behind these numbers

Short notes on what the sweep turned up, each measured rather than argued. The long form is
in `benchmarks/BENCHMARK_RESULTS.md`.

**Common-subplan reuse was refusing the shape it exists for.** A `ROLLUP` is one `GROUP BY`
per level over one common input, so the input repeats once per level -- exactly what the
reuse pass is for, and on four TPC-DS queries it chose nothing. Three defects: a constant
group key (`nullif(k, k)`, how both front-ends mark a rolled-up key) estimated at 10,000
distinct values instead of 1; the size and cost gates reading the plan *as written*, where a
`WHERE` still sits above the join tree, so q80's largest repeated subtree measured 1.6e13
rows against its real 687; and the saving formula short by a factor of the appearance count,
which made the bar *harder* the more there was to gain. Fixing those alone made TPC-DS
**worse**, which exposed the fourth: materializing forfeits the fusion each appearance had
with its parent, and that cost is a width. q18's subtree carried 133 columns where the plan
read 11. Materializing only what the plan reads took the suite to 0.98.

**The H2O `groupby` gap is the aggregate, not overhead.** Profiled, the native call is 8.25
of q4's 9.76 ms. It is also not the `int32`-to-`int64` boundary widening: handed the same
rows already as `int64`, Batcher measures 9.75 ms against 9.71 ms.

**The few-group loss is measured and its cause is still open.** It is worth saying what it is
*not*, because two plausible mechanisms were checked and neither holds. It is not the
chunked-partial concatenation: that path is guarded by `width_from_sample`, which only offers
it when a morsel *fails* to reduce, and a 100-group aggregate over 16,384-row morsels reduces
164:1, so the aggregate takes the per-morsel path and never concatenates. (This entry
previously said otherwise, on a reading of the concatenation and its gate that skipped the
guard above them.) And it is not overhead or type widening, per the two measurements above.

The morsel-width candidate is refuted too, and by the measurement that already exists: the
patch narrowing the planned morsel to the columns a query reads moved this suite **1.02 to
1.05**, the wrong way. It is recorded separately as a real defect that ClickBench is 2.1x
faster for; it is not this one.

**What the shape actually looks like, decomposed.** `count(*) GROUP BY k` at 100 groups over
a one-column table, swept from 625 K to 10 M rows, separates into a fixed cost and a
per-row one:

| | fixed | per row |
|---|---:|---:|
| Batcher | 1.77 ms | 0.257 ns |
| DuckDB | 1.04 ms | 0.117 ns |
| ratio | 1.71x | 2.19x |

Both halves are roughly 2x, which is the useful part: there is **no single hotspot**. The key
mapping is already optimal for this shape (a non-nullable `int32` key of span 100 takes the
dense direct-map path, no hashing), the plan rewrite the window question needs already fires,
and the parallel aggregate takes the per-morsel path rather than concatenating. What is left
is a broad constant-factor difference in the scan-and-accumulate inner loop and in per-query
fixed cost. That is an engineering programme, not a defect to find.

**Two competitors returned wrong answers.** See below.

## Reproducing it

```bash
python benchmarks/run.py --benchmark <suite> --engines batcher,<engine>
```

Pairwise rather than all-at-once because five engines holding TPC-DS sf1 simultaneously was
killed at 71 GB. Spark needs `BENCH_SPARK_DRIVER_MEMORY=12g` for TPC-DS on a 92 GiB box.

## See also

- {doc}`/benchmarks/methodology`: the correctness gate, the quiet-box rule, and the hardware.
- {doc}`/benchmarks/comparisons/index`: one page per engine, with the architectural reason.
