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
the whole query in 5.2 ms.

**The boundary widening is a live suspect, and an earlier entry here wrongly dismissed it.**
That entry compared the same rows handed in as `int32` against `int64` and found 9.71 ms
against 9.75, concluding the widening costs nothing. The comparison cannot show that: narrow
integers are normalized to `int64` once at the Arrow boundary (`bc_py::normalize_to`), so
**both arms run the identical `int64` kernel** and agreeing tells us only that normalization
itself is cheap. A `perf` profile of the 100-group aggregate settles what is actually
running: the hottest symbol, at **33.8% of samples**, is
`bc_runtime::agg::group::assign::int_group_ids::<Int64Type>` -- the dense direct-map over
**eight-byte** keys, for a column the user declared as four. DuckDB groups the same column
at its declared width.

Testing it properly needs a narrow-key path rather than a narrower input, which is a
wire-contract change. Until then this is an open lead, not a finding.

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

**Where the cost actually is: not the aggregate.** The engine's own measured plan settles
it. For `k, count(*) GROUP BY k` over 10 M rows:

```
operators        79us     2%  of 3.9ms  (2.4ms cpu across workers)
elsewhere       3.8ms    98%  planning, optimization, admission, FFI crossing, result assembly
```

**Ninety-eight per cent of the query is not operators.** Isolating the fixed part by running
the identical query over **three** rows gives a **1.47 ms per-query floor** -- more than half
of DuckDB's entire 10 M-row time (2.76 ms) spent before any data is touched. SQL parsing and
plan building are not it: a pre-built plan collects in 4.61 ms against 4.58 for one parsed
each time, so both are already cached.

It is tempting to say that single fact explains the shape of the whole suite, and the
arithmetic says otherwise: the floor is 15.5% of q4 but only 1.4% of q8, so removing it moves
q4 from 1.76 to 1.49 and q8 from 1.48 only to 1.46. The larger losses are real work.

What the floor *is* is the lever on the **geometric mean**, because it shaves every one of
ten ratios a little and a geomean compounds them. Removing it entirely takes the suite from
1.038 to **0.988**, and the sensitivity says how much has to go:

| floor removed | 0.00 | 0.50 | 0.75 | 1.00 | **1.25** | 1.47 |
|---|---:|---:|---:|---:|---:|---:|
| geomean | 1.038 | 1.021 | 1.013 | 1.004 | **0.995** | 0.988 |

**The suite turns at about 1.25 ms of the 1.47** -- an 85% cut. The obvious route is a fast
path for a plan that has already run, skipping work that cannot have changed. Priced by
stubbing each subsystem out entirely, that route **does not reach it**:

| stubbed | saved |
|---|---:|
| event-log write | 0.408 ms |
| learning-loop close | 0.100 ms |
| `learn_column_stats` | -0.046 ms (no effect) |
| `recommended_config` memoized | 0.097 ms |
| **total** | **0.559 ms of 1.534** |

Removing four whole subsystems -- including the cross-query learning loop, which is the
stated moat -- buys 0.56 ms against the 1.25 required, leaving the geomean near 1.019. The
remaining 0.975 ms is diffuse orchestration with no single owner: 418 `dict.get`, 486
`isinstance` and 325 `getattr` per query across many small functions.

So **this suite is not winnable by reducing per-query overhead**, and that is worth recording
as a closed direction rather than an open one. The event log is still the single largest item
in the floor at 27% of it and worth fixing on its own merits; it just does not win the suite.
What would is a smaller constant *and* a faster aggregate, which is two programmes rather
than one.

Three corollaries, each measured:

* **It does not scale with cores**, because a fixed cost cannot. Pinned with `taskset` on an
  idle box: 34.27, 11.85, 8.25, 5.26, 4.99, 4.43 ms at 1, 4, 8, 16, 32 and 46 cores -- a
  7.7x ceiling, which Amdahl's law fits with a ~4.4 ms serial section. Not an artifact of the
  probe: OpenBLAS pinned to one thread gives the same ladder.
* **It is flat across table width** -- 4.48, 4.50, 4.49, 4.96, 4.92 ms from 1 to 17 columns.
  An earlier entry here suggested otherwise and was reading noise from a loaded box.
* **The scan underneath is seven times faster than DuckDB's** (0.20 ms against 1.39 for a real
  filtered count over the same rows). There is nothing wrong with the data plane on this
  shape.

**What the floor is made of, and what it is not.** Profiled over 200 collects of the 3-row
query, the native call is **0.14 ms** of it -- the rest is Python orchestration. Two things
in that orchestration touch the filesystem on *every* query and look like the answer:
`carbonite.memory.probe._cgroup_total_bytes` opens 1.48 files per collect, and
`api.terminal.event_log._prune` unlinks one. Priced by stubbing both, they are worth
**0.082 ms of 1.513** -- five per cent. Recorded as a measured negative so the next reader
does not spend the afternoon caching a cgroup read.

What is left is diffuse: 418 `dict.get`, 486 `isinstance` and 325 `getattr` per query, spread
across many small functions with no single owner. That agrees with the independently recorded
finding that the control plane is a flat ~2.2 ms per query of which four named subsystems
(event log, adaptive morsel sizing, the learning loops, column-stat learning) account for
only 0.80 ms between them. Lowering it is broad control-plane work, not one change.

**An earlier entry wrongly dismissed the `int32` to `int64` boundary widening**, comparing the
same rows handed in as `int32` and as `int64` (9.71 against 9.75 ms). The comparison cannot
show what it claimed: narrow integers are normalized to `int64` at the Arrow boundary, so
*both arms ran the identical `int64` kernel*. A `perf` profile confirms what runs -- the
hottest symbol is `agg::group::assign::int_group_ids::<Int64Type>` for a column declared
`int32`. It remains untested rather than refuted, and given the finding above it is a
second-order lead: the kernel is 2% of this query.

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
the whole query in 5.2 ms.

**The boundary widening is a live suspect, and an earlier entry here wrongly dismissed it.**
That entry compared the same rows handed in as `int32` against `int64` and found 9.71 ms
against 9.75, concluding the widening costs nothing. The comparison cannot show that: narrow
integers are normalized to `int64` once at the Arrow boundary (`bc_py::normalize_to`), so
**both arms run the identical `int64` kernel** and agreeing tells us only that normalization
itself is cheap. A `perf` profile of the 100-group aggregate settles what is actually
running: the hottest symbol, at **33.8% of samples**, is
`bc_runtime::agg::group::assign::int_group_ids::<Int64Type>` -- the dense direct-map over
**eight-byte** keys, for a column the user declared as four. DuckDB groups the same column
at its declared width.

Testing it properly needs a narrow-key path rather than a narrower input, which is a
wire-contract change. Until then this is an open lead, not a finding.

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

**Where the cost actually is.** Adding one layer of work at a time to the same 10 M rows, on
a quiet box, localizes it to a single layer rather than spreading it:

| shape | Batcher | DuckDB | |
|---|---:|---:|---:|
| `count(*)` | 0.11 ms | 0.78 ms | 0.14x |
| `count(*) WHERE k < 50` (a real scan) | 0.20 ms | 1.39 ms | 0.14x |
| `sum(k)` | 0.13 ms | 1.16 ms | 0.12x |
| `k, count(*) GROUP BY k` | 4.42 ms | 2.76 ms | 1.60x |
| `k, sum(k) GROUP BY k` | 5.19 ms | 2.75 ms | 1.89x |

**The scan is seven times faster than DuckDB's.** What costs is the group-by layer on top of
it: adding `GROUP BY k` over 100 groups costs Batcher **4.2 ms** where it costs DuckDB
**1.4 ms**. The whole gap is there, and the scan underneath it is a large win.

That is worth stating because two other readings were tried and are wrong. It is not table
width -- swept from 1 to 17 columns the time is flat (4.48, 4.50, 4.49, 4.96, 4.92 ms), and
an earlier measurement suggesting otherwise was noise from a loaded box. And the key mapping
is already the good one: a non-nullable `int32` key of span 100 takes the dense direct-map
path, with no hashing and no probe.

**And the layer stops scaling at about sixteen cores.** Pinned with `taskset` on an otherwise
idle box, the same query:

| cores | 1 | 2 | 4 | 8 | 16 | 32 | 46 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ms | 34.15 | 18.59 | 16.74 | 10.20 | 4.52 | 4.44 | 4.43 |

Past sixteen it is flat: 46 cores buy 2% over 16. Amdahl's law fits the curve closely -- a
34.15 ms single-core time with a **4.4 ms serial section** predicts a 7.8x ceiling against
the 7.69x measured -- and that serial section on its own is larger than DuckDB's entire
query. It is worse at higher cardinality: at 10,000 groups the query is flat from 8 cores
(12.41 ms) to 46 (13.98 ms), so whatever is serial grows with the group count.

The partial merge is the obvious suspect and is **not** it: at 610 morsels x 100 groups the
merge sees 61,000 partial rows, well past the 11,776 at which `combine` takes its
radix-parallel path on this machine. The curve is also not an artifact of the probe -- with
OpenBLAS pinned to one thread (its spin-server shows up in the profile at 28.6% of samples)
the ladder is unchanged: 34.27, 11.85, 8.25, 5.26, 4.99, 4.43 ms.

What the profile does name, besides the group-id assignment above: `agg::dispatch::accumulate`
at 9.4%, and `HyperLogLog::add_array_fast` at 1.4% -- the cross-query sketch running inside a
`count(*)`.

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
