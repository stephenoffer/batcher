# Loss backlog — every case measured slower than a competitor

One row per benchmark case where Batcher is slower than the fastest competitor that ran it.
Re-swept **2026-09-13** on a 48-core box, replacing the 2026-09-02 sweep of a 16-core one. The
two are not comparable row for row and the new board is not smaller: it has 40 rows against 34,
because the suite gained four operator families in between and because a different box moves
every ratio (`BENCHMARK_RESULTS.md`: ratios from different machines "differ by an order of
magnitude"). Read this board, not the arithmetic between them.

## How to read it

**Ordered by absolute gap, not by ratio.** A 2.5x on a 1.4 ms query is 2 ms of engineering value
and a 1.2x on a 100 ms one is 20 ms; sorting by ratio puts the wrong work first. `gap ms` is
`batcher_ms` minus the *best* competitor time, so it is what a perfect fix would return.

**The `kind` column is the one that decides what to work on**, and it needs the `on Arrow`
column beside it to be read. Every timed comparison here is against DuckDB on its **native,
compressed** storage — the right headline bar, and not a comparison of execution engines.
`duckdb_arrow` is the same DuckDB over the *same Arrow buffers Batcher reads*:

* **execution** — an Arrow-native engine (DuckDB-on-Arrow, or Polars) also beats Batcher. The
  gap is in this engine and is worth fixing here.
* **storage** — both Arrow-native engines are *behind* Batcher and only DuckDB-native is ahead.
  The gap is bytes read: dictionary encoding, zone maps, and block statistics Batcher has no
  equivalent for. `cb-q41` is the shape of it — 6.1 ms against DuckDB-native's 4.3 and
  DuckDB-on-Arrow's **43.4**. These close by adopting `StringView` and keeping dictionaries
  through the kernels (ceiling 2), not by faster kernels.

**Twenty-one of the forty rows are execution and nineteen are storage.** The split is not
uniform: ten of ClickBench's fifteen losses are storage (its `hits` table is wide and
string-heavy, which is what a dictionary is for), four of H2O groupby's six, and one of TPC-H's
six. The other five ClickBench rows are losses to *Polars*, which reads the same Arrow — those
are execution.

**Box**: 48-core (24 physical + SMT), 92 GiB, shared. Engines: batcher, duckdb, duckdb_arrow,
polars, best-of-five, one process per case. **A single-pass per-case ratio on this board has a
median spread of 11%** (`--repeat 5` over TPC-H), against 1.4-2.3% for a suite geomean — so a
row here is a target, not a measurement, and a 10% move in either direction means nothing until
it is repeated.

## Where the suites stand

| suite | vs duckdb | vs duckdb_arrow | vs polars | losing cases |
|---|---:|---:|---:|---:|
| TPC-H sf1 | 0.72 | 0.25 | 0.54 | 6 of 22 |
| operator mix | 0.75 | 0.47 | 0.16 | 13 of 46 |
| ClickBench | 0.65 | 0.16 | 0.37 | 15 of 43 |
| H2O groupby | 1.05 | 0.83 | 0.53 | 6 of 10 |
| H2O join | 0.63 | 0.58 | 0.51 | **0 of 5** |
| JSON | 0.35 | 0.32 | 0.01 | **0 of 5** |

## The board

| # | case | suite | batcher ms | best rival ms | vs | ratio | gap ms | on Arrow | kind |
|---:|---|---|---:|---:|---|---:|---:|---:|---|
| 1 | `h2o-gb-q8` | h2o-groupby | 104.9 | 70.8 | duckdb | 1.48x | 34.1 | 83.2 | execution |
| 2 | `tpch-q21` | tpch sf1 | 85.6 | 60.9 | duckdb | 1.41x | 24.7 | 191.9 | execution |
| 3 | `op-except` | operators | 46.6 | 24.7 | polars | 1.89x | 21.9 | 32.6 | execution |
| 4 | `h2o-gb-q7` | h2o-groupby | 66.8 | 50.7 | duckdb | 1.32x | 16.1 | 56.7 | execution |
| 5 | `op-join-build-large` | operators | 70.6 | 55.7 | duckdb | 1.27x | 14.9 | 60.5 | execution |
| 6 | `h2o-gb-q3` | h2o-groupby | 65.3 | 53.9 | duckdb | 1.21x | 11.4 | 70.1 | storage |
| 7 | `tpch-q17` | tpch sf1 | 16.0 | 5.3 | polars | 3.02x | 10.7 | 80.9 | execution |
| 8 | `tpch-q5` | tpch sf1 | 21.1 | 13.2 | polars | 1.60x | 7.9 | 140.6 | execution |
| 9 | `h2o-gb-q2` | h2o-groupby | 34.2 | 26.4 | duckdb | 1.30x | 7.8 | 35.3 | storage |
| 10 | `tpch-q8` | tpch sf1 | 17.6 | 10.4 | polars | 1.69x | 7.2 | 92.6 | execution |
| 11 | `op-sort-float` | operators | 42.7 | 36.6 | polars | 1.17x | 6.1 | 52.3 | execution |
| 12 | `op-sort-string-limit` | operators | 11.1 | 5.9 | duckdb | 1.88x | 5.2 | 6.4 | execution |
| 13 | `op-str-like-prefix` | operators | 8.4 | 3.4 | duckdb | 2.47x | 5.0 | 17.6 | storage |
| 14 | `h2o-gb-q9` | h2o-groupby | 44.7 | 39.9 | duckdb | 1.12x | 4.8 | 45.3 | storage |
| 15 | `h2o-gb-q4` | h2o-groupby | 9.6 | 5.3 | duckdb | 1.81x | 4.3 | 16.3 | storage |
| 16 | `cb-q38` | clickbench | 9.2 | 5.2 | polars | 1.77x | 4.0 | 35.3 | execution |
| 17 | `op-expr-date-part` | operators | 6.0 | 2.4 | duckdb | 2.50x | 3.6 | 10.2 | storage |
| 18 | `op-str-like-contains` | operators | 13.3 | 10.3 | duckdb | 1.29x | 3.0 | 11.5 | execution |
| 19 | `tpch-q2` | tpch sf1 | 9.5 | 6.7 | polars | 1.42x | 2.8 | 60.1 | execution |
| 20 | `op-str-upper-group` | operators | 19.8 | 17.3 | duckdb | 1.14x | 2.5 | 21.2 | storage |
| 21 | `op-expr-date-arith` | operators | 20.0 | 17.7 | duckdb | 1.13x | 2.3 | 17.9 | execution |
| 22 | `op-str-substring-group` | operators | 25.7 | 23.5 | duckdb | 1.09x | 2.2 | 24.0 | execution |
| 23 | `cb-q37` | clickbench | 11.4 | 9.5 | duckdb | 1.20x | 1.9 | 80.0 | storage |
| 24 | `cb-q41` | clickbench | 6.1 | 4.3 | duckdb | 1.42x | 1.8 | 43.4 | execution |
| 25 | `cb-q30` | clickbench | 6.3 | 4.5 | duckdb | 1.40x | 1.8 | 35.6 | storage |
| 26 | `op-expr-cast-chain` | operators | 6.3 | 4.6 | duckdb | 1.37x | 1.7 | 5.9 | execution |
| 27 | `cb-q14` | clickbench | 6.3 | 4.6 | duckdb | 1.37x | 1.7 | 36.7 | execution |
| 28 | `op-str-length` | operators | 4.6 | 3.0 | duckdb | 1.53x | 1.6 | 10.9 | storage |
| 29 | `op-expr-conditional` | operators | 7.3 | 5.8 | duckdb | 1.26x | 1.5 | 6.1 | execution |
| 30 | `cb-q31` | clickbench | 6.8 | 5.5 | duckdb | 1.24x | 1.3 | 36.9 | storage |
| 31 | `cb-q19` | clickbench | 1.7 | 0.5 | polars | 3.40x | 1.2 | 8.3 | execution |
| 32 | `tpch-q6` | tpch sf1 | 6.2 | 5.1 | duckdb | 1.22x | 1.1 | 20.0 | storage |
| 33 | `cb-q25` | clickbench | 3.4 | 2.3 | duckdb | 1.48x | 1.1 | 34.3 | storage |
| 34 | `cb-q39` | clickbench | 44.7 | 43.7 | duckdb | 1.02x | 1.0 | 101.1 | storage |
| 35 | `cb-q40` | clickbench | 5.4 | 4.8 | duckdb | 1.13x | 0.6 | 42.9 | storage |
| 36 | `cb-q07` | clickbench | 1.9 | 1.3 | duckdb | 1.46x | 0.6 | 22.8 | storage |
| 37 | `cb-q26` | clickbench | 3.3 | 2.7 | duckdb | 1.22x | 0.6 | 34.5 | storage |
| 38 | `cb-q01` | clickbench | 1.0 | 0.5 | polars | 2.00x | 0.5 | 21.9 | execution |
| 39 | `cb-q24` | clickbench | 3.3 | 3.0 | duckdb | 1.10x | 0.3 | 35.7 | storage |
| 40 | `cb-q16` | clickbench | 9.6 | 9.3 | duckdb | 1.03x | 0.3 | 11.7 | storage |

Everything else is a win: 33 of 46 `operators`, 16 of 22 `tpch` sf1, 28 of 43 `clickbench`,
4 of 10 `h2o-groupby`, and all of `h2o-join` and `json`.

### The four largest, and what is known about each

1. **`h2o-gb-q8`** (34 ms) — top-2 per group as a window. The bounded top-k kernel already
   exists and is tuned (`bc-runtime/src/window/topk.rs`); `perf` attributes 66% of the case to
   the hash bucketing around it, and the two obvious ways to remove that were built and measured
   slower. Open.
2. **`op-except`** (22 ms) — the tagged-union group-by form. A semi/anti join is the right
   algorithm and needs a **single** null-safe key; built with a two-column key it was 5.6x
   slower (see `BENCHMARK_RESULTS.md`), so this wants either an engine-level null-equal join
   mode or a proof that the keys hold no NULLs.
3. **`h2o-gb-q7`** (16 ms) — high-cardinality grouped `max`/`min`. Batcher uses **less** CPU
   than DuckDB here (1,673 ms against 1,725) and more wall clock: the gap is parallel scaling of
   the aggregate, not kernel work, and it is not thread count — 30 workers beat both 23 and 46
   on this shape.
4. **`op-join-build-large`** (15 ms) — already improved from 107 ms to 68 by routing the
   unshardable plan to the materializing executor. What remains is the streaming flat probe
   against a 1.2M-row build, which wants a cost comparison that reads the probe size, not a
   smaller constant (a smaller one was tried and cost the semi/anti cases 30%).

## Distributed GPU inference at 95 GiB, the one cluster-scale loss (2026-09-10)

Everything above is single-node. This row is not, and it is the largest absolute gap in the
file by two orders of magnitude.

| case | batcher | best rival | vs | ratio | gap |
|---|---:|---:|---|---:|---:|
| `map_batches(GPU model) -> agg`, 100M rows x 256 f32 (95.4 GiB Parquet) | 175.34 s | 142.08 s | ray-data | **1.23x** | **33.3 s** |

8 x T4 + 8 x 16-core CPU nodes; answers agree to a 1.2e-12 checksum spread. Harness
`benchmarks/gpu_backend/vs_raydata_parquet_inference.py`.

**The cause is known and priced**, which is unusual for this file. The two engines read at the
identical 162.2 s; Batcher then **adds** its 13 s of device time where Ray Data **overlaps**.
Batcher's own staged route already overlaps -- the same work with a `write.parquet` terminal
instead of an aggregate is 50.2 s against 64.4 s at 30M rows, 1.28x, while additionally
writing output the aggregate does not. The aggregate shape simply never reaches that route:
`_is_linear_map_pipeline` is false with an aggregate on top, so the dispatcher never asks
`split_into_resource_stages` -- which, asked directly, splits the plan correctly into three
stages.

The fix is a dispatcher branch plus a partial-aggregating consumer stage, worth ~20% here on
the evidence of one ratio measured at a third of the size -- enough to put this row at or near
parity, not enough to name a figure. See the dated entry in `BENCHMARK_RESULTS.md` for
the hypotheses that died first (it is not the reader, and it is not where the read runs).

## Not measured, and why

**These belong to the retired 2026-09-02 sweep**, on the 16-core 30 GiB box, and are kept for
the semantic findings in them rather than for their status. The 2026-09-13 board above ran
every H2O groupby question to completion on 92 GiB, so the first row is closed by the machine
rather than by any change: `q6` is now a 0.60x win, `q9` a 1.12x loss and `q10` a 0.83x win.
Daft is not in the current lineup, so its rows stand as recorded.

| case | status |
|---|---|
| `h2o-gb-q6`, `q9`, `q10` | **KILLED** on the old box — OOM at 30 GiB, shared. Ran clean on the 2026-09-13 sweep; see the board above. |
| `tpch-q6` | **DIVERGENT** — Daft folds `0.06 + 0.01` in IEEE double and drops every `l_discount = 0.07` row. Batcher and DuckDB match the published TPC-H answer. A recorded semantic difference, not a defect. |
| `tpch-q21`, `q22` | PARTIAL — Daft errored (row-group pruning bind failure; unsupported `SUBSTRING` syntax). The DuckDB/Polars comparison stands. |
| `op-dedup-keyed-ordered`, `op-window-*` | PARTIAL — Daft OOM/unsupported. Batcher wins all of them against the engines that ran. |
| tpcds, job, scan, images | **not yet swept** — batch 2. `BENCHMARK_RESULTS.md` records 28 ClickBench losses and four catastrophic TPC-DS queries (q77 16.4x, q45 8.3x, q5 7.8x, q80 6.5x) from earlier runs on other boxes. |

## The next lever on `tpch-q18`, located but deliberately not taken (2026-09-08)

q18 is the largest single-node gap (11.2x DuckDB at sf10) and **99% of its operator time is one
node**: `GROUP BY l_orderkey`, 60M rows to 15M groups (`explain(analyze=True)`). That node is the
build side of the `SEMI` join, and `stream/builds.rs::collect_builds` runs every build side through
`parallel::run` — the *streaming* executor.

The same subtree, standing alone, was measured at **4,660 ms streaming against 736 ms
materializing** (both orders, minimum of three; read 347 ms + aggregate 505 ms accounts for the
second figure). It is join-free, so `materializing_aggregate_is_faster` already admits it and
Kyber already sets `prefer_materializing_aggregate` on it — the routing simply is not consulted
for a build side, which is decided one level up for the plan as a whole.

So the change is small to write: in `collect_builds`, route a build side the aggregate guard
admits to `par::execute_parallel_with` instead of `parallel::run`.

**The obstacle is not the routing, it is the options**, and it is bigger than the memory pool
alone. `collect_builds` receives `budget: usize`, `workers` and `meter` — not the caller's
`ExecOptions`. Synthesizing one there would run the materializing aggregate without:

* `pool` — the process-wide accounting that forces a breaker to spill instead of pushing toward
  OOM. Dropping it on q18 at 30 GiB is the same trade that took q17/q18/q20/q21 from completing to
  `SIGKILL` earlier the same day (`BENCHMARK_RESULTS.md`);
* `agg_spill.dir` — `SpillOptions` **requires** a path, and the streaming executor is never given
  one, so the subtree could not spill where the caller configured;
* `agg_spill.codec`, `op_budgets`, and `cancel` — the last meaning a ten-minute build could not be
  cancelled.

So the prerequisite is threading `ExecOptions` through the streaming build path: **19 signatures
taking a bare `budget: usize` across 5 files** (`breaker`, `builds`, `folds`, `mod`, `parallel`),
in the executor's hot path. Verified safe in one respect — `par` does not call back into `stream`,
so the sub-execution cannot recurse.

**Expected value, quantified rather than guessed.** q18 is 7,755 ms at sf10 and the build subtree
standing alone is 4,660 ms streaming, so the aggregate is ~60% of its wall time. At 736 ms it
would take q18 to roughly 3,800 ms — **11.2x DuckDB down to ~5.5x**. A large improvement on the
board's biggest row, and **still not a win**: q21 (6.0x) and q17 (5.4x) are untouched, and both are
the algorithm gap this file already records. Beating DuckDB outright at sf10 needs those too, so
this is worth doing on its own merits and should not be sold as closing the gap.

## What the top of the board already has a cause for

Only these. Every other row needs a profile before a patch — this repo's records hold four
separate cases where a plausible mechanism was implemented and moved nothing.

- **`tpch-q21`** — recorded at 3.8x DuckDB's CPU while using *more* of the machine, i.e. an
  algorithm gap rather than a localized one. It also declines streaming (it is a self-join), so
  the runtime join filter never reaches it. The largest single gap on this board.
- **`op-sort-float`** — new to the board because the case is new. It beats DuckDB (0.83x) and
  loses to Polars, which is the first time Polars has set the bar on a fixed-width sort here.
- **`h2o-gb-q2`/`q4`** — the string group key. Profiled previously as `assign_groups` 18%,
  memory movement 14%, rayon scheduling 15%: no single villain. `combine_ndv` under-estimates
  q9's 10,000 groups at ~100.

## Retracted 2026-09-08 — the defect below is not one, and the entry sent the wrong work first

**The seeded base-table NDV does reach the cardinality estimator, on the first run, inside the
optimizer that chooses the plan.** Re-measured by tracing every call to
`kyber.stats.columns.scan_columns` in one fresh process, on the same shape the entry uses (a
200,000-row table with 25 brands and 40 containers, filtered on one of each — 203 rows):

| call | learned ndv | reached from |
|---|---|---|
| 1 | no | `rules/extra/runtime_filters/skipping.py::drop_filter_conjunct_implied_by_zonemap` |
| 2 | no | `metadata_answer.py::_root_stats` <- `api/terminal/metadata_answer/aggregate.py` |
| 3 | no | `metadata_filter_count/answers.py::_child_stats` <- the same |
| 4 | **yes** — `brand 25.019`, `container 40.049` | `rules/extra/agg_rules.py::_child_stats`, inside `optimize_full` |

`seed_column_ndv` runs immediately before `kyber.optimize_full`, exactly where
`api/orchestration/run.py::_optimize` puts it, so the three blind calls are all *earlier* passes:
the metadata-answer attempt for a `count(*)`, and a rule pass above it. None of them picks a join
order or a build side. The one call inside the plan-choosing optimizer has the seeded counts.

**The 6,325-cold / 998-warm pair the entry read as "cold plans blind" is a different thing**, and
the two numbers say so themselves. `6,325 = 0.1 x 0.1^0.5 x 200,000` is the *default* equality
selectivity under exponential backoff — the metadata-answer path, calls 2 and 3. And
`998 = (1/40.049) x (1/25.019)^0.5 x 200,000` is the *same backoff rule over the seeded counts*,
which is what the optimizer used cold. Reproduced to the digit: predicted 998.39, observed
998.39.

So the estimate the optimizer works from is 998 against an actual 203 — **4.9x over, with the
statistics correct and present**. Independence over the same two counts gives 199.6, within 2% of
the truth. That is `stats/selectivity/combine.py::_exponential_backoff` (`s1 * s2^(1/2) * s3^(1/4)
...`) doing exactly what its docstring says it does: it "lifts the estimate toward the correlated
case", and these two columns are independent.

**And it is not worth fixing — that was A/B'd, not assumed.** Patched in-process so the arms
differ in exactly one expression, TPC-H sf1, all 22 correct in every arm, control run twice:

| arm | conjunction rule | b/duckdb | b/polars |
|---|---|---:|---:|
| backoff (today) | `s1 * s2^(1/2) * s3^(1/4) ...` | 0.87 | 0.72 |
| independent | `prod(s_i)` | 0.87 | 0.73 |
| sqrt_all | `prod(s_i^(1/2))` | 0.86 | 0.70 |
| backoff (repeat) | the control again | 0.86 | 0.69 |

The control's own spread between its two runs is as large as any difference between arms. A rule
that is provably 4.9x wrong on a two-equality conjunction is worth nothing on the geomean, because
a cardinality error costs wall time only when it changes a *decision*, and here it does not change
enough of them to see. Repro: `backoff_ab.py` in the session scratchpad — 30 lines that patch
`combine._exponential_backoff` and defer to `benchmarks/run.py` for everything else. **Better per-column statistics cannot fix it, and that was measured rather than argued.** The
obvious cheaper lever is the Misra-Gries `mcv` seeded alongside the ndv by the same pass:
`_equality_selectivity` already consults it, and a literal it covers has a *measured* frequency
that needs no independence assumption. It does not fire on this shape because
`optimizer.cardinality.mcv_min_fraction` is **0.05** while 25 uniform brands sit at 0.04 and 40
containers at 0.025 — and TPC-H `part` has exactly those cardinalities. Lowering the floor and
re-measuring:

| `mcv_min_fraction` | mcv recorded | estimate | actual |
|---|---|---:|---:|
| 0.05 (default) | none | 998.4 | 203 |
| 0.02 | `Brand#07` 0.0396, `BOX03` 0.0250 — both exact | **995.9** | 203 |
| 0.005 | same | **995.9** | 203 |

0.25%. The floor is doing what it should and the frequencies it then records are right; the answer
does not move because backoff **discards the second conjunct by construction** — it takes its
square root, so `0.0250 x 0.0396^0.5 = 0.004975` whatever the two numbers are worth. A rule that
square-roots its inputs cannot be repaired by sharpening them. Anyone reaching for a statistics
fix here should read this row first: it is the experiment that says the statistics are not the
problem.

What follows from the retraction is only this: **do not spend a day on the write->read path.** It
works. The rows the entry lists as estimate-sensitive (`tpch-q17`, `q5`, `q8`, `q3`, and JOB) are
still estimate-sensitive; the estimate they are sensitive to is arriving with correct inputs.

Repro: `scratchpad/ndv_trace.py` and `ndv_cold.py` in the session that wrote this; both are ~30
lines over a synthetic table and need no TPC-H.

## The original entry, superseded by the retraction above

**The seeded base-table NDV is computed correctly and never reaches the cardinality
estimator.** Reproduced on `SELECT count(*) FROM part WHERE p_brand = 'Brand#23' AND
p_container = 'MED BOX'` (actual answer: 204 rows of 200,000):

1. `seed_column_ndv` runs where its docstring says it does — immediately before
   `kyber.optimize_full`, on 4 resident sources, wanting exactly the right columns
   (`ndv_columns(plan)` = `p_brand`, `p_container`, `p_partkey`, `l_partkey`).
2. It computes the right numbers: `core.column_ndv` returns **`p_brand: 25.02`,
   `p_container: 40.05`** — TPC-H `part` has exactly 25 brands and 40 containers.
3. `kyber.stats.columns.scan_columns` nevertheless receives **`learned.ndv=None`** for those
   same columns, on the first run and every run after.
4. So the filter is estimated at 6,325 rows cold and 998 warm against an actual 204, with
   `Provenance.DEFAULT` throughout. `1/ndv` on the two equalities would give
   `200,000/25/40 = 200`.

The consequence is not confined to one estimate: `explain(analyze=True)` on `tpch-q17` shows
**every** node 4.3-4.9x over ("the plan above it was chosen for 29.8K rows and ran on 6.1K"),
and the three `join build side` decisions are taken on those numbers.

**The mechanism is not identified and this entry does not guess at one.** Two candidates were
tested and eliminated: the cell budget (`ndv_sketch_max_cells` is 2,147,483,648, so nothing is
skipped for size) and the `take_ndv = measured.ndv is not None and existing.ndv is None` gate in
`scan_columns` (not reached — there is nothing in `learned` to gate). What is left is the path
between the seeder's write and `load_learned_stats`' read, which was not traced because
`api/terminal/_metadata.py` is being edited by another session and the accessor could not be
reached from outside it.

Worth fixing before any of the estimate-sensitive rows above (`tpch-q17`, `q5`, `q8`, `q3`, and
JOB as a whole, which exists to measure exactly this), because a cardinality fix moves all of
them at once and a per-query fix moves one.

**(Retracted — see the section above. Steps 1 and 2 are correct and step 3 is not: traced per
call, `scan_columns` does receive the seeded counts inside `optimize_full`. The `4.3-4.9x over`
observation on `tpch-q17` stands as a measurement; its cause is the conjunction rule, not a
missing statistic.)**
