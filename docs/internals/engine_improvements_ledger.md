# Engine improvements ledger: competitor techniques, landed and open

**Status:** running record, opened 2026-07-24.

`competitor_technique_review.md` is the parts list: which mechanisms DuckDB, Polars,
DataFusion, Spark, Daft and Ray Data have that Batcher does not. This file is the build
log against it. One row per improvement, with what changed, why it is safe, and how it was
measured. Nothing is listed here until it compiles, passes its tests, and has a number or a
correctness argument behind it.

**Counting rule.** An entry is a change to engine behaviour that is verified. A refactor, a
doc, or a test with no behaviour change is not an entry, and neither is a change that is
written but unproven. That rule exists because the temptation on a "make N improvements"
task is to inflate the count with work that cannot fail, and an inflated ledger is worse
than a short one.

## Landed

### 1. Short-circuiting conjunctive filter

**Source:** DuckDB `ExpressionExecutor::Select`
(`src/execution/expression_executor/execute_conjunction.cpp:61`).

**Was:** `Expr::eval` evaluated every conjunct of an `AND` over every row and
`and_kleene`'d the masks, so a five-predicate filter keeping 1.65% of its rows paid five
full-width passes.

**Now:** `Expr::short_circuit_filter_mask` (`crates/bc-expr/src/select.rs`) evaluates the
cheapest conjunct at full width, then *compacts* — gathers the surviving rows of only the
columns the remaining conjuncts name — and evaluates the rest against that. Arrow has no
selection vector, so compaction is the equivalent move; gathering the two columns a
predicate names instead of the seventeen a fact table carries is what makes it pay.

**Safe because:** null-folding composes (`filter_record_batch` already treats a null mask
entry as false), every reachable kernel is elementwise, and
`Expr::is_infallible_predicate` admits only conjuncts whose failures are schema-driven, so
a skipped row can never have been the row that raised. Exhaustive match, no wildcard arm.

**Measured** (`cargo test --release -p bc-expr --test short_circuit_filter -- --nocapture`,
64 morsels x 16,384 rows, 17 columns), whole Filter operator including the gather both
paths pay:

| Predicate | Kept | Mask | Mask + gather |
|---|---|---|---|
| TPC-H q6 (5 cheap conjuncts) | 1.65% | 1.90x | 1.47x |
| `IN` list + 2 ranges | 2.87% | 1.28x | 1.14x |
| Cheap guard + `contains` | 1.14% | 3.83x | 2.42x |
| Cheap guard + `LIKE` | 1.14% | 2.56x | 1.84x |
| Cheap guard + `regexp_matches` | 1.14% | 7.68x | 4.63x |

### 2. Scalar broadcast for non-numeric literal comparisons

**Source:** the general vectorized-engine practice of comparing against a scalar rather
than a materialized column; Arrow's own `Datum` API.

**Was:** `try_scalar_binary` (`crates/bc-expr/src/eval/binary.rs`) broadcast a one-element
`Scalar` only for `Int64`/`Float64` columns against `Int`/`Float` literals. Every other
literal comparison fell to the array path, where `Literal::to_array(n)` materializes *n*
copies of the value — for a string, offsets **and** bytes — once per morsel per evaluation.
`o_orderpriority = '1-URGENT'` and `l_shipdate < DATE '1995-03-15'` are the two most common
predicate shapes in the benchmark suite, and both paid it.

**Now:** `Utf8`, `Boolean`, `Date32` and naive-`Timestamp` columns broadcast the same way,
on an **exact** type match only.

**Safe because:** the array path applies three adjustments before comparing, and an exact
type match makes each provably the identity — `align_date_timestamp_for_cmp` only fires on
a Date-vs-Timestamp pair, `align_decimals_for_cmp` only on two decimals of differing scale,
and `canon_float_array` is an `Arc::clone` for anything that is not a float. A mixed pair
(`LargeUtf8` column, tz-aware timestamp against a naive literal, `Int64` column against a
`Date` literal) is declined and keeps the array path's coercion, which is pinned by
`a_mismatched_literal_type_is_declined`. Bit-for-bit parity over every comparison operator,
both operand orders and null-bearing columns is pinned by
`scalar_path_equals_array_path_for_non_numeric_literals`.

### 3. Measurement-driven conjunct ordering

**Source:** DuckDB `AdaptiveFilter::AdaptRuntimeStatistics`
(`src/execution/adaptive_filter.cpp:107`), reached by a different mechanism.

**Was:** improvement 1 opened with the *static* cost order (cheapest first), which is what
DuckDB's `ExpressionHeuristics` computes. Cost is not selectivity, so the case it cannot
fix even in principle is two conjuncts of identical cost where only one is selective: the
order stays as written, and a broad predicate written first removes nothing, so no
compaction is possible and the selective one runs at full width too.

**Now:** `ConjunctOrder` (`crates/bc-expr/src/select.rs`) accumulates rows-in, rows-out and
nanoseconds per conjunct across a Filter operator's morsels, and each morsel orders by
`time per row / fraction of rows removed` — cheapest expected work first. Built once per
operator and shared across workers at `par.rs:534`, `par.rs:1982` (the fused pipeline),
`stream/mod.rs` and `stream/parallel.rs`. The first two of those are the materializing
parallel path; the last two are the engine **default**, which never carries a JIT for a
filter (measured at 1.01x, `stream/mod.rs:272`) and so is exactly where a measured order
has no competition.

DuckDB instead swaps a random adjacent pair and keeps it if *total* runtime improved over
the next ten batches, halving that position's swap likeliness when it did not. That is a
hill-climb on an aggregate signal and needs tens of batches to walk a permutation. Because
the short-circuit evaluates conjuncts one at a time anyway, Batcher can attribute rows and
time to each conjunct individually and reach the implied order after a *single* morsel. The
trade is that the measurement is conditional — a conjunct running third only sees rows the
first two kept — which biases the estimate without destabilizing it, because the order it
produces feeds the next measurement.

**Safe because:** the conjuncts of an `AND` commute, so every order yields the identical
mask. That is why the state needs no lock (two workers may briefly disagree about the best
order, at no cost) and why an adaptive ordering is not a correctness surface here.
`a_measured_order_never_changes_the_mask` asserts it against the whole-batch oracle over
eight rounds; `a_measured_order_promotes_the_selective_conjunct` asserts the convergence
actually happens; `a_mismatched_order_width_is_ignored` covers a caller pairing state with
the wrong predicate. The sequential oracle deliberately stays on the static order.

**Measured:** **1.73x** on the adversarial shape (two `regexp_matches` of identical static
cost, broad one written first), 48.64 ms -> 28.18 ms, verified against the oracle on all 64
morsels first. No change on the shapes where the static order was already right, which is
most of them.

### 4. Row-encoded multi-key sort, from three keys on

**Source:** the sort-key / row-encoding idea DuckDB and Polars both use, via
`arrow::row::RowConverter`.

**Was:** a multi-key sort went through `arrow::compute::lexsort_to_indices` with an ascending
row-index **column appended to the key list**. The column exists only to force a total order,
because `lexsort_to_indices` is unstable and the sequential oracle, the parallel sample-sort
and the external merge sort each sort a differently-sized slice and must agree on tied rows.

**Now:** `bc_arrow::row_sort::stable_lexsort_indices` encodes the key columns to
order-preserving bytes once and sorts `memcmp`-style, with the row index moved out of the
comparands and into the **comparator** (`rows[a].cmp(rows[b]).then(a.cmp(b))`). Identical
permutation — the encoding is order-preserving and the index is the last comparand either way
— and the sort carries one fewer key. Wired into `sort_indices_of`, which keeps arrow's path
as the fallback for any type the row encoder rejects.

**Safe because:** the permutation is asserted equal to the exact path it replaces, over every
direction and null placement, at 65,536 rows with dense ties
(`stable_row_sort_equals_the_reference_permutation_at_scale`) and in unit tests for mixed
directions, string-plus-numeric keys and floats. Ties resolving to input order is asserted
directly, ascending *and* descending. The 46-test streaming oracle and the
`parallel_top_n_matches_eager` pair — which pin seq == par on sort — stay green.

**Measured, and this is the part that shaped the change:** row encoding is *not*
unconditionally better, because `lexsort_to_indices` is a comparator sort with monomorphized
specializations for 2 to 4 columns, not a row-encoding sort as assumed. At 1,048,576 rows:

| Keys | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Ratio | 0.76x | 0.97x | **1.49x** | **1.34x** | **1.50x** |

So it is gated at `MIN_KEYS_FOR_ROW_ENCODING = 3`, and a one- or two-key sort keeps arrow's
path untouched. Without that gate this would have been a **regression on the most common sort
there is**, which is the whole reason the crossover was measured before wiring rather than
after.

**Not shipped:** a row-encoded *top-N* was written and measured at 0.60x to 0.66x at every
limit tried, because arrow's `partial_sort` keeps a bounded region of size `limit` and barely
touches the tail while the encode pass is `O(n)` in row width regardless. It was deleted
rather than left in behind a flag, and `sort_indices` carries a comment saying so, so the next
person does not re-derive it. A bounded heap over encoded rows might close that gap; nothing
has measured it.

### 5. Top-N sort keys evaluated once per morsel instead of twice

**Was:** `parallel_top_n` evaluated each `ORDER BY` expression twice per morsel — once inside
`top_k_indices` to select the local top-`k`, and again to gather the candidates' key columns.

**Now:** the keys are evaluated once and threaded through. `top_k_indices` was split into
`top_k_indices_of`, which takes already-evaluated, already-normalized key arrays, mirroring how
`sort_indices_of` was split out for the same reason. The wrapper was deleted rather than kept,
because after the split nothing called it.

**Why it mattered more than it looks:** for a bare column the second evaluation is an `Arc`
clone, but `normalize_sort_key` calls `canon_float_array`, which **scans the whole column**
looking for `-0.0`/NaN. So a float sort key was scanned twice per morsel, and a computed key
(`ORDER BY a + b`) was computed twice.

**Safe because:** it is the same arrays, passed rather than recomputed. `normalize_sort_key` is
a pure function of the array. The two tests that pin top-N against the eager oracle
(`parallel_top_n_matches_eager`, `parallel_top_n_float_key_matches_eager`) stay green, and the
float one is exactly the case where double normalization was happening.

### 6. Top-N morsel skip on a shared key-range bound

**Source:** DataFusion's `TopK::update_filter`
(`datafusion/physical-plan/src/topk/mod.rs:542`), which republishes the heap's boundary row as a
real predicate through a `DynamicFilterPhysicalExpr` so the scan can prune on it.

**Now:** `bc_runtime::topn::TopNBound` holds a value that `k` rows are known to beat, published
lock-free as each morsel finishes a full candidate set. Before selecting a morsel,
`parallel_top_n` reads its first key's `[min, max]` in one min/max pass; if the whole range is
*strictly* worse than the bound, the morsel is dropped without selecting anything. No plan
rewrite, no scan plumbing, no new operator.

**Safe because:** the bound is the weaker claim "at least `k` rows are at least as good as `v`",
which the max of the first key over any `k` candidates satisfies and which can be read off an
unordered candidate set. A row strictly worse than `v` is worse than each of those `k`, and the
first key dominates the comparison, so it cannot place under any tie-break. Every comparison is
**strict**, because Batcher breaks key ties by original `(morsel, row)` and a `>=` test would
silently drop a tied row that belongs in the answer. Nulls disqualify both publishing and
skipping, since a null has no magnitude to compare. The bound only ever tightens, so a racing
worker reading a stale one skips *less* — which is why it needs no lock.

**Measured, and the measurement changed the design.** 256 morsels x 16,384 rows
(`cargo test --release -p bc-interp topn_bound_tests -- --ignored --nocapture`):

| Key distribution | Morsels skipped | Cost of a skip vs a selection |
|---|---|---|
| Clustered (time-ordered / key-partitioned) | **255 / 256** | 0.009 ms against 0.33 ms — 36x cheaper |
| Uniformly random | **0 / 256** | check switches itself off |

The random result is the important one, and the first version of this shipped nothing to guard
it. A morsel's *minimum* sits at a far smaller quantile than its own `k`-th best, so on random
keys no morsel's range is ever wholly worse than a bound derived that way: the probe excludes
nothing and costs 2.5% of the selection it hoped to replace. So `excludes_range` now watches
itself and latches off after `PROBE_MORSELS = 32` checks that have excluded nothing, capping the
loss at a bounded constant (~0.3%) while leaving the clustered win untouched. That is the same
self-disabling shape `bc_interp::stream::runtime_filter`'s `Gauge` already uses on a runtime join
filter, for the same reason. Pinned by `a_useless_check_switches_itself_off`,
`a_productive_check_never_switches_off` and
`checks_before_a_bound_do_not_count_toward_giving_up`.

### 7. The streaming join morselizes its output

**Source:** not a competitor technique — found by measuring the range-join gap (open item 0) and
noticing that streaming, which is on by default, did not bound its memory.

**Was:** the streaming hash-join probe was `stream.map(|morsel| gather_join_output_with(...))`,
so one input morsel produced exactly one output `RecordBatch` however many rows that was. A join
*multiplies* rows, so this silently made peak memory proportional to `probe_rows x fanout`. It
broke the streaming executor's headline property — "peak memory is the breakers' state plus one
morsel per worker, a constant, independent of the input size" — for every high-fan-out join: a
cartesian, or an ordinary equi-join on a skewed key with many build-side duplicates.

It survived the 46-test streaming oracle because every one of those tests is *correct*. None of
them fans out, and a correctness suite cannot see a memory property. That is the lesson worth
keeping from this entry.

**Now:** the probe emits as many `DEFAULT_MORSEL_ROWS`-bounded morsels as the fan-out needs,
gathering each from a slice of the join indices.

**Safe because:** when the whole result fits one morsel — the overwhelmingly common case,
including every 1:1 foreign-key join — it is gathered from the **unsliced** indices, so
`gather_join_output_with`'s identity-permutation fast path (a full column copy replaced by an
`Arc::clone`) still recognizes them and that path is byte-for-byte unchanged. Slicing would
defeat it even when the slice covers everything, which is why the special case is explicit.
Metrics keep their contract: probe `rows_in` is carried by the first chunk only so chunking
cannot inflate the input count, while `rows_out` accumulates as it should; the seven
`metrics_contract` tests pin that. `peak_bytes` becomes *true* rather than merely defined.

**One thing this cost, worth remembering.** The regression test was first written into
`tests/stream_memory.rs`, which installs a **process-wide counting global allocator** and had
been a deliberately single-test binary. A second test in the same binary runs concurrently and
perturbs those counters, which turned that file's `peak - base` into an arithmetic underflow —
intermittently, in two runs out of three. It presented as a flake in a *pre-existing* test and
was neither pre-existing nor a flake. The fan-out test now lives in its own binary
(`tests/stream_join_fanout.rs`) with the reason written at the top, and both were re-run five
times to confirm.

**Measured:** on a cartesian join over two 20,000-row tables, **13,139 MB → 5,179 MB peak RSS
(2.5x) and 17,104 ms → 10,287 ms (1.66x)**, identical results. The speedup came free with the
memory — the predicate above the join now runs over cache-resident morsels instead of a 10 GB
batch. Pinned by `a_high_fanout_join_emits_bounded_morsels`, which uses a *skewed equi-join*
rather than a cartesian (the general case), and asserts both the morsel cap and row-for-row
equality with the materializing oracle.

### 8. The streaming join sizes its probe from measured fan-out

**Source:** Daft's pluggable batching strategy
(`daft-local-execution/src/dynamic_batching`), pointed at a join's fan-out rather than a scan's
row width.

**Was:** entry 7 morselized the join's *output*, which bounds the gathered batch — the larger
term, since a gathered row carries every output column. It does nothing for the `JoinIndices`,
two `u32` arrays over the *whole* probe result that are built before the first chunk can be
emitted. On the measured cartesian those were the residual 2.6 GB of a 5.2 GB peak.

**Now:** `ProbeSlicer` takes the probe morsel in slices sized so the result stays morsel-scale,
using the fan-out the previous slice actually produced (`rows_out / rows_in`, rounded up). The
first probe of an operator takes `INITIAL_PROBE_ROWS = 256` rather than a full morsel, because
until something has been measured a fanned-out morsel is exactly the unbounded case; a floor of
`MIN_PROBE_SLICE = 64` stops an extreme fan-out from degenerating into one probe call per row.

**Costs nothing on the common case:** a 1:1 foreign-key join observes a fan-out of 1 on its
first 256 rows and opens straight to a full morsel, so the mechanism costs one extra probe call
per operator, once. The slice is not even materialized when it covers the whole morsel, so the
identity-permutation fast path entry 7 preserved is still reached.

**Safe because:** slicing the probe changes only how many rows are handed to `probe()` at a
time. A morsel is a contiguous in-order row range, so a slice of one is too, and the streaming
executor's ordering argument ("probing morsels in order emits what slicing the concatenated
relation would") applies unchanged one level down. A stale fan-out estimate makes a slice the
wrong *size*, never the result wrong. Seven unit tests cover the estimator — the 1:1 case, a
row-dropping semi-join that must not inflate the slice, proportional shrink, the floor, tracking
in both directions, and an empty probe that must teach nothing — plus two `const` assertions
that fail the *build* if the constants are ever set inconsistently.

**Measured**, on the cartesian join over two 20,000-row tables, and the two entries compose:

| | Peak RSS | Time |
|---|---|---|
| Before entry 7 | 13,139 MB | 17,104 ms |
| + output morselization (7) | 5,179 MB | 10,287 ms |
| + probe slicing (8) | **728 MB** | **5,360 ms** |

**18x less memory and 3.2x faster**, identical results. Against DuckDB the range-join ratio goes
from ~13x to ~3.5x. It does **not** fix the algorithm: the join is still quadratic (4x the pairs
is still ~3.6x the time), so this buys headroom rather than removing the wall. See
`competitive_architecture.md` ceiling 7.

### 9. A range-join algorithm — the cartesian product is gone, the gap to DuckDB is not

**Source:** DuckDB `PhysicalIEJoin` (`src/execution/operator/join/physical_iejoin.cpp`),
which implements Khayyat et al., *Lightning Fast and Space Efficient Inequality Joins*
(VLDB 2015).

**Was:** every inequality, interval-containment and band join lowered to a *materialized*
cartesian `HashJoin` on a synthetic `__cross_key` with the predicate as a `Filter` above it.
The intermediate was `|L| x |R|` rows however few survived — quadratic in time and in memory.
This was ceiling 7 in `competitive_architecture.md` and item 0 on this ledger's own open list,
"the largest gap on the board".

**Now:** a real `RangeJoin` operator, end to end:

- `bc_runtime::join::range_join_indices` (`crates/bc-runtime/src/join/range.rs`) produces the
  same `JoinIndices` relation the hash join does, for every join type, by one of two
  output-sensitive algorithms. **One inequality**: sort the right side once; each left row's
  matches are a contiguous *suffix* found by binary search, emitted directly with no per-pair
  predicate evaluation. **Two inequalities**: IEJoin — sort the union of both sides on each
  axis, sweep the second axis with a monotone cursor marking right rows in a bit array indexed
  by first-axis rank, and read each left row's answer off as the set bits in a suffix of that
  array. The bit array carries a one-level summary so an empty 4,096-bit region is skipped in
  one word read.
- `bc_ir::RelOp::RangeJoin` is the new wire node, executed by both the sequential oracle and
  the parallel executor.
- `derive_range_join` (`kyber/rules/joins/range_join.py`) is the rewrite: on a `Filter` over a
  *purely cartesian* inner join it moves up to two crossing inequality conjuncts into the join
  and leaves everything else in the filter.

**Safe because:** the algorithm is pinned against the plan it replaces, not against intuition.
`one_inequality_matches_the_cross_product_oracle` and
`two_inequalities_match_the_cross_product_oracle` fuzz it against a brute-force nested loop over
all four operators (and, for two conditions, all sixteen operator pairs) x all six join types x
null-bearing keys. The rewrite declines whenever declining is right: an equi-conjunct anywhere in
the predicate stands it down entirely (a hash join beats any range algorithm, and
`derive_join_keys` will take that equality), a key pair whose Arrow types differ is left in the
filter, and a computed operand (`a.x + 1 < b.y`) has no key column to sort on.

**A divergence the differential test caught, and it is the reason to write both kinds of test.**
The first implementation excluded NaN keys on IEEE grounds — every comparison with NaN is false.
That is wrong *here*: Batcher's one float-identity contract (`bc_arrow::float_ident`) is a
**total** order in which all NaNs are equal and rank above every number, and the cross-product
path this replaces uses it. The Rust unit tests all passed, because their oracle was integers.
`test_nan_never_matches_and_negative_zero_equals_zero` failed against DuckDB, which agrees with
Batcher and not with IEEE. Recorded because a "safe, conservative" reading of a semantic
produced a silently narrower result set.

**Measured** — interval containment (`pt.x >= iv.lo AND pt.x < iv.hi`, ~10 matches per interval,
both sides `n` rows), release build, identical results at every size, both engines warmed.

Two DuckDB columns, following the convention `benchmarks/TPCH_FINDINGS.md` already set, because
they measure different things and only reporting one of them is how this entry got it wrong the
first time:

- **native** — the data ingested with an untimed `CREATE TABLE`. DuckDB then has real
  statistics and `EXPLAIN` shows `IE_JOIN`. **This is DuckDB at its best and it is the number
  that decides the competitive claim.**
- **arrow** — the same zero-copy Arrow tables Batcher reads. DuckDB's cardinality estimate for a
  registered Arrow table falls below `merge_join_threshold` (default 1,000), so it *declines its
  own IEJoin* and plans `NESTED_LOOP_JOIN`.

| n | Batcher | DuckDB (native, `IE_JOIN`) | DuckDB (arrow, `NESTED_LOOP_JOIN`) |
|---|---|---|---|
| 10,000 | 35 ms | **15 ms** | 492 ms |
| 20,000 | 42 ms | **19 ms** | 1,886 ms |
| 40,000 | 49 ms | **38 ms** | 7,561 ms |
| 100,000 | 78 ms | **78 ms** | 47,503 ms |
| 200,000 | 146 ms | **97 ms** | *(not run)* |
| 500,000 | 251 ms | **138 ms** | *(not run)* |
| 1,000,000 | 476 ms | **274 ms** | *(not run)* |
| 2,000,000 | 1,093 ms | **383 ms** | *(not run)* |

**Batcher loses to DuckDB's IEJoin at every size, by 1.3x to 2.9x, and the gap widens with `n`.**
An earlier draft of this entry claimed 15-643x *faster*; that was measured against the `arrow`
column — DuckDB's nested-loop fallback — and is retracted in full. The scorecard row in
`competitive_architecture.md` has been corrected with it.

What the two columns say, separately:

- **Execution**: DuckDB is ahead and pulling away. From 100,000 to 2,000,000 rows (20x) DuckDB
  takes 4.9x longer and Batcher takes 14x. That is the `~ L x n / 4096` mark-scan term in this
  implementation's cost, and DuckDB does not pay it: `PhysicalIEJoin` decomposes the sorted
  union into blocks and prunes block *pairs* whose key ranges cannot intersect, so its inner
  loop never walks a suffix that holds no answers. That decomposition is the work this entry
  did not do, and it is now the specific, named next step rather than a vague one.
- **Planning**: on the input both engines actually share, Batcher wins by 14x to 600x, because
  DuckDB will not choose its own range join without statistics it only has for its own storage.
  That is a real and reportable property of the pair — a user handing DuckDB an Arrow table gets
  the quadratic plan — but it is DuckDB's *planner* declining, not Batcher's execution winning,
  and it must not be quoted as the latter.

**What did unambiguously improve**, and it is the reason the entry stands at all: the plan is no
longer a materialized cartesian product. Ceiling 7 measured 13.1 GB RSS for `n = 20,000` and "at
`n = 100,000` it does not run"; `n = 2,000,000` now completes in a second. Batcher's own before
and after, same shape, same box:

| n | Batcher before (ceiling 7) | Batcher now |
|---|---|---|
| 10,000 | 3,380 ms | **35 ms** |
| 20,000 | 15,573 ms | **42 ms** |
| 40,000 | 64,249 ms | **49 ms** |
| 100,000 | did not run | **78 ms** |
| 2,000,000 | did not run | **1,093 ms** |

**Where the remaining time goes.** At `n = 10,000` a repeated `collect()` in a warm process takes
**14 ms**, against the 35 ms single-shot above; plan construction is 0.01 ms and optimization
1.4 ms, so the difference is cold-path cost (JIT, allocator, rayon pool), not planning. The
single-shot number is what a user sees and is the one tabled, but the gap at small `n` is
substantially warm-up rather than algorithm.

**Parallelism, and what it was worth.** Both axis sorts run on rayon above 32,768 rows, and the
sweep splits the left rows into contiguous slices of axis-2 order, each worker rebuilding the
mark array for its slice's start with one binary search and one prefix pass. Slices are folded
back in slice order, so the output is *identical* to the sequential sweep's rather than merely
equivalent — `the_parallel_paths_agree_with_an_analytic_answer` pins that against a
single-threaded rayon pool. Together they were worth **1.8x at `n = 100,000`** (141 ms -> 78 ms)
and 1.6-1.8x at the sizes around it.

**What this buys the AI and multimodal side.** Every "which of these spans touches which of
those" question is a range join, and they are the joins a multimodal pipeline is built out of:
aligning ASR spans to detected video scenes, windowing event logs against sessions, IoU blocking
for object-detection dedup, band joins for entity resolution. All of them were the materialized
cartesian product. On the temporal-overlap shape (`seg.s_start < tr.t_end AND seg.s_end >
tr.t_start`, ~0.9 matches per segment, both sides `n` rows) Batcher runs `n = 20,000` in
31 ms, 50,000 in 56 ms and 100,000 in 102 ms — sizes at which the old plan materialized 400
million to 10 billion candidate pairs. Those three figures were taken **before** entry 14's
rewrite, which made the operator 2.4x to 8.1x faster, so they are conservative rather than
current; the point they carry (these shapes went from not running to tens of milliseconds) is
unaffected, and the interval numbers in entry 14 are the ones to quote for speed. A 2-D bounding-box overlap is four inequalities: two go
into the join and two stay in the filter, which is the designed behaviour. Both shapes are pinned
against DuckDB in `test_diff_range_join.py` (`test_temporal_overlap_of_media_segments`,
`test_bounding_box_overlap`). No competitive ratio is quoted for them: they were measured against
DuckDB's Arrow-scan fallback, and the native-storage comparison above is the one that counts.

**The named next step.** The remaining gap to DuckDB is one mechanism: **block decomposition**.
Partition the sorted union into blocks, prune the block *pairs* whose key ranges cannot
intersect, and the `~ L x n / 4096` mark-scan term disappears — which is exactly the term that
makes Batcher's curve steepen past a million rows. It is also what would make the operator
**distributable**, since a distributed range join needs the same "which block pairs can
intersect" pruning to decide what to ship where. Today the distributed planner has no range-join
staging and executes the operator whole, which satisfies single-node == distributed
(`test_distributed_equals_single_node`) without scaling it out. Both are the same piece of work
and are left together rather than half-built.

### 10. The plan cache could hand one query the plan built for another schema

**Source:** found while writing entry 9's tests, not from a competitor.

**Was:** `LogicalPlan.content_key()` fingerprints a plan by its lowered IR, and
`kyber.plan_cache` memoizes optimized plans on that key. But `Scan.to_ir()` is
`{"op": "scan", "source_id": n}` and nothing more — the engine reads column types off the Arrow
batches it is handed, so the schema is deliberately not on the wire. Two runs of one query text
over sources with the same column *names* and different column *types* therefore collided, and
the second run was handed the plan optimized for the first one's types.

Silently, and after every schema-dependent decision had already been made: key-type validation,
cast folding, and — the way it was found — a range join's row encoding. The failing observation
was that `SELECT ... WHERE a.x < b.y` got a `RangeJoin` when run alone and a `Filter` when a
differently-typed pair had been optimized first in the same process.

**Now:** each node contributes an `identity_suffix()` alongside its IR, and `Scan` returns its
schema there. The wire contract is untouched; the key still hits for a genuinely identical plan,
which `test_the_plan_cache_cannot_hand_one_schema_the_other_schemas_plan` pins in both
directions.

### 11. A range join's cardinality is estimated, not defaulted

**Was:** the new node fell through `StatsEstimator.estimate` to `unknown_rows`, so every range
join reported `1e12` — the same number regardless of input size, which is worse than imprecise.
Join ordering and memory sizing above the operator could not tell a ten-row range join from a
ten-million-row one.

**Now:** `|L| x |R| x prod(selectivity_i)`, where each inequality's selectivity is the closed
form `P(X < Y)` for two independent uniform distributions over the columns' measured `[min, max]`
ranges. That is a real integral rather than a constant: a join whose ranges barely overlap is
estimated small and one whose ranges nest is estimated near the full product. System R's 1/3 —
the constant DuckDB, Postgres and Spark all still use — remains the fallback when either range
is unknown.

**Safe because:** the closed form is checked against a 200,000-draw Monte Carlo integration on
three range configurations, and against six hand-computed cases including both point-mass limits.
The result is floored above zero: a zero estimate is a *proof* of emptiness, and a distribution
assumption may not make one.

**Honest limit:** conditions are combined by independence, which is optimistic for exactly the
shape the operator exists for — interval containment, where `lo` and `hi` are strongly
correlated. The estimate runs high there, which errs toward over-provisioning.

### 12. A computed range-join operand is materialized instead of giving up

**Was:** entry 9's rewrite required both operands of an inequality to be bare columns, because
the join sorts key *columns*. `a.ts - 5 < b.ts AND a.ts + 5 > b.ts` — the canonical temporal
proximity join, "events within a window of each other" — has no column to sort, so it stayed
the materialized cartesian product. So did `a.x + 1 < b.y` and every band join written with an
offset.

**Now:** the rule computes such an operand in a hidden column beneath the join
(`_HiddenKeys` in `kyber/rules/joins/range_join.py`) and points the condition at it. The
expression is first rewritten from the join's output aliases into that side's source column
names, and its inferred Arrow type is checked against the other side's key before the condition
is accepted, because the engine encodes both with one row converter.

**Safe because:** it is the same per-row work the filter over the cartesian product was already
doing, on the same rows — as long as the expression cannot *raise*. That caveat is the whole
soundness argument and it is not hypothetical: if the other side is empty the cartesian product
is empty and the filter never runs, so hoisting a division below the join would turn an empty
result into an error. The guard is `_is_push_safe`, **imported** from
`push_projection_through_join` rather than restated — the two rules move a computation onto a
join input for the same reason and must not drift about which operations qualify.
`test_a_raising_computed_operand_is_not_hoisted` pins that `a.ts / 2 < b.ts` still declines, and
`test_hoisting_does_not_leak_the_hidden_column` pins that the hidden key never reaches the
output.

**Measured** — temporal proximity (`a.ts - 50 < b.ts AND a.ts + 50 > b.ts`, ~1 match per row,
both sides `n` rows, identical results):

| n | Batcher | DuckDB (arrow, `NESTED_LOOP_JOIN`) |
|---|---|---|
| 20,000 | 75 ms | 1,117 ms |
| 50,000 | 96 ms | 6,897 ms |
| 100,000 | 141 ms | 27,862 ms |
| 200,000 | 210 ms | 109,559 ms |

**Read the Batcher column only.** Before this entry the same query had no range join at all and
ran the cartesian plan, so what this shows is the difference between running and not. The DuckDB
column is its Arrow-scan `NESTED_LOOP_JOIN` fallback and **no ratio is quoted from it** — see
the correction in entry 9, where DuckDB's native-storage `IE_JOIN` is 1.3-2.9x *faster* than
Batcher on the interval shape.

### 13. A heuristic that sounded right, measured, and discarded

Entry 12 restructured the rule so condition *selection* is a separate step from
materialization, in order to try one idea: a 2-D bounding-box overlap has four crossing
inequalities and IEJoin takes two, so pick one per **dimension** rather than two on the same
one, on the theory that two constraints on one dimension are partly redundant.

It is wrong, and the measurement is the entry. Over 25 million random pairs (boxes of side 80
in a range of 4,000):

| Pair taken into the join | Fraction of pairs reaching the filter above |
|---|---|
| both x conditions (an *axis pair*) | **3.94%** |
| one x + one y (the "per-dimension" pick) | **27.12%** |
| *(true answer)* | 0.156% |

Two conditions on one axis express *overlap on that axis* and select `~2w/R`; a lone
inequality on an axis selects about a half, so the mixed pick is nearly the product of two
coin flips. Seven times the intermediate for a rule that reads as obviously sensible.

Written order is therefore kept, and that is now a *considered* choice rather than an
accident: a user writes an overlap as an adjacent pair of conditions, so written order
preserves the pair. An independence-assuming estimator would have chosen the losing pick too,
because the selectivity of an axis pair comes entirely from the correlation between `lo` and
`hi` that independence cannot see — which is why this is left as written rather than costed.
The selection seam stays, because it is what made the hypothesis testable.

### 14. The range join was slower than DuckDB's everywhere; now it wins below a million rows

Entry 9 landed the operator and the correction beneath it recorded the outcome honestly:
DuckDB's `IE_JOIN` was **1.3-2.9x faster** on interval containment, and the gap widened with
`n`. This entry closes most of it, and the interesting part is that the first two hypotheses
were wrong and the measurement said so each time.

**Hypothesis 1: the mark-array scan.** Each left row walks the axis-1 suffix from its bound to
the end, a `~ L x n / 4096` term. A third summary level on `MarkSet` cuts that 64x, to
`L x n / 262144`. Landed (and fuzzed against a naive bitmap over three level boundaries) —
and it moved a 2,000,000-row join by **1%**, from 1,093 ms to 1,080 ms. The term was not the
bottleneck.

**Then measure instead of guessing.** `report_range_join_phases` is a committed `#[ignore]`d
timing study, per this ledger's convention. It said the two axis **sorts were 69%** of the
time: 409 ms and 349 ms of 1,100 ms. Not the sweep at all.

**What that bought.** Three changes, all aimed at the sorts and at what the sweep does per
comparison:

- **`AxisKeys::Fast`** — an order-preserving `u64` per key value for every primitive type
  (ints, floats, dates, times, timestamps, durations), so a sort comparison is a register
  compare instead of deriving an `arrow::row::Row` and `memcmp`ing it. Null-keyed rows are
  already excluded from the universe, so no null byte is needed and the key fits in 64 bits.
  Floats use the standard total-order transform, which lands exactly on this engine's float
  contract because `canonicalize_float_keys` has already folded `-0.0` and every NaN. Strings,
  decimals and booleans keep the encoder, so nothing is declined for want of a fast path.
- **`dense_ranks`** — equal keys share a rank, so a `u32` compare means exactly what a key
  compare means, ties included. After this the sweep touches no encoded key at all: the
  cursor test and both binary searches are integer compares over flat slices. Produced in the
  *same pass* as the sort order, because computing it afterwards is a random-access gather
  over the key array, which measured as the largest phase after the sorts themselves.
- **`IndexBuf`** for the output instead of `Vec<Option<u32>>` — 4 bytes per row per side
  rather than 8, reusing the hash join's existing buffer and the reasoning already written on
  it ("a 60 M-row join writes 960 MB of scratch").

**Hypothesis 2: worker count.** With 96 cores each sweep worker rebuilds its own mark array, so
the *total* rebuild work is `O(workers x n)`. Replacing it with one sequential walk that
snapshots the bitmap at each slice boundary is `O(n)` — asymptotically better, and **20%
slower** (541 ms -> 653 ms at two million rows), because 96 copies of a half-megabyte bitmap
are serialized where the rebuilds are not. Reverted, with the measurement written into the
code so the next reader does not re-derive it. A worker cap stayed, justified by *memory*
rather than speed: it holds the peak at 8 MB of bitmap instead of 48 MB, at no measurable
cost.

**Measured** — same interval-containment shape, DuckDB in native storage (`IE_JOIN`), best of
three runs each so run-to-run noise does not decide:

| n | Batcher | DuckDB (`IE_JOIN`) | Batcher is | spread over 5 runs |
|---|---|---|---|---|
| 10,000 | 3.7 ms | 9.5 ms | **2.6x faster** | 2.0-2.7x |
| 100,000 | 19.1 ms | 56.4 ms | **3.0x faster** | 2.6-3.0x |
| 500,000 | 87 ms | 132 ms | **1.5x faster** | 1.3-1.7x |
| 1,000,000 | 194 ms | 188 ms | **0.97x — parity** | 0.94-1.12x |
| 2,000,000 | 418 ms | 303 ms | 0.73x | 0.72-0.89x |
| 5,000,000 | 1,493 ms | 656 ms | 0.44x | 0.41-0.50x |
| 10,000,000 | 4,416 ms | 1,966 ms | 0.45x | one run |

The tabled column is the final run, taken once the box was **quiet** — 7 GB of 184 GB in use,
against 73 GB while the neighbouring session's benchmark was running. The spread column is
every run including the contended ones, because a single column would imply a precision these
numbers did not have for most of the session. The shape is stable across all of them:
**Batcher wins below a million rows, ties at a million, and loses above it.**

Against entry 9's numbers the operator is **2.8x to 8.1x faster** than it was, and it now
**beats DuckDB up to a million rows** where it previously lost everywhere. Past that it still
loses, and the gap widens: 0.89x at two million, 0.50x at five, 0.45x at ten.

**Four further hypotheses about the large-`n` gap were tried and none paid.** They are listed
because the pattern is the point — every one was plausible, and the measurement said no:

| Idea | Reasoning | Measured |
|---|---|---|
| A third bitmap level | cuts the suffix-walk term 64x | 1% at 2M |
| Snapshot the mark state once instead of per worker | `O(n)` instead of `O(workers x n)` | **20% slower** — serialized bitmap copies |
| Derive the level count from the size (four levels at 10M) | the term does dominate at 5M | no change |
| Replace the per-row binary search with a lookup table | ~23 random probes into 40 MB, five million times | no change |
| Build each segment's marks in parallel, combine by prefix union | the marks are then set once in total, not `workers` times | **~10% slower** — each segment needs a universe-sized bitmap, and zeroing `workers + 1` of them costs more than the rescan |

Two are kept anyway, on the grounds that both are *simpler* than what they replaced — a
self-tuning level hierarchy instead of a hardcoded depth, and an array read instead of a binary
search — but they are recorded as **performance-neutral**, not as wins. The two that were
slower are reverted, with the measurement written into the code beside the thing they would
have replaced.

**What five failures in a row say.** They are not five independent misses; they are the same
answer arriving five times. Every one of them traded a sequential pass over a flat `u32` array
for something with better asymptotics and worse locality. At five million rows a side the
operator touches roughly eight arrays of 4-40 MB and is **memory-bandwidth-bound across all of
them**, with no single hot spot left to remove — which is exactly the regime DuckDB's block
decomposition is designed for, because it keeps a block pair's working set in cache instead of
streaming the whole universe. That is the structural answer, and after this evidence it is the
*only* remaining one worth trying.

What that leaves: the sorts and the sweep are each roughly a third of the time at five million
rows, and the remaining third is the per-worker mark rebuild. DuckDB's block-pair pruning is
still the structural answer and is still the named next step.

**A measurement caveat that applies to this whole entry.** The box was shared with another
session running a 5-hour benchmark at ~200% CPU. Every number above is best-of-three for both
engines, which is robust to that; the single-shot phase studies are not, and two of the "no
change" rows above sit inside the noise band a competing process creates.

The module outgrew the 800-line Rust limit on the way and was split on its seams rather than
allowlisted: `range/marks.rs` (the bitmap and its levels, a self-contained structure with its
own invariant), `range/keys.rs` (sortable key forms and dense ranking), `range/mod.rs` (the
operator). Pure code movement — clippy, the fuzz oracle and the differential suite all
unchanged across it.

### 15. What actually caps a short query is the optimizer, not the operator

Found while chasing the small-`n` end of entry 14, and it is worth more than the join work
because it applies to every query rather than one operator.

At `n = 10,000`, median over 12 never-seen query shapes, DuckDB in native storage:

| Query shape | Batcher | DuckDB | optimizer runs |
|---|---|---|---|
| `count(*)` over a filter | 25.8 ms | 10.0 ms | **2** |
| `sum()` over a filter | 17.8 ms | 9.2 ms | 1 |
| plain projection, no aggregate | 18.1 ms | 11.2 ms | 1 |
| filter only, one table | **9.2 ms** | **2.0 ms** | 1 |

A *repeated* shape hits `kyber.plan_cache`, costs 0.12 ms of planning, and the same range-join
query then runs in 4.5 ms against DuckDB's 9.4 ms — **2x faster**. The entire gap is cold.

**One optimizer pass is 5-8 ms**, and per-phase timing says where: of a 5.34 ms optimize on the
simplest query, the phase carrying **282 rules is 4.19 ms — 78%**; the other six phases total
0.19 ms. The driver already fuses node-local rules into one traversal and leaf `Expr -> Expr`
rules into one expression walk within it. What is left is that all 282 leaf rewrites are
offered *every expression node*, each opening with its own `isinstance` — about 2,800 Python
calls for a two-predicate filter.

**Four fixes were tried; none paid, and that is the entry.**

| Attempt | Reasoning | Measured |
|---|---|---|
| Collapse `optimize_full` / `optimize_logical` into one memo entry | they looked like the same rewrite keyed twice | no change — a stack trace showed they optimize *different plans*, the metadata layer running on a sub-plan |
| Cache the eligible-leaf list per node type | it is rebuilt from every rule at every plan node | no change (~0.1 ms of 5.3) |
| Memoize the whole fused expression pass per node | the node rules' own `noop` argument (pure rules, fixed `ctx`, immutable nodes) applies to it verbatim | 3% (5.32 -> 5.15 ms) |
| Index the node rules by node type per pass | ~14,000 `matches` tests a query before any rule fires | **slower** (5.44 ms) |

All four reverted. The first also made every `Facts` accessor pay physical wrapping it does not
need, which is a regression rather than a wash.

**Instrumentation, rather than a fifth guess, is what settles it.** The fused expression pass
runs **4 times per query** with 427 leaf slots in total — so the 94 leaf rewrites are not where
the time is. The 188 *node* rules are, spread over **19 plan traversals per query**, with
nothing dominant inside them.

That makes it an **architectural** cost, not a hot spot: 282 rules over 7 phases, iterated to a
fixpoint, in Python, for a plan of four nodes. The candidates are structural — converge
detection per rule *family* instead of per phase, so a settled family stops being re-offered;
or moving the fixpoint loop out of Python entirely. Neither is a tuning exercise, and this is
now `competitive_architecture.md` ceiling 8, the largest single latency item on the board.

**A narrower, separate cost:** `count(*)` over a filter runs the optimizer *twice* — the
metadata-answer layer optimizes the aggregate's input to test whether the surviving count is
derivable from statistics, then execution optimizes the root. That is the only shape that does
it, worth ~8 ms, and the fix is sequencing in `api/terminal` rather than anything in Kyber.

### 16. An inequality-correlated `EXISTS` no longer raises

**Was:** `WHERE EXISTS (SELECT 1 FROM b WHERE a.x < b.y)` raised
`NotImplementedError: correlated subqueries not supported`. Only *equality*-correlated
`EXISTS` decorrelated — an inequality is neither a correlation key nor a local predicate, so
it fell through to `_reject_correlated` and the query had no plan at all. DuckDB answers it.
This is a "situation" where Batcher did not merely lose; it declined.

**Now:** it decorrelates to a range **semi** join (anti for `NOT EXISTS`), which is exactly
what it means — for one *or two* correlations, so
``EXISTS (SELECT 1 FROM b WHERE b.y > a.lo AND b.y < a.hi)``, the `EXISTS` spelling of interval
containment, works too. Two is the engine's ceiling; a third inequality declines the whole
shape rather than dropping one, because a semi join emits no right columns and so cannot carry
a residual filter. `range_semi_join` (`api/_join_helpers.py`) builds the `RangeJoin`;
`_inequality_correlation` (`_sql/parser/subquery.py`) recognizes the shape, keeps every other
predicate as the inner relation's own filter, and projects the subquery down to the joined
column — the same treatment the equality path already gives its correlation keys.
`_reject_correlated` still runs afterwards, so any *other* outer reference is declined as
before rather than silently mis-planned.

**Cheap because the engine was already there.** `bc_ir::RelOp::RangeJoin` carries a
`join_type`, and `range_join_indices` implements `Semi` and `Anti` and is fuzzed against the
brute-force cross-product oracle for both — this is simply the first caller to emit a
`RangeJoin` that is not an inner join. The control-plane side is ~90 lines.

**Safe because** the semantics are pinned against DuckDB across all four comparison
operators, both operand orders, `EXISTS` and `NOT EXISTS`, one and two correlations, an inner
local predicate alongside, and null-bearing keys on both sides — 21 parametrized differential
cases. One of them exists specifically to rule out the
tempting shortcut: `test_correlated_exists_preserves_outer_duplicates` pins that three
identical qualifying outer rows survive as three, which a cross-join-filter-`DISTINCT`
rewrite over the existing operators would have collapsed to one.

**Verified:** 55 differential tests on the range-join surface and **789 passing SQL /
subquery / correlated differential tests**, with the five import-linter layer contracts kept
(`_sql` is layer 6 and may call `api`, which is layer 5 — the direction this uses).

**A structural note, because the split is load-bearing.** The addition pushed
`_sql/parser/` past the 12-files-per-directory limit and `subquery.py` past 500 lines. Rather
than allowlist either, the three subquery modules were package-ized into
`_sql/parser/subquery/` (`core`, `neq`, `range`) — they were already one responsibility
family, which is exactly the case the maintainability rule says to group. The public import
path `batcher._sql.parser.subquery` is unchanged.

That move broke **725 tests** on its first attempt, and the cause is worth recording: the
translator reaches several helpers as *attributes of the module*
(`subquery._decorrelate_scalar_subqueries`), not through `from ... import`, so a package
facade that re-exports only the obviously-public names is not a drop-in. `__init__` now
re-exports every module-level name `core` defines. A package-ization that "only moves code"
is exactly where this bites, and the 783-test subquery suite is what caught it.

### 17. What is still refused, found by probing rather than by guessing

Entry 16 closed one shape. A sweep of ten related SQL constructs against DuckDB says which
others Batcher declines, and — as importantly — which now work:

**Work, matching DuckDB row for row:** a range join under `GROUP BY`, under `LEFT JOIN`, as a
self-join, with a third inequality left as a filter, and beneath a `QUALIFY`/`row_number`
window. The operator composes with the rest of the engine without special cases, which is the
property that matters and is not obvious from the operator's own tests.

**Still refused** (each raises where DuckDB answers), in rough order of how tractable they look:

1. **`EXISTS` with an equality *and* an inequality correlation** (`b.g = a.g AND b.y > a.x`) —
   a band join. Not reachable through `RangeJoin`, which has no equality axis; it needs a hash
   semi join carrying a residual, and a semi join cannot carry one.
2. **Two inequality correlations where one operand is computed** (`b.y < a.x + 10`). The
   two-correlation case landed in entry 16 for bare columns; a computed *outer* operand needs
   hoisting into the outer relation as a hidden column, then excluding it from the join output.
3. **Inequality-correlated `IN`** — the same decorrelation as `EXISTS`, applied at the `IN`
   site.
4. **Inequality-correlated scalar subquery** (`(SELECT max(y) FROM b WHERE b.y > a.x)`) — a
   per-outer-row aggregate over a range join.
5. **`LATERAL` over a correlated subquery with a `FROM`** — supported today only for a
   per-row compute with no `FROM`.

Recorded as a list rather than fixed, because each is a distinct decorrelation and entry 16's
was the one the engine was already equipped for.

## Verification state

Entries 1-8 landed together and were verified against a settled tree:

- **Rust workspace: 1247 tests, 0 failures.** Includes the 46-test streaming oracle
  (streaming == materializing per operator), the `seq == par == JIT` differential tests, and
  the two `parallel_top_n_*_matches_eager` tests that pin top-N against the eager sort.
- **Clippy clean across the whole workspace** (`--all-targets -D warnings`), `cargo fmt` clean.
- **DuckDB differential suite: 5550 passed, 0 failed**, the whole of `tests/differential/`
  including the distributed operator matrix. This is the number that matters: it is the
  correctness spine, and every change above is on a path it exercises.

Entries 9-17 landed together and were verified the same way. (The parallel sorts and the
parallel sweep are folded into entry 9 rather than counted separately, even though each is a
verified behaviour change with a number behind it — they are one operator, and this ledger's
counting rule exists to stop a "make N improvements" task inflating itself.)

- **Rust workspace green**, clippy clean `--all-targets -D warnings`, `cargo fmt` clean,
  `lint-guardrails` clean, all five import-linter layer contracts kept.
- **`tests/unit` + `tests/differential`: 13,948 passed, 4 skipped, 0 failed** (8,100 + 5,848,
  run as two halves — see the caveat below) — the correctness spine, including the 27
  pre-existing theta-join differential tests that now run through the new operator without a
  single change to what they assert. Re-run on the settled tree after every rewrite in
  entries 14 and 16, not once at the start.
- **A caveat on that number, because it is the one that matters.** The tree and the box were
  shared with another session throughout, and full-suite runs kept dying part-way. Two
  distinct causes, both environmental and both worth naming:

  - It rebuilt `_native.abi3.so` while a suite was running — the hazard
    `.claude/rules/concurrent-agents.md` documents, which pulls the mapped pages out from
    under every process that has imported the engine.
  - More often, the **kernel OOM-killed the run**. `dmesg` records a `python3` reaped at
    138 GB anon-rss in this cgroup, and that session's benchmark was holding 73 GB of the
    184 GB limit while a suite needed the rest. The suite died at ~67% every time, which is
    where its own peak lands.

  Runs killed by either cause are *no result*, not a bad one — the distinction
  `concurrent-agents.md` insists on. Splitting the run (`tests/unit`, then
  `tests/differential`) halves its peak and fits alongside a memory-hungry neighbour, which is
  how the 13,948 above was finally obtained: 8,100 + 5,848, both green.
- The new coverage is **10 Rust unit tests** (two fuzzing the algorithm against a brute-force
  cross-product oracle across every operator x join-type combination, one fuzzing the mark
  bitmap against a naive one across four level boundaries, one pinning the parallel sweep
  byte-for-byte against a single-threaded rayon pool) and **53 Python tests** — 34 differential
  against DuckDB, 19 plan-shape/estimator — plus a committed `#[ignore]`d phase-timing study.
- `MAP.md` current, `lint-structure`/`lint-guardrails`/`lint-duplication` clean, `ruff` clean,
  all five import-linter contracts kept, `cargo fmt` clean. `range.rs` outgrew the Rust size
  limit during entry 14 and was **split into a package on its seams** (`marks`, `keys`, the
  operator) rather than allowlisted.
- `tools/surface_snapshot.py` shows exactly the boundary change intended and no other:
  `ir_tags` gains `RANGE_JOIN=range_join`, `kyber_rules` gains `derive_range_join` (455 ->
  456), and **`public_api` and `native_ffi` are unchanged** — the operator rides the existing
  `execute_plan` FFI and adds no user-facing name.

Two conventions worth keeping if this ledger grows. The timing studies behind the
measurements are committed `#[ignore]`d tests with the command in their doc comment, not
remembered numbers — `report_the_row_encoding_crossover` and `report_the_top_n_skip_saving`.
And where a measurement contradicted the design, the ledger says so (entries 4 and 6 both
record what was measured, discarded, and why; entry 7 records a property the test suite could
not see), because the alternative is a ledger that only
records successes and therefore cannot be trusted about them.

## Open, in dependency order

Carried from `competitor_technique_review.md`, with what this pass learned added.

0. **A range join that beats DuckDB's.** Entry 9 landed the operator and removed the quadratic
   plan, but DuckDB's `PhysicalIEJoin` is still **1.3-2.9x faster and pulling away**. One
   mechanism closes it: **block decomposition** — partition the sorted union into blocks and
   prune the block pairs whose key ranges cannot intersect, which removes the `~ L x n / 4096`
   mark-scan term that makes Batcher's curve steepen past a million rows. The same pruning is
   what a *distributed* range join needs to decide what to ship where, so this is one piece of
   work, not two. It is the largest remaining item on this list.
1. **`StringView` end to end** (DuckDB `string_t`, Polars). Still the largest single-node
   item and still the most invasive: the representation has to survive scan, project, join
   key construction and sort to be worth having, and a half-adoption that converts at every
   boundary would be slower than what exists. arrow-rs 56 is already pinned and ships the
   kernels.
2. **Dictionary survival past the leaf.** `decode_dict`
   (`crates/bc-expr/src/eval/dispatch.rs:44`) still casts at the `Col` leaf; only `InList`
   and `try_dict_compare` keep the dictionary. Project, join keys and sort do not.
   Compaction now preserves a dictionary through a gather
   (`a_dictionary_column_stays_a_dictionary_through_compaction`), so the filter half of
   this is done.
3. **Top-K heap threshold as a dynamic filter** (DataFusion `TopK::update_filter`).
   Batcher's quickselect top-N is already late-materialized; what is missing is the
   feedback edge to the scan. `stream/runtime_filter.rs` is the transport it should reuse.
4. **Skew salt derived from measured partition sizes** (Spark `OptimizeSkewedJoin`:
   `median * factor`, floored at an absolute threshold, split toward the mean of the
   non-skewed partitions). Batcher has more learning machinery and less default behaviour:
   `skew_join_salt` defaults to 0.
5. **Adaptive morsel sizing** (Daft `BatchingStrategy`). Lowest value of the set; a latency
   story more than a throughput one.
