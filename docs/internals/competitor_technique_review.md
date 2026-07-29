# Competitor technique review: what to take from DuckDB, Polars, DataFusion, Spark, Daft and Ray Data

**Status:** review, 2026-07-24. Every competitor claim below cites a file that was read in
this pass. Every claim about Batcher was checked against Batcher's code, not its docs.

`competitive_architecture.md` is the *scorecard*: where Batcher wins and loses, and why.
This document is the *parts list* behind it. It answers a narrower question: given the
competitors' source, which specific mechanisms does Batcher not have, and which of them are
worth building.

## Method and its limits

Read from `/mnt/shared_storage/ref`: DuckDB (C++), Polars, DataFusion, Spark (Scala),
Daft, plus Apache Arrow (the C++ tree; Batcher's Arrow is arrow-rs 56, pinned in
`crates/bc-arrow`). Ray Data was read from the installed `ray.data` package.

Paths in this document are relative to the competitor's own tree unless they begin with
`crates/bc-` or `python/batcher/`, which are Batcher's.

Two limits worth stating, because they bound how much weight these findings carry:

1. The trees are large and this pass targeted the mechanisms the scorecard already names as
   Batcher's open gaps, plus the execution-layer machinery around them. It is not an
   exhaustive read of six engines. Where a technique was identified from a file inventory
   rather than read in depth, the table below says so.
2. "Batcher does not have this" means a grep of `crates/` and `python/batcher/` found
   nothing, and the relevant code path was read to confirm. Those greps are recorded
   inline so they can be re-run.

## The shortlist

Ranked by value against the mandate, with the cheapest genuine win first.

| # | Technique | Best source | Batcher today | Value |
|---|---|---|---|---|
| 1 | Short-circuiting conjunctive filter, conjuncts ordered by cost | DuckDB | **Landed this pass** | 1.3x to 5.7x on multi-predicate filters |
| 2 | German strings (`StringView`) end to end | DuckDB, Polars | Absent entirely | The other half of the single-node gap |
| 3 | Online adaptive reordering of filter conjuncts | DuckDB | Absent | Fixes the case a static cost model gets wrong |
| 4 | Top-K heap threshold pushed down as a filter | DataFusion | Absent (heap itself is good) | Large on `ORDER BY ... LIMIT k` over big inputs |
| 5 | Skew detected from measured partition sizes, split automatically | Spark AQE | Machinery exists, off by default | Removes a config the user cannot be expected to set |
| 6 | Dictionary encoding surviving past the leaf | DuckDB, Arrow | Comparisons only | Compounds with 2 |
| 7 | Adaptive morsel sizing as a pluggable strategy | Daft | Fixed 16,384 rows | Small, and mostly a latency story |
| 8 | A range-join algorithm (IEJoin, or a binned rewrite) | DuckDB | Cartesian product + filter | **Largest single gap found**: 12–32x, and OOMs where DuckDB runs |

Items 1 and 3 are two halves of the same DuckDB mechanism. Item 1 is the part that is
provably safe without cross-morsel state, which is why it went first.

## 1. Short-circuiting conjunctive filter (landed)

**What DuckDB does.** `ExpressionExecutor::Select`
(`src/execution/expression_executor/execute_conjunction.cpp:61`) walks the conjuncts of an
`AND` against a selection vector. Conjunct `n + 1` is evaluated only at the positions
conjunct `n` kept, and the walk order comes from an `AdaptiveFilter` permutation whose
opening value is a static cost heuristic (`src/optimizer/expression_heuristics.cpp`).

That heuristic is worth reading, because it is somebody else's calibration and it is free.
A leaf costs a multiplier times a per-type weight, and the weights are 1 by default, 2 for
float or double, 5 for varchar: a column reference uses multiplier 8, a constant 1
(`ExpressionCost(PhysicalType, idx_t)`, `:173`). A comparison is
`Cost(left) + 5 + Cost(right)`. `IS NULL` adds 5, `NOT` adds 10, and `IN` adds
`(n - 1) * 100`. Scalar functions come from a lookup table: `+` and `-` are 5, `*` 10,
`/` 15, `round` 100, and `LIKE`, `regexp_matches` and `||` are 200, with **1000** for any
function not in the table. The shape of it is what matters: comparisons are near-free, a
regex is two orders of magnitude past them, and an unrecognized function is assumed worse
than a regex.

Batcher's `Expr::eval_cost` follows that shape with smaller numbers, and departs from it
in one place deliberately. DuckDB charges `IN` per element because it lowers to a
comparison chain; Batcher's `InList` is hash-set membership and is dictionary-native, so
its cost is capped rather than linear in the set size. One refinement not taken: DuckDB
weights a leaf by its *type*, so a varchar comparison sorts behind an integer one.
`eval_cost` is type-blind because it does not take a schema, and adding one for a
second-order ordering effect was not worth the signature.

**What Batcher did.** `ops::filter_batch_jit` (`crates/bc-interp/src/ops/mod.rs`) evaluated
the whole predicate tree with `Expr::eval`, so `and_kleene` composed a full-width mask per
conjunct. A five-conjunct filter that kept 1.65% of its rows paid five full-width passes.
A grep for `SelectionVector` or `selection_vector` across `crates/` returned nothing.

**What Batcher does now.** `Expr::short_circuit_filter_mask`
(`crates/bc-expr/src/select.rs`) evaluates the cheapest conjunct at full width, then
*compacts*: it gathers the surviving rows of only the columns the remaining conjuncts name,
and evaluates the rest against that narrower and shorter batch. Arrow has no selection
vector, so compaction is the equivalent move, and gathering two named columns rather than
the seventeen a fact table carries is what makes it pay.

The measured effect, from
`cargo test --release -p bc-expr --test short_circuit_filter -- --nocapture` over 64
morsels of a lineitem shape (1,048,576 rows, 17 columns):

| Predicate | Rows kept | Mask | Mask plus gather |
|---|---|---|---|
| TPC-H q6 (5 cheap conjuncts) | 1.65% | 7.83 to 4.12 ms (1.90x) | 11.57 to 7.70 ms (1.50x) |
| `IN` list plus 2 ranges | 2.87% | 7.94 to 6.20 ms (1.28x) | 13.96 to 12.37 ms (1.13x) |
| Cheap guard plus `contains` | 1.14% | 14.03 to 3.18 ms (4.41x) | 16.58 to 5.70 ms (2.91x) |
| Cheap guard plus `LIKE` | 1.14% | 8.58 to 2.94 ms (2.92x) | 11.14 to 5.49 ms (2.03x) |
| Cheap guard plus `regexp_matches` | 1.14% | 38.89 to 4.44 ms (8.75x) | 42.26 to 7.38 ms (5.73x) |

Quote the "mask plus gather" column. It is the whole Filter operator, gather included, and
the gather is work both paths do. The mask column is the part this change actually alters
and is the more flattering of the two.

### Why it is not a semantic change

The interpreter is the correctness oracle, so a filter that returns a different mask is a
wrong answer, not an optimization. Three properties carry the equivalence, and they are
written out in the module docs so a future change has to argue with them:

- **Composition.** `filter_record_batch` keeps a row when the mask is valid and true, so a
  predicate's nulls are already indistinguishable from false. ANDing masks that have had
  their nulls folded into false gives the identical keep set for any nesting or ordering of
  the `AND`s.
- **Position independence.** Every kernel reachable from an eligible conjunct is
  elementwise, so evaluating it over a gathered subset writes, at each surviving position,
  what the full-width pass would have written.
- **Skipping cannot hide an error.** This is the part that needs a guard.
  `Expr::is_infallible_predicate` admits only conjuncts whose failures are *schema*-driven,
  never *row*-driven: comparisons, boolean combinators, null tests, `IN` lists, `TRY_CAST`,
  and the six boolean-returning string predicates. Checked arithmetic overflows on one row
  and not its neighbour; `Div` divides by zero on one row; a strict `CAST` rejects one
  string. Those are excluded, and their predicates take the whole-batch path unchanged.

Two details keep that argument airtight rather than nearly so. The surviving set is never
compacted to zero rows, so a conjunct whose type error needs a non-empty input still gets
one. And any error at all abandons the fast path, so the caller re-evaluates the predicate
as written and raises exactly the error, with exactly the message, it always did.

The string-predicate case is worth dwelling on, because it is where the ordering rather
than the skipping is what pays, and it is also where the safety argument is subtlest.
`Contains`, `StartsWith`, `EndsWith`, `Like`, `Ilike` and `RegexpMatches` return a boolean,
so they never reach `try_map_str`, whose "result exceeds the maximum string length" is a
real per-row failure of every string-*producing* function
(`crates/bc-expr/src/eval/str/mod.rs:1021`). And the input must be a column that is already
UTF-8, because evaluation casts a `Binary` input to `Utf8` and *that* rejects one row's
bytes at a time. That is why the classifier takes the schema: the identical expression is
safe over `Utf8` and unsafe over `Binary`.

### Where it applies, and where the JIT keeps precedence

Short-circuiting is gated on `jit.is_none()`. A fully compiled predicate is already a single
fused pass with no intermediate mask to save, so splitting it would trade the better fast
path for the lesser one, and `bc-codegen` does compile `And`, `Or` and the comparisons
(`emit.rs:145`).

That gate matters less than it looks, because the **default** executor never passes a JIT
here. `bc-interp/src/stream/mod.rs:272` records why, and it was measured rather than
assumed: wiring the JIT into the streaming Filter and Project path was tried and came out at
1.01x over TPC-H with five queries slower, because Arrow's compare and boolean kernels are
already SIMD and a scalar Cranelift loop has nothing to win on a predicate. So the streaming
path, which is the default, takes the short-circuit unconditionally; `par.rs` takes it
whenever the JIT declined the predicate, which is every predicate touching a string.

## 2. German strings

DuckDB's `string_t` (`src/include/duckdb/common/types/string_type.hpp:24`) is a 16-byte
value: a 4-byte length, then either 12 inline bytes or a 4-byte prefix plus a pointer. Short
strings never allocate, and a comparison that differs in the first four bytes never
dereferences. Polars uses the same representation.

Batcher has none of it. A grep for `Utf8View`, `StringView`, `BinaryView` or `ByteView`
across `crates/` and `python/batcher/` returns zero hits. This is not a missing
dependency: arrow-rs 56 is already pinned in `crates/bc-arrow` and ships
`StringViewArray` with comparison, `take` and `filter` kernels.

This remains the largest single-node item, exactly as `competitive_architecture.md` roadmap
step 2 says. It is also the most invasive, because the representation has to survive scan,
project, join key construction and sort to be worth having, and a half-adoption that
converts at every boundary would be slower than what exists now. It was not attempted in
this pass for that reason, not because it is lower value than what was.

## 3. Online adaptive conjunct reordering

DuckDB's `AdaptiveFilter::AdaptRuntimeStatistics`
(`src/execution/adaptive_filter.cpp:107`) is a randomized hill-climb over the conjunct
permutation, run on measured time. After a five-iteration warmup it alternates: for
`execute_interval` (20) iterations it records the mean runtime of the current order, then
swaps a randomly chosen adjacent pair and observes for `observe_interval` (10) iterations.
If the mean improved it keeps the swap and resets that index's swap likeliness to 100; if
not it reverses the swap and halves the likeliness, never below 1, so a position that keeps
losing stops being tried without being ruled out forever. Reordering is disabled outright
when any conjunct `CanThrow()`.

This is the part of the mechanism that fixes what a static cost model gets wrong: cost is
not selectivity, and a cheap unselective predicate ordered ahead of an expensive selective
one is the worst case for item 1. It is also a genuine fit for Batcher's stated moat, which
is adaptation on *measured* quantities, and there is precedent for it in the Rust data
plane already: `bc-interp/src/agg_par.rs` switches aggregate strategy on a measured
reduction ratio.

What it needs that item 1 did not: state that outlives a morsel. `filter_batch_jit` is
called per morsel from five sites (`crates/bc-interp/src/lib.rs:135`,
`crates/bc-interp/src/par.rs:532` and `:1934`, `crates/bc-interp/src/stream/mod.rs:285`,
`crates/bc-interp/src/stream/parallel.rs:501`) with no per-operator slot to hang a
permutation on. The `Jit` value threaded through `par.rs` is the natural place to put one,
since it is already compiled once per operator and shared across rayon workers. That is the
next increment.

Note that DuckDB's `CanThrow()` guard and Batcher's `is_infallible_predicate` are the same
idea, arrived at for the same reason. Batcher's is stricter, because Batcher reorders and
skips where DuckDB skips unconditionally and only guards the reorder.

## 4. Top-K threshold as a dynamic filter

DataFusion's `TopK::update_filter` (`datafusion/physical-plan/src/topk/mod.rs:542`) turns
the heap's boundary row into a real predicate and republishes it through a
`DynamicFilterPhysicalExpr` that the scan re-reads. For `ORDER BY a DESC, b ASC LIMIT 3`
with heap `[(1,5), (1,4), (2,3)]` the filter becomes
`(a > 1 OR (a = 1 AND b < 5)) AND ...`, built with explicit `nulls_first` handling in
`build_filter_expression` (`:596`). Later batches are then skipped before they reach the
heap, and a Parquet scan can prune row groups on it.

Batcher's top-N is already good on its own terms: `ops::parallel_top_n` and
`top_k_indices` (`crates/bc-interp/src/ops/mod.rs:591`) select rather than sort, using a
quickselect, and materialize late, with the tie-break pinned against the eager oracle by
`parallel_top_n_matches_eager` and `parallel_top_n_float_key_matches_eager`. What is absent
is the feedback edge: nothing derived from the heap reaches the scan.

Batcher has the delivery mechanism for this already, which is what makes it attractive.
`crates/bc-interp/src/stream/runtime_filter.rs` sinks a join's build-side key set down the
probe pipeline to the scan, with a keep-rate gauge that switches a useless filter off. A
top-K threshold is the same shape of object travelling the same path. The correctness care
is in ties and nulls, and DataFusion's `build_filter_expression` is the reference for
getting the `nulls_first` and equal-value cases right.

## 5. Skew detected from measured sizes

Spark's `OptimizeSkewedJoin` calls a shuffle partition skewed when its size exceeds both
`median * SKEW_JOIN_SKEWED_PARTITION_FACTOR` and an absolute threshold
(`getSkewThreshold`, `OptimizeSkewedJoin.scala:65`), then splits it toward a target size
that is the mean of the *non-skewed* partitions, floored at the advisory partition size
(`targetSize`, `:75`). It knows which sides may be split per join type
(`canSplitLeftSide`, `canSplitRightSide`). None of this asks the user anything.

Batcher has more learning machinery than Spark here and less default behaviour.
`python/batcher/dist/skew.py` persists measured hot join keys so a repeated shape salts
without re-running detection, and distinguishes "measured, not skewed" from "never
measured". But `skew_join_salt` defaults to `0`
(`python/batcher/config/config.py:998`), and the learned path is only consulted when it is
positive. So out of the box a skewed distributed join is not mitigated, while a config
field suggests otherwise.

The fix is not new machinery, it is a default: derive the salt from the measured partition
size distribution the way Spark derives its threshold, using the sketches Kyber already
consumes. This is the same finding `competitive_architecture.md` ceiling 6 records; what
this pass adds is Spark's actual formula.

## 6. Dictionary survival past the leaf

`decode_dict` (`crates/bc-expr/src/eval/dispatch.rs:44`) casts a `Dictionary` column to its
value type at the `Col` leaf, so no downstream kernel sees a dictionary. Two exceptions
exist and both are recent wins: `InList` and `try_dict_compare` compare the distinct values
and gather through the keys. Filters and group-by are dictionary-native; project, join keys
and sort are not. This compounds with item 2 rather than competing with it.

## 7. Adaptive morsel sizing

Daft's local executor makes batch sizing a strategy trait
(`src/daft-local-execution/src/dynamic_batching/mod.rs`) with three implementations,
including a latency-constrained one, recomputed from execution metrics after each worker
completes. Batcher's morsel is 16,384 rows with a `morsel_target()` knob. This was
identified from the module's trait definition and its three strategy filenames, not from a
full read of the strategies. Listed for completeness; it is a latency story more than a
throughput one, and it is the lowest-value item here.

## 8. A range-join algorithm (landed)

This one was not in the original shortlist because the earlier pass did not look at joins whose
condition is not an equality. It is the largest gap the review has turned up.

**What Batcher does.** `bc_ir::RelOp` has two join nodes, `HashJoin` and `AsofJoin`. Every
inequality, interval-containment and band join is lowered to a cartesian `HashJoin` on a
synthetic `__cross_key` with the predicate as a `Filter` above it. That is deliberate and its
*correctness* is well covered — `tests/differential/test_diff_theta_join.py` has twelve tests
and its docstring says so explicitly.

**Why it is nonetheless a ceiling.** The cross product is materialized before the filter runs,
so the intermediate is `|L| x |R|` however few rows survive. Measured against DuckDB on
`pt.x >= iv.lo AND pt.x < iv.hi` with about ten matches per interval, identical results
throughout: 32.1x at n=5,000, 13.7x at n=20,000, 13.2x at n=40,000 — quadratic in `n`, and a
fresh process peaks at **13.1 GB RSS** for n=20,000 on ~500 KB of input. At n=100,000 it does
not run. Full numbers in `competitive_architecture.md` ceiling 7.

**What DuckDB has.** `IEJoin` (Khayyat et al.): sort both sides on the two inequality
attributes, sweep with a bit array, `O(n log n)`. That is the general answer and it is a
substantial operator.

**Landed as `RelOp::RangeJoin`** — see `engine_improvements_ledger.md` entry 9 and
`competitive_architecture.md` ceiling 7. The IEJoin option below was taken over the binned one:
the binned rewrite turns on a bucket width `W` nobody could choose safely at plan time, and a
badly chosen `W` is a pessimization rather than a win.

**The result is mixed and the ledger says so.** The quadratic *plan* is gone — `n = 2,000,000`
went from not running at all to 1.1 s — but DuckDB's own IEJoin is still **1.3-2.9x faster**, and
the gap widens with `n`, because `PhysicalIEJoin` decomposes the sorted union into blocks and
prunes block pairs by range where this implementation walks a suffix of a bit array. An earlier
draft of this note claimed 15-643x in Batcher's favour; that was measured against DuckDB's
`NESTED_LOOP_JOIN` fallback over an Arrow scan and is retracted. The rest of this item is kept
as the record of what was considered.

**The cheaper option that was not taken, and why it looked like it suited Batcher.** A *binned* range join is a plan
rewrite over operators that already exist. Bucket each point by `floor(x / W)`; expand each
interval across the buckets it spans with `sequence` plus `Unnest`; equi-join on the bucket;
re-apply the original predicate to drop false positives. Properties worth noting:

- **No duplicates, structurally.** A point has exactly one bucket, so a surviving pair can be
  produced at most once. That is what makes the rewrite safe without a distinct.
- **No misses.** For integer `x`, `lo <= x < hi` implies
  `floor(lo/W) <= floor(x/W) <= floor((hi-1)/W)`, so a matching pair always shares a bucket.
- **Nothing new to build.** It lowers to a hash join, so it inherits streaming, spill and
  distribution unchanged — which is exactly the property a new physical operator would have to
  re-earn.

The hard part is `W`. Too small and the interval side replicates explosively; Databricks makes
it a user-supplied hint (`RANGE_JOIN`) for precisely this reason. Batcher has a better answer
available in principle — the measured-cardinality adaptive layer could set it at the stage
boundary rather than guessing at plan time — but that is the part that needs designing, and a
rule that fires with a badly chosen `W` would be a pessimization rather than a win. It was
therefore recorded rather than half-built.

## Things Batcher already has, so do not "add" them

Recorded because each is a technique a competitor is known for, and each is easy to
mistake for a gap:

- **Runtime filters / sideways information passing.** DataFusion and Spark both do this.
  `crates/bc-interp/src/stream/runtime_filter.rs` does it, with a soundness argument
  restricting it to `Inner` and `Semi` probe sides, and a per-filter gauge that disables a
  filter whose keep-rate shows it is not earning its cost.
- **Late-materialized top-N by quickselect.** See item 4.
- **Morsel-driven parallelism with a streaming pipeline.** Polars' newer engine
  (`polars/crates/polars-stream/`) and Daft's swordfish are both this shape.
  `crates/bc-interp/src/stream/` is Batcher's, and it is the default.
- **Credit-based backpressure.** Ray Data has backpressure policies and a
  throughput-based resource allocator (`ray/data/_internal/execution/`, identified by file
  inventory). Batcher's `bc-transport` credit scheme with a proven in-flight bound is at
  least as strong, and it bypasses the object store, which is the more important
  difference.
- **Adaptive aggregate strategy on measured reduction ratio.** `bc-interp/src/agg_par.rs`.

## Backlog, in dependency order

0. ~~A range-join algorithm (item 8).~~ **Landed** — the largest gap the review turned up, and
   the one it recorded rather than half-built. See `engine_improvements_ledger.md` entry 9.
1. Online adaptive conjunct reordering (item 3). Smallest remaining win, and it completes
   the mechanism item 1 started.
2. `StringView` adoption (item 2), with dictionary survival (item 6) alongside it, since
   both are about not destroying a compact string representation at the leaf.
3. Top-K threshold as a dynamic filter (item 4), reusing the runtime-filter transport.
4. Skew salt derived from measured partition sizes (item 5).
5. Adaptive morsel sizing (item 7), if latency ever becomes the complaint.
