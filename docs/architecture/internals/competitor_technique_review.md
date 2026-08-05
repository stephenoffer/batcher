# Competitor technique review: what to take from DuckDB, Polars, DataFusion, Spark, Daft and Ray Data

**Status:** review, 2026-07-24; **status of every Batcher-side claim re-checked against the code
2026-07-29**, which moved three items to landed and one from "partly closed" to "unreachable on
the live path". Every competitor claim below cites a file that was read in the original pass.
Every claim about Batcher was checked against Batcher's code, not its docs.

**Measured pass, 2026-08-04.** Item 2 (`StringView`) was benchmarked rather than assumed, which
re-valued it and narrowed where it can pay; item 4 gained the record of a top-N change that was
built, proven correct, measured a regression and reverted; and a new item 9 records the largest
gap `ORDER BY` has, together with three fixes for it that were measured and lost. The bias of
this pass is deliberate: **prefer a number that closes off a direction to a plausible technique
that opens one**, because the previous two passes both lost time to work that turned out to be
already built or unreachable.

**Operator sweep, same day (item 10).** A second pass read the competitors' operator
inventories rather than following the scorecard, and found the one gap in this document that
is *asymptotic* rather than a constant factor: `DISTINCT`/`GROUP BY` under a `LIMIT` never
stops early (10a, 0.15x at 16M rows and widening). It also found a query every competitor
accepts and Batcher raises on (10b), and **ruled four candidates out** — DuckDB's perfect-hash
aggregate and perfect-hash join, time-bucketed grouping, and expression CSE are all already
built here (10e). Note what that ratio says: of nine operator-level candidates examined, four
were already done. Check before building.

The sweep then extended to Polars' streaming-node inventory, which contributed 10f (sortedness
is tracked and propagated, and no operator specializes on it) and 10g — a methodological
finding worth as much as any gap: **two attempts to time a plan-level optimization measured
Kyber's shortcut machinery instead of the mechanism**, because it answered both probe shapes
from metadata 15-22x faster than DuckDB executed them. DataFusion, Daft and Ray Data followed
(10h) and turned up almost nothing: `WITH RECURSIVE`, Daft's whole AI axis including a WARC
reader, and Ray Data's `train_test_split`/`streaming_split`/shuffle are all already built.

**The honest summary of the whole sweep is that Batcher's operator coverage is broad.** Across
the five trees, three genuine openings came out of it — 10a (asymptotic, and the one to build),
10b (cheap), 10f (unmeasured, memory-shaped) — against roughly a dozen candidates that were
already implemented. An agent reading this document for work should start at the backlog, not
at a competitor's file listing.

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

1. The trees are large and the 07-24 and 07-29 passes targeted the mechanisms the scorecard
   already names as Batcher's open gaps, plus the execution-layer machinery around them. They
   were not an exhaustive read of six engines. Where a technique was identified from a file
   inventory rather than read in depth, the table below says so. **Item 10 works the other way
   round** — it reads the competitors' operator *inventories* and checks each entry against
   Batcher, which is how it found the one asymptotic gap and also ruled two candidates out.
2. "Batcher does not have this" means a grep of `crates/` and `python/batcher/` found
   nothing, and the relevant code path was read to confirm. Those greps are recorded
   inline so they can be re-run.

## The shortlist

Ranked by value against the mandate, with the cheapest genuine win first.

| # | Technique | Best source | Batcher today | Value |
|---|---|---|---|---|
| 1 | Short-circuiting conjunctive filter, conjuncts ordered by cost | DuckDB | **Landed** | 1.3x to 5.7x on multi-predicate filters |
| 2 | German strings (`StringView`) end to end | DuckDB, Polars | Absent entirely | **Re-valued 2026-08-04**: the win is `take`/`filter` (3-13x), *not* comparison (parity) or sort (0.75-0.81x). Only pays scan-native; a boundary conversion loses at morsel size |
| 10a | `DISTINCT`/`GROUP BY` + `LIMIT` stops once `k` groups exist | DuckDB | Absent — `RelOp::Distinct` carries no limit | **0.15x at 16M rows on a high-cardinality key, widening with scale.** The only *asymptotic* gap in this document |
| 9 | A faster string `ORDER BY` | DuckDB | Loses; **magnitude unmeasured** — this machine's noise is 5.3x | Now tracked by `benchmarks/.../ordering.py`; three candidate fixes measured and rejected, two leads left |
| 10b | `row_number() OVER ()` with no `ORDER BY` | DuckDB, Spark, Polars | **Raises** on both the DataFrame and SQL surfaces | Cheap; a query every competitor accepts |
| 3 | Online adaptive reordering of filter conjuncts | DuckDB | **Landed** (`ConjunctOrder`) | Fixes the case a static cost model gets wrong |
| 4 | Top-K heap threshold pushed down as a filter | DataFusion | **Half landed** (`TopNBound` skips morsels; nothing reaches the scan) | Large on `ORDER BY ... LIMIT k` over big inputs |
| 5 | Skew detected from measured partition sizes, split automatically | Spark AQE | Machinery exists, off by default | Removes a config the user cannot be expected to set |
| 6 | Dictionary encoding surviving past the leaf | DuckDB, Arrow | **Unreachable** — decoded at the FFI boundary, so the dict-native kernels never see a dictionary from Python | Compounds with 2 |
| 7 | Adaptive morsel sizing as a pluggable strategy | Daft | Fixed 16,384 rows | Small, and mostly a latency story |
| 8 | A range-join algorithm (IEJoin, or a binned rewrite) | DuckDB | **Landed**, and since re-tuned | **Largest single gap found**: 12–32x, and OOMs where DuckDB runs |

**Status correction, 2026-07-29.** Items 3, 4 and 8 were recorded as absent or open and are
landed; item 6 was recorded as partially closed and is in fact *inert on the live path*. Three
of the six open items in the backlog at the bottom of this document therefore described work
that already existed, which is a real cost: an agent working from this list re-implements what
is there, or "fixes" what is not broken. Each row above now names the type that implements it
so the claim can be checked in one grep. The per-item sections carry the evidence.

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

### What the arrow-rs 56 view kernels actually win, measured 2026-08-04

The paragraph above described the *representation*. Before adopting it, the kernels Batcher
would actually run were benchmarked against their `Utf8` equivalents: 4M rows, single
thread, best of five, `arrow` 56.2.1. **The result does not match the shape of the claim,
and it should change what gets built.**

| operation | 1 char | 7 chars | 27 chars | 60 chars |
|---|---|---|---|---|
| `filter`, 1% kept | **3.01x** | **3.84x** | **9.44x** | **7.58x** |
| `filter`, 50% kept | **6.23x** | **3.52x** | **7.39x** | **13.74x** |
| `take`, random permutation | **4.82x** | **7.02x** | **8.19x** | **9.63x** |
| `sort_to_indices` | 1.12x | 1.51x | *0.81x* | *0.75x* |
| `RowConverter` encode | 0.99x | *0.81x* | 0.93x | 0.93x |
| `eq` vs a literal | *0.88x* | 0.99x | 0.98x | *0.74x* |
| `lt` vs a literal | 1.70x | 1.03x | 1.07x | *0.87x* |

Read the columns, not the average. **The entire win is `take` and `filter`; comparison is
parity and sorting is a small loss on long strings.** That is the opposite of the usual
German-strings story, which sells the inline 4-byte prefix as a comparison optimization —
arrow-rs's `eq`/`lt` against a literal do not exploit it, so the prefix buys nothing here.
What views *do* buy is that `take` and `filter` move 16-byte descriptors instead of chasing
and copying bytes, which is why `take` is flat at ~114 ms across every string length while
`Utf8` climbs from 658 ms to 1,107 ms.

Two further facts bound any adoption. Views are **larger in memory**, not smaller, for every
shape measured (27 chars: 150 MB against 198 MB), because the 16-byte descriptor is paid per
row on top of the bytes. And conversion is not free: `cast(Utf8 -> Utf8View)` costs 60-88 ms
per 4M rows and the reverse 44-187 ms.

### Converting at an operator boundary does not pay, and the crossover says why

The cheap version of this — leave the schema alone, convert to a view inside one operator,
convert back — was measured directly as `cast -> take -> cast` against a plain `take`, over
input sizes, output/input ratios, and both index orders:

| input rows | out/in | index order | round trip is |
|---|---|---|---|
| 4,000,000 | 1.0 | random | **2.22x** |
| 4,000,000 | 1.0 | sorted | 0.43x |
| 4,000,000 | 0.1 | random | 1.08x |
| 1,000,000 | 1.0 | random | **1.87x** |
| 100,000 | 1.0 | random | 0.53x |
| **16,384** (one morsel) | 1.0 | random | **0.45x** |

The round trip wins only on a *large* input, gathered *whole*, in *random* order. Batcher is
morsel-driven at 16,384 rows, where it is 0.45x at best — so an operator-local conversion is
a pessimization on the default path, and no gate on selectivity rescues it. Note also the
`sorted` column: `take(Utf8)` with ascending indices is already 6-11x cheaper than with a
random permutation (92 ms against 984 ms at 4M x 27 chars), because the value buffer is then
read sequentially. A filter's gather is ascending by construction, which is exactly why the
filter path has nothing to gain here.

**What that leaves.** `StringView` is still worth having, but as a representation the *scan*
produces and the engine never converts — not as a boundary trick, and not for the reason
usually given. Its value is `take`/`filter`, its cost is memory and a conversion at every
edge that is not view-native, and the honest expected win should be argued from the `take`
and `filter` rows above rather than from the comparison story. This remains the most invasive
single-node item; what the measurement changes is that a *partial* adoption is now known to
lose rather than merely suspected to.

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
since it is already compiled once per operator and shared across rayon workers.

**Landed as `bc_expr::ConjunctOrder`** (`crates/bc-expr/src/select.rs`), and it took a
stronger form than the mechanism it was modelled on. DuckDB hill-climbs on an *aggregate*
signal — swap an adjacent pair, keep the swap if total runtime improved over the next ten
batches — which needs tens of batches to walk a permutation. Because
`short_circuit_filter_mask_with` already evaluates the conjuncts one at a time, Batcher can
attribute rows and time to each conjunct *individually* and jump to the implied order after a
single morsel. The rank is time-per-row divided by rows-removed-per-row, which is the quantity
that actually matters and which cost alone cannot express. It needs no lock: every counter is a
`Relaxed` atomic and each morsel derives its own permutation, because the conjuncts of an `AND`
commute so every order yields the identical mask.

It is wired at four sites, each building the state **once per operator** and capturing it in
the per-morsel closure — the distinction that decides whether this works at all, since a
per-morsel `ConjunctOrder` would measure and then discard: `stream/mod.rs:289`,
`par.rs:567`, `par.rs:2094`, `stream/parallel.rs:619`. The default (streaming) executor never
carries a JIT on this path, so it takes the measured order unconditionally.
`a_measured_order_beats_the_static_one_and_agrees_with_the_oracle` pins both halves.

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

**Half of this is landed, and the half matters.** `bc_runtime::topn::TopNBound` is wired into
`ops::parallel_top_n` (`crates/bc-interp/src/ops/mod.rs:768`): once any morsel has produced a
full set of `k` candidates it publishes its cut-off, and a later morsel whose entire first-key
range is strictly worse than that is dropped for the price of one min/max pass over the key
column. The bound only ever tightens, so a stale read costs a missed skip and never a wrong
answer, and an `is_off()` gauge retires a bound that is not earning its cost. The key
expressions are evaluated once per morsel and reused for the selection, the bound check and the
candidate gather. `report_the_top_n_skip_saving` measures the trade it makes.

**A related change was tried on the string key and reverted, 2026-08-04.** For a *single*
sort key `top_k_indices_of` answers a top-N by running the specialized full sort over each
morsel and slicing it to `k`. For the integer radix that is sound because the radix is linear.
For a **string** key it is not: `stable_sort_indices_str` is a comparison sort, so the branch
pays `O(n log n)` to keep `k` rows, and the code comment asserting both were linear was simply
wrong. Replacing it with an `O(n)` quickselect under the identical total order was built,
proven result-identical (16 differential cases against DuckDB plus a Rust oracle test), and
**measured a regression**: at 10M rows `LIMIT 10` was a wash (15 ms either way), `LIMIT 100000`
went 893 -> 1139 ms (0.78x) and `LIMIT 1000000` went 1434 -> 4267 ms (**0.34x**).

The reason is downstream of the selection. `parallel_top_n` concatenates every morsel's
candidates and re-sorts them with a limited `lexsort`; once `k` exceeds the 16,384-row morsel
every row becomes a candidate, and handing that merge 610 **sorted runs** is worth far more
than the per-morsel ordering costs. At a morsel of 16,384 rows the ordering was never the
bottleneck in the first place. The engine code is unchanged; the comment now records the real
reason the branch exists, and the tests were kept because the string key was the one key type
with no top-N oracle test of its own.

The lesson generalizes past this operator: **asymptotics measured per morsel can invert once
the merge is included**, and Batcher's morsel is small enough that `O(n log n)` inside one is
often free.

What is **not** there is the edge to the scan. The bound skips morsels the engine has already
read; nothing derived from it reaches a Parquet reader, so no row group is pruned and no I/O is
avoided. That is the expensive half for `ORDER BY x LIMIT k` over a large file on disk, and it
is what item 4 should now be read as meaning. It needs the predicate republished through the
`runtime_filter.rs` transport and then through `io/predicate.py`'s pushdown, which is a
boundary-crossing change rather than a data-plane one.

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
and gather through the keys. This compounds with item 2 rather than competing with it.

**Correction, 2026-07-29: "filters and group-by are dictionary-native" is true of the kernels
and false of the engine.** A `Dictionary` column is decoded **at the FFI boundary**, one level
above every one of those fast paths: `bc_py::normalize_to`
(`crates/bc-py/src/normalize.rs:84`) rewrites `Dictionary(_, value)` to `value` for every input
column, alongside the narrow-numeric widening, so **no dictionary ever reaches the engine from
Python**. `try_dict_compare`, the `InList` dictionary path and `assign_groups`' dictionary
grouping are all reachable only from Rust callers that construct a dictionary directly — which
is what their unit tests do, and which no query does.

So the headline number attached to this axis in `competitive_architecture.md` ceiling 2 — a
low-cardinality string filter going 144.9 ms to 7.4 ms, 19.6x — is a real measurement of a
kernel that a user cannot reach, and the same applies to the ~7x quoted for grouping on codes.
It is not a wrong measurement; it is a measurement of the wrong scope, and it should not be
quoted as an engine result until the boundary preserves the encoding. `normalize.rs` is honest
about this in its own `NOTE`; the scorecard is not.

The blocker `normalize.rs` records is real and is not the kernels. The plan's schema treats a
column by its Arrow type, so a preserved `Dictionary` propagates into intermediate schemas, and
an operator that decodes one while reusing its input's schema then fails Arrow's own
`RecordBatch::try_new` validation on the type mismatch. That is RFC
`rfc-streaming-executor.md` Proposal 3: separate the plan's *logical* type (the value type)
from the morsel's *physical* encoding (the dictionary).

The shape of the fix worth recording, because it is smaller than the RFC implies: **decode on
egress, not on ingress.** `normalize_to` keeps the `Dictionary` and normalizes only its value
type; a new egress pass decodes `Dictionary` to its value type on the batches handed back to
Python; and `plan/types/lattice.py::widen` then needs **no change at all**, because it already
reports the value type — which is precisely what keeps `Dataset.schema` truthful and keeps a
dictionary column joinable against a plain string one.

### Built, measured end to end, and reverted — with the numbers

**The operator half is done and committed.** `bc-interp/tests/dictionary_operators.rs` runs
twelve operators over a dictionary against the decoded oracle on all three executors, both null
encodings, and empty/single-row batches. Nine were already correct; three were fixed
(`project_field::output_field` took its type from the input schema instead of the evaluated
array; `keys::decode_dict_keys` now reconciles the two sides of a hash join and a range join).
So a future attempt starts from an engine that handles dictionaries correctly.

**The boundary half was then built and benchmarked, and the result retires the headline claim.**
Preserving a canonical `Dictionary(Int32, Utf8)` on ingress and decoding on egress works exactly
as designed — `Dataset.schema` still reports `string`, `collect()` still returns `string`, and
results are correct. Measured at six million rows, best of five, against the same data
pre-decoded (which is what the old boundary produced):

| query shape | dictionary | plain string | |
|---|---|---|---|
| filter + `sum`, 25 x 63-char values | 3.4 ms | 9.6 ms | **2.80x** |
| filter + `sum`, 1000 x 72-char values | 3.3 ms | 9.1 ms | **2.75x** |
| filter + `sum`, 25 x 7-char values | 3.2 ms | 3.8 ms | **1.20x** |
| filter 1/25, return the column | 5.8 ms | 6.3 ms | 1.08x |
| filter 24/25, return the column | 7.5 ms | 7.0 ms | **0.93x** |
| `SELECT <string col>`, no filter | 7.4 ms | 4.6 ms | **0.63x** |

So the win is real but it is **1.2x to 2.8x and scales with string length**, not 19.6x — and it
is a **regression on any query that returns the column rather than consuming it**, because the
decode moves from the input to the (equally large) output while the engine carries the encoding
in between. `SELECT a_string_column FROM t` is not a corner case, and a regression there is a
blocking failure, so the change was reverted rather than shipped.

**What that changes about this item.** It is no longer "flip one decode and collect 19.6x". It
is a **planner** decision — preserve the encoding only when the plan *consumes* the column
(filter, group-by, join key) and decode at the leaf when it *projects* it — which is a cost
model, not a boundary edit. Valued honestly it is worth ~1.2x on TPC-H-shaped short codes and
~2.8x on long low-cardinality strings, on consuming shapes only. Anything sequenced behind it
(notably `StringView`, item 2) should be re-argued against those numbers rather than the
retired one.

The 19.6x itself is not reproducible end to end and should stop being quoted: measured here, a
plain-string filter over 6M rows costs 3.8 ms, not the 144.9 ms that figure divides into.

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

**Landed as `RelOp::RangeJoin`** — see `competitive_architecture.md` ceiling 7. The IEJoin option below was taken over the binned one:
the binned rewrite turns on a bucket width `W` nobody could choose safely at plan time, and a
badly chosen `W` is a pessimization rather than a win.

**The result is mixed and the ledger says so.** The quadratic *plan* is gone — `n = 2,000,000`
went from not running at all to 1.1 s — but DuckDB's own IEJoin is still **1.3-2.9x faster**, and
the gap widens with `n`, because `PhysicalIEJoin` decomposes the sorted union into blocks and
prunes block pairs by range where this implementation walks a suffix of a bit array. An earlier
draft of this note claimed 15-643x in Batcher's favour; that was measured against DuckDB's
`NESTED_LOOP_JOIN` fallback over an Arrow scan and is retracted. The rest of this item is kept
as the record of what was considered.

**The block-decomposition diagnosis was measured and does not hold, 2026-07-29.** A phase study
split the operator's non-sort time into the sweep and the seven passes that build the sweep's
inputs. At five million rows a side the *passes* cost 781 ms against a single-threaded sweep of
638 ms — and the sweep already fanned out to rayon while the passes ran on one core of 96. The
suffix walk that block-pair pruning removes was already the smaller half, because `MarkSet`
derives its level count from the universe size and each level multiplies the span one word read
dismisses by 64, so an empty suffix costs one read per level (four at ten million entries)
rather than one per word.

Fixing what was actually there — fusing the three passes that re-read `order2` into one rayon
pass, and parallelizing the `pos1` inverse permutation (disjoint by construction, so the slot
type carries the disjointness as an atomic), `sorted_drank1` and `left_at` — took the whole
operator from **107 ms to 91 ms at 500,000 rows a side, 558 ms to 386 ms at 2,000,000, and 1.9 s
to 1.3 s at 5,000,000** (controlled A/B against `HEAD`, best of three each way, same box). Every
change is a reindexing of the same values, so the arrays handed to the sweep are byte-for-byte
what the sequential passes produced.

Block decomposition remains the route to making the operator **distributable**, since that needs
the same "which block pairs can intersect" pruning. It is no longer a single-node speed argument.

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

## 9. String sorting — the largest measured single-node gap, and three fixes that lost

Added 2026-08-04. This was not on the shortlist because the earlier passes measured the
string *filter* and *group-by* paths and not `ORDER BY`.

**The gap.** Against DuckDB's own storage (not a registered Arrow table — an Arrow scan makes
DuckDB look slow for reasons that have nothing to do with the operator, and quoting that is
how the retracted 15-643x range-join number happened), 10M rows, 27-char values, 200k
distinct:

Each row below is one run, and **every absolute number in this table carries the caveat at the
end of this section**: on this machine the same case re-run four times spanned 5.3x. Treat the
*direction* of each row as the finding and the milliseconds as indicative only.

| shape | Batcher | DuckDB | Batcher is |
|---|---|---|---|
| full `ORDER BY <string>` | 620-655 ms | 188-216 ms | 0.30-0.33x |
| `DISTINCT <string>` | 544 ms | 196 ms | 0.36x |
| `GROUP BY <string>` | 570 ms | 274 ms | 0.48x |
| `ORDER BY <string> LIMIT 10` | 14-17 ms | 6-7 ms | 0.41x |
| `ORDER BY <string> LIMIT 10000` | 17-19 ms | 53-55 ms | 2.81-3.22x |
| filter on a string + `sum` | 30 ms | 22 ms | 0.72x |

**Now tracked by a committed benchmark.** The table above came from a scratch script, which is
how the gap stayed invisible: `benchmarks/suites/operators/ordering.py` had exactly one case and
it sorted a *float* on *three* keys, so nothing in the suite touched the single string key path
at all. Three cases now do. Reproduce with:

```bash
python benchmarks/run.py --benchmark operators --only sort-string --engines batcher,duckdb,polars --scale 1
```

At scale 1 (6M `lineitem` rows, read from parquet, best of five, 16 cores, correctness gate
passed on all three):

| case | batcher | duckdb | polars | vs DuckDB |
|---|---|---|---|---|
| `op-sort-string` (high-card `l_comment`) | 1068 ms | 908 ms | 1239 ms | 1.18x slower |
| `op-sort-string-lowcard` (`l_shipmode`, 7 distinct) | 411 ms | 128 ms | 205 ms | **3.22x slower** |
| `op-sort-string-limit` (top-100, tie-broken) | 47 ms | 24 ms | 1311 ms | 1.97x slower |

**Retracted 2026-08-04, same day: the 3.22x in that table is not reliable and neither is any
other single-run figure taken on this machine.** Re-running the identical case four times
across two builds gave `op-sort-string-lowcard` at 580 ms, 110 ms, 115 ms and 397 ms — a 5.3x
spread with no relation to which build was running. The box carries two other sessions' cargo
builds and test suites, and at 16 cores under a load average of 20-35 a benchmark measures the
scheduler. The *ratios within one run* are still fair (both engines see the same load), which
is why the correctness gate and the relative shape of the table survive; the absolute
milliseconds and any cross-run comparison do not.

What can be said without a quiet box: the low-cardinality string sort is **structurally**
disadvantaged, because seven distinct values give the sampled quantile boundaries nothing to
cut on, so the sample-sort's ranges come out lopsided however many are asked for. That is a
property of the code, not of a timing. Whether it costs 1.1x or 3.9x is exactly what needs
re-measuring somewhere quiet.

The full sort is the big one, and it is the *ordering* rather than the materialization:
holding the sort key fixed and changing only the payload moves nothing (sorting by a string
and returning an `int` costs 973 ms; returning the string costs 941 ms), while holding the
payload fixed and changing the key moves everything (sorting by an `int` and returning that
same `int` costs 220 ms). So ~750 ms of a ~950 ms sort is attributable to the string key.

**Where that 750 ms sits.** The sample-sort's phases, replicated at 10M rows over 64 ranges
on 16 threads:

| phase | high-card 27 ch | low-card 7 ch | shared 20-char prefix |
|---|---|---|---|
| sample boundaries | 5 ms | 4 ms | 3 ms |
| route (binary search per row) | 82-139 ms | 120 ms | 110-113 ms |
| bucket | 75-103 ms | 51-56 ms | 76-87 ms |
| gather each range's key column | 154-157 ms | 35-49 ms | 152-195 ms |
| **per-range sort** | **288-537 ms** | 35-65 ms | 47-63 ms |

The per-range comparison sort is ~53% of the total on the shape that loses, and routing plus
bucketing is another ~23%.

**Three fixes were measured and all three lost.** Recording them because each is the obvious
next idea, and each costs a day to re-derive:

1. **LSD radix over the packed prefix** — DuckDB's own shape (fixed-width radix key, tie-break
   only where the key ties). Measured *slower* than the existing comparison sort on the
   high-card shape. `sort_live`'s packed prefix already removes the pointer chase that a radix
   is meant to avoid, and pdqsort over a 12-byte key beats eight scatter passes plus a
   run-repair pass. This also explains why `radix_sort.rs` declines above 2^18 rows for
   floats: same cause, already known, and it applies here too.
2. **Skipping the per-range key gather** — build the packed prefix once over the whole column
   and let each range sort in place through global row indices. This looks like it removes
   154 ms of pure copying. It costs **2,655-2,735 ms**: the gather is not overhead, it is what
   makes every subsequent read of that range sequential. Do not remove it.
3. **A narrower sort key** — one `u64` per row holding the first *four* bytes above the row
   index, so the key array halves and sorts with no comparator at all, with runs sharing four
   leading bytes repaired afterwards. Verified to produce the **identical permutation** on
   every shape. Interleaved medians over seven repetitions: **1.69x** on high-card 27-char
   (361 -> 214 ms), but **0.87x** on low-card 7-char and **0.78x** on 10M-distinct 12-char,
   where four bytes collide often enough that the repair pass dominates. A fixed width is
   therefore wrong; an *adaptive* one chosen from the sample `prefix_discriminates` already
   takes is the version worth building, and it is the one open lead this section leaves.

**A caveat on all of the above.** These were measured on a 16-core box carrying another
session's build and test load; single-pass numbers swung as much as 2.3x on the same code, so
every figure here is a median of interleaved repetitions and the equivalence checks (not the
timings) are what the conclusions rest on. Re-measure on a quiet box before acting on a
margin under ~1.3x.

## 10. Operator-by-operator sweep, 2026-08-04

The passes above targeted mechanisms the scorecard already named. This one went the other
way: read the competitors' *operator inventories* and check each entry against Batcher. The
useful output is as much what it **ruled out** as what it found — two of DuckDB's named
operators turned out to be already built here, and an agent working from an operator list
alone would have re-implemented both.

### 10a. `DISTINCT` / `GROUP BY` with a `LIMIT` never stops early

**What DuckDB does.** `PhysicalLimitedDistinct`
(`src/execution/operator/aggregate/physical_limited_distinct.cpp`) is planned from
`plan_limit.cpp:96` for a `DISTINCT`/`GROUP BY` with no aggregates under a `LIMIT`. Its
`Sink` returns `SinkResultType::FINISHED` as soon as the hash table holds `limit` groups
(`:128`), and the flag is an `atomic<bool>` shared across threads, so every worker stops.

**What Batcher does.** `bc_ir::RelOp::Distinct { input }` carries no limit at all — the
`Limit` is a separate operator above a dedup that has already consumed the whole input.

**Measured, and it is asymptotic rather than a constant factor**, which is why it survives
this machine's noise. `SELECT DISTINCT g FROM t LIMIT 5`, all values distinct so the early
exit is worth the most:

| key | 4M rows | 16M rows |
|---|---|---|
| `Int64` | 0.40x | **0.15x** |
| `Utf8` | 0.29x | **0.14x** |

Batcher's time grows ~7x for a 4x row increase; DuckDB's grows 2.6x. The gap widens with
scale, so it is worse at every size above these.

**Where it does not apply, measured:** on a *low*-cardinality key (1000 distinct over 32M
rows) Batcher is **4.9x faster** than DuckDB, because `agg::group::assign`'s dense
direct-map dedups the whole column faster than DuckDB reaches its early exit. So this wants
a rule that fires on an *estimated-high-cardinality* key, not an unconditional one — and
Kyber already has the sketch-backed cardinality estimate to decide it.

### 10b. `row_number() OVER ()` is rejected, and every competitor accepts it

`SELECT x, row_number() OVER () FROM t` runs in DuckDB, Spark and Polars. Batcher refuses it
on both surfaces:

- DataFrame: `PlanError: window ranking function 'row_number' requires order_by keys`
- SQL: `NotImplementedError: window ranking function 'rownumber' requires ORDER BY`

The refusal looks principled — without an `ORDER BY` the numbering has no defined order, and
this engine cares about determinism more than most. But the conclusion does not follow.
Batcher already defines a deterministic order for exactly this situation everywhere else: its
sorts resolve ties to *input order*, and `str_sort`/`radix_sort` exist to guarantee it. Numbering
an unordered window in input order is the same rule, is what DuckDB effectively produces, and
costs a ported query nothing. Refusing is the one option that is neither compatible nor more
deterministic than the alternative.

### 10c. No streaming window path

`PhysicalStreamingWindow::IsStreamingFunction`
(`physical_streaming_window.cpp:179`) admits a window with **no** `PARTITION BY`, **no**
`ORDER BY` and no `EXCLUDE`, when it is either a running total (`UNBOUNDED PRECEDING` to
`CURRENT ROW`) or a function that declares itself streamable. Those compute in one pass with
no materialization. Batcher's `RelOp::Window` is a full breaker on every shape.

Measured on `sum(x) OVER ()`: **0.45x at 4M rows, 0.62x at 16M**. Both scale linearly, so this
is a constant-factor and memory-footprint item, not an asymptotic one — worth less than 10a,
and listed so it is not mistaken for more.

### 10d. Shuffle partitions are sized from history, not from the shuffle just written

Spark's `CoalesceShufflePartitions` reads the **materialized** sizes of the shuffle it just
wrote and merges adjacent small partitions before the next stage reads them. Batcher's
equivalent is cross-*run*: `dist/executors/map.py:769` persists "a run's measured total rows
for `source` so the **next run's** partition count can" be chosen, and
`dist/adaptive_sizing/sizing.py` keeps an EMA across runs.

Recorded because of where it sits rather than its size. Cross-query learning from measured
quantities is a genuine advantage over Spark and `competitive_architecture.md` is right to
claim it. But the specific claim of *stage-boundary re-optimization on measured cardinalities*
is weaker here than it sounds: for partition sizing the measured quantity consulted is the
previous run's, so a first-seen shape gets a prior rather than a measurement, and a shape whose
data volume changed run-to-run gets a stale one. Spark, with no learning at all, uses the
fresher number for this one decision.

### 10f. Sortedness is tracked, and no operator specializes on it

Polars' streaming engine carries a sortedness flag and has **operators that consume it**:
`nodes/sorted_group_by.rs`, `nodes/sorted_unique.rs`, `nodes/is_first_distinct.rs`,
`nodes/merge_sorted.rs`. On sorted input a group-by is a linear scan over adjacent runs — no
hash table, `O(1)` state, and it cannot spill.

Batcher has the *property* and propagates it well. `RelStats.sorted_by` carries a canonical
ascending/nulls-last ordering, `kyber/properties.py` defines `PhysicalProperties` with a
`satisfies` relation that knows a `(a, b)` ordering satisfies a request for `(a)`, and
`project_ordering` threads it through projections. Two things consume it: `kyber/rules/ordering.py`
**eliminates a redundant sort** when the wanted columns are a prefix of the delivered ordering
(`:41-55`), and `kyber/cost/shuffle.py` plus `dist/executor.py` use it to cost and avoid
shuffles.

What consumes it nowhere is aggregation. A grep for a sorted-aggregate or adjacent-dedup
operator across `crates/` returns nothing, and no rule rewrites `Distinct`/`Aggregate` on the
strength of a sorted input. So a lakehouse table with a declared sort key — the case
`shortcuts/ordering.py` exists to describe — still pays a full hash aggregate to group by its
own sort key.

**Unmeasured**, and honestly stated as such: it needs a source that declares `sorted_by`, which
is more setup than the other items here. The reason to rank it anyway is that the win is
*memory* before speed. A sorted group-by has bounded state, so it is the one shape that
converts a spilling aggregate into a streaming one.

### 10g. A caution about probing this optimizer with a stopwatch

Two attempts to measure whether Batcher reuses a repeated *subplan* (as Polars' `multiplexer`
node and Spark's `ReuseExchange` do) failed, in an instructive way. A CTE self-joined on its own
group key answered in **7 ms** against 122 ms for the aggregate alone, and a regex-extraction
subplan `UNION ALL`-ed with itself answered in **22 ms** against 359 ms for one copy. Both times
Kyber *solved* the query from metadata rather than executing it, so the probe measured the
shortcut and not the thing it was aimed at.

Recorded because it changes how the next person should measure. A timing probe of a plan-level
optimization has to use a shape the shortcut machinery cannot answer, or it reports a number
about the wrong mechanism entirely. Note also what the two figures say on their own account:
on both shapes Batcher beat DuckDB by 15x and 22x, because DuckDB executed what it was asked.

Expression-level CSE is present and is a different thing —
`kyber/rules/extra/cse.py` binds a subexpression repeated across a `Project`'s outputs to one
synthetic column. Plan-level reuse remains **unestablished in either direction**.

### 10h. DataFusion, Daft and Ray Data — one small gap, and a lot ruled out

The sweep finished across the remaining three trees. The result is lopsided and worth
reporting as such: **almost everything checked was already built.**

DataFusion's `physical-plan/src` contributes `recursive_query.rs` + `work_table.rs`, i.e.
`WITH RECURSIVE`. Batcher has it — `_sql/parser/translator.py:63` even documents the iteration
cap it applies against a missing stop condition, and
`WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 5) SELECT sum(x) FROM c`
returns 15.

Daft's crate list is the sharpest test of the AI axis, since that is what Daft is for. Its three
most distinctive crates all have Batcher counterparts: `daft-ai` against `ml/embed_api.py` and
`ml/llm/`; `daft-functions-tokenize` against `plan/functions/prompt/` (token budgeting) and the
`.str` n-gram/normalization family; and `daft-warc` — a WARC reader for web-crawl corpora, which
is niche enough to be a plausible gap — against `io/formats/unstructured/warc.py`.

Ray Data's `Dataset` surface (93 public methods) is likewise covered where it matters:
`train_test_split` is `api/dataset/ml.py`, `streaming_split` is `ml/loader/lazy.py`, and
`random_shuffle` is `Dataset.shuffle(seed=)` — the Feistel permutation the scorecard already
claims as a strength.

**The one gap, and it is small.** `split_at_indices` and `split_proportionately` have no
Batcher equivalent. More useful than the two methods: **no Ray Data spelling appears in the
compat guidance table** (`api/dataset/compat/guidance/_dataset_table.py`), which does carry
Polars, pandas and Spark names — `group_by_dynamic` is mapped there, `random_shuffle` is not.
Batcher's scorecard positions Ray Data as a primary competitor and claims 50-450x over it, so
the porting surface for that specific audience being the unmapped one is a mismatch worth
closing. It is a table entry, not an engine change.

### 10e. Ruled out — already built, do not re-implement

Both were on the shortlist of DuckDB operators worth taking, and both already exist:

- **Perfect-hash aggregate.** `CanUsePerfectHashAggregate` (`plan_aggregate.cpp:121`) direct-
  indexes an array for integer group keys whose observed range fits a bit budget (default 12).
  Batcher: `bc-runtime/src/agg/group/assign.rs`, a "**dense direct-map** path [that] drops the
  hash entirely when the key's value range is small".
- **Perfect-hash join.** `perfect_hash_join_executor.cpp`. Batcher:
  `bc-runtime/src/join/dense.rs`, "a perfect hash for a small-range integer build key",
  explicitly written as the join-side counterpart of the group-key one, with a measurement of
  the build cost it removes.
- **Time-bucketed grouping.** DuckDB's `time_bucket`, Polars' `group_by_dynamic` /
  `group_by_rolling`, Spark's `window()`. Batcher: `plan/functions/temporal.py` and
  `expr_ir/func_nodes.py:274` give `window_start(ts, width, origin)` for tumbling buckets and a
  list of overlapping buckets for sliding ones, and `group_by_dynamic` is already mapped to the
  Batcher spelling in `api/dataset/compat/guidance/`.
- **Expression CSE.** `kyber/rules/extra/cse.py`, which binds a subexpression repeated across a
  `Project`'s outputs to one synthetic column. (Plan-level *subplan* reuse is a different
  question, and 10g explains why it is still open.)

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

Re-ordered 2026-07-29, after three items turned out to be already built. What is struck through
is done; what remains is ordered by value against the mandate.

0. ~~A range-join algorithm (item 8).~~ **Landed**, and its follow-up re-tuned — see item 8 for
   why the block-decomposition step it named is a *distribution* prerequisite and not a
   single-node speed fix.
1. ~~Online adaptive conjunct reordering (item 3).~~ **Landed** as `ConjunctOrder`, in a
   stronger form than DuckDB's (per-conjunct attribution rather than an aggregate hill-climb).
2. **Preserve the dictionary across the FFI boundary (item 6).** Promoted to the top of the
   remaining work, because it is not an optimization to build — the kernels exist and are
   tested. It is one decode, in `normalize_to`, standing between them and every real query.
   Decode on egress instead of ingress; `plan/types/lattice.py::widen` needs no change. The
   audit is every operator that builds an output batch from its input's schema.
3. **The early exit for `DISTINCT`/`GROUP BY` under a `LIMIT` (item 10a).** The only
   *asymptotic* gap the review has found, measured at 0.15x on 16M rows and widening with
   scale — and, unlike everything below it, confirmable without a quiet box, because the row
   count decides it rather than a timing margin. Gate it on Kyber's existing cardinality
   estimate rather than firing unconditionally: on a *low*-cardinality key Batcher's dense
   direct-map already beats DuckDB 4.9x, and an unconditional early exit gives that back.
4. **`row_number() OVER ()` (item 10b).** A query DuckDB, Spark and Polars all accept and
   Batcher refuses on both surfaces. Cheap, and the deterministic answer — number in input
   order — is the rule Batcher's sorts already apply to ties.
5. **The low-cardinality string sort (item 9).** The cause is structural rather than a
   kernel, which is why it is here despite the magnitude being unmeasured: seven distinct
   values give the sampled quantile boundaries nothing to cut on, so the sample-sort's ranges
   come out lopsided however many are asked for. `sample_sort.rs` already solves exactly this
   on the *multi-key* path — `composite_part_of` re-routes by the full composite key when
   `max_bucket > 3 * fair_share` — and the single-key path has no equivalent fallback.

   **The fix is written and was not landed, deliberately.** Extending that gate to one key
   needs the composite to carry a trailing ascending row index, or it ties for every row and
   routes identically; `(keys…, row index)` is exactly the order this sort produces, so
   splitting a tie group across ranges is sound. It passed the serial oracle on every
   shape. It is not landed because "no performance regressions" is a gate and this machine
   could not measure it either way. Land it from a quiet box, with the benchmark above.
6. **An adaptive-width sort key for the per-range sort (item 9).** Proven
   permutation-identical and worth 1.69x on high-cardinality 27-char keys, but a *fixed*
   4-byte head loses 0.78-0.87x on two other shapes, so the width has to come from the sample
   `prefix_discriminates` already draws. Needs a quiet box: the margin is inside this
   machine's noise band.
7. `StringView` adoption (item 2), alongside 2, since both are about not destroying a compact
   string representation. Still the largest and most invasive single-node item — but now
   argued from `take`/`filter` only, and known to lose if adopted anywhere short of
   scan-native.
8. **The scan half of the top-K dynamic filter (item 4).** The morsel-skip half is landed; what
   is missing is republishing the bound so a Parquet reader can prune row groups, which is where
   the I/O saving is.
9. Skew salt derived from measured partition sizes (item 5).
10. Post-shuffle partition coalescing from the sizes just written, not the previous run's
    (item 10d).
11. Adaptive morsel sizing (item 7), if latency ever becomes the complaint.

**A process note, since this document exists to direct work.** Three of the six items in the
previous version of this list described work that already existed, and one described a win the
engine cannot reach. Every row of the shortlist now names the type that implements it, so the
next reader can check the claim with one grep rather than trusting the prose. A parts list that
has drifted is worse than no parts list: it spends effort re-deriving what is there and defends
numbers the live path never produces.
