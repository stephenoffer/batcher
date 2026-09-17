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
stops early (10a, 0.15x at 16M rows and widening). It also found a SQL-portability item (10b — whose
first draft wrongly claimed Spark accepts the query; Spark's own test asserts it does not, and
the entry now carries the correction), and **ruled four candidates out** — DuckDB's perfect-hash
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

A final pass compared **optimizer passes** rather than operators (10i), on the reasoning that a
pass DuckDB has and Kyber does not is worth more than an operator, Kyber being the moat. It
found one: common *subplan* elimination, which also settles the question 10g had to leave open.

**The honest summary of the whole sweep is that Batcher's coverage is broad.** Across the five
trees plus DuckDB's optimizer, three genuine openings came out of it — 10a (asymptotic, and the
one to build), 10i (a pass Kyber lacks), 10f (unmeasured, memory-shaped) — against roughly
fifteen candidates that were already implemented, one of which (partial aggregate pushdown) is
argued more carefully here than in the pass it matches. Plus one item (10b) that shrank to an
error message once its competitor claim was actually run instead of recalled. An agent reading this document for work should start at the backlog, not
at a competitor's file listing.

**Status pass, 2026-08-06.** Two of those three openings are now built (10a and 10i), as is the
low-cardinality half of item 9 and the one capability gap in 10h. **10f — a sorted-input
aggregate, the memory-shaped one — is the last of the three still open**, and it remains the
most interesting item here, because its win is bounded state rather than speed: it is the one
shape that turns a spilling aggregate into a streaming one.

**Update, 2026-08-13.** 10f is now *half* built: the run-scanning group assignment is in
(`agg/group/runs.rs`), and it is worth 1.0-1.2x rather than the 6.0x its microbenchmark
promised — the discrepancy, and the A/B method that exposed it, are recorded in 10f and are
more useful than the feature. **The bounded-state half is still open, and it is still the
reason this item ranks where it does.** What is left below it is either
large and invasive (`StringView`), a boundary-crossing planner decision (dictionary survival,
top-K to the scan), or waiting on a quiet box. See the backlog for the per-item evidence.

**A fresh sweep the same day, into source no earlier pass had opened** — DuckDB's
`src/function/window/` and `src/execution/operator/join/` — added 10j and 10k. Both are worth
noting for *what kind* of finding they are, because it is not the kind the first nine items
are. Neither is a technique to copy: DuckDB's window segment tree is **worse** than what
`bc-runtime/src/window/frame/` already does, and its delimited joins turn out to cover shapes
Batcher covers too. What the comparison surfaced instead were defects **inside Batcher**, at
the seam between the SQL front-end and the engine — seven window aggregates the engine
computes and SQL could not spell, and a frame silently dropped on three others, which is a
**wrong answer** rather than a missing feature and which every gate passed. The lesson
generalizes: reading a competitor's implementation of something Batcher already has is worth
as much as reading one it lacks, because the comparison is what makes an internal gap visible.

**Status pass, 2026-08-25 — and this one moves the board rather than an entry (item 19).** The
operator suite was re-run at `HEAD` before any work was started, and **every sort row entry 15
listed as a loss is now a win**: Batcher beats DuckDB on **19 of 21** operator cases and the two
remaining losses total 2.4 ms. The ranking this document has directed sort work from since
2026-08-18 is therefore stale, and an agent following it today would optimize four operators
that already win. Item 19 has the table. The same pass read the two sorters entry 18 stopped
short of (DuckDB sorts with `vergesort` over `ska_sort`), **took the first** — natural-run
detection, now `ops/run_sort.rs`, worth 1.09-1.32x on partly-ordered input with the random
control unchanged — and **ruled out the second** together with entry 18's proposed crossover
gate, on the measurement that the packed radix never sees a large input in the first place.
It also records one unbuilt candidate nobody has read before: Polars' `polars-ooc` global spill
arbiter, which picks a spill *victim* by measured score.

**All six reference engines are now covered (item 25 closes Spark and Arrow):** thirteen of
Spark's AQE/execution techniques have a Batcher site and the fourteenth
(`CoalesceShufflePartitions`) is deliberately *not* built, because the measurement that would
justify it turned out to be an artifact of stale shuffle files. Arrow's compute kernels
contribute nothing to the backlog.

**The pass then widened from operators to the whole surface (items 23-24):** TPC-DS's 99
queries measure **0.97x against DuckDB overall**, its one correctness failure is diagnosed to
the bit as a harness artifact rather than an engine defect, a plausible `ROLLUP` bottleneck was
measured and **refuted** (Batcher's rollup scales *better* than DuckDB's — do not "fix" it), and
the capability census's top item is re-checked and scoped.

**Five claims were found stale in this one pass, all in the same direction** — the sort ranking
(19), the range join's ceiling (20), the sf100 OOM's location (21), the two rows `StringView`
was ranked on (22), and `generate_series` in the capability census (24). Four of the five are
now *wins* or built. The fourth is worse than recorded: the
sf100 OOM is current, it has simply moved to q9. **Start at item 19 and re-measure before
building anything from the older entries.** The one thing this pass found that is unambiguously
open, and structural, is the >=100M-row memory profile in item 21.

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
| 2 | German strings (`StringView`) end to end | DuckDB, Polars | Absent entirely | **Re-valued twice.** 2026-08-04: the win is `take`/`filter` (3-13x), not comparison or sort, and only scan-native. 2026-08-25: **the two join rows it was ranked on are now wins** — see item 22 before starting it |
| 10a | `DISTINCT`/`GROUP BY` + `LIMIT` stops once `k` groups exist | DuckDB | **Landed** (`RelOp::Distinct { limit }` + `fuse_limit_into_distinct`) | Was the only *asymptotic* gap; now **13x over DuckDB** on the committed `op-distinct-limit` case |
| 9 | A faster string `ORDER BY` | DuckDB | **Won as of 2026-09-11** — `op-sort-string` 0.91x on a quiet 48-core box | The low-cardinality half landed as `split_constant_ranges`; the high-cardinality half landed as a tie-probability admission rule plus a monomorphic key read (220 -> 176 ms). The adaptive-width key that was this item's open lead is **measured and closed** — see the backlog entry |
| 10i | Common **subplan** elimination (signature-matched, plan-level) | DuckDB | **Landed** (`kyber/common_subplan.py` + `api/subplan_reuse.py`) | 1.95x on a shared aggregate feeding both sides of a join |
| 10b | `row_number() OVER ()` with no `ORDER BY` | DuckDB, Polars (**Spark rejects it too**) | **Done**: SQL lowers it to `with_row_index`; the DataFrame error now names that | Small: SQL portability and an error message, not a capability gap |
| 10h | Ray Data's positional split family, and its names in the compat table | Ray Data | **Landed** (`split_at_indices` / `split_proportionately`, 63 guidance entries, a migration page) | The last *capability* gap the six-engine sweep found |
| 10j | The window aggregate vocabulary, reached from SQL and framed honestly | DuckDB `src/function/window/` | **Fixed** — 7 aggregates were unreachable from SQL; 3 silently ignored an explicit frame | A **wrong answer** (running result returned for a framed query), plus 7 aggregates SQL could not spell. DuckDB's segment tree itself is **not** worth taking — see 10j |
| 3 | Online adaptive reordering of filter conjuncts | DuckDB | **Landed** (`ConjunctOrder`) | Fixes the case a static cost model gets wrong |
| 4 | Top-K heap threshold pushed down as a filter | DataFusion | **Half landed** (`TopNBound` skips morsels; nothing reaches the scan) | Large on `ORDER BY ... LIMIT k` over big inputs |
| 5 | Skew detected from measured partition sizes, split automatically | Spark AQE | **Landed** — detection runs unasked above ~8.4 M rows and the fan-out is sized from the measured share; the residual is 10d (per-key, not per-written-partition) | Removes a config the user cannot be expected to set |
| 6 | Dictionary encoding surviving past the leaf | DuckDB, Arrow | **Unreachable** — decoded at the FFI boundary, so the dict-native kernels never see a dictionary from Python | Compounds with 2 |
| 7 | Adaptive morsel sizing as a pluggable strategy | Daft | Fixed 16,384 rows | Small, and mostly a latency story |
| 8 | A range-join algorithm (IEJoin, or a binned rewrite) | DuckDB | **Landed**, and since re-tuned | **Largest single gap found**: 12–32x, and OOMs where DuckDB runs |
| 26 | Composite integer keys hashed **column at a time**, and that table presized | DuckDB, Polars | **Landed** (`hash_columns_i64` + `GroupGrowth` in `assign_groups_int64_multi`) | 1.4x-6.2x on a multi-key integer `GROUP BY` with many groups; 1.07x-1.12x at low cardinality. Generalizing either half to the *other* group paths loses — see item 26 |
| 11 | Compressed materialization — narrow a key to its *measured* range | DuckDB | **Landed for the multi-key sort** (`packed_multi_sort_indices`); **measured and reverted** for the composite group key | 1.5-3.0x on a multi-key `ORDER BY`; 0.86x-1.04x on grouping, so not taken there — see 11 |
| 27a | A top-N bound proved from row-group statistics | DuckDB `RowGroupPruner` | **Landed** (`kyber/learned_tuning/topn_footer.py` + `api/orchestration/topn_seeding.py`) | `ORDER BY ts DESC LIMIT 10` on 40M clustered rows: first run 422 -> 134 ms, warm 433 -> 45 ms, DuckDB 92 |
| 27d | Float bounds that cannot see NaN | DuckDB `can_have_nan` | **Fixed** — a wrong answer at four layers, invisible to the DuckDB oracle at its default | Correctness: NaN rows returned by every terminal |

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

### The string group-by gap is real, and a wider integer pack does not close it (2026-08-07)

The measurement that says this axis still matters, isolated so the key is the only variable.
H2O db-benchmark group-by table, 1e7 rows, the identical `sum(v1)`, 100,000 groups either way:

| key | bytes | Batcher | DuckDB | ratio |
|---|---|---:|---:|---:|
| `id6` (`int32`) | — | 26.0 ms | 31.6 ms | **0.82x** |
| `id3` (`'id0000039083'`) | 12 | 54.5 ms | 32.0 ms | **1.70x** |

Same cardinality, same aggregate, same everything but the key's type: the string key costs
Batcher **2.1x** what the integer one does, while DuckDB pays the same either way. That single
comparison accounts for most of Batcher's H2O group-by losses (q2 2.78x on two string keys,
q7 2.04x, q3 1.39x — against wins on every integer-keyed case).

`bc_runtime::agg::group::assign` already packs a null-free string of **≤ 7 bytes** into a `u64`
and groups on the integer, which is why the 5-byte `id1` wins and the 12-byte `id3` does not.
The obvious next step — widen the pack to an `i128` and cover 8-15 bytes, where most real
categorical keys live — was built, measured, and **reverted: it is 1.13x slower** (`id3`
49.7/53.0 ms -> 57.6/59.2 ms over two interleaved rounds, one tree, two `.so`s differing only
in that change).

Two costs swamp the saving, and both are properties of a wide key rather than of the
implementation. The packing is a **second full pass** that materializes a 160 MB `Vec<i128>`
the hash path never allocates. And `int_group_ids`' inline-key table goes from 4 bytes a slot
to 20, so on a 100,000-group probe it loses more to cache misses than it gains by comparing
registers instead of slices — the byte path is not naive, it already keeps each group's
representative *slice* beside its id precisely so its comparison costs no indirection.

So the gap is not about how many bytes fit in a register. It is the representation, which is
what this item is about: a `StringView` leaf whose comparison never leaves the array that was
scanned. The negative result is recorded on `pack_short_bytes` itself so the next reader finds
it before rebuilding it.

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
`runtime_filter.rs` transport and then through `io/predicate/`'s pushdown, which is a
boundary-crossing change rather than a data-plane one.

## 5. Skew detected from measured sizes

Spark's `OptimizeSkewedJoin` calls a shuffle partition skewed when its size exceeds both
`median * SKEW_JOIN_SKEWED_PARTITION_FACTOR` and an absolute threshold
(`getSkewThreshold`, `OptimizeSkewedJoin.scala:65`), then splits it toward a target size
that is the mean of the *non-skewed* partitions, floored at the advisory partition size
(`targetSize`, `:75`). It knows which sides may be split per join type
(`canSplitLeftSide`, `canSplitRightSide`). None of this asks the user anything.

Batcher has more learning machinery than Spark here. `python/batcher/dist/skew.py`
persists measured hot join keys so a repeated shape salts without re-running detection, and
distinguishes "measured, not skewed" from "never measured" — a state Spark has no
representation for, and the reason a shape known to be uniform never pays the pre-pass
twice.

**This entry recorded a gap that has since been closed, and the recommendation it made is
no longer the work to do.** It read `skew_join_salt: int = 0` as "off" and concluded that
out of the box a skewed distributed join is not mitigated. That is no longer what `0` means.
`resolve_hot_keys` now takes the hot values from, cheapest first: the set learned for this
join shape, else the column statistics Kyber already holds, else a Misra-Gries pre-pass that
it runs **on its own initiative** above `_DETECT_MIN_INPUT_ROWS` (~8.4 M rows across both
sides), because past that size one pass costs ~4% on a join that turns out uniform against
5.8x for an undetected 40% hot key. The fan-out is then sized from the key's *measured*
share by `salt_factor`, which is Spark's `f · P` overload argument derived rather than
tabulated. The three values are: positive forces the pre-pass and pins the fan-out, `0`
(the default) leaves both to the measurement, and **negative** is the off switch.

So the default behaviour Spark's `OptimizeSkewedJoin` provides is present, and one thing
Spark's is not: the decision is remembered across runs. What remains genuinely open is
narrower than this section originally claimed — skew is detected per *join key*, not from
the measured sizes of the partitions a shuffle just wrote, which is what entry 10d is about.

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

**A third strategy landed 2026-09-12, for the shape neither of the two above serves.** IEJoin
and the band both sort the *left* side, which is the wrong algorithm against a bucket table —
price bands, IP ranges, a histogram's edges, a date dimension. With at most 32 right rows the
join is now `|R|` vectorized comparisons over the left key column, chunked and parallel, with no
sort at all: the operator mix's `op-join-range` (six discount bands against `lineitem`) went
**329 ms to 80 ms**, from 3.68x DuckDB to 0.89x. It also covers the mirror of the band —
one left key bounded by two right columns — which `band::bounds` structurally cannot see. See
`competitive_architecture.md` ceiling 7.

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

**Closed 2026-09-11, and it was the wrong lead.** Measured on a quiet 48-core box, an adaptive
four-byte key loses 0.26x-0.72x above a quarter of a million rows and no sample-drawn width can
rescue it; the full numbers are in `report_the_packed_prefix_alternatives`. The 53% the table
above attributes to "per-range sort" was real, but its cause was `prefix_discriminates`
refusing to build the eight-byte pack **at all** on every one of the thirty ranges -- it scored
pack *distinctness* (~29% within a range, which is a narrow lexicographic band) where the cost
depends on pack *tie probability* (<1%, so the pack settles ~99% of comparisons). With the rule
scoring the right quantity, and with the sort no longer reading each key through the
`ByteKeyColumn` five-arm match it had already resolved, `op-sort-string` is **220 -> 176 ms,
1.13x DuckDB -> 0.91x**.

The **top-N** path kept the resolution rule, deliberately: `top_k_live` keeps the rows whose
pack is at or below the `k`-th best, so a pack that ties half the column hands it half the
column as candidates and it declines at its budget after packing every row — the 146 -> 212 ms
late decline already recorded on `packed`. Cheapening a comparison and narrowing a field are
different questions and are now two rules.

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

### 10b. `row_number() OVER ()` is rejected — **and this entry was wrong when first written**

**Correction, same day.** The first version of this entry said the query "runs in DuckDB, Spark
and Polars" and called Batcher's refusal indefensible. Only DuckDB and Polars had actually been
run; **Spark was asserted from memory and is the opposite of the truth.** Spark's own test suite
pins the failure:

```scala
// sql/core/src/test/scala/org/apache/spark/sql/DataFrameWindowFunctionsSuite.scala:72
test("window function should fail if order by clause is not specified") {
  ... intercept[AnalysisException](df.select(row_number().over(Window.partitionBy("value")))),
  condition = "WINDOW_FUNCTION_FRAME_NOT_ORDERED",
  parameters = Map("wf_name" -> "row_number", ...)
```

with the message defined at `common/utils/.../error-conditions.json:9426`.

**The verified state**, all four checked rather than recalled:

| engine | `row_number() OVER ()` | |
|---|---|---|
| DuckDB | accepts | run |
| Polars | accepts | run |
| **Spark** | **rejects** (`WINDOW_FUNCTION_FRAME_NOT_ORDERED`) | its own test |
| Batcher | rejects | run |

So it is a 2-2 split, and Batcher sides with the *other distributed engine* — for the same
stated reason. `plan/logical/window.py:211` argues that "without an order there is no 'previous'
row: the result would depend on arrival order, which a morselized/distributed scan does not
fix." That is Spark's argument, and it is sound: DuckDB and Polars are single-node and can
define arrival order cheaply, where an engine whose contract is single-node == distributed
cannot.

**What survives is much smaller than the original claim.** The *capability* is already present
under Polars' own spelling — `Dataset.with_row_index(name, offset=)` at
`api/dataset/frame.py:1933`, plus `with_row_count`. What is missing is only SQL portability: a
ported DuckDB or Polars-SQL query using `row_number() OVER ()` fails, and the error names the
missing `ORDER BY` without pointing at `with_row_index` as the deterministic alternative. That
is an error-message and porting-guidance item, not a capability gap, and it is ranked
accordingly below.

Recorded at this length because the failure mode matters more than the item: this document's
whole purpose is to be checkable, and a competitor claim taken from memory rather than run is
exactly what it must not contain.

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

### 10f. Sortedness is tracked, and no operator specializes on it — **half built, and the win was not where this entry said**

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

**Re-verified still open, 2026-08-06**, and it is now the last of the three openings this sweep
found (10a and 10i are built). The greps, so the next reader does not repeat them:
`grep -rniE "sorted_group|sorted_agg|adjacent_dedup|sorted_distinct|sorted_unique" crates/`
returns **nothing**, and the only `sorted_by` consumers under `kyber/rules/` are
`ordering.py`'s two sort-elimination rules (`sort_elimination_from_ordering`,
`topn_over_sorted_input_to_limit`) — neither of which touches `Aggregate` or `Distinct`.

Worth stating what building it costs, because it is larger than its position here suggests and
the cost is mostly *contract*, not algorithm. The adjacency scan is easy; what it needs around
it is a two-sided IR change in one commit (a flag on `Aggregate`/`Distinct` saying the input
arrives ordered by the group keys), a **mergeable** form so the sorted path does not become a
second semantics — invariant #7, and the exact shape of "a stateful operator that works
single-node and silently caps at one machine" — and agreement across all three executors. So
it wants a session that can run the distributed suite, which is the one thing a contended box
cannot currently do.

**Built and measured, 2026-08-13 — and the result is a caution, not a victory.**
`bc_runtime::agg::group::runs` now assigns group ids by scanning runs of equal adjacent values
instead of hashing, and `assign_groups` tries it first. Because `assign_groups` is shared, this
reaches every `GROUP BY`, every `DISTINCT` and every partitioned window at once.

Two things about it are worth keeping; one is worth *not* repeating.

**Worth keeping (1): it establishes the ordering instead of trusting or buying it.** The first
draft of this entry said Polars "reads a flag", which is wrong, and the correction is the most
useful thing in this section. Polars' planner *tracks* sortedness
(`polars-plan/src/plans/optimizer/sortedness.rs`), derives it from a `Sort` in the plan or an
explicit user hint, and deliberately does **not** derive it from a scan — `IR::Scan => None`.
When the keys are not already sorted, `try_build_sorted_group_by` **inserts a `Sort`** and runs
the same node. Polars therefore either knows the order because the plan produced it, or pays to
create it. Neither is careless, and this document should not have implied otherwise.

Batcher's equivalent declaration is `RelStats.sorted_by`, and that one *does* carry a lakehouse
table's declared sort key — metadata **nothing enforces on write**. Believing it when false is
not a slow answer but a silently wrong one: one key split across two non-adjacent runs is
emitted as two groups. So this takes a third option Polars cannot take as cheaply, because it
sits below the planner: establish the ordering per batch, which costs nothing when it does not
hold, needs no `Sort`, and is the reason **no IR flag and no Kyber rule were needed** for this
half.

**Worth keeping (2): a sampled gate is not a cheap exact scan, and the difference is 4x.** The
first implementation gated on a fixed 64-pair prefix before committing to a detection pass. A
key cycling `0,1,…,99,0,1,…` is ordered across any short prefix, so the gate passed it and paid
for a full pass that then failed — **0.24x, four times slower than simply hashing**, on input
that is not sorted at all. Replacing the gate with a chunked scan that exits at the first
violating chunk took every unsorted shape to exactly **1.00x**. Rejection being free is what
makes this safe to attempt unconditionally, and it is the property to defend if anyone
re-tunes it.

**Not worth repeating: the number this was first justified with.** At the level of one
`partial()` call over 6M rows the path measures **6.0x** on an all-distinct sorted integer key,
1.56x on a sorted string key, 1.40x on a sorted composite one. An A/B of two *builds* over the
same data then measured **1.0-1.2x** end to end, and at full parallelism the difference sat
inside the noise band. The engine morselizes, so it never makes the 6M-row call the
microbenchmark made; at 16,384 rows the win is 1.1-1.2x, which is a few percent of a query that
also scans, accumulates, combines and finalizes.

That is **10g happening again to someone who had read 10g** — a mechanism-level probe
disagreeing with the query. The correction that worked, and which the next measurement here
should copy: A/B two builds in *one process* over identical data. Cross-run timings on this box
disagreed with themselves by 30%, and two shapes where the path *declines* read as faster with
it enabled, which is impossible and is what exposed the noise.

So this half is a **small win on sorted input and free on everything else**, tested against the
hash oracle in `runs.rs` and against DuckDB in
`tests/differential/test_diff_agg_sorted_input.py` (which includes the case that matters most:
a key sorted *within* each morsel but not across them, where runs must stay partials and only
`combine` may make them groups).

**What remains open is the half this entry called the reason to rank it: memory.** The state is
still `O(groups)`, because bounded state needs the streaming aggregate to *emit* a group when
its run closes, and that is not a detection problem, it is a soundness one. A group emitted
early is unrecoverable if a later batch reintroduces its key, and per-batch verification cannot
prevent that — it only detects it afterwards. So early emission is sound only where the
ordering is a fact about the *plan* (a `Sort` the engine itself performed) rather than a claim
about the data, and that distinction — not the adjacency scan — is what the remaining work has
to get right. `stream/breaker.rs` is where it lands: the streaming aggregate does not spill, it
refuses (`"the streaming aggregate does not spill"`), so bounding its state is what converts a
refused query into a completed one.

**And the value of that half is smaller than this entry has claimed since 2026-08-04, which is
worth settling before anyone spends a session on it.** The claim has been that a sorted
group-by "turns a spilling aggregate into a streaming one". Check what it is being compared
against: the *materializing* aggregate already spills — `bc_runtime::agg::spill` is real and
`spill_split.rs`, `join_par.rs`, `window_spill.rs`, `distinct_on_spill.rs` and `dist.rs` all
use it. Large aggregates are therefore not failing today; they are paying disk. So the prize is
**avoiding a spill**, not rescuing a query, everywhere except the narrower case where the
streaming path was chosen and then refused.

That reframes the cost/benefit sharply, because of the soundness constraint above. Early
emission is sound only when the ordering is a fact about the plan, and the way to *make* it a
fact is to sort. **Polars is the precedent to read here, not Spark**, and it is a closer one
than this document realized: `try_build_sorted_group_by` inserts a `Sort` when the keys are not
already ordered, which is exactly the design the paragraph above arrives at. Note what Polars
does with it, though — behind `POLARS_FORCE_SORTED_GROUP_BY` when the keys are *not* already
sorted, and taken unasked when they are. That gating is the honest reading of the trade:
sorting in order to avoid a spill costs at least as much as the spill it removes, so it pays
where the plan was going to sort anyway and is opt-in otherwise.

What is left that is unambiguously worth having is therefore narrow and should be scoped that
way: an aggregate **directly above a `Sort` the engine itself performed**, where the ordering
is free and already paid for. Anything wider needs a number first — specifically, the cost of
sorting against the cost of the spill it removes, on a shape where the streaming aggregate
currently refuses.

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

**Closed, 2026-08-06.** All three parts, and the porting surface turned out to be the larger
half exactly as this entry predicted.

- **The two methods** are `Dataset.split_at_indices(indices)` and
  `Dataset.split_proportionately(proportions)`, matching Ray Data's names, validation and edge
  cases (an index past the end gives an empty part, a repeated index gives an empty part
  between the two, and the proportional form nudges colliding boundaries apart so no part comes
  out empty). They add **no operator and no IR**: both lower to `with_row_index` plus a range
  filter, the shape `tail` and `gather_every` already use, so they inherit streaming, spill and
  distribution rather than re-earning them. They also improve on the original in the way the
  architecture forces — Ray Data returns a `MaterializedDataset` and these stay lazy, so a
  pipeline consuming one part never computes the others. Covered by
  `tests/differential/test_diff_split_positional.py` (27 cases, DuckDB as the oracle via
  `ORDER BY k LIMIT n OFFSET lo`, ordered comparisons throughout because an order-independent
  one cannot see a boundary in the wrong place).
- **The guidance table** now carries a `DATASET_RAY_DATA` family of 63 entries
  (`_dataset_naming.py`), so a Ray Data name gets the Batcher spelling out of the traceback
  instead of a bare `AttributeError`. `tests/unit/test_guidance_is_executable.py` holds every
  suggestion to naming something that exists, and caught one invented method while these were
  written, which is the whole reason that test exists.
- **The docs** gained `docs/getting-started/migration/ray-data.md`, since the migration section
  covered pandas, Polars, PySpark, DuckDB and Daft but not the competitor the scorecard ranks
  highest. It leads on the difference that actually changes how a job is tuned: bulk Arrow
  never enters the Ray object store, so there is no object store to size.

### 10j. The window vocabulary — a capability that existed and could not be reached from SQL

Added 2026-08-06, from a pass that deliberately went where the earlier ones had not. Items
1-9 read the competitors' *execution mechanisms*, 10a-10h their *operator inventories*, and
10i their *optimizer passes*. This one read DuckDB's `src/function/window/` — the window
implementation — which no previous pass had opened.

**What is there, and what it is not.** DuckDB gives framed window aggregates a segment tree
(`window_segment_tree.cpp`), a merge-sort tree (`window_merge_sort_tree.cpp`) and a dedicated
distinct aggregator (`window_distinct_aggregator.cpp`). **The segment tree is not worth
taking**, and it is worth saying why so nobody takes it later: it is `O(log n)` per row over
an arbitrary frame, and `bc-runtime/src/window/frame/` is already `O(1)` amortized, because
SQL frame edges are both non-decreasing in row position, so the frame is a FIFO queue rather
than an arbitrary range. `count`/`sum` keep a running accumulator, `min`/`max` a monotonic
deque, and float `sum`/`avg` a two-stack `FifoSum` that never subtracts — which is *more*
numerically careful than a segment tree, not less. Batcher wins this one on the algorithm.

**What the comparison did turn up is a seam defect, in two directions.** Reading DuckDB's
window function list against Batcher's produced a discrepancy inside Batcher rather than
between the two:

1. **Six window aggregates the engine computes were unreachable from SQL.**
   `_sql/parser/windowing/translate.py::_window_func` mapped exactly five names
   (`sum`/`avg`/`min`/`max`/`count`), so `bit_or(x) OVER (...)` failed with "unsupported
   window function: **bitwiseoragg**" — sqlglot's node name for a typed aggregate, which only
   the *aggregate* front-end translated. Nothing was missing but the mapping:
   `WINDOW_AGGREGATES` lists all of them, the runtime computes them, and
   `col("x").bit_or().over(...)` returned DuckDB's own values throughout. Fixed by making the
   two five-name copies one table; `bit_and`, `bit_or`, `bit_xor`, `bool_and`, `bool_or`,
   `stddev` and `variance` now match DuckDB in both the running and whole-partition forms.

2. **A frame was silently dropped on three aggregates, which is a wrong answer rather than a
   missing feature.** `api/dataset/_build.py` reduced the frame to `None` for anything outside
   `WINDOW_FRAMEABLE`. That is correct for a function SQL gives no frame either (ranking,
   `lag`/`lead`, the fills) and wrong for an aggregate, where SQL defines the framed result:
   `window(frame=(-1, 0), functions={"w": ("stddev", "f")})` returned the **running** standard
   deviation — DuckDB's answer to the query *without* the frame. `stddev`, `var` and
   `count_distinct` are exactly the set in that gap, and they now raise.

   This is the shape `CLAUDE.md` warns about and no mechanical check catches: the column that
   comes back is a perfectly good number, so every gate stayed green and an order-independent
   comparison could not have seen it. It was **pre-existing** — the line is unchanged at
   `HEAD` — and reachable from the public DataFrame API, not only from SQL.

`tests/differential/test_diff_window_agg_vocabulary.py` covers both, and includes a case
asserting that each framed answer *differs* from its running one, because a frame test whose
two answers coincide proves nothing — it caught two of its own cases doing exactly that.

### 10k. Correlated `EXISTS` with a *mixed* equality-and-inequality correlation — **landed**

Added 2026-08-06, from reading DuckDB's `src/execution/operator/join/`, which no pass had
enumerated. Its delimited joins (`physical_delim_join.cpp`, plus the left/right variants) are
the machinery DuckDB decorrelates subqueries with, so the question they raise is what Batcher
refuses.

**Almost nothing, and the first draft of this entry was wrong about it.** "Correlated `EXISTS`
is not supported" is what the error message says and what a single probe suggested; running
the four shapes separately says otherwise:

| correlation inside `EXISTS` | Batcher | matches DuckDB |
|---|---|---|
| equality only (`x.g = t1.g`) | works | yes |
| `NOT EXISTS`, equality only | works | yes |
| inequality only (`x.v > t1.v`) | works | yes |
| **equality *and* inequality together** | **`NotImplementedError`** | — |

So the gap is one shape, not a feature: `subquery/core.py::_apply_exists` decorrelates an
equality correlation into a tagged left join, and `subquery/range.py` handles a pure
inequality, but a predicate carrying both falls through to `_reject_correlated`. The fix is to
keep the equalities as the join keys and apply the inequality as a residual predicate on the
joined rows before the distinct — a semi-join with a residual, which is a real change to the
join path rather than a parser edit, which is why it is recorded rather than built here.

Recorded at this length for the same reason 10b is: the error message overstates the
limitation, and a reader who trusted it would rebuild machinery that already exists. Correlated
*scalar* subqueries and correlated `IN` both work too, and were also checked rather than
assumed.

**Built 2026-08-13** as `_sql/parser/subquery/specialized.py`, and the entry's own prescription
turned out to be the wrong plan. It said the fix "needs a semi-join with a residual predicate
rather than a parser edit". The engine has no such operator and should not grow one for this:
`bc_ir::RangeOp` deliberately excludes `=` (an equality is a hash join), and a semi join emits
no right columns, so there is nothing for a residual to read. Adding an equi-prefix to
`RangeJoin` would be a two-sided IR change for one SQL shape.

The general decorrelation needs no engine change at all. Tag each outer row, inner-join on the
equality keys, apply the inequality as an ordinary filter on the joined rows, and reduce the
survivors to the set of tags that matched. **The tag is the load-bearing part**: without it the
final `DISTINCT` runs over outer *values* and collapses two identical outer rows into one,
where `EXISTS` must keep both — the same trap `test_correlated_exists_preserves_outer_duplicates`
already pins for the `range` path. The cost is that the join materializes matching pairs the
filter then discards, where a semi join would stop at the first match; that is the price of the
shape having had no plan at all, and it is bounded by the equality keys rather than being a
cross product.

Two things the work turned up that the entry did not predict:

- **`NOT EXISTS` with a mixed correlation was refused too.** The table above tabulated
  `NOT EXISTS` only for the equality-only case, so the row read as working. Both are fixed.
- **The refusal was one dispatch away from a bug of its own.** `core._apply_exists` tested the
  two specialized shapes in two separate blocks, and the second had to go before the
  *uncorrelated* branch while the first went after it. Consolidating them into one call
  (`decorrelate_correlated_exists`) and then placing it wrongly silently sent every
  inequality-only `EXISTS` down the uncorrelated path — caught by the regression case in the
  new differential file, which exists precisely because every neighbouring shape already
  worked.

The boundary is now stated rather than implied: a correlation on an *expression*
(`x.v < t1.v + 100`) is still refused, because `RangeCondition` carries column names on both
sides, and `test_a_correlation_on_an_expression_is_still_declined` pins that it refuses rather
than mis-plans.

### 10l. A construct-by-construct SQL probe — 35 of 38, and three that raise

Run 2026-08-14, after 10k, on the reasoning that made 10j and 10k productive: **reading a
competitor's implementation of something Batcher already has is what makes an internal gap
visible**, and the gap is usually at the seam between the SQL front-end and an engine that can
already do the work. 10j found seven window aggregates the engine computed and SQL could not
spell; 10k found one correlation shape out of four. So this pass stopped reading source and
asked the question directly: **38 SQL constructs, run against both engines, compared.**

The result is the strongest evidence in this document for the "coverage is broad" claim, and it
is worth stating before the gaps. Thirty-five answered and matched, including several this
document would not have assumed: `FILTER` on an aggregate, `string_agg`/`array_agg` with an
inner `ORDER BY`, `INTERSECT ALL` and `EXCEPT ALL`, `QUALIFY`, `LATERAL`, `IS NOT DISTINCT
FROM` as a join condition, `DISTINCT ON`, `WITH RECURSIVE`, `GROUP BY ALL`, `ORDER BY ALL`,
`SELECT * REPLACE`, named windows, and a `RANGE` window frame.

Three raised where DuckDB answered:

| construct | status |
|---|---|
| `<expr> IN (SELECT …)` — an expression, not a column, on the left | **Fixed** (`subquery/in_expr.py`) |
| `EXCLUDE CURRENT ROW` / `TIES` / `GROUP` on a window frame | open, engine work |
| `count(DISTINCT (a, b))` — a row value inside an aggregate | open, front-end |

**The one that was fixed is the one worth reading**, because the restriction was not where it
looked. `_apply_in_subquery` reads the left side as a *column name* so it can hand it to a
semi/anti join; an expression has no name to hand over. Everything past that point — the
correlation split, the multi-column row value, and the three-valued `NOT IN` — was already
general. So the fix names the value (evaluate it into a synthetic column, rewrite the predicate
to name that column, recurse once into the case that does not come back) rather than adding a
second `IN`.

That choice is load-bearing rather than tidy. `x NOT IN (S)` is **not** an anti join when `S`
can yield NULL — `_not_in_antijoin` implements the three-valued answer — and a separate
implementation for expressions would have had to restate that rule, would have looked correct
on every input without a NULL, and is exactly the "second implementation of the same semantics"
this repository's defects cluster in. Routing back through the one path makes restating it
impossible.

The two left open are recorded rather than built, with their reasons. Frame `EXCLUDE` needs the
window kernels to skip rows inside a frame, which is engine work and not a parser edit — the
one case in this table where the deferral reason is real, unlike 10k's. `count(DISTINCT (a,b))`
needs a composite key inside an aggregate; the `IN` path already accepts a row value, so the
vocabulary exists on one side of the front-end and not the other, which is the same shape of
gap as 10j.

### 10m. The whole function catalog, enumerated rather than sampled — 367 of 516

Run 2026-08-14. Every earlier pass compared *operators*, *optimizer passes* or *streaming
nodes*. None compared the widest surface either engine has: the scalar and aggregate function
catalog. DuckDB exposes its own (`duckdb_functions()`), so this one can be enumerated instead
of sampled, which is the only pass here that is exhaustive over its surface rather than
representative.

**Method.** Take DuckDB's distinct scalar + aggregate function names, drop what is not
user-facing (`__internal_*`, `duckdb_*`, `pragma_*`, `variant_*`, the ~160 `icu_collate_*`
locale entries, and operator spellings), leaving **516**. Call each from Batcher's SQL across
thirteen argument shapes, and count a name unreachable only when *every* shape reports an
unknown function.

**Result: 367 reachable, 149 not.** Of the 149, most are DuckDB-specific plumbing rather than
portable SQL — sequences (`nextval`, `currval`), settings and transactions (`current_setting`,
`txid_current`), logging, `enum_*`, `union_*`, `bar`, `stats`, `switch`, and `st_*` spellings
Batcher already serves under its own names.

**Read the count with the correction attached, because the first version of this entry was
wrong by 2.4x.** Probing each function with a *single* argument reported **362** unreachable.
That number is an artifact: Batcher answers a known function called at the wrong arity with
`unknown function 'gcd'`, so a one-argument probe of a two-argument function reads exactly like
a missing one. `gcd(x)` says unknown; `gcd(n, 4)` returns an answer. Probing across arities cut
the figure to 149. **Any future probe of this surface has to vary arity**, and the same caution
applies to reading Batcher's error message as evidence of anything.

That error message is itself the smallest finding here and worth fixing on its own account: it
reports an arity mismatch as a missing function, which sends a user to `bt.register_function`
for something the engine already computes. It is the same class as 10b — an error that
overstates the limitation — and it misled a probe written by someone who knew to be careful.

**One true "the engine has it and SQL cannot spell it" case** was found, which is the 10j shape:
`hash` exists as an `Expr` method and no SQL spelling reaches it, while `md5`/`sha1`/`sha256`
are reachable from both. The reverse also occurs (`md5` is reachable from SQL and is not a
bare `Expr` method), so the two surfaces have drifted in both directions rather than one.

**Everything else in the 149 is absent from both surfaces**, checked against the `Expr` API
rather than assumed. Grouped, and worth having in roughly this order:

- `printf`/`format` — no format-string function on either surface.
- The list higher-order tail: `list_reduce`, `list_zip`, `list_aggregate`, `list_resize`,
  `list_where`. `list_transform` and `list_filter` are both built, so this is a partial family.
- The struct vocabulary: `struct_insert`, `struct_values`, `struct_concat`, `struct_extract_at`.
  The `.struct` accessor has **three** methods against `.list`'s 74, which is the widest
  namespace asymmetry on the surface.
- `to_json` / `from_json` / `row_to_json`.
- Interval constructors (`to_seconds`, `to_minutes`, `to_months`, …) and `age`, `datepart`.
- `signbit`, `bit_position`, `set_bit`; `strip_accents`, `nfc_normalize`, the grapheme-aware
  string ops; `like_escape` (the `ESCAPE` clause).

### The same pass across Polars and Ray Data, and why its raw numbers are worthless

Polars' `Expr` surface (432 names across its accessors) and Ray Data's `Dataset` (93 methods)
were enumerated the same way and diffed against Batcher's — 692 `Expr` names and 177 `Dataset`
methods. **Do not read those diffs as gaps.** A bare-name diff reported 138 Polars names and 63
Ray Data names "missing", and a verified sample of thirty found the great majority to be
spelling or placement differences:

- Ray Data's `select_columns`, `rename_columns`, `drop_columns`, `add_column`,
  `random_shuffle`, `random_sample`, `take`, `write_parquet`, `materialize`,
  `iter_torch_batches` and `train_test_split` are all present under Batcher's own names.
- `num_blocks`, `zip` and `input_files` are **deliberate refusals** whose error messages name
  the alternative ("Spelled `ds.repartition`", "There is no positional column …").
- Polars' `Expr.list.explode` is `ds.explode` — a Dataset-level operation here.
- `streaming_split` **is** built (`ml/loader/lazy.py`, exported from `batcher.ml`), exactly as
  10h says. The probe missed it by looking for a `Dataset` method rather than a module
  function, and briefly "found" a gap that 10h had already settled.

What survived verification is short, and every entry was called before it was written down:
**`bottom_k`** (while `top_k` is built — the cleanest asymmetry on the surface), `null_count`,
`cum_prod`, `implode`, `hist`, `list.index_of`, `json.decode`, `str.extract_groups`.

**The methodological finding is worth more than the list, because three different probe designs
each produced false gaps, and all three erred the same way — overstating what is missing.** A
one-argument call read an arity error as a missing function (2.4x). A bare-name diff read a
spelling difference as an absence (roughly 5x). An attribute probe read a module-level function
as an absence. A pass over a competitor's surface is not evidence until each survivor has been
*called*; the cost of not doing that is a backlog of work that is already done, which is the
failure this document has recorded against itself three times already.

**The headline, though, is the 367.** Batcher's `Expr` surface is 236 methods plus roughly 450
across ten accessor namespaces, and the great majority of DuckDB names this pass first read as
missing turned out to be present under Polars-style spellings that the SQL front-end already
maps. That is the strongest evidence in this document for the coverage claim, and it is the
reason the remaining list is short enough to read.

### 10n. `top_k` means two different things, and one of them is silent

Found while checking whether `bottom_k` was worth adding (Polars has it, Batcher does not). It
is not the missing name that matters:

```
values = [5, 1, 1, 1, 9, 2, 2]

Dataset.top_k(2, by="x")   ->  [9, 5]      the two highest-ranked rows
Expr.top_k(2)              ->  [1, 2]      the two most FREQUENT values
Polars  top_k(2)           ->  [9, 5]
DuckDB  approx_top_k(x, 2) ->  [1, 2]
```

Both Batcher spellings are individually correct and individually documented — `Dataset.top_k`
follows Polars, `Expr.top_k` follows DuckDB's `approx_top_k` and its docstring says so. The
defect is that **one name means two different things across the two surfaces of one API**, and
the mismatch is silent: a Polars migrant reaching for `bt.col("x").top_k(2)` expecting the
largest values gets a frequency ranking, of the right type and a plausible length, with no
error.

That places it in this repository's worst category rather than its naming-nit category — a
wrong answer no gate can see. The `compat` guidance machinery cannot help either: it fires from
`Expr.__getattr__` for names Batcher does **not** carry, and this name is carried.

**Not fixed here, because the fix is a public-API decision rather than a defect repair**, and
`plan/expr_ir/core.py` is another session's open file. The options, in the order this pass would
rank them:

1. Rename the aggregate to `most_frequent` (what it computes) and leave `top_k` to mean what it
   means everywhere else. It is the only option that removes the ambiguity rather than
   documenting it.
2. Keep both and make `Expr.top_k`'s docstring open by contrasting itself with `Dataset.top_k`,
   which today it does not mention.
3. Add `Expr.bottom_k` — **do not do this first**. Whichever meaning it took would deepen the
   collision, and the question of which meaning it should take is the same question as 1.

Recorded rather than built for the same reason 10b was: the cost of guessing is larger than the
cost of writing it down, and this one needs an owner's decision rather than a patch.

### 10i. The optimizer pass list — one real gap, and it settles 10g

The sweep above compared *execution operators*. This compares **optimizer passes**, which is
the more pointed comparison: Kyber is the moat, so a pass DuckDB has and Kyber does not is
worth more than an operator. `duckdb/src/optimizer/` holds ~40 of them.

**The gap: common subplan elimination.** `common_subplan_optimizer.cpp` computes a *signature*
per subplan, canonicalizing column indices so that two structurally identical subtrees compare
equal even when their bindings differ, and rewrites the duplicates to one computation. Kyber has
no equivalent — `rules/extra/cse.py` is expression-level, within a single `Project`, and the only
`grep` hit for "subplan" under `kyber/rules/` is an unrelated docstring in `joins/order.py:159`
about tracing a column to its leaf.

This also **settles the question 10g left open**. Two timing probes there failed to establish
whether Batcher reuses a repeated subplan, because Kyber answered both shapes from metadata. The
source answers it directly: it does not, and DuckDB thinks the technique is worth a dedicated
pass with signature-based matching. 10g's caution about probing this optimizer with a stopwatch
still stands; it was the method that was wrong, not the question.

**Related, and already on the backlog rather than new.** `late_materialization.cpp` rewrites
`SELECT * ... ORDER BY x LIMIT k` so the scan fetches only the key plus a row locator, then
fetches the wide columns for the `k` survivors alone. Batcher already does the *operator*-level
half — `ops::parallel_top_n` emits "only its top-k **sort-key values** plus a `(morsel, row)`
locator" and gathers the payload once. What DuckDB's version additionally saves is the *I/O* of
reading those columns at all, which needs a positional fetch from the scan. That is the same
family as backlog item 8 (the scan half of the top-K dynamic filter): both are about getting
information down to the reader so it reads less. Sequence them together.

**Ruled out here, both already present and one of them better than DuckDB's.**
`partial_aggregate_pushdown.cpp` against `kyber/rules/agg_pushdown.py`, which distinguishes
`eager_aggregation` (min/max, "idempotent under the join's row duplication, so it is correct for
*any* fan-out") from `pre_aggregation_through_join` (sum/count, admitted only when the other side
is "provably unique on the join key") — a fan-out safety argument stated more carefully than the
pass it matches. And `empty_result_pullup.cpp` against `limit_extra.py`'s empty-marker machinery,
covered by three differential files (`test_diff_empty{,_propagation,_relation}.py`).

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

### 10j. Ray Compiled Graphs — measured, and ruled out for this data plane

Ray's compiled graphs (`ray.dag` + `.experimental_compile()`) are the technique a Ray user asks
about first, so the reason not to adopt them belongs on the record with a number attached rather
than as an opinion.

**What they buy, and why most of it is already banked.** A compiled graph pre-allocates the
channels between a *fixed* set of actors, so a repeated call skips task submission, skips the
object-store put/get, and can move GPU tensors over NCCL instead of host memory. Two of those
three are things Batcher's design already does not pay: bulk Arrow moves worker-to-worker over
Flight and never enters the object store (the A1–A14 row of the scorecard above), and every
GPU fan-out task reads its own shard straight from storage and returns a *mergeable partial*,
so there is no large intermediate to route. What is left is the task-submission latency of the
small control messages — `(addr, ticket)` in, a published count out.

**Measured on this cluster, Ray 2.56**, against the exact call shape the streaming pipeline's
hottest hop uses (a tuple in, an int out, 2,000 calls, warm):

| Path | Per call |
|---|---|
| Plain actor call | 822 µs |
| Compiled graph | 546 µs |
| Saving | 276 µs (34%) |

**Why that is not worth taking.** The saving is real and it is small against what a hop
actually does: a morsel is 16,384 rows, and the inference forward pass it triggers is tens to
hundreds of milliseconds, so 276 µs is a fraction of a percent. Against it, a compiled graph
requires a static actor set and a static topology, and the streaming scheduler's whole value is
that neither is static — it re-deals a morsel to whichever actor went idle, replaces a preempted
actor and replays its subtree, and (since this pass) grows a stage that falls behind. There is no
partial adoption: compiling the hop freezes the assignment that makes all three possible.

**The vendor's own field guidance says the same, on both counts.** Anyscale's
`foundations/core/accelerated-dag.md` (`../optimization-guides`) carries a "When to Use" table
whose two "No" rows are exactly this workload: *"Batch processing (steps > 100 ms) — No,
scheduling overhead is negligible"* and *"Dynamic DAG structure (changes per request) — No,
compilation overhead not amortized"*. Its "Yes" rows are the complement — sub-10 ms GPU steps
and a fixed DAG at high QPS — which is a serving shape, not an analytics one. The same page
lists the feature's open limitations as of Ray 2.54 (no NCCL broadcast, request blocking in
multiples of the read timeout under network instability, no pinned memory for host/device
copies), any one of which would be a new failure mode inside a path that currently recovers
from preemption by construction.

**Where it would be worth revisiting.** If a stage ever holds tensors resident on the device
across operators — the open half of G4, which Ray has not solved either (its GPU object store is
RFC-only, ray#51173) — then the NCCL transport becomes the point rather than the latency, and
this decision should be re-taken against that. Until then the residual is a driver-side
concurrency limit, not a per-call one: at ~822 µs and four control calls per morsel, one driver
saturates at roughly 300 morsels/s, and the fix for *that* is fewer driver round trips rather
than faster ones.

## 11. Compressed materialization — narrow a key to its measured range (2026-08-18)

**Source: DuckDB, `src/optimizer/compressed_materialization/`.** DuckDB rewrites a column to
`value - min` at the smallest integer width its catalog statistics allow, immediately before any
operator that *materializes* the value: `compress_aggregate.cpp`, `compress_comparison_join.cpp`,
`compress_distinct.cpp`, `compress_order.cpp`. An `INTEGER` column whose statistics say
`[1000, 1050]` becomes a `UTINYINT` offset, so the hash table entry, the sort payload and the
join key all shrink.

Nothing in this document had covered it: a grep of both ledgers for "compressed
materialization", "zone map", "late materialization" and "radix partition" returned zero before
this pass.

Batcher can do the same narrowing **better in one respect and worse in another**, and both
follow from where it would live. DuckDB's is a *planner* rewrite driven by catalog statistics, so
it reaches only columns a catalog describes and only as well as the statistics are current.
Batcher's data plane already holds the rows, so a min/max scan measures the range *exactly*, on
every input, including intermediates no catalog has ever seen — but it must pay for that scan
every time, where DuckDB's is free at plan time.

That trade decides the result, and it came out differently for the two operators tried.

### 11a. The multi-key sort — **landed, 1.5x to 3.0x**

`ORDER BY <a>, <b>` had no fast path at all. `radix_sort_indices` takes a single column, so every
multi-key sort fell to the row-encoded comparison sort: an encode of every row into arrow's
comparable byte format, then `O(n log n)` memcmps over it. `ORDER BY o_orderdate,
o_shippriority` is about 2,400 and 5 distinct values — fifteen bits, against the ninety-six the
declared types claim and the twelve-plus bytes the encoder writes.

`radix_sort::packed_multi_sort_indices` measures each key's live range, gives it
`ceil(log2(range))` bits, and packs the tuple into one `u64` **most-significant key first** — so
an integer comparison of two packed keys is a left-to-right comparison of their fields, which is
the definition of lexicographic order. The sort is then the LSD radix that already existed, and
`lsd_radix` skips a constant byte, so a fifteen-bit key costs two counting passes rather than
eight.

Measured on 8 M rows, best of three **interleaved** runs (base and candidate alternating, so a
busy box's drift lands on both arms):

| Shape | Before | After | |
|---|---|---|---|
| `ORDER BY <date>, <priority>`, 12,000 distinct pairs | 1,878 ms | 625 ms | **3.00x** |
| `ORDER BY <int>, <int>`, 3 M distinct pairs | 112 ms | 59 ms | **1.92x** |
| `ORDER BY <int> DESC, <int>` | 90 ms | 60 ms | **1.48x** |
| `ORDER BY <int>, <int>, <int>`, narrow | 133 ms | 71 ms | **1.86x** |
| two full-width `Int64` keys (declines) | 66 ms | 61 ms | 1.10x |
| `ORDER BY <string>, <int>` (declines) | 199 ms | 200 ms | 1.00x |
| single `Int64` key (untouched) | 45 ms | 43 ms | 1.05x |

Four things carry the equality with the comparison sort, and all four are pinned by
`packed_multi_key_tests`, which compares against arrow's own `lexsort_to_indices` with the
row-index tie-break appended — the path this replaces, sharing no code with it:

* **Direction lives in the key.** Each field holds the order-preserving `u64` the single-key
  radix already uses, which folds `descending` in by inverting, so a sort may mix directions per
  key and the packed integer still sorts ascending.
* **Nulls are encoded inside their field**, not partitioned out, because a multi-key sort's nulls
  are per column and interleaved. A null takes the field's lowest value under `nulls_first` and
  its highest otherwise; the field widens by one value only when the column has a null.
* **A constant column takes zero bits** — it cannot separate two rows, so it costs nothing.
* **The radix is stable**, so full ties keep input order, which is the tie-break every other path
  gets from its trailing row-index column.

**Nobody else does this for a sort, and the reason is where each of them puts it.** DuckDB's
narrowing is a *planner* rewrite off catalog statistics, so it does not reach a table registered
from Arrow, an intermediate, or a stale-statistics column — the three cases that dominate here.
Polars encodes multi-key sorts through its row format
(`polars-core/src/chunked_array/ops/sort/arg_sort_multiple.rs::_get_rows_encoded`), and
DataFusion hands them to arrow's `lexsort_to_indices`
(`physical-plan/src/sorts/sort.rs:905`); both are the path this replaces. So this is DuckDB's
idea moved into the data plane, where the range is a fact about the rows rather than a claim
about the table.

Two measured lessons are worth more than the ratios, because both were mistakes first:

1. **Decide before materializing.** The first version measured each column's width from a
   materialized rank array, which made the *decline* the expensive case: two full-width `Int64`
   keys over 8 M rows built 128 MiB of ranks and then rejected the budget, at **1.41x slower**
   than not trying. Widths now come from two values — the extreme ranks are the ranks of the
   extreme values, because the encoding is monotone — so the scan allocates nothing and the
   ordering rule is still stated once.
2. **A prefix can reject but never accept.** A range measured over a subset can only widen, so
   4,096 rows are enough to *prove* a key will not fit, while a prefix that fits proves nothing
   and the exact scan still runs. That one-sidedness is what makes the guard safe to consult
   before the real measurement rather than instead of it, and it takes the decline from 0.96x to
   1.10x.

### 11b. The composite group key — **built, measured, reverted**

The same narrowing applied to `assign_groups`' composite-integer path, whose own notes record a
probe costing **362.7 ns/row** at 1.7 M groups because verifying a hash hit reads the
representative row in *every* key column. Packing the tuple into one `u64` and handing it to the
single-integer grouper — whose table stores the key inline beside the group id — should have
removed that read outright.

It did not pay, on any shape, and the reason is worth recording so the next reader does not
rebuild it:

| Shape, 8 M rows | Base | Packed | |
|---|---|---|---|
| `GROUP BY <int>, <int>`, 8 M groups | 40.7 ms | 41.2 ms | 0.99x |
| `GROUP BY <orderkey>, <orderdate>, <shippriority>` (TPC-H q3) | 52.5 ms | 61.0 ms | **0.86x** |
| `GROUP BY <str>, <str>` via ranked codes | 33.6 ms | 32.2 ms | 1.04x |
| `GROUP BY <int>, <int>`, 3 M groups | 42.4 ms | 43.2 ms | 0.98x |
| low-cardinality composite (dense map, untouched) | 12.0 ms | 11.4 ms | 1.05x |

**The premise was wrong: that per-column read is not on the common path.** `eq_rows` runs only
when a probe *matches* — hashbrown's SIMD tag check dismisses a non-match without ever calling
it. A high-cardinality group-by is mostly *inserts*, so the read the packing removes barely
happens; what remains is the pack's own passes, which is why the three-key shape, paying three of
them, is the worst row in the table. The 362-469 ns/row that motivated this is real, and it is
a *match*-heavy measurement.

A row floor (skip the packing below 2^17 rows, so the per-morsel partial aggregate keeps the
existing path) removed the regression at morsel scale and left the whole-relation regroup, where
the three-key shape still read 0.86x. There is no gate that rescues it, because the shape it
would need — many matches, a table too large for cache, and wide key columns at once — is not
one the measurements found.

So: the aggregate/distinct/window grouper keeps its existing dispatch, which is already six
specialized paths deep and good. **Do not re-derive this.** DuckDB's `compress_aggregate.cpp`
looks like the same win as its `compress_order.cpp` and is not, because DuckDB's narrowing is
free at plan time and Batcher's costs a scan — and a scan is only repaid by an operator that
would otherwise touch the key many times per row. A sort does. A hash insert does not.

### 11c. What the benchmark case found: a temporal key never parallelized at all

Adding a case for the shape 11a optimizes is what made this visible, and it is the larger
finding of the two. `op-sort-multikey-narrow` (`ORDER BY l_shipdate, l_suppkey` over 6 M
`lineitem` rows) read **37.13x against DuckDB** — 1,590 ms to DuckDB's 43 — *after* 11a had
already taken 3x off it.

The cause is one predicate. `sample_sort::parallel_sort_batch` routes its leading key with
`dt if dt.is_integer()`, and `arrow::datatypes::DataType::is_integer()` is **false for
`Date32`, `Date64`, `Timestamp`, `Time32/64` and `Duration`**. Every temporal leading key fell
through to the `_ => return Ok(None)` arm, so `ORDER BY <date>` — one of the commonest sorts an
analytic query writes — ran **serially, on a 96-core box**, and had done since the sample-sort
was written.

Measured on 6 M rows, before the fix, which is what a missing parallel path looks like from
outside:

| Sort | Time |
|---|---|
| `ORDER BY <date>`, 2,500 distinct values | 643 ms |
| `ORDER BY <int64>`, **10,000** distinct values — more work, same shape | 38 ms |

Admitting the integer-ordered temporals to the same `i64` routing (`is_integer_ordered_temporal`)
is the whole fix. Those types are physically signed integers whose numeric order *is* the type's
order, in every unit and with or without a time zone — a zone changes the wall-clock rendering,
never the instant.

**Two types stay out, and the second one nearly shipped as a regression.** `Interval` is not one
integer at all: `MonthDayNano` is three fields in 128 bits and no `i64` orders it. `Time32` *is*
one integer, and **arrow will not widen it** — `can_cast_types(Time32(_), Int64)` is false in
arrow-rs, and arrow C++ refuses `date32 -> int64` too, so this is a gap the two implementations
already answer differently rather than a hypothetical. The first version of this change admitted
`Time32`, which would have turned `ORDER BY <a time column>` over 2^17 rows from a working serial
sort into a **raise** from inside the routing. What caught it was a test asserting that every type
the predicate admits is one arrow can actually cast — a property no correctness test over *data*
would have found, because the shape simply never appeared in one. The routing now also declines on
a failed cast, so the list being wrong costs a missed optimization and never an error.

End to end, on the committed benchmark case, against DuckDB on its native store:

| Case | Before | After | vs DuckDB before | vs DuckDB after |
|---|---|---|---|---|
| `op-sort-multikey-narrow` | 1,590.4 ms | **52.1 ms** (30.5x) | 37.13x | **1.28x** |
| `op-sort-multikey-wide` (both paths decline) | 68.6 ms | 71.6 ms | 0.97x | 1.17x |

The two fixes compound and neither subsumes the other: on 6 M rows the date sort goes
643 -> 50 ms from parallelism alone (12.9x), and the two-key sort 1,827 -> 58 ms (31.6x) from
parallelism plus the packed key. The `-wide` row moved inside the box's noise band — DuckDB's own
time on it swung 70.8 -> 61.1 ms across the same two runs — and is the decline path, which does
no work either version did not.

**The lesson is the one this document keeps relearning.** The gap was not subtle, it was 37x,
and it survived because no benchmark case sorted on more than one fixed-width key. Item 9 records
the same thing about string sorts. A shape with no case is a shape with no floor.

One thing worth recording because it is the opposite of what you would guess: **the
*distributed* sort had temporal keys all along.** `dist/executor.py::_range_partitionable_sort_key`
admits `pa.types.is_temporal` explicitly and routes it through the order-preserving integer
backing. So the single-node parallel path was the outlier, and the two had disagreed on which
sorts parallelize for as long as both existed — a query that scaled out on a date key ran serially
on one node.

One thing this pass nearly recorded and should not have: that `lowcard::rank_part_of` — the
rank routing that rescues a sort on a *seven-value* key — being string-only leaves a gap for
low-cardinality integer and date keys. **The first measurement said so and was confounded**: it
sorted a five-column table, so what it actually compared was the cost of gathering a string
column. Repeated with the same payload on both sides, `ORDER BY <a seven-value Int64>` is
**37.6 ms** against **51.6 ms** for the string equivalent *with* rank routing — so the plain
boundary path integers already take is the faster of the two, and generalizing the rank path to
them has no evidence behind it. What the clean measurement did surface is a different anomaly,
recorded as backlog item 14.

## 12. The six-engine operator census, 2026-08-18

Item 10 read the competitors' operator *inventories* once. This pass repeats it against every
engine at once and, unlike that one, **checks each answer with a measurement rather than a grep**
— which is what changed three of the conclusions below, in both directions.

The sources are the physical-operator directories, which is where an engine's real vocabulary
lives: `duckdb/src/execution/operator/{aggregate,join,order,projection,scan,set}`,
`datafusion/datafusion/physical-plan/src`, `polars/crates/polars-stream/src/nodes`,
`spark/sql/core/.../execution`, `Daft/src/daft-local-execution/src/{intermediate_ops,sinks}`,
`arrow/cpp/src/arrow/acero`, and the installed `ray/data/_internal/execution/operators`.

### What the inventory found

**Almost nothing is missing.** Batcher's `Dataset` carries 177 public methods and its IR 18
`RelOp` variants; every operator named by DuckDB, Polars, DataFusion, Spark, Daft, Acero and Ray
Data has an equivalent, with four exceptions, all small:

| Present elsewhere, absent here | Source | Verdict |
|---|---|---|
| `positional_join` (join two relations by row position) | DuckDB `physical_positional_join.cpp` | Expressible as `with_row_index` + an equi-join. Not worth an operator. |
| `merge_sorted` (merge two already-sorted inputs) | Polars `merge_sorted.rs`, Acero `sorted_merge_node.cc` | **Measured and ruled out** — the shape it serves is already 5.3x faster here than in DuckDB. See 12b. |
| positional `zip` of two relations | Ray Data `zip_operator.py`, Polars `zip.rs` | Same as `positional_join`. |
| `cte_inlining`, `unnest_rewriter` | DuckDB optimizer | Parser-level; `_sql` already inlines non-recursive CTEs by substitution. |

Polars' streaming node list is the one worth reading in full, because it is the longest and the
most specialized (`forward_fill`, `backward_fill`, `interpolate`, `ewm`, `cum_agg`, `rle`,
`rle_id`, `shift`, `peak_minmax`, `rolling_group_by`, `dynamic_group_by`, `is_first_distinct`,
`sorted_unique`, `sorted_group_by`, `top_k`, `gather_every`, `with_row_index`). **Batcher has
every one of them**, mostly as window functions rather than as nodes.

DuckDB's `src/optimizer/` is the same exercise for plan-level work, and it is the better
checklist because it is a flat list of 38 named passes. Kyber has an equivalent for all but two
(the two named above), including the ones that are easy to assume are missing:
`join_elimination`, `outer_join_simplification`, `partial_aggregate_pushdown`
(`kyber/rules/agg_pushdown.py::eager_aggregation`), `remove_duplicate_groups`,
`common_aggregate_optimizer`, `regex_range_filter`, `in_clause_rewriter`, `sampling_pushdown`,
`topn_window_elimination`, `window_self_join`, `limit_pushdown`, `build_probe_side_optimizer`.

### Three conclusions the measurements reversed

The inventory is not the interesting part. These are, and each one is a case where reading the
code gave the wrong answer:

1. **`late_materialization` looked like a real gap and is not.** DuckDB has a whole optimizer
   pass for it and a grep of Batcher for the term returns zero. Measured on 4 M rows x 31
   columns, `SELECT * ORDER BY k LIMIT 10` costs **12.4 ms** against **8.0 ms** for
   `SELECT k ORDER BY k LIMIT 10` — thirty extra columns for 4.4 ms, which is not a payload
   being carried through a sort. The top-N path already ranks on the key and gathers `k` rows at
   the end. (DuckDB is 7.5 ms and 3.1 ms: faster in absolute terms, by the fixed per-query
   margin ceiling 8 records, and with the *same* scaling in column count.) Polars is 95.9 ms.

2. **The window `rank_limit` looked like an asymptotic gap and Batcher wins it anyway.**
   `qualify_to_partition_topn` fuses `rank <= k` into `Window.rank_limit`, but that bound is a
   post-hoc mask: `window_batch_with` computes the full ranking for every row — sorting every
   partition — and `filter_by_rank_limit` then drops the rest. Only `k = 1` escapes, via
   `rank1_window_to_distinct_on`. Spark built a whole operator to avoid exactly this
   (`WindowGroupLimitExec`), and Daft has `window_partition_and_dynamic_frame`. So the
   expectation was a loss. Measured over 6 M rows in 200,000 partitions:

   | top-k per group | Batcher | DuckDB | Polars |
   |---|---|---|---|
   | k = 1 | **33.2 ms** | 666.0 ms | 333.6 ms |
   | k = 3 | **93.2 ms** | 1,252.9 ms | 306.5 ms |
   | k = 10 | **131.5 ms** | 1,384.5 ms | 322.6 ms |

   Batcher leads by 10-20x over DuckDB and 2.5-10x over Polars. The `k = 1` row is 2.8x faster
   than `k = 3` on the same data, which is the `DISTINCT ON` rewrite showing through — so a
   bounded-heap window would still be worth roughly 2x for `k > 1`. That is an *improvement to a
   win*, not a gap, and it should be ranked as one.

3. **The H2O `groupby` losses are not an aggregate-kernel gap.** Six of ten queries lose, and the
   shape of the table invites the conclusion that some aggregate is slow — `q4` (three `avg`s)
   reads **1.98x** while `q5` (three `sum`s, same shape) **wins at 0.65x**. Isolated on 10 M rows
   with the aggregate as the only variable:

   | | `sum` x3 | `avg` x3 | `avg` | `sum` | `max`-`min` |
   |---|---|---|---|---|---|
   | 100 groups | 1.02x | 1.72x | 1.63x | 1.11x | 0.80x |
   | 1 M groups | **0.35x** | **0.44x** | **0.41x** | **0.34x** | **0.39x** |

   At a million groups Batcher wins every aggregate by 2.3-3x. At a hundred groups the whole
   query is 7-11 ms and the gap is a 3-5 ms *constant* — the per-query fixed cost ceiling 8
   already measures, arriving on a query too fast to hide it. No aggregate function is slow.
   What is left of the H2O row is what the scorecard already says it is: **string keys**, which is
   ceiling #2 and the largest open item in the repository.

### 12c. What the census pass verified, across the four execution scopes

Both optimizations this pass landed (item 11a's packed sort key, item 12's `eager_aggregation`
gate) reach every executor by construction rather than by a per-scope implementation, which is
what the mergeable/one-`Expr` design buys — but "by construction" is the claim that has been
wrong here before, so it was checked:

| Scope | How it is reached | Evidence |
|---|---|---|
| Batch, single-node serial | `sort_indices_of` is the oracle every path is compared against; Kyber rules are plan-level | 374 Rust tests; 18,819 unit tests |
| Batch, single-node parallel | the sample-sort's per-range sorts and `window_with`'s per-bucket kernel call the same entry points | seq == par oracle tests |
| Out-of-core | the external sort calls `sort_indices_of` per run | spill tests in the Rust suite |
| Streaming | `stream/breaker.rs` and `stream/parallel.rs` both dispatch to `ops::parallel_sort_batch`/`sort_batch`; a Kyber rewrite is upstream of every executor | **32/32** across `test_stream_batch_parity`, `test_streaming_executor`, `test_streaming_rule_parity`, `test_streaming_pushdown_parity`, `test_diff_streaming_aggregate`, `test_diff_stream_interval_join` |
| Distributed | per-partition sorts and the same plans | `test_distributed_aggregate_over_join_grouped_by_non_key` **2 passed on a fresh Ray cluster**; the sort files pass individually |

`test_stream_batch_parity` is the one that matters most for the mandate's "batch and streaming"
clause: it asserts the two paths agree, so a rewrite that changed one and not the other would
fail there rather than in production.

Two failures seen along the way were run to ground rather than attributed. `test_diff_agg_arg_extreme`'s
distributed case fails against the **shared** Ray cluster and passes in 14 s against a fresh one
(`RAY_ADDRESS=local`) — the stale-`.so` mode `concurrent-agents.md` documents.
`test_distributed_multi_table_join_matches_single_node[flight]` times out identically on
unmodified `HEAD` (`exit=124` on both sides, one dot each), so it is pre-existing.

### 12e. The census re-verified in the *other* direction, by calling

Item 12d's lesson cuts both ways and the second half had not been done. Five false *gaps* were
found and corrected in this pass, all from asking whether a name exists. The claim that carried
the most weight in item 12 — "Polars' streaming node list is the longest and most specialized of
the six, and **Batcher has every one of them**" — rested on exactly the discredited method: a
grep of the codebase, not a query.

So each was called. The result confirms the claim, and the *way* it nearly did not is the
interesting part: on the first attempt five of thirteen looked missing, and every one of those
five was the probe's fault.

| Node | Verdict |
|---|---|
| `forward_fill`, `backward_fill`, `interpolate`, `ewm_mean`, `rle_id` | **present** — they take their ordering from `.over(order_by=...)`, not an `order_by=` kwarg. Called wrongly, all five raise, and the raise reads like an absence |
| `is_first_distinct` | **present** — needs `order_by`; a bare call is an arity error, the 2.4x overstatement mode this document already names |
| rolling aggregates | **present** — `.over(order_by=…, frame=(-1, 0))`; the frame is `frame=`, not `rows=` |
| `shift`, `peak_max`, `gather_every`, `with_row_index`, session/dynamic windows, `top_k` | **present**, called directly |
| `rle` as a struct | **deliberate refusal**, and the error names the composition: number the runs with `rle_id().over(order_by=…)` and group by that |

**A raise is not an absence.** Three distinct failure modes have now produced false gaps here — a
bare-name diff, an arity error, and a competitor's spelling for a capability that exists under
Batcher's own name (12d, where the error message printed the answer and it was read past). The
only probe that has never lied is a query that returns rows.

That list of survivors was itself checked, and shrank again:

* **`list.argsort` — false gap number six.** Batcher spells it `list.arg_sort`, with the
  underscore, consistent with `arg_min`/`arg_max` beside it; `bt.col("l").list.arg_sort()`
  returns `[[1, 2, 0]]`. Polars' `argsort` and DuckDB's `list_grade_up` were the names checked.
  Same mode as 12d, for the sixth time.
* **`equi_width_bins`** — `Expr.cut(breaks)` does the binning with Polars/pandas semantics
  (`['(-inf, 3]', '(3, 7]', '(7, inf]']`). DuckDB's function computes *boundaries* the user then
  bins with, so this is a convenience over `cut`, not a capability.
* **`list.resize`** — genuinely absent; `list.slice` covers truncation but not padding. Minor.
* **A bounded-memory mode for the exact `top_k`** (12d) — the one substantive item.

**Six false gaps in one pass, every one from checking a competitor's spelling instead of calling
the capability.** The names that misled were `implode`, `list.index_of`, `min_by`/`max_by`,
`approx_top_k`, the `.over(order_by=…)` fill family, and `argsort`. That is the finding: the
distance between these engines and Batcher is mostly *vocabulary*, and any future pass should
budget for the probe being wrong far more often than the engine is.

### 12d. `approx_top_k` was already built — the fifth false gap, and the worst one

**This entry previously specified `approx_top_k` as "the one adoption left" and ranked it 0 in
the backlog, with a nine-point implementation plan. It is already built, and has been.** The
plan was written, committed, and would have sent the next reader to build a duplicate aggregate
across the wire contract. That is the failure this document has now recorded against itself four
times, and this is the most expensive instance because the work had already been *specified* on
top of the false premise.

What exists at `HEAD`, end to end:

* `bc_ir::AggFunc::ApproxTopK` with its serde tag, and `bc_runtime::agg::AggFunc::ApproxTopK(u16)`
  carrying `k` — `finalize.rs` dispatches it to `finalize_top_k`;
* `Expr.top_k(k)` in `plan/expr_ir/core.py`, lowering to `AggExpr("approx_top_k", param=k)`, with
  a Google-style docstring and a runnable doctest;
* DuckDB's own spelling wired in the SQL parser
  (`_sql/parser/expressions/aggregates.py`: `"approx_top_k": lambda x, p: x.top_k(int(p))`);
* the metadata-answer path (`api/dataset/meta/approx.py`) and a Kyber shortcut
  (`kyber/shortcuts/approx.py`).

Called rather than grepped, which is the whole lesson:
`SELECT g, approx_top_k(v, 2) FROM ds GROUP BY g` returns `{'g': ['a','b'], 't': [['x','y'],
['p','q']]}`.

**How the probe failed this time is worth recording, because it is a new mode.** The earlier false
gaps came from bare-name diffs. This one survived a diff *and* a call: `hasattr(Expr, "approx_top_k")`
is `False`, and the attribute error even printed the answer — *"Did you mean 'top_k'?"* — which
was read past. Batcher's spelling is `top_k`; `approx_top_k` is the DuckDB name it answers to in
SQL. **Calling the competitor's name is not calling the capability**; the check has to be "does
this query work", not "does this attribute exist".

**The one real residue, and it is much smaller than the entry it replaces.** Batcher's is
deliberately *exact*: the docstring says so, and the reasoning is sound — the aggregate already
holds every value of the group in `median_state`, so a sketch could only lose accuracy for no
saving. Ties break to the smaller value, which is what keeps it mergeable and partition-order
independent.

That leaves exactly one thing a Misra-Gries state would buy that the current one does not:
**bounded memory under skew.** Exact top-k keeps every value of every group, so one hot group can
OOM — the same argument `hll.rs` makes for `approx_count_distinct` against exact
`COUNT(DISTINCT)`. `bc_sketches::FrequentItems` is `Mergeable` but has no `to_bytes`/`from_bytes`,
which `hll.rs`'s state-in-a-`Binary`-column pattern needs. So the open item is not "add the
aggregate", it is "give the existing aggregate a bounded-memory mode for the skewed case" — a
smaller, differently-shaped piece of work, and one that must not cost the exact path its accuracy.

### 12b. `merge_sorted` is a capability gap that is not a gap

The inventory table lists `merge_sorted` as one of four operators the competitors have and
Batcher does not, on the reasonable ground that the k-way merge exists inside the external sort
and is not reachable as a verb. Polars ships it as a streaming node and Acero as
`sorted_merge_node.cc`, so it looked like a clean technique to adopt — and it has the property
this document keeps asking for, being naturally streaming, batch, single-node and distributed at
once.

**Measure the shape before adding the operator.** Merging two 4 M-row sorted relations on an
`int64` key, which is exactly what the operator is for:

| | time |
|---|---|
| Batcher, `a.union(b).sort("k")` | **33.5 ms** |
| DuckDB, `(a UNION ALL b) ORDER BY k` | 177.6 ms |
| NumPy, stable sort of the concatenated key arrays alone | 100.1 ms |
| Batcher, the same 8 M rows *unsorted* (what a general sort costs) | 248.9 ms |

Batcher is **5.3x faster than DuckDB** on the shape, and faster than a hand-written NumPy sort of
just the keys. The last row is the one that settles it: the same 8 M rows unsorted cost 248.9 ms,
so the pre-sorted concatenation is already running **7.4x cheaper than a general sort**. The
engine is *already* exploiting the sortedness — `radix_sort`'s `is_ordered` check, its skipping of
constant key bytes, and the sample-sort's ranges arriving pre-ordered — without being told about
it and without a dedicated operator.

So the verb would add public API surface, a `RelOp`, a wire-contract change and four executor
paths, to make a shape faster that is already the fastest of the three engines measured. That is
the speculative generality `maintainability.md` forbids, and the inventory row is corrected rather
than acted on.

Worth separating two things the row conflated. Batcher genuinely cannot *spell* `merge_sorted`,
and a user porting a Polars script has to write `union(...).sort(...)` instead — a one-line
translation that belongs in the migration guide, not in the engine. What it cannot do is
**assert** the inputs are sorted and skip the verification; that is the only real difference, it
is worth a linear scan, and nothing measured suggests the scan is what costs.

### 12a. Spark's `WindowGroupLimitExec` — **landed**, after being reverted once

`Window.rank_limit` was a post-hoc mask over a fully ordered partition: Kyber folded
`rank <= k` into it, and `window_batch_with` then ranked every row — ordering every partition —
before `filter_by_rank_limit` threw all but `k` away. Only `k = 1` escaped, by the `DISTINCT ON`
rewrite. Spark built `WindowGroupLimitExec` and Daft `window_partition_and_dynamic_frame` to
avoid exactly that.

`bc_runtime::window::topk` is Batcher's: one flat `groups x k` max-heap array with no
per-partition allocation, ordering on `(packed order key, row index)` so the `k` rows kept are
the `k` smallest under the *same* total order the ordering path produces.

**It was built, measured 2-4x slower, and reverted — and the reason it lost is the reason it now
wins.** The first version hooked the selection above `window_with`, over the whole batch, which
traded the operator's bucketed parallelism for the better complexity: `O(n log k)` on one core
against `O(n log n)` on ninety-six, at 0.24-0.50x. `rank_limit` is now threaded through
`window_with` -> `window_parallel` -> `window_serial`, so the selection runs *inside* the
per-bucket kernel and each worker heaps only the partitions it owns.

The plumbing is also why the first attempt was abandoned rather than fixed, and that call was
wrong. It was costed against putting `rank_limit` on `WindowCall` — **49 construction sites, 47
in tests** — without costing the alternative. Threading it through three signatures is **16 call
sites**, most of them a literal `None` in a test. A rejected design was priced and the cheaper one
next to it was not.

Measured on 6 M rows, interleaved best-of-three:

| Partitions | `k` | Ordering | Bounded | |
|---|---|---|---|---|
| 100 x 60,000 rows | 10 | 63.2 ms | 45.1 ms | **1.40x** |
| 100 x 60,000 rows | 3 | 60.8 ms | 45.2 ms | **1.35x** |
| 2,000 x 3,000 rows | 10 | 58.0 ms | 44.4 ms | **1.31x** |
| 2,000 x 3,000 rows | 3 | 56.7 ms | 45.3 ms | **1.25x** |
| 50,000 x 120 rows | 3 | 65.6 ms | 52.5 ms | **1.25x** |
| 200,000 x 30 rows | 3 | 91.9 ms | 75.2 ms | **1.22x** |
| 1,000,000 x 6 rows | 2 | 116.7 ms | 98.4 ms | **1.19x** |
| `k = 1` (takes `DISTINCT ON`, untouched) | 1 | 26.8 ms | 27.6 ms | 0.97x |
| `rank()` instead of `row_number` (declines) | 3 | 101.5 ms | 97.4 ms | 1.04x |

The win grows with partition size, which is what `O(n log k)` against `O(n log n)` predicts, and
the two control rows are flat — `k = 1` goes down Kyber's `DISTINCT ON` route and never reaches
this path, and a declining shape falls through to the ordering path unchanged.

Seven tests pin the equivalence **through the real operator at both thresholds**, so the bucketed
parallel path is exercised rather than only the serial kernel: every `k`, both directions,
tie-heavy input (where the row-index tie-break is the thing that can silently disagree), a string
partition key, an unpartitioned window, a declining shape, and the property that a non-survivor
can never pass a `rank <= k` mask — it is marked `k + 1`, because a zero or a null would pass and
silently keep every row of every partition.

One subtlety the tests caught rather than the author: a *declining* shape returns the ordering
path's **unmasked** ranks from `window_serial`, since the mask lives in `bc_interp`. Comparing a
masked bounded result against an unmasked ordering one fails on a shape where nothing is wrong.
The comparison masks both sides, which is also the operator's real contract — the rows kept and
their ranks must agree, not the intermediate column.

Keep it in proportion: Batcher already led this shape by 10-20x over DuckDB and 2.5-10x over
Polars, so this makes a win larger rather than closing a gap.

### The measured census: 96 queries, four suites, ranked by what they actually cost

The inventory says almost nothing is missing, so the bottlenecks have to be found by
measurement. Every suite that runs in reasonable time was run on one 96-core node against
DuckDB's **native store** (the harder of the two bars) with a two-engine lineup: H2O `groupby`
(10), H2O `join` (5), ClickBench (43) and TPC-H sf1 (22).

**Batcher wins 66 of the 80 queries** where a ratio is meaningful. What follows is the other
fourteen, and the ordering is the point.

**Rank by ratio and you optimize the wrong thing.** The three worst ratios in the whole census
are `cb-q19` (2.64x), `h2o-gb-q2` (1.98x) and `h2o-gb-q4` (1.98x) — worth **1.2 ms, 18.3 ms and
4.0 ms**. Ranked by the time actually lost:

| Query | Batcher | DuckDB | Ratio | **Lost** | Shape |
|---|---|---|---|---|---|
| `h2o-join-q5` | 378.5 ms | 271.8 ms | 1.39x | **+106.7 ms** | `x JOIN big USING (id3)`, 1e7 x 1e7, `SELECT x.*` + 5 cols |
| `h2o-join-q4` | 214.7 ms | 114.9 ms | 1.87x | **+99.8 ms** | `x JOIN medium USING (id5)`, `SELECT x.*` + 4 cols |
| `h2o-gb-q8` | 115.0 ms | 69.2 ms | 1.66x | +45.8 ms | `row_number() OVER (PARTITION BY id6 ORDER BY v3 DESC) <= 2` |
| `h2o-gb-q7` | 75.2 ms | 42.1 ms | 1.79x | +33.1 ms | `max(v1)-min(v2) GROUP BY id3` (string key) |
| `h2o-gb-q3` | 72.5 ms | 45.5 ms | 1.59x | +27.0 ms | `sum,avg GROUP BY id3` (string key) |
| `h2o-gb-q2` | 37.1 ms | 18.8 ms | 1.98x | +18.3 ms | `GROUP BY id1, id2` (two string keys) |
| `h2o-gb-q9` | 40.4 ms | 27.4 ms | 1.47x | +13.0 ms | `corr GROUP BY id2, id3` (string keys) |
| `cb-q32` | 31.1 ms | 21.9 ms | 1.42x | +9.2 ms | `GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10` |
| `tpch-q13` | 47.4 ms | 41.1 ms | 1.15x | +6.3 ms | customer-order count distribution |
| `h2o-gb-q4` | 8.1 ms | 4.1 ms | 1.98x | +4.0 ms | `avg x3 GROUP BY id4` (100 groups) |
| `tpch-q5` | 31.2 ms | 27.3 ms | 1.15x | +3.9 ms | six-way join |
| `tpch-q6` | 6.3 ms | 4.5 ms | 1.38x | +1.8 ms | filter + `sum` over `lineitem` |
| `cb-q25` | 3.4 ms | 2.2 ms | 1.50x | +1.2 ms | `ORDER BY <string> LIMIT 10` |
| `cb-q19` | 2.0 ms | 0.8 ms | 2.64x | +1.2 ms | `WHERE UserID = <literal>` |

Read that way the census says something the per-suite geomeans do not: **two H2O join queries are
the largest single-node losses in the repository, at about 100 ms each, and together they cost
more than the other twelve rows combined.**

#### The q2/q4 pair, instrumented — and it is not the join

`h2o-join-q2` (`x JOIN medium USING (id2)`) is a **win** at 0.74x while `h2o-join-q4`
(`x JOIN medium USING (id5)`) is a 1.87x loss, on the same two tables with nearly the same
projection. The pair is a free controlled experiment, so it was run: separate the *join* from
the *materialization* by asking each query for a `count` as well as for its rows. The H2O join
schema matters here — `id1`, `id2`, `id3` are `int32` and `id4`, `id5`, `id6` are `string`.

| | count only | `SELECT x.*` | materialization | DuckDB count |
|---|---|---|---|---|
| `x JOIN medium USING (id2)` — int32 key | 22.0 ms | 93.8 ms | **+71.8 ms** | 3.6 ms |
| `x JOIN medium USING (id5)` — string key | 32.3 ms | 195.4 ms | **+163.1 ms** | 10.6 ms |
| `x JOIN big USING (id3)` — int32, 1e7 x 1e7 | 78.1 ms | 390.5 ms | **+312.4 ms** | 100.0 ms |

Two things fall out, and the first is the largest single finding in this census.

**1. Three quarters of these queries is the output gather, not the join.** Materialization is
77%, 83% and 80% of the three totals. On the big-by-big join Batcher's *join* is **faster than
DuckDB's** (78.1 ms against 100.0 ms) and the query still loses by 107 ms, because 312 ms of it
is gathering nine million rows of a wide, string-bearing result through join indices. This is the
same ~2.5 GB/s permuted read backlog entry 13 measures on the sort, arriving through a different
operator — and it unifies three separate rows of the table above (`h2o-join-q4`, `h2o-join-q5`,
`cb-q32`) with the sort finding under **one root cause**. Together those are worth more than
every other row of the census combined, which makes **selection vectors / deferred payload
materialization** (RFC proposal 2) the highest-value structural item on the board by a wide
margin, ahead of the string-key work it was previously ranked behind.

**2. The small-build-side row was measuring the optimizer, not the join — corrected.** The first
draft of this entry read the 22.0 ms against DuckDB's 3.6 ms as a join-kernel gap and proposed
building DuckDB's `perfect_hash_join_executor.cpp`. Both halves were wrong, and the correction is
the more interesting finding.

`crates/bc-runtime/src/join/dense.rs` **is** that executor — "a perfect hash for a small-range
integer build key", the join-side counterpart of the dense direct map `agg::group::assign` uses.
It was there the whole time, so the premise was false; a grep would have settled it and the
measurement was trusted instead.

And the 22.0 ms was not the join. Varying only *which column the aggregate reads*, on the same
join:

| `SELECT count(...) FROM x JOIN medium USING (id2)` | | |
|---|---|---|
| `count(v2)` — a right-side column, so nothing can be pushed | **8.9 ms** | plain join |
| `count(id2)` — the join key, so `eager_aggregation` pushes | 21.1 ms | **2.4x slower** |
| `count(v1)` — a left-side column, likewise | 23.2 ms | **2.6x slower** |

`kyber/rules/agg_pushdown.py::eager_aggregation` pre-aggregates a join side to shrink its input,
gated on `_MIN_PREAGG_REDUCTION = 8.0` — a **row-reduction ratio**. Here the ratio is ~1000x
(10 M rows to 10,000 groups) so it fires with room to spare, and it is 2.4-2.6x slower than not
firing.

The gate is measuring the right quantity and comparing it against nothing. A 1000x reduction
still costs a **full 10 M-row hash aggregate**, and what it buys back is a *broadcast probe
against a 10,000-row build side*, which is almost free. No reduction ratio, however large, can
justify the push when the join it replaces costs nothing — the missing term is what the join
would have cost. The rule's own comment anticipates the shape of this ("the cost model cannot
catch it — so the guard lives here, on the reduction the rewrite must actually achieve") and then
picks the one quantity that cannot distinguish a saved shuffle from a saved broadcast.

**What remains true about the join after the correction is much smaller**: `sum(v2)` over the
same join is 8.8-13.3 ms against DuckDB's 4.7 ms. A ~2x on a small-build broadcast, not a 6x, and
not the kernel.

**Do not read finding 1 as "the join is slow".** It is not; it is the materialization, and the
same materialization is behind the sort rows, the `cb-q32` row, and `op-filter-project`'s
sequential-write cousin. Optimizing the join kernels would move none of it.

Everything below the join rows falls into three buckets already named elsewhere, which is a
useful result in itself — the census found **no new engine-level gap**:

* **String group keys** (`q2`, `q3`, `q7`, `q9`, +91.4 ms together) — ceiling #2, the largest
  open item, and the reason the H2O `groupby` geomean is above 1.
* **Permuted materialization** (`h2o-join-q4/q5`, `cb-q32`) — item 11's backlog entry 13.
* **Per-query fixed cost** (`h2o-gb-q4`, `cb-q19`, `cb-q07`, `tpch-q6`) — ceiling 8's warm gap.
  These produce the largest *ratios* and the smallest *losses*, because a 1-4 ms constant is
  most of a 2-8 ms query. `cb-q00` is 0.1 ms against DuckDB's 0.8 ms on the same dataset, so the
  constant is not uniform and not a floor; it is work proportional to the plan, not the data.

## 13. The state `mode` and `top_k` were carrying — **landed, 31x and 287x** (2026-08-18)

Found by measuring the aggregate 12d had just finished proving already existed. `approx_top_k`
was built, correct, wired end to end — and 22.9x *slower* than DuckDB on the shape it is for,
holding 854 MB to return three values. Being present is not the same as being good, and this
document had spent four entries checking presence.

The cause was a shared state, not an algorithm. `mode` and `top_k` were listed alongside
`entropy`, `mad` and `quantile_disc` as "the value-list family: they read a group's whole value
list, so they share `Median`'s state and differ only in `finalize`". True of the other three.
False of these two, which never read the list -- they only ever ask how *often* each value
occurs. So the partial retained every value of every group and the counting happened at
finalize, over ten million rows, to return three.

The fix is `bc-runtime`'s `agg::counted`: per group, the distinct values and their counts, as
two parallel `List` state columns. `O(distinct)` instead of `O(rows)`, still exact, and still
mergeable because counts add -- so `combine` re-groups the pairs and sums, and the parallel,
spilling and distributed paths inherit it with no second implementation. No IR or wire change:
same `AggFunc`, same tags, only the state behind them.

10M rows, one group, 50 distinct values, both builds `--release`, best of 3, all three engines
agreeing on the answer:

| | time | peak RSS |
|---|---|---|
| `top_k(3)` before | 547.6 ms | +687 MB |
| **`top_k(3)` after** | **17.5 ms** | **+224 MB** |
| DuckDB `approx_top_k(v, 3)` | 705.3 ms | — |
| `mode()` before | 3,988.3 ms | +20 MB |
| **`mode()` after** | **13.9 ms** | **+4 MB** |
| DuckDB `mode(v)` | 109.4 ms | — |

**31x and 287x** against the previous state; 40x and 7.9x against DuckDB. Peak RSS falls 3.1x
rather than to nothing because `partial` still row-encodes the batch it is handed: the state is
`O(distinct)`, the transient encoding is still `O(batch rows)`.

Two divergences from DuckDB surfaced while testing, both **pre-existing** and both verified
identical on a build without this change, so neither is a regression this introduced:

* `mode` over NaN. Batcher folds every NaN into one value and answers `nan`; DuckDB answers
  `2.5` on `[nan, nan, nan, 2.5, 2.5]` -- while its *own* `GROUP BY` on that column reports
  `(nan, 3), (2.5, 2)`. DuckDB contradicts itself here; Batcher agrees with its own grouping,
  which is the property worth keeping. Pinned in `test_diff_agg_counted_state.py` including
  DuckDB's contradicting answer, so the test fails loudly if DuckDB ever fixes it.
* `top_k` over an all-null group yields `[]` where DuckDB yields NULL. Recorded, not hidden.

**The lesson for this document.** Entries 12d, 12e and 10n each ended "already built, do not
re-implement", and that conclusion was right every time. It was also incomplete: none of them
asked whether the thing that existed was any good. A census that checks presence finds false
gaps and stops; the one bottleneck worth 287x was inside a feature already marked done. Measure
the capabilities you rule out, not just the ones you rule in.

## 14. Item 0a re-measured, and it is the wrong lever — the gap is the string *representation* (2026-08-18)

Entry 13 ended with "measure the capabilities you rule out, not just the ones you rule in."
Applying that to this document's own top item retires it.

**0a said**: deferred payload materialization (selection vectors) is the highest-value
structural item, worth 77-83% of the two largest single-node losses (`h2o-join-q4/q5`), more
than the other thirteen census rows combined. Re-measured at `HEAD` on the same shapes and the
same bar the census used (DuckDB's **native store**, 1e7 x 1e7, best of 3):

| | batcher | duckdb | ratio |
|---|---|---|---|
| `x JOIN big USING (id3)`, full projection | 369.6 ms | 204.8 ms | 1.80x |
| ...`count(*)` only, so the join without the gather | 87.9 ms | 98.5 ms | **0.89x** |
| `x JOIN medium USING (id5)`, full projection | 231.7 ms | 74.2 ms | 3.12x |
| ...`count(*)` only | 101.3 ms | 23.4 ms | **4.34x** |

The census's structural claim survives on q5 and only q5: the join beats DuckDB, the query
loses by 165 ms, and 76% of Batcher's time is the gather. **But three measurements say the
proposed fix cannot collect it.**

**1. Deferring cannot help a query that returns every row.** `SELECT x.*, ...` over an inner
join emits all nine million rows. A selection vector defers materialization; it does not remove
it. It pays where a *downstream* operator discards rows — `cb-q32`'s `LIMIT 10`, a sort's
top-k — and those shapes are worth single-digit milliseconds each in the census. The two rows
that made 0a the top item are exactly the two it cannot help.

**2. The gather kernel is already 15x faster than arrow's.** Gathering the ten output columns
of q5: `bc_runtime::gather` ~282 ms against `pyarrow.compute.take` at 4,254 ms on the identical
indices. There is no ordinary optimization left in it.

**3. The gap is concentrated in strings, and by a factor.** Same join, **same number of output
columns**, varying only their type — so nothing is subtracted and no assumption about what
`count(*)` skips is needed:

| 3 output columns | batcher | duckdb | ratio |
|---|---|---|---|
| numeric (`int32`) | 105.8 ms | 101.1 ms | 1.05x |
| **string** | **260.3 ms** | **145.1 ms** | **1.79x** |
| **string premium over numeric** | **+154.5 ms** | **+44.0 ms** | **3.51x** |

Numeric materialization is at **parity**. The whole gap is the string columns, whose mean length
is **six characters**.

*A correction to this entry's first draft.* It reported 4.21x, obtained by subtracting each
engine's `count(*)` time to isolate materialization. That subtraction assumes the join costs the
same with and without a projection, which it does not — DuckDB's `count(*)` skips fetching
payload columns entirely — and it inflated the number. The controlled form above needs no
subtraction and gives 3.51x. The direction was right and the magnitude was not; the method is
the part worth copying.

### Why six characters is the whole answer

DuckDB's `string_t` is a 16-byte struct with the first 12 bytes of the string stored *inline*.
For a column averaging six characters, every value is inline, so DuckDB's gather is a
**fixed-width 16-byte copy** — no character buffer, no offset prefix-sum, no second read.
Batcher's `Utf8` gather copies characters, which is a scattered read of the source values plus
a dependent prefix-sum over the offsets, and that is what the 4.21x is.

Arrow's `Utf8View` is the same layout and the same 12-byte inline rule. The floor is measurable
even though pyarrow 19 ships no `take` kernel for it: on the same 9 M indices, `take` on `Utf8`
is 1,756 ms while `take` on an 8-byte fixed-width column is 130 ms. A 16-byte view gather is
two of those, and it is the kernel `gather::take_fixed_width_parallel` already implements.

### The string length is the proof, and it says where the fix pays

The same premium measured on a *different* string column settles the mechanism. TPC-H's
`l_comment` averages **26.5** characters, past the 12-byte inline window, and there the gap
halves — a sort over an identical random permutation, varying only the payload type:

| payload | batcher | duckdb | premium |
|---|---|---|---|
| `l_orderkey` (`int64`) | 32.3 ms | 31.6 ms | -- |
| `l_comment` (26.5 chars) | 129.9 ms | 90.4 ms | +97.6 ms vs +58.8 ms = **1.66x** |

**6-character strings: 3.51x. 26.5-character strings: 1.66x.** That length dependence is the
inline prefix and nothing else — past 12 bytes DuckDB has to copy characters too, and most of
its advantage disappears. So `Utf8View` is worth roughly 3.5x on short string columns (keys,
codes, identifiers, categoricals — the columns joins and group-bys are actually built on) and
roughly 1.7x on prose. Size the work by that, not by the headline.

### A hypothesis this measurement killed, recorded so nobody rebuilds it

The obvious cheaper fix is a **cache-conscious partitioned gather**: radix-partition the indices
by source region so each pass's reads stay inside cache, which is what Velox and DuckDB do for
random access. It does not apply here, because **Batcher's gather is already almost
locality-insensitive**:

| gather of a 159 MB string column, 6 M rows | pyarrow | batcher | duckdb |
|---|---|---|---|
| random source order | 968.2 ms | 129.9 ms | 90.4 ms |
| sorted / in-order source | 172.0 ms | 109.6 ms | 40.6 ms |
| **penalty for random access** | **5.63x** | **1.19x** | **2.23x** |

pyarrow pays 5.63x for scattered reads and a 64-bucket partitioning pass recovers 3.41x of it.
Batcher pays **1.19x**, so the entire prize is ~20 ms of a 130 ms gather and a partitioning pass
would spend most of it. The parallel chunked gather already solved this problem. What is left is
not locality, it is the per-value work of a variable-length layout — which is the representation,
which is `Utf8View`.

### What this changes on the board

**`StringView` (arrow `Utf8View`) replaces deferred materialization as the top structural
item.** `CLAUDE.md` already names "no `StringView`" as one of the four specific gaps against
DuckDB single-node; this measurement is what it costs, on the queries that cost the most.

Deferred materialization is **not** retired, it is **re-ranked**: it is the right mechanism for
`cb-q32` and the sort-plus-limit shapes, which are worth a few milliseconds each, not the
~165 ms rows it was promoted on.

And q4 needs a second thing besides: its `count(*)` alone loses 4.34x, so a string *join key*
is slow independently of the output gather. That is the string-key work already on this list,
and this is the first measurement that separates it cleanly from the gather it was tangled with.

**Scope, stated honestly.** `Utf8View` is not a contained change: it is a new physical type
through `bc-arrow`, every `bc-expr` string kernel, the row encoder, the FFI boundary, and the
IR's type vocabulary, with a planner decision about when to produce it (the same shape as the
dictionary item 6, which was built, measured and reverted for want of exactly that decision).
It should be started as its own piece of work with that precedent in view, not bolted onto a
gather patch.

## 15. The operator suite, re-run at HEAD and ranked by time lost (2026-08-18)

`python benchmarks/run.py --benchmark operators`, one 96-core node, TPC-H sf1, five engines,
**every correctness check passed**. Batcher beats Polars, PyArrow and Daft on all 21 cases —
often by one to two orders of magnitude — so the only lineup worth tabulating is DuckDB.

Fourteen of 21 are wins. Ranked by **time lost** rather than by ratio, which is the ordering
entry 12 argued for and this table exists to keep honest:

| case | batcher | duckdb | ratio | **lost** |
|---|---|---|---|---|
| `op-sort-string` | 162.5 ms | 140.5 ms | 1.16x | **+22.0 ms** |
| `op-sort-multikey-narrow` | 50.8 ms | 36.5 ms | 1.39x | **+14.3 ms** |
| `op-sort-multikey-wide` | 67.1 ms | 55.6 ms | 1.21x | +11.5 ms |
| `op-sort-string-lowcard` | 45.7 ms | 34.4 ms | 1.33x | +11.3 ms |
| `op-sort-string-limit` | 9.6 ms | 7.0 ms | 1.37x | +2.6 ms |
| `op-distinct-high-card` | 20.0 ms | 17.6 ms | 1.13x | +2.4 ms |
| `op-groupby-sum` | 5.5 ms | 3.2 ms | 1.73x | +2.3 ms |
| `op-filter-project` | 10.3 ms | 8.1 ms | 1.28x | +2.2 ms |
| `op-groupby-2key` | 5.7 ms | 5.4 ms | 1.07x | +0.3 ms |

**Sorting is four of the top five and about 59 ms of the roughly 70 ms lost in total**, and
three of those four sort a string. Everything that is not a sort costs under 3 ms.

Two things follow, and the second is the one that matters.

**The window operators have become a clear win** — `op-window-runsum` 0.20x, `op-window-lag`
0.37x, `op-window-rank` 0.41x against DuckDB, with Daft OOM-killed on all three (it needs
~22 GB for an ordered 6 M-row window and exceeds 30 GB on `lag`). That was a loss when this
document started tracking it.

**The remaining single-node deficit is one operator and one data type.** Entry 14 measured the
string *payload gather* at 3.51x on short strings; this table adds the other half, the string
*key ordering*, which `op-sort-string` isolates at 1.16x on a 26.5-character key. They are
separate paths (`stable_sort_indices_str` and the string range-partitioner versus
`gather::take_column`) and only the second is addressed by `Utf8View`. The multi-key rows are a
third, unrelated thing: entry 11a's packed radix sort already landed there and they are still
1.21-1.39x, so that path has something left in it that the packing did not reach.

## 16. The distributed path, measured — and one open question about the object-store claim (2026-08-19)

Everything in entries 12-15 measures the single-node engine. The mandate names streaming,
batch, single-node **and distributed**, and `CLAUDE.md` says CI installs no Ray, so the
distributed path is the one nothing routinely runs. This is what it does at `HEAD`, on a
**single-node local Ray cluster (4 CPUs)** at TPC-H sf1 with 8 partitions
(`benchmarks/run.py --benchmark distributed`, `RAY_ADDRESS=local`).

**Correctness first, and it holds:** *"Distributed results match single-node on every query."*
The mergeable algebra does what it claims. What follows is only about time.

| query | single-node | distributed | |
|---|---|---|---|
| `groupby-agg` | 4.8 ms | 635.0 ms | 132x slower |
| `groupby-2key` | 5.3 ms | 669.6 ms | 126x slower |
| `join+groupby` | 34.4 ms | 11,066.3 ms | **322x slower** |
| `distinct` | 6.0 ms | 493.8 ms | 82x slower |

**Distributing an sf1 query is not supposed to pay** — the adaptive loop's per-cut size floor
(5M rows, or roughly 320 MB, per pipeline breaker) keeps it away from queries this small
precisely because of this, and no one should read these ratios as a defect on their own.
Two things underneath them are worth an owner.

### The group-by warms up and the join never does

Repeating each query four times in one session, with the session fleet on (the default):

```
group_by       run1=11063ms  run2= 721ms  run3= 661ms  run4= 643ms
join+group_by  run1=11571ms  run2=11035ms run3=11267ms run4=24196ms
```

The group-by pays a large first call and then settles at ~650 ms. The join pays ~11 s **every
run**. `ShuffleFleet.spawn` was traced through all six runs and fired **zero** times, so neither
number is fleet spawn: the first call is Ray cluster init, and the join's 11 s is the query.

It is also flat in partition count — 11.2 s, 10.9 s, 11.8 s, 11.4 s at 2, 4, 8 and 16 partitions
— so it is not per-partition work, and it grows with both row count and column width
(a 2-column synthetic join of the same 6 M x 1.5 M shape costs 3.8 s against the 16-column
`lineitem`'s 11 s). That points at bytes moved, which is the next section.

**Caveat that limits how far this can be read.** These inputs are driver-resident Arrow tables,
so distributing them means shipping ~850 MB out to the workers before any shuffle. A real
distributed job scans into the workers and never pays that. The measurement is honest about
what it is: the cost of distributing an in-memory table, not of a cluster-native scan.

### The transport numbers, and the claim they qualify

`benchmarks/run.py --benchmark shuffle`, 8 partitions x 64 batches = 128 MiB, same box:

| transport | | |
|---|---|---|
| Ray object store | 108.6 ms | 1,178.4 MiB/s |
| Carbonite Flight, **network path** (`locality=0.00`) | 331.8 ms | **385.8 MiB/s** |
| Carbonite **co-located** (`DIRECT_MEMORY`, `locality=1.00`) | 6.4 ms | **19,890.9 MiB/s** |

`competitive_architecture.md` states that bypassing the Ray object store "is the single biggest
reason Batcher beats Ray Data 50-450x". The co-located row supports that emphatically — 16.9x
Ray's object store. The network row does not: on same-node loopback it is **0.33x**, three times
*slower* than the thing it bypasses. The harness labels that row "same-node loopback" itself, and
loopback is not Flight's design point, so this is not a refutation of the claim.

**It is an open question about which path the operators actually take.** The 51x gap between
those two rows is entirely a locality decision, and a single-node cluster is the case where
locality should always win. A distributed join on that same cluster costs ~11 s for roughly
850 MB, which is the arithmetic of the 386 MiB/s row and not of the 19.9 GiB/s one. Either the
operators are not reaching the co-located path where the microbenchmark can, or the join's time
is going somewhere else entirely. **Nothing here settles which**, and the next person on this
should settle it before optimizing anything: instrument which transfer mode the join's shuffle
actually uses (`carbonite/transfer/server.py` already counts bytes served over the same-node
fast paths), on a cluster-native scan rather than driver-resident tables.

Recorded rather than fixed, deliberately. `CLAUDE.md` requires a recorded cluster run for any
`dist/` change, this box is a 4-CPU single node, and a locality fix validated only here would be
validated on the one topology where locality is trivially true.

## 17. The string-keyed join was serial — **landed, up to 22.4x** (2026-08-19)

Found by chasing the one number entries 14-16 kept stepping over. Entry 14 measured
`h2o-join-q4`'s `count(*)` at 4.34x DuckDB and set it aside as "the string-key work already on
this list"; nobody had asked *why*. Varying only the key type on an otherwise identical join
answered it immediately:

| `count(*)` only, no output gather | batcher | duckdb | ratio |
|---|---|---|---|
| `x JOIN big USING (id3)` — int32, 1e7 x 1e7 | 95.2 ms | 101.0 ms | **0.94x, a win** |
| `x JOIN big USING (id6)` — **string**, same shape | **4,999.4 ms** | 116.5 ms | **42.91x** |

Same rows, same cardinality, same result — 52x more time for a string key than an integer one.
That is not a comparison cost, and the scaling curve said so before any code was read: Batcher
spent a **flat ~650 ns per row** from 250 k to 10 M rows while DuckDB's fell from 127 to 15. Flat
per-row cost at every scale is what a serial loop looks like. Confirmed directly — 96 cores,
4 M rows, `cpu_time / wall_time`:

| key | wall | CPU | parallelism |
|---|---|---|---|
| `int64` | 38.6 ms | 968 ms | 25.1x |
| `Utf8` | 2,234.9 ms | 4,692 ms | **2.10x** |

**The cause is a trait bound, not a decision.** `radix_join_scalar` — the cache-radix join both
integer fast paths use — is generic over `O: Copy`, because the partition pass carries the key
inline beside the row index. A string is not such a value, so a string key fell to the flat
build-and-probe path, which is serial. Nothing in the code says "strings run on one core"; it
follows from a bound, which is exactly why it survived. `ops::byte_sort` records the identical
failure for sorts — "`sample_sort` reads the same type list, so the sort then ran on **one core**
whatever the machine" — so this is the second instance of one pattern: **a type list or trait
bound that silently withholds the parallel path from byte keys.** Worth grepping for a third.

**The fix is not a second join algorithm.** A byte key of <= 15 bytes packs losslessly into one
`u128` (length in the top byte, value bytes below), which *is* `Copy`, so it reaches the same
radix join unchanged. The packing is injective, so "packs equal" and "bytes equal" are the same
predicate and the matches are identical by construction. Longer keys keep the flat path. Fifteen
bytes covers what join keys are: ids, codes, SKUs, ISO dates, categoricals.

| rows per side | before | after | duckdb | before | after |
|---|---|---|---|---|---|
| 250,000 | 117.3 ms | **22.3 ms** | 23.4 ms | 3.69x | **0.96x** |
| 1,000,000 | 633.1 ms | **46.1 ms** | 53.9 ms | 7.12x | **0.86x** |
| 4,000,000 | 2,617.3 ms | **124.2 ms** | 59.7 ms | 29.27x | 2.08x |
| 10,000,000 | 6,835.6 ms | **304.8 ms** | 129.7 ms | 46.74x | 2.35x |

**Up to 22.4x**, two scales from loss to win, parallelism 2.10x -> 19.38x, and the integer
control unmoved (38.6 -> 39.1 ms). The H2O case that started it: 4,999.4 ms -> **284.8 ms**,
42.91x -> **2.50x**.

`x JOIN medium USING (id5)` barely moves and should not — its build side is 10 k rows, below
`RADIX_MIN_BUILD_ROWS`, so the flat path stays and is right for a build table that fits in cache.

### What this says about the entries before it

Entries 14 and 15 concluded the remaining single-node deficit was "one operator and one data
type", with `StringView` as the top item. That is still true of the *gather*, and this changes
none of those measurements. But the largest string number in this document was never the gather:
it was a join that was accidentally serial, worth 42.91x where the gather is worth 3.51x, and it
sat inside a row (`h2o-join-q4`) that three separate entries had already looked at.

Both times the lesson repeats: **the census ranked capabilities by whether they existed, then by
ratio, and the biggest wins were inside things already marked done.** Entry 13 found 287x inside
a "built" aggregate; this found 42.91x inside a "built" join. Neither needed a new feature.

## 18. DuckDB's sort key, read rather than measured — and three hypotheses it killed (2026-08-19)

Every entry above measures competitors as black boxes. This one reads one, because entry 17's
win came from a *mechanism* (a trait bound withholding parallelism) that no timing could have
named. The remaining single-node deficit is sorting (entry 15: four of the top five rows,
~59 ms of ~70 ms), so the thing to read was DuckDB's sort.

### What DuckDB actually does

`src/common/sort/sort.cpp` lowers the whole `ORDER BY` list into **one** `create_sort_key`
expression, and `src/function/scalar/create_sort_key.cpp:52-67` decides its physical type:

```cpp
bool all_constant = true;
idx_t constant_size = 0;
for (idx_t i = 0; i < arguments.size(); i += 2) {
    auto physical_type = arguments[i]->GetReturnType().InternalType();
    if (!TypeIsConstantSize(physical_type)) { all_constant = false; }
    else { constant_size += GetTypeIdSize(physical_type) + 1; }  // +1 validity byte
}
if (all_constant && constant_size <= sizeof(int64_t)) {
    function.SetReturnType(LogicalType::BIGINT);   // else: BLOB
}
```

A single `BIGINT` normalized key when every key column is fixed-width and their **declared**
widths plus one validity byte each fit in eight bytes; otherwise a `BLOB` and a comparison sort.

**Batcher's rule is strictly stronger, and there is nothing here to adopt.**
`radix_sort::packed_multi_sort_indices` (entry 11a) packs from the keys' *measured value ranges*
rather than their declared widths. The benchmark case is the proof:
`ORDER BY l_shipdate, l_suppkey` is `4+1 + 4+1 = 10` bytes by DuckDB's rule, so DuckDB takes the
BLOB path; Batcher measures ~2,500 days and ~10,000 suppliers, packs 26 bits into one `u64`, and
counting-sorts it.

### And yet Batcher loses that case 2.06x, so three hypotheses were tested and all three failed

`ORDER BY l_shipdate, l_suppkey` over 6 M rows, varying only the projection:

| returned | batcher | duckdb | ratio |
|---|---|---|---|
| 1 narrow column | 54.1 ms | 26.3 ms | **2.06x** |
| both keys | 53.6 ms | 31.1 ms | 1.72x |
| both keys + 2 more | 83.8 ms | 65.7 ms | 1.28x |
| *delta for the two added columns* | **+30.2 ms** | +34.5 ms | — |

1. **Not the key encoding.** Batcher takes the better of the two paths, as above.
2. **Not the payload gather.** Adding two columns costs Batcher **less** than DuckDB (+30.2 ms
   against +34.5 ms). The gather is a Batcher strength here, and the ratio *improves* as the
   projection widens because the fixed deficit is amortized.
3. **Not parallelism** — which is what entry 17 predicted, and it was wrong. Entry 17 closed by
   naming these multi-key rows as the likely third instance of "a type list or bound that
   silently withholds the parallel path". Measured, `cpu_time / wall_time` on 96 cores:

   | sort | wall | cpu | parallelism |
   |---|---|---|---|
   | 1 key, `date32` | 47.3 ms | 782 ms | 16.55x |
   | 1 key, `Utf8` | 169.2 ms | 4,394 ms | 25.97x |
   | **2 keys, the 2.06x case** | **51.1 ms** | **1,304 ms** | **25.49x** |
   | 2 keys, `(int32, int64)` | 45.3 ms | 1,686 ms | 37.20x |

   Every shape is properly parallel. **The prediction is retired; there is no third instance in
   the sort.**

### What is left, stated as what it is

The entire gap is a fixed ~28 ms of *work*: Batcher spends 1,304 ms of CPU to finish in 51 ms
where DuckDB finishes in 26 ms. That is not a missing technique, a wrong path, or a serial
loop — it is CPU efficiency, and it is the cost `CLAUDE.md` already names when it says Batcher
"buys several of its wall-clock wins with 1.4-4.4x more CPU". An eight-pass LSD radix over 6 M
`u64` keys moves ~384 MB and does 48 M scattered writes; DuckDB's comparison sort over 10-byte
keys stays in cache within each block and merges. Below some row count the radix's guaranteed
`O(n)` stops paying for its memory traffic, and 6 M rows on this machine is evidently below it.

**So the next move on multi-key sorts is a crossover, not a new algorithm**: measure where the
packed radix loses to the row-encoded comparison sort and gate on it, exactly as
`ops::byte_sort::radix_sort_live` already does for byte keys (it declines above 2^18 rows for
this very reason, and `report_the_byte_radix_crossover` records the measurement). The packed
multi-key path has no such ceiling and should.

## 19. The operator suite is no longer the deficit, and DuckDB's *other* two sorters (2026-08-25)

Two findings, and the first one retires most of the second's motivation.

### The item-15 table is stale, and the direction it points is wrong now

Entry 15 (2026-08-18) is the ranking this document has used to direct sort work: "sorting is
four of the top five rows and about 59 ms of the roughly 70 ms lost in total". Re-run at
`HEAD`, same suite, same box, same TPC-H sf1, five engines, **every correctness check passed**:

| case | batcher | duckdb | ratio | entry 15 said |
|---|---|---|---|---|
| `op-sort-string` | 146.1 ms | 153.6 ms | **0.95x** | 1.16x |
| `op-sort-string-lowcard` | 31.4 ms | 42.4 ms | **0.74x** | 1.33x |
| `op-sort-multikey-narrow` | 31.2 ms | 43.5 ms | **0.72x** | 1.39x |
| `op-sort-multikey-wide` | 59.8 ms | 72.6 ms | **0.82x** | 1.21x |
| `op-sort-limit` | 7.9 ms | 8.0 ms | 1.00x | — |
| `op-sort-string-limit` | 10.6 ms | 9.4 ms | 1.13x | 1.37x |
| `op-filter-project` | 10.2 ms | 9.0 ms | 1.14x | 1.28x |

**Every sort row flipped from a loss to a win.** Batcher now beats DuckDB on **19 of 21**
cases, and the two it does not total **+2.4 ms**. The work between the two runs — entries 11c,
17, the per-shard probe blooms, `split_constant_ranges` — did it; no single entry claims the
whole flip, which is why nothing in this document recorded that the board had changed.

That is the fourth time this document has been found overstating what is left, and the first
time in the *helpful* direction. The instruction stands and now cuts both ways: **re-measure
before building, and expect the ranking to have moved.** An agent that had taken entry 15 at
its word today would have spent the session optimizing four operators that already win.

### DuckDB's in-memory sorter, read rather than measured

Entry 18 read `sort.cpp` and `create_sort_key.cpp` — how DuckDB *builds* a sort key. It did
not read what DuckDB then *sorts* it with, which is one line in
`src/common/sort/sorted_run.cpp:269`:

```cpp
duckdb_vergesort::vergesort(begin, end, std::less<SORT_KEY>(), fallback);
// fallback = duckdb_ska_sort::ska_sort(fb_begin, fb_end, ska_extract_key)
```

Two mechanisms Batcher did not have, and they are worth different amounts.

**Vergesort — natural-run detection. Taken; see below.** It finds the maximal ordered runs and
merges them, sorting only what lies between. What makes it worth copying is not the merge, it
is the *detection*: vergesort does not scan for runs, it **strides** `n / log2(n)` positions,
tests the pair it lands on, and only then expands outward. A run shorter than the stride is not
worth having, so failing to find one costs about two comparisons — roughly `3 * log2(n)` over the
whole input, sixty-odd on six million rows. Batcher had only the degenerate form of this
(`is_ordered`: the whole column, or nothing).

**ska_sort — an in-place MSD radix. Not taken, and the reason is entry 18's own measurement.**
Entry 18 attributed the multi-key gap to "CPU efficiency… an eight-pass LSD radix over 6 M
`u64` keys moves ~384 MB and does 48 M scattered writes", and proposed a crossover gate as the
next move. **That model is wrong, and the measurement above is why it no longer matters.**
`packed_multi_sort_indices` is not reached with six million rows: above 2^17 rows the sample-sort
range-routes first, so the radix runs per range over ~94,000 rows, which is cache-resident. The
proposed crossover would have gated a path that was already below its own cliff. `op-sort-multikey-narrow` is now 0.72x with no such gate.

### What natural runs are worth here, measured

Built as `bc-interp/src/ops/run_sort.rs`, wired into both radix paths (single fixed-width key
and the composite pack). One tree, two `.so`s differing only in whether detection runs, two
interleaved rounds, 6 M rows with a `float64` payload:

| key structure | detection off | detection on | speedup |
|---|---|---|---|
| random (**the control**) | 36.1 ms | 34.2 ms | 1.06x |
| fully sorted | 20.0 ms | 18.3 ms | 1.09x |
| strictly descending | 23.6 ms | 17.8 ms | **1.32x** |
| two sorted halves concatenated | 23.1 ms | 18.4 ms | **1.26x** |
| eight sorted parts | 25.3 ms | 21.2 ms | 1.19x |
| sixteen sorted parts | 23.6 ms | 21.7 ms | 1.09x |
| sixty-four sorted parts | 25.9 ms | 25.9 ms | 1.00x |
| sorted, 1% late arrivals | 24.3 ms | 23.1 ms | 1.05x |
| two keys, two sorted halves | 30.1 ms | 27.6 ms | 1.09x |

**Read the control row first.** Random input is unchanged, which is the property the whole
technique rests on and the reason it needs no gate, no IR flag and no Kyber rule.

**Read the ceiling second, because it says the work here is finished rather than promising.**
A fully sorted key costs 18.3 ms, and that is almost entirely the payload gather — entry 13's
point restated by a different route. The run shapes land at 18.4-21.7 ms, i.e. **within 20% of
a floor that is not the sort at all.** There is no second increment to collect on this operator
by ordering better; what is left is the gather, and that is `StringView` and selection vectors.

**Why sixty-four runs measures at parity, which is the useful mechanism note.** Above 2^17 rows
the sample-sort routes by key range *first*, so each range holds rows drawn from every run in the
relation — sixty-four relation-level runs become sixty-four fragments of ~1,470 rows inside a
94,000-row range, below that range's own `min_run` of ~5,500, and detection correctly declines.
Range-routing destroys run structure. Detecting on the whole relation *before* routing would see
it, but the table above prices that at 18.3 ms against 18.4 ms, so there is nothing there.

**The one place it departs from vergesort is stability**, and it has to. Vergesort feeds
`std::sort` and therefore reverses any *non-ascending* run; Batcher's sort is stable by
contract, so reversing a run holding two equal keys would emit the later row first. A descending
run is exploited only when **strictly** descending. `a_descending_key_with_ties_still_sorts_stably`
is the test that would catch a regression here, and it is the only failure mode in the module
that no timing would reveal.

Covered by `tests/differential/test_diff_sort_ordered_runs.py` (**47 cases**: seven run
structures x two row counts x both directions, plus nulls, temporal keys, the composite key,
`iter_batches`, `collect(spill=True)`, and the degenerate 0/1/2-row inputs) and ten unit tests
against a written-out stable oracle rather than against the radix, so a shared misunderstanding
cannot pass.

**And on a real cluster, because this is exactly the shape invariant #7 exists for.** Detection
sees whatever slice it is handed, so one node and three workers take genuinely different routes
— different run structures over the same rows — to what must be the identical permutation.
`test_diff_distributed_sort.py::test_distributed_run_structured_sort_equals_single_node` pins
the row *sequence* across three run structures x both directions; **9 passed in 27.9 s** on a
fresh `RAY_ADDRESS=local` cluster (CI has no Ray, so a recorded run is the only evidence).

### A candidate this pass turned up and did not build

`polars/crates/polars-ooc/` is a **global spill arbiter with scored victim selection**, which no
earlier pass has read. Every spillable registers a context; when global usage exceeds the budget
a manager picks *which* state to spill by a measured score — byte-seconds of relief per unit of
I/O time, with an exploration bonus for unmeasured contexts and an exponential penalty (5 s
half-life) for anything read back soon after being spilled
(`polars-ooc/src/{memory_manager,spill_context/stats}.rs`). Batcher spills per operator against
an envelope; it does not choose a victim across the operators alive at once. That is a Carbonite
concern by the layering ("Carbonite protects"), it is the same bandit shape `kyber/learning.py`
already uses, and it is unbuilt here. Not measured, so it is a candidate and not a
recommendation.

## 20. The range join had no benchmark, and the shape with no benchmark was the broken one (2026-08-25)

The range join is the only stateful operator with **no case in the committed suite**. Its
numbers have lived in a docstring and in `competitive_architecture.md` ceiling 7, taken ad-hoc,
and the two had drifted apart: the scorecard reports 1,493 ms at five million rows a side where
`join/range/mod.rs` reports 522 ms for the same shape after a rewrite neither document
propagated. This entry is what happened when that measurement was made repeatable.

`benchmarks/internals/operators/range_join.py` is the case. **It ingests both tables into
DuckDB's native store with an untimed `CREATE TABLE` and asserts `IE_JOIN` appears in the plan
before it times anything**, because DuckDB picks its own range join only when its cardinality
estimate clears `merge_join_threshold` and a table registered from Arrow never does — the
Arrow-registered comparison is a `NESTED_LOOP_JOIN` and reporting it would be a two-order
falsehood in Batcher's favour. Selectivity is held constant as `n` grows, so the emit term does
not swamp the algorithm.

### Two of the three shapes had already overtaken DuckDB, and ceiling 7 says otherwise

Best of two, both engines back to back on the same data:

| shape | n/side | batcher | duckdb | |
|---|---|---|---|---|
| band | 5,000,000 | 403.8 ms | 773.7 ms | **0.52x** |
| general IEJoin | 5,000,000 | 377.5 ms | 801.7 ms | **0.47x** |

Ceiling 7 says the operator "loses above ~1,000,000" and prices 5 M at 0.44x *against* Batcher.
Both halves of that are now wrong: the general IEJoin **wins at every size measured**, and the
band with it. That is the second stale competitive claim this pass has found (item 19 is the
other), from the same cause — the work landed and the document that directs work did not move.

### The third shape was 12x behind, and the tell was internal rather than competitive

| n/side | matches | batcher | duckdb | |
|---|---|---|---|---|
| 500,000 | 1.27 M | 97.6 ms | 38.6 ms | 2.53x |
| 1,000,000 | 3.86 M | 191.5 ms | 49.8 ms | 3.85x |
| 2,000,000 | 5.44 M | 378.9 ms | 58.8 ms | 6.44x |
| 5,000,000 | 13.51 M | 1,054.6 ms | 85.5 ms | **12.34x** |

**A single inequality — the simplest shape the operator serves — cost more than the general
two-inequality IEJoin over the same data** (378.9 ms against 178.4 ms at two million). A fast
path slower than the general path it exists to avoid is a defect statement that needs no
competitor at all, and it is the reading that located this: the ratio against DuckDB says
"slow", the ratio against the operator's own general path says *where*.

A second measurement said what kind of slow. Holding `n` at two million and varying only the
match count, both engines' times tracked the matches — so DuckDB is not avoiding the emit — but
Batcher carried a **~300 ms fixed cost against DuckDB's ~47 ms**. Fixed, not per-match, so it
was the setup rather than the output.

**The cause was one serial loop, and this module's own header had already documented its cost
for the neighbouring shape.** `single_condition` did a `partition_point` per left row, serially,
through the generic comparator. `band.rs`'s header records exactly that algorithm as having cost
*"11.5 s of an 11.8 s join"* at five million rows a side — "23 random probes into a 40 MB array,
five million times" — before [`Right::bounds_by_merge`] replaced it with a parallel monotone
merge. **The band was fixed and the single inequality was not**, and nothing noticed for the same
reason nothing noticed the drift above: no case measured it.

A single inequality is a band with an open upper bound, so it wants the identical machinery. It
now takes it (`band::run_single`), with the emission shared through `emit_slices` and the open
bound carried as an absent array rather than a repeated value:

| n/side | before | after | | vs DuckDB |
|---|---|---|---|---|
| 500,000 | 97.6 ms | **34.1 ms** | 2.9x | 2.53x -> **1.08x** |
| 1,000,000 | 191.5 ms | **67.4 ms** | 2.8x | 3.85x -> 1.43x |
| 2,000,000 | 378.9 ms | **109.3 ms** | 3.5x | 6.44x -> 2.25x |
| 5,000,000 | 1,054.6 ms | **251.1 ms** | **4.2x** | 12.34x -> **3.81x** |

Re-measured once more on the final build, all three shapes at five million rows a side, best of
three: `one` **233.8 ms** against DuckDB's 58.8 (3.97x), `band` **382.6** against 793.6 (0.48x),
`ie` **450.4** against 1,167.6 (0.39x). So the band and IEJoin paths did not regress through the
shared-emission refactor, and the single inequality holds its 4x improvement. 508 `bc-runtime`
tests pass, including `one_inequality_matches_the_cross_product_oracle`, and the benchmark's own
gate compares every count against DuckDB before timing it.

**It still loses ~4x at five million.** Batcher emits about 54,000 matches per millisecond
against DuckDB's 205,000 — roughly 17.7 ms per million pairs against 2.8 ms.

### A fix for that which was built, measured and reverted — do not rebuild it

The obvious reading of those per-match numbers is that `Out::pair` is the cost, and the obvious
culprit inside it is the **gather**: `order` holds universe entries, so every emitted pair paid
`rmap[order[p] - nl]`, a scattered read into `rmap` charged per *match* rather than per row. A
left row's matches are a contiguous slice, so resolving `order` to row indices once turns the
emission into `extend_from_slice` plus a fill — a `memcpy` instead of thirteen million gathers.

It was built (`IndexBuf::extend_rows`/`extend_repeat`, `Out::pairs_run`, an `order_rows`
precompute in `emit_slices`), it is correct, and **it is worth nothing.** Two `.so`s differing
only in that change, interleaved over two rounds at five million rows a side, each row best of
three — and normalized by DuckDB's time *in the same round*, because the box drifted 8% between
them and the raw numbers otherwise say whatever the load says:

| shape | per-pair gather | resolved once |
|---|---|---|
| `one` | 4.44x DuckDB | 4.37x |
| `band` | 0.459x | 0.453x |
| `ie` | 0.520x | 0.525x |

Neutral on all three. Reverted, because it also allocates a 20 MB array at this size for that
nothing. The raw numbers looked like a 9% win on `band` and were not: DuckDB ran 8% faster in
the same rounds, which is the whole of it. **An A/B against a competitor's own time in the same
round is the only form of this measurement worth taking on a shared box**, and the first
uninterleaved pass at these numbers said "5% better" where the interleaved one says "nothing".

So the per-match cost is not the gather, and the next reader starts from a narrower question
than "the emission is slow": something inside `Out::pair` or the loop around it costs ~6x what
DuckDB's does, and it is not memory locality.

**The generalizable finding is the missing benchmark, not the missing parallelism.** Three
shapes, one broken, and the broken one was invisible for as long as it was unmeasured — which is
the same sentence item 9 wrote about the string sort. An operator with no committed case has no
regression protection and no drift protection, and this one had neither while its documentation
disagreed with itself by 3x.

## 21. TPC-H at sf100, run rather than recalled: the OOM moved, it did not go (2026-08-25)

`competitive_architecture.md` has carried "**L at >=100M rows (2-11x, OOM on q3/q4/q5 at
sf100)**" since the streaming executor landed, with the caveat that sf100 "is untested here".
It is the oldest untested claim in either document, and it is the one that decides whether
"any data scale" is true. This entry tests it.

**Method.** 600,037,902 `lineitem` rows mirrored to local NVMe with canonical column names
(the repo's `tools/mirror_bench_data.py` writes each table with a single `COPY`, which at sf100
is a 600M-row conversion that was OOM-killed on this shared box; one source file per `COPY`
bounds the peak and the destination glob already accepts many parts). Then
`run.py --benchmark tpch --scale 100 --scan --engines batcher,duckdb`, with the process's RSS
sampled every two seconds.

**The three queries the claim names now complete.** q1 through q8 all ran. q3, q4 and q5 — the
three recorded as OOM-killed at 133 GB — finished, with q3 peaking at about **55 GB** and q8 at
**102 GB**. The streaming executor's structural win is real, and the specific claim about those
three queries should be retired.

**q9 is OOM-killed, twice.** First on a shared box at **127 GB**, which is not evidence on its
own; then again with the box quiesced and nothing else of consequence running, at **125 GB**.
Both deaths are a `SIGKILL` — no traceback, no benchmark row, the process simply gone — after a
climb through 71, 75, 80 and 125 GB over roughly forty minutes. So the honest restatement is:

> The sf100 OOM is **current**, and it has **moved**. It is not q3/q4/q5 any more; it is q9.

**Two further observations, both from the same run, and both worth more than the timings.**

*The memory profile is the story, not the wall clock.* q8 completing at 102 GB is not a win in
any sense that matters — DuckDB answers the same query in single-digit GB. A 100 GB working set
completes on a 184 GB box and OOMs on every smaller one, which is most of them. The gap at
sf100 is memory, and the queries that die are simply the ones whose working set crosses whatever
the box happens to have.

*And q9 was not even using the machine.* Sampled repeatedly while it ran: **~5.2 cores of 96
busy**, RSS flat for minutes at a time, 750 threads of which two were runnable. Whatever q9
spends forty minutes doing, it is doing it nearly serially. That is a different defect from the
memory one and probably the more tractable of the two, and no timing on a completing query
would have surfaced it.

**What this does not establish.** Both engines run in one process here, so the serial phase
cannot be attributed to Batcher rather than DuckDB from these samples alone — the RSS is far
more consistent with Batcher, but the next run should be `--engines batcher` so the question
does not need arguing. Nothing here is a timing claim: the box was shared for the first run and
q9 never produced a row in either, so **this entry reports capability and memory, not ratios.**

**The scorecard needs the same correction and did not get it in this pass**, because
`competitive_architecture.md` was being edited by another session throughout and editing it
would have swept that work into this one's commit. The rows to fix are the "Single-node >=100M
rows" scorecard line and ceiling 1's closing claim; the evidence is above.

## 22. The two measurements that made `StringView` the top structural item no longer reproduce (2026-08-25)

Entry 14 promoted `StringView` (arrow `Utf8View`) to "the new top structural item", and
`competitive_architecture.md`'s roadmap ranks it second only to the streaming executor. Its
case rests on two H2O join rows, both measured 2026-08-18 against DuckDB's native store at the
1e7 tier. **Both are queries in the committed suite**, so they can simply be re-run:

| | entry 14 (08-18) | run A (08-25) | run B (08-25) |
|---|---|---|---|
| `h2o-join-q5` — `x JOIN big USING (id3)`, full projection | 369.6 vs 204.8 = **1.80x** | 212.5 vs 252.0 = **0.84x** | 205.8 vs 249.4 = **0.83x** |
| `h2o-join-q4` — `x JOIN medium USING (id5)`, the **string** key | 231.7 vs 74.2 = **3.12x** | 83.9 vs 101.8 = **0.82x** | 77.7 vs 102.9 = **0.76x** |

Both flipped from losses to wins, on two independent runs, with every correctness check passing.
**q4 is the string-keyed join** — entry 14's single strongest datum, the 3.12x that the whole
"the gap is the string representation" argument was sized from — and Batcher now wins it.
Batcher wins all five join questions in both runs (0.37x-0.84x), and beats Polars on all five
in the run that included it.

The group-by half moved the same way. Entry 2 recorded "q2 2.78x on two string keys, q7 2.04x,
q3 1.39x"; re-run, those are **1.78x, 1.19x and 1.10x**, and Batcher beats Polars on all ten
questions while beating DuckDB on four.

### What this does and does not retire

**It does not refute entry 14's mechanism.** DuckDB's `string_t` really does inline twelve
bytes, Batcher's `Utf8` gather really does chase offsets, and the length-dependence entry 14
measured (3.51x on six-character strings against 1.66x on 26.5-character ones) is a property of
the two layouts rather than of any particular commit. If that argument is re-made it should be
re-made on its own terms.

**It does retire the ranking.** "Worth roughly parity-to-a-win on the two largest single-node
losses" is now false in its premise: those two queries are not losses. `StringView` is a new
physical type through `bc-arrow`, every `bc-expr` string kernel, the row encoder, the FFI
boundary and the IR's type vocabulary — a grep for `as_string`/`StringArray` finds **90 files** —
and it is a *partial* adoption that entry 2 already measured as losing. **Nothing in this
document currently justifies starting that, and an agent that started it on entry 14's numbers
would be optimizing two queries that already win.**

### The caveat, stated because it bounds the claim

This box is shared, and DuckDB's own times here are 20-37% higher than entry 14 recorded
(249 ms against 205 for q5), so **absolute times are not comparable across the two documents** —
only ratios measured back to back inside one run are. That is exactly why the table above prints
both engines for both runs rather than Batcher alone. What survives that caveat is the direction
and its size: Batcher's own q4 time is a third of what entry 14 recorded, the ratio flipped in
two independent runs, and no reading of the load explains a 3.12x loss becoming a 0.76x win.

**The generalizable point is the one items 19, 20 and 21 also make.** Four separate competitive
claims in these two documents have now been found stale in a single pass, all in the same
direction — the work landed and the documents that direct work did not move. The remedy is not
to write them more carefully; it is that **every claim worth directing work from should name a
committed benchmark case that reproduces it**, the way this entry could only be written because
q4 and q5 are in the suite. Item 20's range join had no such case, and its shape stayed broken
for as long as it was unmeasured.

## 23. The 99-query sweep: where the operator surface actually loses (2026-08-25)

Items 19-22 each re-measured one operator. This one measures the widest operator surface the
suite has — **TPC-DS, all 99 queries, sf1, batcher against duckdb** — because a per-operator
reading cannot say which operators a real workload spends its time in.

**Result: 3,293 ms against DuckDB's 3,386 ms, 0.97x overall.** 44 queries win on ratio and 55
lose, which sounds bad and is not: the wins are large (q36 0.04x, q64 0.06x, q75 0.08x, q74
0.10x, q8 0.10x) and the losses are small. Ranked by **time lost** rather than by ratio — the
ordering entry 12 argued for:

| query | batcher | duckdb | ratio | lost |
|---|---|---|---|---|
| q22 | 188.1 ms | 77.0 ms | 2.44x | **+111.1 ms** |
| q78 | 161.2 ms | 63.9 ms | 2.52x | **+97.3 ms** |
| q47 | 133.7 ms | 56.6 ms | 2.36x | +77.1 ms |
| q5 | 81.7 ms | 22.0 ms | 3.72x | +59.7 ms |
| q77 | 66.4 ms | 12.7 ms | 5.22x | +53.7 ms |
| q88 | 70.0 ms | 28.4 ms | 2.47x | +41.6 ms |

### A hypothesis this sweep killed — do not "fix" the rollup

Four of the six worst queries use `ROLLUP`, and Batcher lowers `ROLLUP` to one ordinary
`GROUP BY` per level stacked with `UNION ALL` (`api/multi_group.py`) where DuckDB computes every
grouping set in a single pass. That is an obvious-looking `O(levels x input)` against `O(input)`,
and it is wrong. Measured on 4 M rows, each shape against **its own** plain group-by so the
level count is the only variable:

| shape | batcher | duckdb | b/duck | x its own plain cost |
|---|---|---|---|---|
| `GROUP BY k1,k2,k3,k4` | 54.3 ms | 27.7 ms | 1.96x | b 1.0 / d 1.0 |
| `ROLLUP(k1,k2)` (3 levels) | 9.3 ms | 18.1 ms | **0.51x** | b 0.2 / d 0.7 |
| `ROLLUP(k1,k2,k3)` (4 levels) | 22.1 ms | 24.2 ms | **0.91x** | b 0.4 / d 0.9 |
| `ROLLUP(k1,k2,k3,k4)` (5 levels) | 83.3 ms | 75.0 ms | 1.11x | b 1.5 / d **2.7** |
| `CUBE(k1,k2,k3)` (8 levels) | 37.2 ms | 56.0 ms | **0.66x** | b 0.7 / d **2.0** |

**Batcher's rollup scales better than DuckDB's, not worse.** At five levels it costs 1.5x its
plain group-by against DuckDB's 2.7x, and it wins `CUBE` outright. The stacked-union lowering
gets the coarser levels almost free because each one aggregates a *smaller* input than the level
below it, which the "N passes over the input" intuition misses entirely.

So the rollup is not the bottleneck. **The plain multi-key group-by underneath it is** — 1.96x
on four integer keys with two aggregates, and that is the term every one of those queries pays
once per level. That is where the remaining TPC-DS time is, and it is `agg_par` / `assign_groups`
rather than anything grouping-sets-specific.

### Three hypotheses about *why*, all eliminated by measurement (2026-08-26)

Chased far enough to rule out the obvious causes, so the next reader starts past them.

**1. "There is a cost cliff at three keys."** Holding the group count at 65,536 and varying only
the key count, Batcher looked flat to two keys (1.08x, 1.03x) and then jumped to 1.73x at three
while DuckDB stayed level — a discrete threshold, which reads like a fast path switching off.
It is an artifact of comparing *different data layouts* at each key count, not a threshold.

**2. "The dense composite direct-map declines at three columns."** `dense_multi_span` gates on
the product of the columns' value ranges against `dense_budget`, with no column-count limit, so
this was checkable directly: run each shape twice, once with tiny spans (dense eligible) and
once with the same group count but spans multiplied out of the budget (dense refused). It
reported the dense path **21% slower than the fallback** at three keys — the same defect shape
as item 20's range join, where a fast path lost to the general path it exists to avoid.

**3. "So the budget is too generous and should be tightened."** This is the one worth recording,
because it is where the discipline mattered. The apparent loss did not survive repetition. A
second sweep showed *every* shape winning, including the 2-key/65,536-group case that had just
measured 0.93x. Interleaving the strongest apparent loss — 4 keys, 65,536 groups, the 0.74x
reading — five times on the same data gives **1.03x, 1.11x, 1.12x, 1.13x, 1.04x: a win in every
round.**

`dense_budget` is also shared with the single-key dense path, which measures as a clear win, so
tightening it on the strength of that 0.74x would have degraded a good path on the basis of one
noisy sample. **The gate is correct as written and should be left alone.**

What remains true is only the top-level number: a four-key integer `GROUP BY` runs at roughly
1.55-1.96x DuckDB, the dense-versus-hash decision is not the reason, and locating the real cause
needs Rust-level profiling rather than black-box probing from Python. That is the honest state of
this bottleneck.

**Followed up and found, 2026-08-29 — see item 26.** Reading the code rather than probing it
from Python located two causes in the same function, and this entry's own framing is worth
correcting on one point: the cost was never in the dense-versus-hash *decision*, which is what
all three hypotheses above tested. It was in the hash path itself, in how a composite key's hash
is built and how the table that holds it grows.

### `tpcds-q67` reports FAILED and the engine is not at fault

The sweep's one correctness failure, diagnosed rather than waved through, because "1 of 99
TPC-DS queries returns a wrong answer" is a serious claim in either direction.

q67 is `rank() OVER (PARTITION BY i_category ORDER BY sumsales DESC)` where `sumsales` is a
**float sum**. Four measurements:

1. `rank()` is correct in isolation — ties, nulls, `DESC`, and all four window functions agree
   with DuckDB exactly.
2. Run against the **raw DECIMAL parquet**, q67 is **byte-identical** between the two engines:
   100 rows, no difference, and both produce 56,692 exact adjacent ties.
3. Batcher is **self-consistent** across repeated runs.
4. Run against the harness's tables, which normalize `ss_sales_price` to `double`, the two
   engines produce **48,878 (DuckDB) against 50,487 (Batcher) exact adjacent ties** over the
   same 505,178 rows, with a largest sum difference of **1.19e-06**.

`rank()` keys on *exact* equality, so ~1,609 tie disagreements in the last bits become integer
rank differences, one of which lands inside `rk <= 100`. The failing row moves between runs
because which near-tie crosses the boundary does.

This is the behaviour `python-control-plane.md` explicitly permits — "floating-point reductions
are identical *up to reassociation*... bound to near the last bits; they do not remove it". The
defect is that **the gate compares an integer rank derived from a float ordering**, which is not
stable under a reassociation the contract allows.

**Deliberately not "fixed".** Weakening a correctness gate to make a run green is the move
`CLAUDE.md` names as the most damaging one available, and the contract's instruction where the
engines legitimately differ is to *surface it as a decision*. This entry is that surfacing. Whoever
owns the harness should decide between excluding `rk` from q67's comparison and accepting a
permanent known-failing row; both are defensible and neither should happen silently.

## 24. The capability census re-checked: the MAP constructor, scoped and pinned (2026-08-25)

`competitor_parity_census.md` names one item as "the next wave's highest-value single item":
`map(...)` has no constructor, which blocks ten Spark/DuckDB names *on the argument* rather than
on the function. That census's own instruction is to re-check before working from it, so this
entry does.

**It is accurate, and the shape is better than it reads.** Probed against DuckDB, every
`map_*` name fails, but the reason is narrow: an Arrow `Map` **column** works end to end
already. On a `map<string,int64>` column read from Arrow, passthrough, `.map.keys`,
`.map.values`, `.map.entries`, `.map.len`, `.map.get` and `.map.contains` all return DuckDB's
answer, nulls included. `ExprTag.MAP` and `bc_expr::Expr::Map { func: MapFunc }` are the
read side and they are complete. **What is missing is only the ability to *build* a map value
in an expression**, which is why every `map_*(map(...))` probe fails on its argument.

Also confirmed missing by the same probe, and still open: `struct_insert`, `struct_values`,
`list_reduce`, `list_zip` (the `ExprTag.LIST_ZIP` tag exists but no accessor or SQL name
reaches it), `equi_width_bins` and `bar`. **`generate_series` is listed as open and is not** —
it answers correctly, so that line of the census is stale.

### Semantics, pinned against the oracle before writing anything

| case | DuckDB |
|---|---|
| `map(['a','b'],[1,2])` | `{'a': 1, 'b': 2}` |
| `map([],[])` | `{}` |
| `map(NULL, NULL)` | `NULL` |
| `map(['a'],[NULL])` | `{'a': None}` — a null **value** is fine |
| `map([NULL],[1])` | **error**: "Map keys can not be NULL." |
| `map(['a','a'],[1,2])` | **error**: "Map keys must be unique." |
| `map(['a','b'],[1])` | **error**: key list does not align with value list |
| `map_from_arrays` | **does not exist** — it is Spark's name, not DuckDB's |

Three of those are errors, which is the part worth pinning: the tempting implementation
silently keeps the last duplicate or truncates to the shorter list, and both produce a
plausible wrong answer rather than a refusal.

### Built (2026-08-26)

`bc_expr::Expr::MakeMap { keys, values }` with the wire tag `make_map`, evaluated in
`bc-expr/src/eval/map_ops/make_map.rs` — its own package beside `list_ops`, because map
*construction* is not a list operation and `map_concat`/`map_from_entries` want the same
home. Surfaced as `bt.map_from_arrays(keys, values)` (Spark's name; Python's `map` builtin
rules out DuckDB's spelling at the top level) and as both `map(...)` and `map_from_arrays(...)`
in SQL. The JIT classifies it `Unsupported` and falls back, which is invariant #6's required
answer for a variant it cannot compile.

Uniqueness is checked on the key's **row-encoded bytes** (`arrow::row::RowConverter`) rather
than by a typed comparison, so one implementation covers every key type the encoder accepts
instead of restating equality per type — which is where a quiet disagreement with the grouping
path would otherwise start.

Verified: 5 Rust unit tests (the null-key/null-value asymmetry, per-row rather than
column-wide uniqueness, and the empty-versus-null map distinction an offsets bug collapses),
14 differential tests against DuckDB including all three refusals, the `test_ir_snapshot`
golden re-baselined (**13 lines, purely the new shape** — no other node moved), a new
user-guide page whose code blocks execute, and `examples/expr_collections/map_columns.py`
extended and run.

**One deliberate divergence, pinned by a test rather than left to be rediscovered.**
`map(NULL, NULL)` with a *bare untyped* SQL `NULL` answers a null map in DuckDB and raises
here: sqlglot types a bare `NULL` as `Int64`, so there is no key or value type to build the
`Map` fields from, and inventing one would put a guessed schema into the plan. A null *list
column*, which carries a type, behaves as DuckDB's does — and that is the case real data has.

### What the scoping said before it was built

It needed a new `bc_expr::Expr` variant — the read side has no constructor to reuse, and
`list_zip` + a cast cannot stand in because the accessor does not exist either. That means
editing `bc-expr/src/lib.rs` (the **wire contract**) and `eval/dispatch.rs`, and
`CLAUDE.md` requires the Python `to_ir()` tag and the Rust `serde` tag to move **in the same
commit**.

Both of those files were being actively edited by another session throughout this pass, and
`bc-expr` did not compile under `clippy -D warnings` because of that in-flight work. Adding a
variant to a contended wire contract — where two sessions could race on the tag set — is the
one change that must not be split across commits, so it was left for a session that can hold
`bc-expr`. The scoping above is the part that does not need to be redone: the semantics are
pinned, the read side is confirmed complete, and the work is one variant plus its eval, the
Python node, the two SQL names, an `test_ir_snapshot` entry and a differential test.

## 25. Spark and Arrow, reviewed against the operator surface (2026-08-26)

Items 19-24 read DuckDB, Polars, DataFusion and Daft. This closes the two the pass had leaned
on earlier entries for, by walking their inventories and grepping Batcher for each entry.

### Spark: AQE and `sql/execution`, thirteen techniques, one gap

| Spark | Batcher |
|---|---|
| `PlanAdaptiveDynamicPruningFilters` (DPP) | `kyber/rules/joins/rewrites.py` + `io/formats/structured/parquet/partitions.py`; and `stream/runtime_filter.rs` sinks the build-side key set **to the scan** |
| `OptimizeSkewedJoin` / `OptimizeSkewInRebalancePartitions` | `dist/skew.py` (item 5 — fires unasked above ~8.4 M rows) |
| `ReuseAdaptiveSubquery` | `kyber/common_subplan.py` (item 10i) |
| `OptimizeMetadataOnlyQuery` | `kyber/metadata_answer.py` |
| `ConvertSortMergeJoinToShuffledHashJoin` | `kyber/rules/selection.py` (+ the UCB1 bandit over join strategies) |
| `DemoteBroadcastHashJoin` | `dist/flight_broadcast.py` |
| `CombineAdjacentAggregation` | `kyber/rules/agg_pushdown.py` (item 0b) |
| `AQEPropagateEmptyRelation` | `dist/executor.py` |
| `ExpandExec` (grouping sets) | `api/multi_group.py` — and item 23 measured it **beating** DuckDB's single-pass form |
| `ExternalAppendOnlyUnsafeRowArray` | `dist/spill/buckets.py`, `ops/external_sort.rs` |
| `InsertSortForLimitAndOffset` | `dist/flight_sort.py`, `TopNBound` |
| `OptimizeShuffleWithLocalRead` | `dist/flight_sort.py` |
| **`CoalesceShufflePartitions`** | **absent** — see below |

### Arrow C++ compute kernels: no gaps found

Walked the `kernels/` families — `aggregate_pivot`, `aggregate_tdigest`, `aggregate_mode`,
`vector_cumulative_ops`, `vector_pairwise`, `vector_run_end_encode`, `vector_rank`,
`vector_select_k`, `vector_selection`, `scalar_nested`, `scalar_random` — and every one has a
Batcher site. Arrow contributes nothing to the backlog.

### The one gap, and why it is *not* being built on this evidence

Spark coalesces adjacent small reducer partitions using **this run's** map-output sizes.
Batcher sizes reducers from the worker count and learns a per-family EMA across runs
(`dist/adaptive_sizing/sizing.py`), which is a different answer to the same problem — and
arguably a better one, since Spark's feature exists largely because its default
`shuffle.partitions=200` is so often wrong, a default Batcher does not have.

So the question is empirical, and the measurement **failed in a way worth recording**.

A first probe globbed the shuffle scratch after a 4 M-row, 200,000-group distributed
aggregate and reported 92 reducers, a 0.19 MB median, and a 59.3 MB maximum — a **308x
skew**, which reads as overwhelming justification to build it. It is an artifact. The bucket
files are written to per-worker scratch and are **not visible from the driver**, so the glob
was counting **stale files from earlier runs in the same directory**. Re-run with per-query
isolation (only files created after the query started), the same probe reports *no new bucket
files visible from the driver* at all.

**The 308x number is false and nothing should be built on it.** Measuring this properly needs
the map task to report its own bucket sizes, which is also what an implementation would need —
the driver cannot stat what it cannot see — so the measurement and the feature share a
prerequisite. Until that exists there is no evidence Batcher's reducers are mis-sized, and the
ledger already ranks this 11th of 14.

The generalizable point is the one item 23 makes about `ROLLUP` and item 22 about
`StringView`: **a competitor having a technique is not evidence that Batcher needs it**, and a
number that arrives conveniently large deserves the same scrutiny as one that arrives
inconveniently small. This pass produced **four** such numbers — a refuted rollup bottleneck, a fabricated shuffle
skew, a bulk-emission win that was the box getting quieter, and a dense-path "loss" that won
every round once interleaved — against three genuine gaps that were built. Every one of the
four looked like a reason to change code. **Repeat the measurement before you do.**

## 26. The composite-integer `GROUP BY` hashed row-major — **landed, 1.4x to 6.2x** (2026-08-29)

Item 23 left the TPC-DS board with one undiagnosed residue: a four-key integer `GROUP BY` at
**1.55-1.96x DuckDB**, the term every level of every `ROLLUP` in that suite pays once. It ruled
out three hypotheses (a cost cliff at three keys, the dense composite map declining, a too-
generous `dense_budget`) and closed by saying that locating the cause needed Rust-level reading
rather than black-box probing from Python. This is that pass.

### What the code was doing

`assign_groups_int64_multi` built each row's composite hash **row-major**: construct an `ahash`
hasher, walk the column list writing every column's value into it, finish it — per row. DuckDB
(`src/common/vector_operations/vector_hash.cpp::TightLoopCombineHash`) and Polars
(`polars-core/src/hashing/vector_hasher.rs::vec_hash_combine`) both do the opposite, and the
same thing as each other: seed one hash per row, then make **one pass per column**, folding
that column's values into the running hashes.

The per-row cost is the smaller half of the difference. The larger half is the group table's
**resize**, and it is invisible in the loop that pays for it. `HashTable` rehashes every entry
it already holds on each growth step, and what an entry held here was a *representative row
index* — so the rehash closure read that row back out of **every** key column, one random
access each into an array of `8 x num_rows` bytes, with representatives scattered across the
whole relation. Keeping the hashes live makes that closure a single array read.

And the table grew far more than it needed to. `group_table_capacity` deliberately starts small
(analytical `GROUP BY`s are overwhelmingly low-cardinality, and sizing at the row count measured
**2.6x** slower), with `GroupGrowth` extrapolating the real group count once that initial
capacity fills. `GroupGrowth` was wired into two of the paths that grow such a table — the
`RowConverter` fallback and `assign_groups_packed_wide` — and **not** into this one, so a
near-unique composite integer key paid the full doubling cascade, each doubling rehashing every
entry through the scattered-read closure above. The two defects multiply, which is why the
column-major hash is worth four times as much before the presize as after it.

### Measured

Interleaved in one process, three implementations alternating over the same data, best of nine
rounds, with the group ids asserted **identical** on every round. Separate runs cannot resolve
anything below roughly 30% on this box, so the A/B alternates rather than comparing two
binaries; this run was also the only test in its process, on a box at load 11 rather than 25.
1,048,576 rows, single-threaded, `assign_groups` only (no aggregate). The middle column is the
old row-major hashing **with** the presize, so the two halves stay attributable:

| key | groups | before | + presize | + column-major | presize alone | both |
|---|---|---|---|---|---|---|
| 2 x `Int64`, near-unique | 1,016,244 | 177.5 ns/row | 56.0 | 48.6 | 3.17x | **3.65x** |
| 3 x `Int64` | 257,417 | 99.3 | 92.7 | 70.5 | 1.07x | **1.41x** |
| 4 x `Int64` | 860,183 | 155.7 | 101.3 | 78.3 | 1.54x | **1.99x** |
| 4 x `Int64`, near-unique | 1,048,576 | 285.6 | 69.8 | 46.4 | 4.09x | **6.15x** |
| 6 x `Int64` | 884,263 | 223.4 | 125.9 | 92.9 | 1.78x | **2.40x** |
| 2 x `Int64`, 1,024 groups | 1,024 | 15.8 | 16.9 | 14.4 | 0.93x | 1.10x |
| 3 x `Int64`, 4,096 groups | 4,096 | 28.4 | 29.0 | 25.3 | 0.98x | 1.12x |
| 4 x `Int64`, 4,096 groups | 4,096 | 35.7 | 36.3 | 33.4 | 0.98x | 1.07x |

**The presize is the larger half, and that is the opposite of what this pass set out to fix.**
It began as "Batcher hashes row-major where DuckDB and Polars hash column-major", which is true
and is worth 1.07x-1.41x on its own. The missing `GroupGrowth` is worth 3.2x-4.1x on the shapes
that matter, and it was not a missing technique at all — it was an existing, tested, documented
one wired into one composite path and not this one. Read that before reaching for a competitor's
design again: the largest number here came from finishing something this repository had already
built.

The two are also not independent, which is why the middle arm was worth running rather than
assumed. Column-major hashing is worth much more *before* the presize than after it (at four
columns near-unique: 285.6 to 69.8 by presizing, then 69.8 to 46.4), because a large part of
what it removes is the scattered per-column read inside the rehash closure — and presizing
removes most of the rehashes. Either fix alone leaves value on the table.

**The low-cardinality rows are the point of the table, not filler**, and they are the reason the
two must ship together. Presizing *alone* is 0.93x-0.98x there — a small loss, in the shape the
small initial capacity was tuned for — and the column-major hash is what turns those rows back
into 1.07x-1.12x. This is also where the one previous attempt on this operator died: delegating
a two-column key to `assign_groups_packed` bought 1.4-1.6x at high cardinality and cost **1.7x**
at low, so it was reverted, and the note recording that is still in `assign_groups_int64_multi`.

Two earlier runs of the same bench, contended, read the four high-cardinality `before` rows at
225.0 / 373.5 / 165.7 / 278.5 and 184.7 / 300.4 / 144.6 / 239.4 ns/row against the
177.5 / 285.6 / 155.7 / 223.4 above. The machine drifted by more than 25% between them; the
**ratios** stayed within about 0.2x throughout. That is the argument for interleaving, restated
as data — and the reason the absolute figures here are quoted only alongside their ratios.

### One smaller thing in the same function

`dense_multi_span` checks the running product of the columns' value ranges against the budget
**inside** its scan loop rather than after it, so a key that cannot be dense stops at the column
that proves it. It reads every row of every key column and is the one part of the composite
decision that is never free; on a high-cardinality key — exactly where the hash path is already
the expensive one — it was scanning all of them to learn what the first one said.

### Three obvious generalizations, all built, all measured, all reverted

Every one of these follows from the entry above by an argument that sounds right, and every one
loses. They are recorded so the next reader does not spend the afternoon this cost.

**Storing the hash in the table entry instead of materializing the array.** The array is 8 bytes
per input row; putting the hash in the entry removes it, makes the rehash closure the identity
rather than an array read, and lets a probe compare a full 64-bit hash before it touches the key
columns. Three wins and a memory saving. Measured against the shipped version it is **0.62x to
0.83x** on the four high-cardinality shapes — the entry goes from 4 bytes to 16, and past a few
hundred thousand groups the wider table costs more cache than any of that saves. The memory
argument that motivated it was also weaker than it looked: `assign_groups_packed` on the very
next path already materializes a `u128` per row, and the `RowConverter` path a whole encoded
row, so 8 bytes per row is *below* what this file already spends.

**Presizing the single-`Int64` hash table the same way.** `int_group_ids` has no `GroupGrowth`
and a near-unique sparse key reaches it, so this reads as the same bug one column narrower. It
is **0.80x to 0.93x** across 500 to 660,000 groups. The reason is the same as the one above and
is worth stating as a rule: this table's entry is `(key, group_id)` — sixteen bytes, chosen
deliberately so a probe does not have to fetch the key back out of the column — and presizing a
sixteen-byte-entry table scatters every probe across the final footprint from row zero, where
doubling into it keeps the live set compact for most of the pass. **`GroupGrowth` pays where the
entry is narrow and the rehash is expensive. It costs where the entry is wide.** The composite
path is the first; the single-key paths are the second.

**Presizing `assign_groups_bytes` and `group_by_u128`.** Same shape again. The string path split
— 1.56x on a near-unique key, 0.89x at 300,000 groups — which is not a result to ship on, and
`group_by_u128` was never measured at all. Both reverted; both are open questions rather than
known wins.

These three were measured in a process that was also running the rest of the crate's 515 tests,
so the magnitudes are soft. The **direction** was consistent across every shape and reproduced
the same explanation each time, which is what the reverts rest on. Anyone reviving one of them
should re-measure in isolation first — and should expect the entry width, not the presize, to be
the variable that decides it.

### What this does not claim

The figures above are `assign_groups` in isolation, single-threaded. A whole query divides them
by everything else it does — the scan, the accumulators, the combine — and the parallel path
calls this per 16,384-row morsel, where the table is small and the cascade barely fires. The
shape that collects the full win is the **whole-relation** call: `combine`'s regroup, a
distributed partition, and `DISTINCT` on a near-unique composite key. The end-to-end number
belongs against `op-groupby-multi-int`, which this pass added because the suite had no composite
*integer* case at all — only a single key and a two-column *byte* key, each of which takes a
specialized path that never reaches this code. A bottleneck with no case in the suite is how
item 9's string sort stayed invisible for months.

That case, at sf1 over `lineitem`, `GROUP BY l_linenumber, l_quantity, l_suppkey, l_partkey`
with a `SUM` and a `COUNT`, correctness gate passed on all three engines:

| batcher | duckdb | polars |
|---|---|---|
| 73.5 ms | 66.9 ms (1.10x) | 218.7 ms (0.34x) |

**Read that as where the shape stands, not as this change's delta.** It is a single run on a box
carrying several other sessions, and the suite geomean's own spread across three repeats was
**42%** — so the third digit is noise and the second is not safe either. The attributable figure
is the interleaved A/B above; this row exists so the shape has a number in the suite at all, and
so the next session has something to A/B a build against. It belongs in
`benchmarks/BENCHMARK_RESULTS.md` and is recorded here instead only because that file was being
rewritten by another session at the time.

## 27. The scan under a top-N, a spill and a float filter — four gaps, one of them a wrong answer (2026-09-16)

A pass started from the operator suite at `HEAD` (46 cases, **0.75x DuckDB and 0.16x Polars** by
geomean, every correctness gate passing) and found the operator surface is not where the
remaining single-node losses are. The largest remaining single-case losses were one string
kernel against DuckDB's inline string prefix (`LIKE 'the%'`, 2.63x) and DuckDB's anti-join for
`EXCEPT` (1.58x), both a few milliseconds. So it read the scan side instead, where DuckDB's
optimizer has a pass Batcher did not.

### 27a. A top-N bound proved from row-group statistics — **landed**

DuckDB's `RowGroupPruner` (`src/optimizer/row_group_pruner.cpp`) orders a table's row groups by
the sort key's zone map for `ORDER BY k LIMIT n`, and prunes the ones that cannot reach the
answer. Batcher had half of this: `kyber/learned_tuning/topn_bound.py` remembers the k-th value
from the *previous* run, so every first run read the whole relation.
`kyber/learned_tuning/topn_footer.py` computes the bound from the footers instead. For a
descending sort it walks the row groups from the largest `min`, summing non-null row counts, and
stops at the group that brings the sum to `k`. Its `min` is a value `k` rows provably reach, so
`key >= min` removes only rows strictly worse than the k-th, and predicate pushdown skips every
row group whose `max` is below it. The conductor applies it in
`api/orchestration/topn_seeding.py`, and on `collect(spill=True)` and `iter_batches()` too,
because a proved bound needs no row count to check.

Measured on a 40M-row, 8-file event table clustered on its timestamp, `ORDER BY ts DESC LIMIT 10`:
the first run went from **422 ms to 134 ms** with nothing learned, and the warm run from 433 to
**45 ms**, against DuckDB's 92 ms and Polars' 1,203 ms. The ascending warm run is 47 ms against
DuckDB's 33. A fresh interpreter's *first* query of any kind costs another ~850 ms, which is not
this bound's and is recorded as its own finding. It declines floats, strings, nanosecond timestamps, nulls-first
orderings, anything but a renaming `Project` between the sort and the scan, and any bound that
would leave more than half the rows. The recorded run is in `BENCHMARK_RESULTS.md`, produced by
`benchmarks/internals/operators/parquet_topn.py`.

### 27b. The learned bound had never worked on a temporal key

Found while measuring 27a. The hub stores JSON, `record_topn_bound` stored the k-th value raw,
and for a `datetime` that raised inside a `try` that recorded the error as a suppressed hint.
So the *most common* top-N shape, the latest events, never seeded: every warm run of the
timestamp query read all 40M rows, 433 ms, while the identical integer query was answered from
35 rows. Non-JSON bounds are now stored as a one-element Arrow IPC stream. Warm: **433 to 39 ms**.
It is recorded here because it is the pattern this document keeps finding: a feature measured
at 19x on the shape it was built on, silently inert on the shape users write.

### 27c. A spilled query never pushed a predicate

Every out-of-core phase reads through `dist/spill/scratch.py::_iter_spill_morsels`, which took a
projection and no predicate. So `collect(spill=True)` decoded every row group that `collect()`
pruned, on exactly the queries that spill because their inputs are large. `map_predicate` now
sits beside `map_projection` for aggregate, join, sort and window. A 5%-selective clustered
filter under a spilled group-by: **1,189 to 95 ms (12.5x)**.

### 27d. A float filter over Parquet dropped NaN rows — **a wrong answer at every layer**

The engine ranks NaN above every number (`bc_arrow::canon_float_array`), so `x > 0.9` is true
for a NaN row. Parquet, Delta and Iceberg keep NaN out of a float column's recorded min/max. Four
separate layers pruned `x > v` on that max: the native reader's row-group and page pruning
(`bc-io/src/predicate.rs`), the pyarrow filter the Parquet source re-applies, manifest file
skipping, and Kyber's zone-map rules via `SourceStatistics.to_relstats`. Over a row group of
`[0.1, 0.5, NaN]`, `collect()`, `count()`, `is_empty()`, `iter_batches()` and `collect(spill=True)`
all answered no rows where the same filter over the same table in memory keeps one.

**Every differential test passed through this, and the reason is the lesson.** DuckDB has the
identical behaviour by default: its Parquet reader trusts the float max unless you pass
`can_have_nan=true` (`extension/parquet/parquet_statistics.cpp`). The oracle agreed with the bug.
The fix follows DuckDB's `can_have_nan=true` semantics everywhere: a float max cannot prove `>`,
`>=` or `!=` empty, and the pyarrow translation Or-s `is_nan` into `>` and `>=`, which makes it
exact under negation too. `<`, `<=` and `=` keep their pruning, because no NaN satisfies them.
The tests compare against DuckDB **with** `can_have_nan=true` and against Batcher's own in-memory
answer.

What it costs: no row group or file is skipped on a float column's max for `>`, `>=` or `!=`.
Those predicates prune well only on a float column clustered across files, which analytic
tables rarely have. The integer, date and timestamp paths are untouched. Measured on TPC-H sf10
read from Parquet, alternating builds with and without it: per-query geomean **1.003**, every
correctness check passing (`BENCHMARK_RESULTS.md`, 2026-09-16).

### 27e. Float predicates now filter inside the Parquet decode

`bc-io`'s row filter decodes the predicate columns first and the rest only for surviving rows,
and it refused floats outright because its comparison order differs from the engine's. It now
admits them as a **superset**: every row the two orders could disagree on, NaN and a zero against
a zero literal, is kept, and the engine's `Filter` above the scan decides. Pinned against the
engine's own canonicalization in `row_filter::tests::a_float_mask_keeps_every_row_the_engine_keeps`.
An interleaved A/B at sf10 put a selective filter on `l_extendedprice` at **163-169 ms before and
101-108 ms after**. q6 did not move (186-199 against 181-214 ms), because its payload is one
column and the saving is the payload's decode.

### 27f. `EXCEPT` as a distinct anti-join — built as a prototype, measured, not taken

`op-except` is 1.58x DuckDB while `op-intersect` is at parity, and DuckDB plans `EXCEPT` as an
anti-join. Batcher lowers both to one membership group-by (`Dataset._set_membership`), so the
obvious move is `left.distinct()` anti-joined to the right side on NULL-safe keys. At the
benchmark's shape (1.5M distinct left keys, 6M right rows over 1.5M keys, in memory): the
membership group-by took **57 ms**, the anti-join **141 ms**, DuckDB **39 ms**. The semi-join form
of `INTERSECT` took 49 ms against the group-by's 55 and DuckDB's 67, which is not worth a second
lowering. The anti-join builds on the 6M-row side; the group-by aggregates both sides once. Do not
rebuild this without a right-side distinct in front of it, and measure that first.

And the float row filter in 27e is not free on a permissive read: q6's filtered read, isolated,
costs 294 ms and 6.3 CPU-seconds with the filter against 144 ms and 3.3 without, because
decoding through a fragmented 12.5% selection costs more than the one payload column it skips.
The end-to-end query does not move, because the filter hands the engine 7.5M rows instead of 60M.
A width-aware admission rule (payload columns against predicate columns) is the open refinement.

### 27g. Date predicates in the native Parquet reader — built a third time, reverted a third time, and why

TPC-H at sf10 read from Parquet is **1.52x DuckDB** (item 27d's A/B), 21 of 22 queries lose, and
its worst ratios all filter on a date. `to_native_predicate` refuses temporal literals, so every
such read falls to pyarrow's per-file filtered scan. The 2026-09-08 attempt at pushing dates was
reverted because that box could not measure it (q3 was bimodal). This box can: two sandboxes,
alternating, two rounds, and the spread between rounds sits inside a few percent.

Built: a tagged `{"date": days}` literal, compared only against a column the file declares a
Parquet `DATE`, in row-group pruning, page-index pruning and the row filter, with Rust tests that a
plain `INT32` column holding the same day numbers is never pruned by it.

Measured on TPC-H sf10: **0.994 overall**, q6 **0.63x**, q14 0.82x, q21 0.83x, and q3 **1.35x**,
q7 **1.31x** slower. Decomposed, the loss is not the read that got the date. q3's `lineitem` read
cost 183 ms natively against 169 ms through pyarrow. It is the *other* reads and the batch shape.
Pushing dates made the native reader take reads that pyarrow's scanner serves better: a
date that keeps a third to a half of `lineitem` is decoded whole by the native reader and then
re-filtered once in Python, where pyarrow decodes and filters in one parallel pass and returns
fewer, larger batches. q7's engine execute time rose from 60 ms to 103 ms on the same rows.

Two follow-ups were built and measured on the way, and both lost. Filtering decoded batches
inside each row group's task, in parallel, made q7's `lineitem` read **455-470 ms** against the
native 322 ms: the filtered batches then come back small and are copied again by the coalescing
step and again by the Python re-filter. And lowering the row filter's gate to the read-only
crossover (2 %) turned q6's 0.63x back into 0.94x.

**What would make date pushdown pay is a routing decision, not a literal.** The native reader wins
when row groups prune or the row filter admits the predicate, and pyarrow's scanner wins on a
permissive predicate over a wide read. Choosing between them per read from footer statistics is
the open item, and it is a cost model in the conductor rather than anything in `bc-io`. Do not
re-add the literal without it.

### 27h. The row filter's selectivity gate was set at half, and the cliff is below a tenth — **landed**

Found while chasing 27g's q3. `bc-io`'s row filter was admitted for any predicate measured under
50 % selective, from two points on sf1. Across the range on sf10 `lineitem` (60M rows, a scattered
integer predicate, filter plus aggregate, two builds alternating), the filter is a win at 5 % and a
loss by 10 %, and at 20-35 % the query ran **1.44-1.47x slower** with it than without. The gate is
now 8 %: unchanged within 2 % at 1-5 %, and **0.68-0.95x** at 10-35 %. The per-point table is on
`MAX_SELECTIVITY`, together with the read-only measurement that pointed at 2 % and the end-to-end
one that overruled it: declining the filter leaves the caller an unfiltered read to re-filter,
which a read-only benchmark does not charge.

### 27i. A fresh process pays ~850 ms before its first query does any work

Recorded, not addressed. In a fresh interpreter any first query costs 830-880 ms, where DuckDB
imports and answers `SELECT 1` in 57 ms. `import batcher` is lazy (1 ms), but resolving `bt.read`
loads **492 Batcher modules** in about 460 ms, and the first `collect()` imports another ~450 ms,
most of it `pyarrow.dataset` and, through pyarrow's own pandas shim, pandas (~200 ms; any pyarrow
array built from a Python list imports it, so this part is pyarrow's). No benchmark here sees it,
because every suite reports a warm best-of-N. It decides every short script, notebook cell and CLI
run, and it is a lazy-import refactor across the reader namespace and the orchestration imports
rather than an engine change.

One part of the first `collect()` was Batcher's own and was not an import. The optimizer asked
each expression type `hasattr(expr, "op")`, and a miss falls through to `Expr.__getattr__`, which
builds the user-facing migration hint by loading the whole migration registry, 42 files. That cost
114-186 ms of CPU per process, measured on a loaded box, for a message `hasattr` discards. The
probe now uses `object.__getattribute__`, which never reaches `__getattr__`, and
`tests/unit/test_expr_dispatch_probe.py` fails on the old probe.

pandas itself is pyarrow's, and not avoidable from here: `pa.array`, `pa.scalar` and `pa.table`
import it even from a numpy array. An unfiltered Parquet read never touches those, and a filtered
one did only because row-group pruning builds its manifest and literals with them (27j). A read
of one morsel or less, 16,384 rows, now skips pruning, since the decode it could save is a
fraction of the 244-272 ms import, and a filtered first query over a small file no longer loads
pandas. `test_diff_parquet_pruned_whole_read.py` checks those reads against DuckDB and pins that
pruning still runs one row above the threshold.

### 27j. A predicated Parquet read decodes its surviving row groups whole — **landed, TPC-H sf10 1.54x to 1.38x**

27g ended on "a routing decision, not a literal", and the oracle for that decision came out
one-sided. Forcing every predicated read to the native reader *unfiltered*, with the engine's
`Filter` doing all row selection, took TPC-H sf10 from Parquet from 1.526x DuckDB to 1.356x
with no query meaningfully slower. The mechanism is CPU, not cores: pyarrow's filtered read of
q1's `lineitem` columns spends 10.6 CPU-seconds where the native decode spends 5.2, and the
engine's filter is a SIMD comparison over morsels already in parallel. So the 2026-09-08 entry's
"native batched + engine filter: three of four slower" does not reproduce on a quiet box, and that
entry had itself withdrawn its per-query numbers as bimodal.

What makes this a routing decision rather than "never filter in the reader" is memory and
clustering, and `io/formats/structured/parquet/routing.py` handles both:

- **Row groups are still pruned first**, from the footers, through the same NaN-aware rules
  `io.stats.file_skipping` applies to lakehouse manifests. A clustered table reads only the
  groups that can match, so 27a's top-N bound keeps its 10x.
- **Only while the survivors fit a quarter of the memory envelope.** A scattered predicate can
  leave every row group alive, and a whole file is up to 100x what its matching rows occupy.
  Past the guard the filtered readers run exactly as before.

Measured, the build with the route against the same build without it, alternating over two
rounds, every correctness check passing: per-query geomean **0.899**, b/duckdb **1.542 -> 1.379**,
q12 0.61x, q3 0.66x, q10 0.69x, q6 0.70x, q1 0.79x.

### 27k. arrow-rs 56 to 60 - **landed, TPC-H sf10 from Parquet 1.39x to 1.26x**

With the read route in 27j, the rest of the scan gap is decode cost, and that code is arrow-rs's.
Batcher pinned 56, four majors behind. A side-by-side benchmark decoding seven sf10 `lineitem`
columns (zstd, dictionary-encoded) put `parquet` 60 at **1.2-1.6x faster decode** with metadata
parsed once, and about 2x once footer parsing is included.

Upgraded: `arrow`, `arrow-pyarrow`, `parquet`, `arrow-avro`, `arrow-flight` 56 -> 60; `pyo3` 0.25 ->
0.29; `object_store` 0.12 -> 0.14; `tonic` 0.13 -> 0.14; `rust-version` 1.85 -> 1.88. API breaks
were mechanical: the page-index types, `ArrowReaderOptions`' page-index policy, pyo3's
`allow_threads`/`with_gil` renames, `ObjectStoreExt`. Three things were not:

- **The new page-index accessors panic on an out-of-range page** where 56 returned `None`, so they
  are bounds-checked to keep "cannot decide, keep the page".
- **object_store 0.14 enables rustls' `aws-lc-rs` backend beside `ring`**, and rustls panics at the
  first handshake with both present. The Flight shuffle installs `ring` once if nothing else has.
- **`az://container/a/b.parquet` now addresses `a/b.parquet`.** object_store 0.12 took the host as
  the container in its builder and *also* stripped the first path segment as a bucket, so it read
  `b.parquet`, silently dropping a directory. This is a fix, and `store.rs`'s test records it.

Measured: builds of the same commit on 56 and 60, alternating over two rounds of TPC-H sf10 read
from Parquet, every correctness check passing: per-query geomean **0.905**, faster on 21 of 22
queries, b/duckdb **1.389 -> 1.255** (each build's best time over DuckDB's best across all four
runs). q21 is the one slower query, at 1.02x; q11 returns no rows at this scale, so its time was
compared and its result was not. The box was shared during the run (load average 22-31), which
widens each query's spread but not the direction. Rust: 2,476 tests passed, 0 failed. The
differential, Parquet IO and Parquet unit suites gave identical results on both builds.

### What this pass did not do, and where the single-node gap now is

The sf10 decomposition in `BENCHMARK_RESULTS.md` (2026-09-08) put the loss in the Parquet reader,
on a 16-core box, as a read and an execute that do not overlap. Re-measured here on 48 cores it
is the read's **total CPU**, not its occupancy: three `lineitem` columns decode in 4.1 CPU-seconds
against DuckDB's 2.6 for decoding *and* summing them, at 34-43 cores busy. A prototype that folded
each file into a partial aggregate concurrently, overlapping read and execute through the
existing mergeable primitives, was **slower** on every shape (sum 40 to 82 ms, q1 about 400 to
540 ms), and is not the lever on a box this wide. What remained was per-value decode cost in
arrow-rs 56, a dependency question that item 27k answered by moving to 60, and DuckDB's
inline-prefix strings, item 2.

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

**Status pass, 2026-08-06 — and it found the same drift a third time.** Every open item below
was re-checked against the code rather than trusted. **Three of the six had been built since the
last pass and were still listed as open** (items 3, 5 and 6), a fourth (item 4) was half-built in
a way the entry did not describe, and the one capability gap the sweep found in 10h has now been
closed. That is the third consecutive pass to find this document overstating what is left, which
is worth more than any single entry: **re-check before building, and expect roughly half of what
is written here to be done already.** Each corrected entry now names the type or file that
implements it, so the next reader can settle it with one grep instead of one day.

0. ~~A range-join algorithm (item 8).~~ **Landed**, and its follow-up re-tuned — see item 8 for
   why the block-decomposition step it named is a *distribution* prerequisite and not a
   single-node speed fix.
1. ~~Online adaptive conjunct reordering (item 3).~~ **Landed** as `ConjunctOrder`, in a
   stronger form than DuckDB's (per-conjunct attribution rather than an aggregate hill-climb).
2. **Preserve the dictionary across the FFI boundary (item 6).** ~~Promoted to the top of the
   remaining work, because it is not an optimization to build — the kernels exist and are
   tested. It is one decode, in `normalize_to`, standing between them and every real query.~~

   **Reframed 2026-08-06 to match item 6's own record, which this entry contradicted.** It is
   *not* "one decode" any more: decode-on-egress was built end to end, measured, and
   **reverted**, because it is 1.2-2.8x on shapes that *consume* the column and **0.63x on
   `SELECT <a string column>`**, which is not a corner case. What remains is a **planner**
   decision — preserve the encoding only when the plan consumes the column (filter, group-by,
   join key) and decode at the leaf when it projects it — so this is a cost model, not a
   boundary edit, and it should be ranked as one. The operator half is done and committed
   (`bc-interp/tests/dictionary_operators.rs`).
3. ~~The early exit for `DISTINCT`/`GROUP BY` under a `LIMIT` (item 10a).~~ **Landed**, end to
   end, and verified 2026-08-06. `bc_ir::RelOp::Distinct` carries a `limit`; all three
   executors honour it (`lib.rs`, `par.rs`, `stream/parallel.rs`), `bc_runtime::agg::distinct`
   implements the early-stopping `DistinctPrefix`, `plan/logical/relational.py::Distinct`
   lowers it, and `kyber/rules/extra/topn_limit.py::fuse_limit_into_distinct` fuses the pair.
   Covered by `tests/differential/test_diff_distinct_limit.py`.

   Two things the implementation settled that this entry had left open. It keeps **the first
   `k` in input order** rather than an arbitrary `k`, because invariant #7 needs one node and
   many to agree where DuckDB is free to take whichever `k` its threads reach first. And the
   gating worry was unnecessary: the early exit is naturally conditional, since a key with
   fewer than `k` distinct values never reaches the threshold and falls through to the dense
   direct-map path that already beat DuckDB 4.9x.

   **Now tracked by a benchmark case**, which it was not — the gap this document calls the
   only asymptotic one had no case in the suite, which is exactly how item 9's gap stayed
   invisible. The case is `op-distinct-limit` in `benchmarks/suites/operators/dedup.py`; note
   that that file is itself uncommitted at the time of writing, so the case lands only when it
   does. At scale 1 over 6M `lineitem` rows, correctness gate passed on all four engines:

   | batcher | duckdb | polars | pyarrow |
   |---|---|---|---|
   | 3.1 ms | 40.4 ms (13x) | 68.6 ms (22x) | 198.1 ms (64x) |

   The case counts the limited distinct rather than returning it, because `DISTINCT … LIMIT k`
   does not say *which* `k` and the engines legitimately differ; counting is deterministic and
   still makes every engine do the same work. Verified as a real early exit rather than a
   metadata shortcut (the trap 10g warns about) by scaling the input: on a near-unique key the
   time is flat at 2.9-5.7 ms from 2M to 32M rows, while on a 3-distinct-value key, where the
   exit *cannot* fire, it scales 9.8 -> 12.1 -> 19.6 ms and returns 3.
4. ~~`row_number() OVER ()` porting guidance (item 10b).~~ **Done**, and the entry understated
   what already existed. The SQL path does not raise at all: `_sql/parser/windowing/translate.py`
   lowers `row_number() OVER ()` to `with_row_index`, so a ported DuckDB or Polars query runs
   and returns `[1, 2, 3]`. What remained was the *DataFrame* path, where
   `bt.row_number().over()` still raised without naming the alternative; the error now does
   (`plan/logical/window.py`). The hint is withheld when `partition_by` is set, because
   `with_row_index` numbers the relation rather than each group, and offering it there would
   trade a clear refusal for a wrong answer.
5. ~~Common subplan elimination (item 10i).~~ **Landed** as `kyber/common_subplan.py` (decides
   which repeated subplans are worth materializing) plus `api/subplan_reuse.py` (runs them and
   splices the results back), keeping the decide/execute split `gating` and `staging` use.
   Measured by its own module docstring at **1.95x** on a 4M-row `GROUP BY` feeding both
   operands of a join, against a floor of "what computing the shared half once costs".
6. ~~The low-cardinality string sort (item 9).~~ **Landed**, and in a stronger form than the
   fix this entry described. `sample_sort.rs:191` gives the single-key path its own fallback,
   `split_constant_ranges`, rather than extending the multi-key composite re-route. The two
   trigger at different thresholds on purpose, and the reasoning recorded there is better than
   the one here: the composite re-route waits for real skew (`3 * fair_share`) because it pays
   for a whole row encoding, while splitting fires as soon as a range exceeds a fair share,
   because **the failure is not skew at all** — twenty-five evenly-sized ranges on ninety-six
   cores are perfectly balanced and still leave seventy-one cores idle, which a skew test
   cannot see and which is exactly the shape `ORDER BY <a 25-value column>` produces.
5a. ~~Correlated `EXISTS` with a mixed equality-and-inequality correlation (item 10k).~~
   **Landed** as `_sql/parser/subquery/specialized.py`. This entry's reason for deferring it —
   "it needs a semi-join with a residual predicate rather than a parser edit" — was wrong, and
   wrong in the expensive direction: it deferred a front-end change by describing it as an
   engine one. The general decorrelation is a row tag, a join, a filter and a distinct, needs
   no new operator, and `NOT EXISTS` was broken for this shape too. See 10k.
6a. **A sorted-input aggregate and adjacent dedup (item 10f).** **The detection half is built
   (2026-08-13); the memory half — the reason this was ranked first — is not.**

   Built: `bc_runtime::agg::group::runs` assigns group ids by scanning runs instead of hashing,
   tried first inside `assign_groups`, so it reaches `GROUP BY`, `DISTINCT` and the partitioned
   window together. It needed **no IR flag and no Kyber rule**, because it proves the ordering
   rather than reading `sorted_by` — see 10f for why trusting that declaration would have been
   a wrong-answer risk rather than a slow one. Worth **1.0-1.2x end to end** on sorted input and
   measured 1.00x on everything else; the 6.0x in the microbenchmark did not survive to the
   query, and 10f records why.

   Still open, and it is the part with the value: **bounded state**. A sorted group-by has
   `O(1)` state, which is the one shape that turns an aggregate the streaming path *refuses*
   (`stream/breaker.rs`: "the streaming aggregate does not spill") into one it completes.
   Polars ships four operators that consume the property (`sorted_group_by`, `sorted_unique`,
   `is_first_distinct`, `merge_sorted`).

   The remaining cost is **not** the adjacency scan, which is now written and tested. It is a
   soundness question the earlier drafts of this entry did not name: a group emitted when its
   run closes cannot be taken back if a later batch reintroduces its key, and per-batch
   verification detects that only *after* the row has gone downstream. So early emission is
   sound only where the ordering is a fact about the **plan** — a `Sort` the engine itself
   performed — rather than a claim about the data. Decide that first; the IR flag, the
   mergeable form and the three executors follow from it, and a session that can run the
   distributed suite is still the right place for them.
7. ~~**An adaptive-width sort key for the per-range sort (item 9).**~~ **Closed 2026-09-11,
   measured on a quiet 48-core box: it does not work, and what was actually wrong was the
   admission rule beside it.** A 4-byte key loses 0.26x-0.72x above a quarter of a million
   rows, because four bytes settle a comparison only while the distinct 4-byte prefixes stay
   comparable to the row count, and past that the repair *is* the pointer-chasing sort the pack
   existed to avoid. A sample cannot even see this coming: 512 rows drawn from 8,000 distinct
   prefixes are essentially all distinct, so a sampled run length reads 1.0 where the true
   figure is 500. The measurement is `report_the_packed_prefix_alternatives` in
   `ops/byte_sort.rs`, which also buries the adjacent idea of packing past the rows' common
   prefix (2.1x-2.5x synthetic, **nothing** on the benchmark, because a sample-sort range of a
   text column has a common prefix of 0-1 bytes -- it is a *band*, not a stem). What the
   profile found instead: `prefix_discriminates` was refusing a pack that settled 99% of the
   comparisons in every one of the sort's thirty ranges, because it read pack *distinctness*
   where the cost depends on pack *tie probability*. Fixing that plus a type-erased key
   accessor the sort should never have been reading through took `op-sort-string` from
   **1.13x DuckDB to 0.91x** (220 -> 176 ms). See `BENCHMARK_RESULTS.md`.
8. `StringView` adoption (item 2), alongside 2, since both are about not destroying a compact
   string representation. Still the largest and most invasive single-node item — but now
   argued from `take`/`filter` only, and known to lose if adopted anywhere short of
   scan-native.
9. **The scan half of the top-K dynamic filter (item 4).** The morsel-skip half is landed; what
   is missing is republishing the bound so a Parquet reader can prune row groups, which is where
   the I/O saving is.
10. Skew salt derived from measured partition sizes (item 5).
11. Post-shuffle partition coalescing from the sizes just written, not the previous run's
    (item 10d).
12. Adaptive morsel sizing (item 7), if latency ever becomes the complaint.
13. **The sort's gather, which is now what is left on this operator (item 11, measured
    2026-08-18).** With the two fixes in item 11 landed, the operator suite's ordering rows sit
    at 1.06x-1.29x of DuckDB and the residue is no longer the comparison. Decomposed on 6 M rows
    with a `float64` payload: `ORDER BY <int64>` returning two columns is **37.6 ms** where the
    same query returning *one* column is the same time, so the cost is not the output width —
    it is 96 MiB of **random-access** gather at about 2.5 GB/s, against the 11 GB/s the same two
    columns copy at with no permutation. DuckDB answers the harder string version of that query
    in 30.4 ms.

    The mechanism is known and is not a tuning question: DuckDB sorts a **row-major payload**
    (`TupleDataCollection`), so its final scan is sequential, where Batcher sorts a permutation
    and then gathers every column through it. That is what the streaming-executor RFC's
    **selection vectors** proposal is for, which makes that proposal the next structural item on
    this operator rather than a leftover.

    Do not read this as "the sort is slow". It is the *permuted* materialization that is.

    **And do not fold `op-filter-project` (2.00x) into the same story**, which an earlier draft of
    this entry did. A filter's compaction is a *sequential* write, not a permuted read, and it
    measures accordingly: on 6 M rows, adding one output column to a filter costs 5.2 ms to read
    48 MiB and write 25 MiB, about 14 GB/s. There is no gather cliff there to find. Whatever that
    row is, it is not this.
14. **A measured anomaly on this operator that is *not* yet diagnosed: a temporal key sorts
    ~1.5x slower than the identical integer key.** 6 M rows, seven distinct values, same
    `float64` payload, both taking the parallel sample-sort after item 11c:

    | leading key | time |
    |---|---|
    | `Int64` | 31.1 ms |
    | `Int32` (normalized to `Int64` at the FFI boundary, so the same path) | 36.8 ms |
    | `Date32` | 47.0 ms |
    | `Date64` | 44.8 ms |
    | `Timestamp(us)` | 40.0 ms |

    **Two of the three candidate causes are now eliminated by measurement, so the next reader
    starts from a narrower question than this entry originally posed.**

    *Not the widening cast.* `arrow::compute::cast` allocates even when the widths already match,
    so the obvious suspect was the `8 x rows` buffer a `Timestamp` key builds on the way into a
    routing that only reads it. A zero-copy reinterpretation was built (`ScalarBuffer` clone plus
    the source's null buffer, pinned array-for-array against the cast it replaced) and measured
    **1.0x**. It was reverted: it removes a real allocation, and the allocation is not the cost.
    The table above already hinted at this and it was not read carefully enough — `Date64` and
    `Timestamp` are `i64`-backed, so their cast is the cheap one, and they cost what `Date32`
    costs.

    *Not the output gather.* Sorting by a `Timestamp` key while projecting an `Int64` **mirror**
    of the same values, so no temporal column reaches the output, still reads **47.4 ms** against
    **37.6 ms** for the same query sorted by the mirror itself.

    *And not a missing parallel path*, which item 11c would have made the natural guess: below
    400,000 rows the two types are at parity (6.5 ms against 7.4 ms) and both show the ns/row
    drop across `PARALLEL_SORT_MIN_ROWS`. The ratio appears between 400,000 and 1.5 M rows and
    holds at 1.38-1.44 above it.

    So it is inside the sort phase, it scales with the input, and it is neither the widening nor
    the gather. That is where instrumentation should start; black-box probing has taken it as far
    as it goes.

    **And check the data before believing a number here.** A first run of this table read
    `Date64` at **5.3 ms**, five times faster than `Int64`, which is not a plausible time for a
    6 M-row sort. The cause was the test data: `Date64` is milliseconds that the Arrow spec
    requires to be a whole number of days, the values used were raw day *numbers*, and every one
    of them therefore fell inside 1970-01-01. The sort was correct, the input was degenerate, and
    the identity short-circuit fired legitimately. The check written to catch exactly that —
    comparing against pyarrow's own sort — **passed**, because it compared the rendered dates,
    which were all equal. Day-aligning the values put the row back at 44.8 ms.

**Re-ranked 2026-08-18 by the census in item 12, which measured 80 queries rather than reasoning
from the code.** Two items move to the top and one new one appears:

0a. **~~Deferred payload materialization (selection vectors), RFC proposal 2.~~ Superseded as
    the top item by `StringView` — see entry 14, which re-measured this at `HEAD`.** Deferring
    cannot collect the two rows it was promoted on: `h2o-join-q4/q5` return every joined row, so
    the materialization is required rather than deferrable, and the gather kernel is already 15x
    faster than arrow's `take`. The measured gap is the string *representation* — 4.21x on string
    columns against 1.0x on numeric ones, over values averaging six characters, which fit inside
    DuckDB's inline `string_t` and inside arrow's `Utf8View`. Selection vectors remain the right
    mechanism for the shapes with a discarding operator downstream (`cb-q32`'s `LIMIT`, sort
    top-k), which the census prices at single-digit milliseconds each.

0a'. **`StringView` (arrow `Utf8View`) — ~~the new top structural item~~. Demoted 2026-08-25:
    item 22 re-ran entry 14's two rows and both are now wins, so the ranking's premise is gone.
    The layout argument still stands on its own; the priority does not.** Entry 14 has the
    measurement and the scope. It is a physical type through `bc-arrow`, the `bc-expr` string
    kernels, the row encoder and the FFI boundary, plus a planner decision about when to produce
    it — the same shape as the dictionary item 6, which was built and reverted for want of that
    decision. Worth roughly parity-to-a-win on the two largest single-node losses.

0b. ~~**Give `eager_aggregation` a cost term, not just a ratio.**~~ **Landed** as
    `_global_aggregate_gains_nothing` (`kyber/rules/agg_pushdown.py`), wired into all three
    pushdown rules. **2.29x / 2.23x / 2.14x** on the measured shapes
    (`count(id2)` 19.6 -> 8.6 ms, `sum(v1)` 20.3 -> 9.1 ms, `min(v1)` 21.0 -> 9.8 ms), with the
    two controls flat (0.94x on the aggregate that could not be pushed, 0.98x on the grouped
    form). The gate is `join_rows > source_rows` for an **ungrouped** outer aggregate only —
    "does the join amplify?", which is the condition eager aggregation always actually needed.
    Pinned by `test_a_global_aggregate_over_a_non_amplifying_join_does_not_push`, its amplifying
    twin, and a third test asserting the grouped path is untouched, so the veto cannot silently
    become "never push". The original entry read below.

    **Give `eager_aggregation` a cost term, not just a ratio.**
    `kyber/rules/agg_pushdown.py` gates the push on `_MIN_PREAGG_REDUCTION = 8.0`, a row-reduction
    ratio with no term for what the join would have cost. On a broadcast join against a
    10,000-row build side it fires on a ~1000x reduction and is **2.4-2.6x slower** than not
    firing (21.1 ms and 23.2 ms against 8.9 ms for the same query whose aggregate reads a
    right-side column and therefore cannot be pushed). The push pays a full 10 M-row hash
    aggregate to save a probe that was nearly free. Cheap to scope, and it is a *regression* the
    rule introduces rather than a missing capability. See item 12.

**A process note, since this document exists to direct work.** Three of the six items in the
previous version of this list described work that already existed, and one described a win the
engine cannot reach. Every row of the shortlist now names the type that implements it, so the
next reader can check the claim with one grep rather than trusting the prose. A parts list that
has drifted is worse than no parts list: it spends effort re-deriving what is there and defends
numbers the live path never produces.
