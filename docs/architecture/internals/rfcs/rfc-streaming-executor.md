# RFC: A streaming (pipelined) executor tier — closing the single-node scale gap to DuckDB

**Status: Proposal 1 (the streaming driver) is IMPLEMENTED and default** (`crates/bc-interp/src/
stream/`, `bc_py::execute_plan`). Proposals 2–5 remain proposed. See
[Proposal 1 — status](#proposal-1--the-streaming-driver-the-load-bearing-change) for what
landed and how it is verified. The rest of this document is the original design; nothing in it
bent an invariant, which is why the highest-risk item shipped behind the same `seq == par ==
streaming` oracle the design called for.

**What landed (2026-07-14).** A pull-based morsel driver: linear runs (`Scan`/`Filter`/`Project`/
`Unnest`/`Unpivot`/`RowId`/`Limit`) never materialize; the hash-join **probe** streams through a
`BroadcastProbe` built once; the aggregate folds incrementally (`partial`→`combine`), state
bounded by the group count. A parallel form shards the driving scan one-pipeline-per-worker and
combines at the root. It is the **default executor** for the in-memory path, with a budget check
that falls back to the spilling (materializing) executor rather than OOM.

- **Correctness:** `execute_streaming == execute` (the oracle), pinned over every operator, both
  serial and sharded, in `tests/stream_oracle.rs`; metrics agree with the oracle in
  `tests/stream_metrics.rs`. The full DuckDB differential suite passes with streaming as the
  default (the one residual failure is B26, a pre-existing IEEE-vs-total-order float-comparison
  bug in the shared kernel, unrelated to the executor).
- **Memory (the whole point):** on the q3/q4/q5 shape, peak is **flat in the input** — 3.4 MB at
  1M rows, 3.3 MB at 4M — where the materializing path grows linearly (198 → 794 MB).
  `tests/stream_memory.rs` guards that it never doubles across a 4× input growth. This is why the
  133 GB sf100 OOM disappears *structurally*.
- **Speed:** on that shape, **1.24× faster** than the materializing parallel path (109 ms vs
  136 ms), because the copies it stops making were not free.

The original design follows.

## Background: the gap is the interpreter, not the architecture

`docs/architecture/internals/execution.md` already states the target: *"A pipeline is a maximal chain of
operators that can run a batch straight through without materializing — scan, filter,
project, probe. A breaker is an operator that has to collect its input."* CLAUDE.md's
performance rule says the same: *"pipeline breakers materialize and are exactly where the
adaptive layer re-optimizes."*

The **Tier-0 interpreter does not do this.** `bc_interp::par::exec()` returns
`Vec<RecordBatch>` — it collects *every* operator's full output, breaker or not. A
`scan → filter → project → join` chain materializes the scan output, then the filter output,
then the project output, then both join inputs, all in RAM at once. Fusion
(`exec_fused`, `exec_agg_fused`) papers over the worst linear cases, but the model is
materialize-everything.

Two measured consequences (2026-07-10, this box, TPC-H local mirror):

1. **sf100 single-node OOM.** q3/q4/q5 (deep join trees) peak at **133 GB** and are
   OOM-killed. Projection pushdown *works* (q3's lineitem scan reads 4 of 16 columns via
   `source_projections`), so this is intermediate blow-up, not a wide scan — the executor
   holds every operator's full output across the whole plan. DuckDB streams the same query in
   a few GB.
2. **The gather tax on high-selectivity filters.** A clean `perf` of TPC-H q1 at sf10
   (in-memory): the aggregate is 50% (`agg::partial` 26%, `assign_groups` 24%) and the
   **filter gather is ~22%** (`memmove` 14% + `filter::extend_offsets_slices` 5% + memset).
   q1's `l_shipdate <= cutoff` passes ~98% of rows, so the engine gathers 59M rows — two
   string columns included — only to feed an aggregate that would have been happy to skip the
   2%. DuckDB carries a selection vector and never gathers.

Neither is an Arrow-contract problem. A morsel is a `RecordBatch`; streaming it instead of
collecting it changes *nothing* about the columnar contract. This RFC implements the
documented pipeline model as a new `core` **Executor strategy** — the sanctioned way to add an
execution tier (CLAUDE.md maintainability rule: *"New execution tier → a `core` Executor
strategy, not new call-site branching"*).

## The invariants this must respect

- **Arrow is the only columnar contract (#3).** Morsels stay `RecordBatch`. The one extension
  (Proposal 2, selection vectors) is itself an Arrow array carried alongside — no bespoke row
  format.
- **One `Expr`, one `RelOp`, `seq == par == JIT` (#6).** The streaming executor is a new
  *scheduling* of the same operator semantics, exactly as `par` is to `execute`. It must
  produce the identical relation (row multiset, and the same order the current path guarantees)
  — pinned by the existing oracle tests, unchanged.
- **Breakers are the adaptive re-optimization points (#10, the moat).** Streaming must *keep*
  breakers materializing so the adaptive loop still measures actual cardinalities and re-plans
  there. Streaming the *linear* runs between breakers is the whole point; the breakers stay.
- **Mergeable algebra / single-node == distributed (#7).** The breaker operators
  (`bc-runtime` `partial/combine/finalize`) are untouched; only how morsels are *driven into*
  them changes.

## Proposal 1 — the streaming driver (the load-bearing change)

**Problem.** `exec()` is a tree-walk that returns the full `Vec<RecordBatch>` of every node.

**Proposal.** Replace the driver for **pipeline (non-breaker) operators** with a pull-based
morsel stream. Concretely, an operator implements:

```rust
/// A source of morsels, pulled one at a time. `None` ends the stream. The morsel is an
/// Arrow RecordBatch (optionally carrying a selection — see Proposal 2). No relational
/// state lives here; breakers still own their state in bc-runtime.
trait MorselStream {
    fn next(&mut self) -> Result<Option<Morsel>, InterpError>;
}
```

- **Pipeline operators** (`Scan`, `Filter`, `Project`, `Unnest`, broadcast-join **probe**)
  become `MorselStream` adapters that pull from their child, transform one morsel, and yield
  — never collecting. A `scan → filter → project` chain is three chained iterators; peak
  memory is *one morsel per stage*, not the whole relation.
- **Breakers** (`Aggregate`, `Sort`, `Distinct`, `Window`, hash-join **build**) *consume* a
  `MorselStream` to build their state (`partial`/insert/sort-run), then *expose* their output
  as a new `MorselStream` (`finalize` streamed out). This is where materialization and the
  adaptive re-plan stay.
- **Parallelism** is unchanged in spirit: morselize the leaf, run one pipeline instance per
  worker over a shard of morsels (rayon), each feeding a thread-local breaker partial; combine
  at the breaker exactly as today. The `par` shuffle/partition logic is reused verbatim.

**What it buys.** The sf100 OOM disappears structurally — a linear run never holds more than
`workers × morsel` bytes, and a breaker holds only its (spillable) state. The deep join trees
(q3/q4/q5) that OOM at 133 GB run in bounded memory. No new spill logic needed; the existing
breaker spill (`agg::spill`, `spilling_hash_join_streaming`) becomes *sufficient* because the
non-breaker intermediates no longer coexist.

**Cost / risk.** This is a real rewrite of `par::exec`'s driver — the highest-risk item.
Mitigations: (a) land it as a *parallel* `Executor` strategy selected by config, with the
current path as the default until parity is proven; (b) the operator *semantics* (the `ops::`
and `bc-runtime` kernels) are reused unchanged, so only the *driver* is new; (c) the
`seq == par == streaming` differential + the DuckDB oracle gate every query shape before it
flips on.

## Proposal 2 — selection vectors, as an Arrow array

**Problem.** A pipeline filter must *gather* to drop rows, even when it passes 98% of them
(the 22% q1 tax). Every downstream operator then re-reads the gathered copy.

**Proposal.** Extend `Morsel` to `(RecordBatch, Option<UInt32Array>)` — the optional array is
a **selection vector** of live row indices. A `Filter` in a pipeline produces a selection
instead of gathering; downstream vectorized kernels honor it; the gather is *deferred* to the
first operator that genuinely needs contiguous data (a breaker's state build, or a sink), and
often elided entirely (an aggregate reads values at the selected indices — no contiguous copy
ever).

- **This is still Arrow.** A `UInt32Array` is a first-class Arrow array; it rides alongside the
  batch the way a validity bitmap does. Invariant #3 says the *columnar contract* is Arrow
  `RecordBatch`; the selection is metadata on the morsel, not a second columnar format. This is
  the one place the RFC *extends* the morsel type, so it needs an explicit maintainer decision —
  but it is the DuckDB/Velox design and it is pure Arrow.
- **Scope it to the fused linear paths first** (`exec_fused`, `exec_agg_fused`), where the win
  is largest (filter→aggregate is TPC-H q1/q6/q12/q14/q19/q20/q22) and the blast radius is
  smallest, before threading selections through joins.

**What it buys.** Kills the ~22% gather tax on high-selectivity filter→aggregate queries — the
exact sf10 shapes where Batcher trails DuckDB-on-Arrow (q1 1.19×, q19 1.27×).

**Corroboration (2026-07-24, sf1, 96 cores, release).** Re-measuring the DuckDB gap found it is
*this* proposal and not twelve separate ones: Batcher trails on 12 of 22 TPC-H queries, and
profiling the three worst put the time in materialization every time — q21's bottleneck is a
filter taking 90 ms to pass 3.79 M of 6 M rows and copy 87 MB; q5's is a hash join taking 204 ms
on a 6 M-row probe emitting 1.2 M rows.

One comparison isolates the cost from the probe itself, which the `perf` profile above cannot
do on its own. The **same** 6 M-row `lineitem` probe costs:

| Query | rows emitted | probe cost |
|---|---|---|
| q17 (first join) | 6,088 | 103 ms |
| q5 | 1,201,113 | 204 ms |

Same input, same operator, same key type — the cost tracks the *output* row count, so it is the
gather and not the hash probe. That is the proposal's premise measured directly rather than
inferred from a profile, and it is the strongest argument available for the morsel-type
extension this section asks a maintainer to decide on.

Also worth recording for whoever takes that decision: two adjacent leads were checked and are
dead. The `interp` backend label on these filters is **not** a JIT fallback — `bc-codegen`
accepts both `int > int` and `date > date` when asked directly, and `stream/mod.rs` records
that wiring Tier-1 into this path measured 1.01× over TPC-H with five queries slower. And
Kyber's join orders on q5/q17/q21 estimate within 2× of actual, so this is not a planning
problem.

## Proposal 3 — Arrow-native encodings (recover the "storage" win without a bespoke format)

**Problem.** DuckDB's native store dictionary-encodes low-cardinality columns and hashes
integer *codes*; Batcher hashes the string bytes (the group-by gap, ~2× at sf10). The FFI
boundary (`bc_py::normalize`) currently *decodes* Arrow dictionaries to plain values, throwing
away exactly the encoding that would make grouping fast.

**Proposal.** Two Arrow-native moves, no bespoke format:

1. **Preserve dictionary/run-end encoding on the Parquet scan** instead of eagerly decoding.
   `DictionaryArray` and `RunEndEncodedArray` are Arrow types; keeping a Parquet
   dictionary-encoded column as an Arrow `DictionaryArray` reads fewer bytes *and* carries the
   codes forward.
2. **Encoding-aware `assign_groups`.** A `DictionaryArray` group key groups on its `keys`
   (integer codes) — the fast integer path — with the dictionary as the representative column.
   Result-identical to grouping the decoded values.

Gate the FFI decode so a *group-key-eligible* dictionary column survives to the grouper (other
operators that don't yet understand dictionaries still decode on demand). This is the
"encoding-aware kernels" work: incremental, one kernel at a time, each with a differential test
(`dictionary key == decoded key`). It closes most of the storage gap inside invariant #3; the
only residual DuckDB keeps is general byte-compression of arbitrary columns, which is small
once dictionary/REE/zonemaps are in (Batcher already does zonemap pushdown on Parquet).

**Status / concrete blocker found (2026-07-10).** The grouper half is done: `assign_groups`
has a dictionary fast path (group on codes, decode-and-fall-back for a non-canonical dict or
null codes), verified `dictionary key == decoded key`. The FFI half is **not** landed, and a
prototype found exactly why. Preserving a string dictionary at the boundary
(`bc_py::normalize`) made `group_by` work *and* fast, but broke `distinct` with
`column types must match schema types, expected Dictionary(Int32, Utf8) but found Utf8`: the
logical plan derives every intermediate schema from the column's **Arrow type**, so a preserved
`Dictionary` propagates through the schema, and an operator that decodes it (distinct's rep
column, a sort output, a join output) produces a `Utf8` batch that fails the plan's `Dictionary`
schema check. The prerequisite is therefore a **logical-vs-physical type split**: the plan/schema
must treat a dictionary column *as its value type* (so all intermediate schemas are `Utf8`),
while the morsel carries the `Dictionary` encoding as an internal optimization the schema does
not see — and each operator either consumes the encoding (grouper) or decodes on demand
(everything else) without the schema disagreeing. That is the real unit of work here; the
grouper fast path is ready for it.

## Proposal 4 — grow the Cranelift tier to vectorized aggregate/hash kernels

**Problem.** At sf10 the raw per-row throughput of `agg::partial` (scatter-add) and
`assign_groups` (hashing) is scalar; DuckDB's is SIMD-vectorized (~2048-value vectors).

**Proposal.** This is what Tier-1 (`bc-codegen`) is *for*. Extend the JIT's supported subset
from numeric arith/compare to: SIMD scatter-add accumulation, vectorized key hashing, and
selection-aware kernels (Proposal 2). Under the existing invariant — *"bit-for-bit identical to
the interpreter, fall back otherwise"* — each kernel lands with a parity test and the
interpreter stays the oracle. Incremental and low-risk; the tier and its fallback discipline
already exist.

## Proposal 5 — lean on the moat to *surpass*, not just match

Proposals 1–4 reach parity with DuckDB's execution. The structural *win* is adaptivity, which
DuckDB (a static planner) cannot do: at a pipeline breaker Batcher has the **measured** actual
cardinality and can re-plan the *rest* of the query — flip a join order, swap the build side,
change a spill decision — mid-flight. This is decisive exactly where static planners are worst:
skew, correlated predicates, stale stats. Streaming (Proposal 1) does not weaken this — it
*preserves* the breakers as the re-plan points, and a streaming engine reaches the first
breaker (and thus the first correction) sooner. Keep the re-optimization hooks (`core`'s
adaptive loop) firing at every breaker; that is what turns "as fast as DuckDB" into "faster
than DuckDB where its plan is wrong."

## Sequencing and validation

1. **Proposal 1 (streaming driver)** — foundational; fixes the OOM (a robustness win, testable
   without timing: *does q3 sf100 complete under a bounded envelope?*). Land as a selectable
   `Executor` strategy, default off, until the `seq == par == streaming` oracle is green on
   every query shape.
2. **Proposal 3 (dictionary-aware grouping)** and **Proposal 4 (SIMD kernels)** — run in
   parallel; incremental, each kernel gated by a differential test. Close the throughput and
   storage gaps.
3. **Proposal 2 (selection vectors)** — scoped to fused filter→aggregate first; kills the
   gather tax.
4. **Proposal 5 (adaptive-first)** — continuous; the differentiator, layered throughout.

Every step is correctness-gated against the DuckDB differential oracle and the Rust
`seq == par == JIT` oracle before any timing is trusted — the same discipline the rest of the
engine holds to. And note the scope boundary that is already a win today: **at multi-node / PB
scale the comparison inverts** — DuckDB is single-node; Batcher's mergeable algebra shards
across a cluster with bounded per-node memory. This RFC is about owning the single-node
small-to-medium range while that stays true.
