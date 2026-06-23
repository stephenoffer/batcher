# Rule: Performance & Competitive Positioning

Batcher exists to win on performance across an unusually wide range: **sub-second
small queries to PB-scale**, **single-node to distributed**, **batch and
streaming**. Every performance-relevant change is measured, and measured against
the systems we claim to beat.

## The competitive mandate

We benchmark against DuckDB, Polars, Spark, and Ray Data — not against our own
last commit alone.

- Run `python benchmarks/run.py` (default 10M rows; pass a row count for other
  scales) for any change to an operator, the runtime, codegen, or a hot path. The
  harness (`benchmarks/harness.py`) checks correctness vs DuckDB/Polars first, then
  reports `batcher_ms | duckdb_ms | polars_ms | ratio`.
- **A regression is a blocking failure.** If a ratio worsens, explain why and fix
  it or justify the trade explicitly. Where Batcher already loses to DuckDB/Polars
  on a shape (e.g. some sort/distinct/window cases today), that gap is a known
  target — don't widen it.
- For distributed/large-scale changes you can't run locally at PB scale, reason
  about scaling explicitly: does the mergeable algebra keep per-node memory bounded?
  does the shuffle stay credit-controlled? Compare the *approach* to Spark
  (stage-boundary AQE) and Ray Data (no optimizer).

## How we win (don't undermine these)

- **Morsel-driven parallelism.** Work is morsels (~16,384-row `RecordBatch`es) so
  scheduling is granular and cache-friendly. Don't introduce whole-relation
  materialization where a streaming/morsel path is possible.
- **JIT-compiled expressions.** The Cranelift fast path compiles once per operator
  and reuses across morsels. Keep the supported subset growing *with parity*; never
  trade correctness for a faster wrong path (it must fall back, not diverge).
- **Single-node == distributed via mergeable algebra.** `partial → combine →
  finalize` is what lets the same operator run on one core or a cluster with bounded
  memory. Adding a stateful operator without a mergeable form caps it at single-node
  — not acceptable.
- **Adaptive intra-query re-optimization** is the moat. Plans re-optimize at
  pipeline breakers using *measured* cardinalities (not just estimates) — the thing
  DuckDB (static) and Spark AQE (stage boundaries only) can't do. Don't regress the
  re-optimization hooks or collapse breakers that feed them.
- **Out-of-core spilling** keeps large queries alive under bounded memory
  (aggregation, join, sort all spill). New stateful operators should have a spill
  story; integration tests under memory pressure (`test_spilling.py`) must stay
  green.
- **Data plane bypasses the Ray object store.** Bulk Arrow batches move via
  `bc-transport` (Arrow Flight) with credit-based backpressure. Never route bulk
  data through Ray objects — that reintroduces the serialization/OOM overhead the
  design removes.

## Scale expectations

- **Small queries**: sub-second, low fixed overhead. Don't add per-query setup cost
  (spinning thread pools, compiling unconditionally, allocating large buffers) that
  hurts the small case to help the large one. Make it adaptive.
- **Large queries / PB**: bounded per-node memory (mergeable + spill), network-aware
  shuffle, work that scales with cores and nodes. Avoid anything `O(rows)` in the
  Python control plane — that's a hot-path tuple touch (see
  `.claude/rules/architecture.md`).

## Positioning cheat-sheet (keep claims honest)

| System    | Their limit                         | Batcher's answer                          |
|-----------|-------------------------------------|-------------------------------------------|
| DuckDB    | static optimization, single-node    | intra-query adaptive re-opt; distributed  |
| Spark AQE | adapts only at stage boundaries     | continuous re-opt at pipeline breakers    |
| Ray Data  | no real optimizer                   | Kyber + Carbonite + learned metadata      |
| Polars    | single-backend, single-node         | mergeable algebra → distributed; adaptive |

Use these as the bar to clear, and verify the claim with `benchmarks/` before
asserting it. Don't ship a positioning statement the benchmark doesn't support.

## Gate before "done"

For any perf-relevant change: correctness gate first (`.claude/rules/testing.md`),
then `python benchmarks/run.py` with no regression vs the prior ratios.
