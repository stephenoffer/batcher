# Rule: Performance & Competitive Positioning

Batcher exists to win on performance across an unusually wide range: **sub-second
small queries to PB-scale**, **single-node to distributed**, **batch and
streaming**. Every performance-relevant change is measured, and measured against
the systems we claim to beat.

## The competitive mandate

We benchmark against DuckDB, Polars, and Spark — not against our own
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
  (stage-boundary AQE).

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
- **The learning loop is the moat — not the within-query loop.** Plans re-optimize at
  pipeline breakers using *measured* cardinalities, which DuckDB (static) does not do at
  all; but that is **the same mechanism and granularity as Spark AQE**, and calling it
  finer is a claim the code does not support. Two things are genuinely differentiated,
  and they are what to protect: the loop runs **single-node**, where AQE needs shuffle
  stages, and what it measures **outlives the query** — sketches, calibrated costs and a
  bandit in `kyber/learning.py` + `kyber/learned_tuning/`, so the same query gets faster
  across runs. Don't regress the re-optimization hooks, collapse the breakers that feed
  them, or break the path that writes measurements back to the `MetadataHub`.
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
| DuckDB    | static optimization, single-node    | re-plans at breakers on measured sizes; distributed |
| Spark AQE | cluster-only, keeps nothing per run | the same grain, single-node too, and learned across runs |
| Polars    | single-backend, single-node         | mergeable algebra → distributed; adaptive |

Note what the middle row does **not** say. Batcher does not re-plan at a finer grain
than AQE, and the within-query loop is off below a size floor
(`api/adaptive/gating.py`), so on most queries it does not run at all. Claim the two
things that are true — single-node availability, and cross-run learning — and nothing
past them.

Use these as the bar to clear, and verify the claim with `benchmarks/` before
asserting it. Don't ship a positioning statement the benchmark doesn't support.
`docs/architecture/internals/competitive_architecture.md` is the code-checked scorecard
and outranks this table; read it before writing any competitive claim.

## Gate before "done"

The canonical gate matrix is in `CLAUDE.md` — run the rows your change touches. Performance
delta: **correctness first**, then `python benchmarks/run.py` with no regression vs the prior
ratios. A fast wrong answer is a bug, so never report a timing on an unverified path.
