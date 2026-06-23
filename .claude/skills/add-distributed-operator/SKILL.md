---
name: add-distributed-operator
description: Recipe to wire an operator through Batcher's distributed path — the mergeable map/shuffle/reduce primitives, Arrow Flight transport, and Carbonite credit-based flow control — so a multi-node result is identical to single-node. Invoke when making a stateful operator scale across Ray workers or touching the shuffle/transport path.
---

# Add a distributed operator

Distribution in Batcher is a **scheduling** concern layered over the *same*
mergeable algebra the single-node engine uses — never a second set of semantics.
The acceptance bar is exact equivalence: a multi-node run must produce the identical
result to a single-node run. Prerequisite: the operator already works single-node
and is mergeable (do the `add-relational-operator` skill first).

Read `.claude/rules/rust-engine.md` (mergeable algebra), `.claude/rules/architecture.md`
(Ray is scheduling only; data plane bypasses the object store), and
`.claude/rules/performance.md`.

## The model

A distributed stateful operator is map → shuffle → reduce over the mergeable
primitives:

- **Map** — each worker runs `bc_interp::dist::partial_aggregate` (or the
  operator's `partial`) on its partition → partial-state batch.
- **Shuffle** — `bc_interp::dist::partition_batches` hash-partitions by key into one
  bucket per reducer; buckets move between nodes via `bc-transport` (Arrow Flight).
- **Reduce** — each reducer runs `bc_interp::dist::combine_finalize` on the partials
  routed to it → output rows.

The invariant: `combine_finalize(partition(partial(pₖ)))` over all partitions ==
the single-node result. If your operator can't be expressed this way, it isn't ready
to distribute — fix the mergeable form in `bc-runtime` first.

## Steps

1. **Confirm the mergeable primitive** exists in `bc-runtime` and its single-node
   invariant test is green. The distributed path adds no new operator math; it only
   surfaces these pieces at orchestrator granularity in
   `crates/bc-interp/src/dist.rs`. Extend `dist.rs` only if the operator needs a
   map/reduce shape not yet exposed (e.g. join's build/probe partials).

2. **Expose the primitives across FFI** in `crates/bc-py/src/lib.rs` if not already
   (`partial_aggregate`, `partition_batches`, `combine_finalize`, and the Flight
   server/fetch functions). Keep `bc-py` thin and zero-copy.

3. **Orchestrate in Python** under `python/batcher/dist/` (`executor.py`,
   `flight_shuffle.py`, `flight_aggregate.py`, `spill.py`). Use Ray to schedule the
   map and reduce tasks/actors and to pass *control-plane* metadata (plan ids,
   tickets, schemas) — **never** to move bulk Arrow batches. Bulk data goes through
   Flight.

4. **Move data over Arrow Flight** (`bc-transport`). One Flight server per node hosts
   shuffle output partitions; reducers `fetch`/DoExchange from each upstream. Tickets
   encode the shuffle coordinate (`p{plan}/s{stage}/{src}/{dst}`). Don't round-trip
   through the Ray object store or an external store.

5. **Respect Carbonite credit-based flow control.** The Flight exchange is
   credit-gated: 1 credit = 1 RecordBatch slot; the producer blocks at 0 credits and
   the consumer grants credits as it drains. Honor this backpressure so a fast
   producer can't OOM a slow consumer — the producer must never buffer more than
   `credits` batches ahead. Wire spill (`dist/spill.py`) for partitions that exceed
   the memory envelope.

## Tests — the hard gate

- **Equivalence test** (`tests/integration/`, alongside `test_distributed.py`,
  `test_flight_shuffle.py`): the operator's distributed result == its single-node
  result, across several partition counts (1, 2, N) and including the empty and
  single-partition cases. This is the acceptance criterion.
- **Spilling** (`test_spilling.py`): the operator stays correct and within memory
  under pressure (partitions spill and merge back).
- **Flight transport**: shuffle round-trip and credit/backpressure behavior under a
  slow consumer.

## Done

Distributed == single-node proven across partition counts, spilling + Flight tests
green, no bulk data through Ray, credits honored. Then `/run-quality-gate`, and for
scale-relevant changes reason about per-node memory and shuffle cost vs Spark/Ray
Data (`.claude/rules/performance.md`).
