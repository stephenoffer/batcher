# Anyscale parity: competing with Ray Data while running on Ray Core

**Status:** audit, 2026-07-26. Internal working document, excluded from the published site.
The hub is `platform_parity_scorecard.md`; this is the Anyscale column in depth.

## The awkward truth, stated first

**Batcher runs on Ray.** `dist/` schedules its work as Ray tasks and actors, and the optional
`[ray]` extra is how a Batcher job goes multi-node at all. So Batcher competes with **Ray Data**
— a library in the Ray ecosystem — while consuming **Ray Core** underneath it.

That is not a contradiction, but it constrains what can honestly be claimed:

- Batcher inherits Ray Core's scheduling behavior, including its pitfalls. Those are catalogued
  separately in `ray_pitfall_parity.md` and are not restated here.
- A statement of the form "Batcher beats Anyscale" is wrong twice over: Anyscale is a platform,
  not an engine, and Batcher depends on the thing it would be beating.
- The defensible statement is **"Batcher beats Ray Data as a data-processing engine, by a margin
  whose mechanism is structural"**, plus "RayTurbo is closed and unmeasured".

## Where the Ray Data margin comes from

`competitive_architecture.md` records 50-450x against Ray Data. That number is only interesting
because the mechanism is explainable rather than incidental:

**The data plane bypasses the Ray object store entirely.** Only `(addr, ticket)` strings transit
Ray; bulk Arrow moves over Arrow Flight with credit-based flow control whose bound is proven by
an in-flight gauge (`crates/bc-transport/src/store.rs`). Ray Data's object-store spill storms are
not a tuning problem — they follow from routing bulk data through the object store, which is what
Ray Data does by design.

Two smaller structural wins:

- **Pre-aggregated partials, not raw rows.** Aggregate mappers publish `partial` state, so the
  hierarchical combiner tree (`dist/flight_aggregate.py`) and bounded reducer memory are cheap.
  Spark shuffles raw rows; Ray Data has no equivalent algebra.
- **Session-warm inference pools.** The model loads once per *session* and is reused across
  `collect()`s. Ray Data respawns per execution.

And one Batcher has that Ray Train does not: **O(1)-memory global shuffle for training**, a
4-round Feistel permutation with cycle-walking (`ml/permutation.py`) that makes the epoch order a
computed bijection rather than a materialized index list, with mid-epoch resume and
world-size-independent ordering.

## RayTurbo

`UNMEASURED`, and it must stay that way. RayTurbo is Anyscale's proprietary accelerated Ray
runtime; it is closed, licensed, and not runnable here. Its published claims are vendor
marketing until someone runs a comparison.

⚠️ Everything a reader might want to say about RayTurbo's accelerated operators belongs behind a
primary citation this file does not have.

## Where Anyscale-the-platform wins

These are platform properties, and Batcher has no answer to any of them because it is a library:

| Capability | Anyscale | Batcher |
|---|---|---|
| Cluster autoscaling | The product | Composes Ray tasks; does not drive scaling |
| Multi-cloud provisioning | Yes | No |
| Managed control plane, dashboards, job submission | Yes | `bt.start_ui()` is a local dashboard on an unauthenticated port |
| Tenant isolation across users | Cluster/namespace level | Process level (see the hub's blocking gap 1) |
| Elastic resize within a job | ⚠️ believed yes | No |

## Where Batcher wins on the enterprise axes

Drawn from the hub, restricted to what is evidenced:

- **Bounded per-node memory by construction.** Mergeable algebra plus spill, one implementation
  for one core / N cores / N machines (`crates/bc-runtime/`). Ray Data's memory behavior on a
  large shuffle is the thing its users complain about.
- **A real optimizer.** Kyber's rules, cost model, and sketch-backed cardinality
  (`kyber/`). Ray Data has no optimizer worth the name; this is the single clearest
  architectural difference between the two.
- **Portable shuffle routing.** Fixed in this pass — `ahash` selected its backend at compile
  time, so a mixed-instance autoscaled cluster (an *Anyscale-shaped* deployment, which is the
  point) could split a `GROUP BY` group silently. Now xxHash64
  (`crates/bc-arrow/src/hash.rs`, `crates/bc-runtime/tests/shuffle_hash_golden.rs`).
- **Recovery is observable.** `RECOVERY` events on the bus (`_internal/events.py`).

## What to measure next

1. Ray Data head-to-head on a **current** Ray release, on a real cluster, with the corpus and
   fingerprint recorded. The existing margin should be re-confirmed rather than re-quoted.
2. The resilience matrix under Ray actor death at N workers. Blocked here: Ray task execution
   does not work in this sandbox — `ray.init` succeeds, and a bare `@ray.remote def add(a, b)`
   then times out at 60 s.
3. Whether Batcher-on-Ray inherits a scheduling pathology at scale that a single-node run cannot
   show.

## Claims not to make

1. "Batcher beats Anyscale." Category error, and Batcher runs on Ray.
2. Any RayTurbo ratio.
3. Any Ray Data ratio without the Ray version, cluster shape, and corpus attached.

## See also

- `platform_parity_scorecard.md` — the hub.
- `ray_pitfall_parity.md` — the Ray Core behaviors Batcher inherits.
- `competitive_architecture.md` — the authority on the Ray Data performance margin.
