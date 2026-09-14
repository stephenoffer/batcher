# SageMaker and SkyPilot: orchestrators, not engines

**Status:** audit, 2026-07-26. Internal working document, excluded from the published site.
The hub is `platform_parity_scorecard.md`.

## Why these two share one file

Because a standalone ledger for either would be about 70% "N/A — not an engine", and that is
the single most important fact about both.

**Neither ships a data-processing engine.** SageMaker orchestrates containers on managed
infrastructure. SkyPilot provisions and manages jobs across clouds. The engine in both cases is
whatever you run *inside* — often Spark, pandas, or Ray, all of which the existing benchmarks
already cover.

So "Batcher is faster than SageMaker" is a category error, and it is listed as a claim not to
make in the hub. What follows is the comparison that *is* meaningful: the one axis these
platforms and Batcher genuinely share.

## The shared axis: elasticity and spot

This is where the comparison has content. All three care about a machine going away mid-job.

### What Batcher has

**Proactive spot-preemption migration** (`carbonite/resilience/preemption.py`). When a worker
signals it is being reclaimed, work is migrated at a **stage boundary** rather than being lost
and recomputed. `competitive_architecture.md` records that Spark has no out-of-the-box
equivalent. ⚠️ No primary source is attached for SageMaker's or SkyPilot's equivalent.

Around it:

- **Lineage recompute** when a worker dies without warning
  (`dist/executors/ray_runtime/reduce.py`).
- **Shuffle replication**, which avoids the recompute entirely — but **only on the flat
  aggregate reduce**. `replicate_shuffle_output` has one caller and hardcodes `stage=0, epoch=0`,
  so the combiner tree, join, sort, and window all fall back to full lineage recompute. The
  combiner tree is the path a *large* cluster takes, which is where node loss is likeliest.
  Slow-but-correct, not wrong. Phase 1E in the plan.
- **Recovery is now observable**: `RECOVERY` events carrying `worker_lost`, `recompute`,
  `straggler_backup`, `backup_won`, `preempt_migrate`, `replica_retired`, `give_up`
  (`_internal/events.py`, `tests/unit/test_recovery_events.py`). Before this pass, none of the
  fault-tolerance machinery emitted anything at all, so an operator could not tell a recovering
  job from a slow one.

### What Batcher concedes

- **Task granularity is roughly a node.** Batcher migrates at stage boundaries between Ray
  tasks. A platform whose unit is the VM can do things Batcher cannot, such as re-bidding a spot
  pool or moving a job to a different region mid-run.
- **No provisioning.** Batcher does not create machines. SkyPilot's entire product is choosing
  the cheapest instance across clouds and moving a job when it is reclaimed. There is no overlap
  to compete on.
- **No cross-cloud anything.** Batcher reads object stores; it does not manage clouds.
- **`UNMEASURED` recovery cost.** Every recovery claim above is mechanism-only. The resilience
  matrix has no timings because Ray task execution does not work in this sandbox: `ray.init`
  succeeds and a bare `@ray.remote def add(a, b)` times out at 60 s. Run that four-line
  reproducer before diagnosing any distributed failure here: it separates "the engine is
  wrong" from "Ray is not running tasks".

## The composition that actually makes sense

These are not competitors, and the useful framing for a reader deciding what to build is that
they **compose**:

| Layer | Product |
|---|---|
| Provisioning, spot arbitrage, cross-cloud | SkyPilot |
| Managed training/inference orchestration | SageMaker |
| Cluster scheduling | Ray / Anyscale |
| **Data-processing engine** | **Batcher** |

Batcher's charter is the bottom row. A SkyPilot-provisioned cluster running Ray with Batcher as
the engine is a coherent stack, not a contest. The competitive question against these two is
therefore not "is Batcher faster" but "does Batcher run well *inside* them" — and the honest
answer is `UNMEASURED`, because neither integration has been exercised here.

## SageMaker specifically

| Dimension | Note |
|---|---|
| Engine | **None of its own.** Usually Spark, pandas, or Ray inside a container. |
| Fault tolerance | Container/job restart. A coarser unit than operator lineage; not comparable. |
| Data processing | SageMaker Processing runs *your* code. Batcher could be that code. |
| Feature store, model registry, endpoints | Product surface. Batcher has an ML front-end (`ml/`) but is not a platform. |
| Enterprise controls | Managed IAM, VPC isolation, KMS. Batcher's boundary is the process. |

The one place a real comparison exists: **batch inference throughput**, where the thing being
compared is Batcher's engine against whatever engine a SageMaker Processing job runs. That is
covered by the existing Ray Data and Spark benchmarks, not by anything SageMaker-specific.

## SkyPilot specifically

| Dimension | Note |
|---|---|
| Engine | **None.** |
| What it is | Cross-cloud provisioning, spot management, job queueing. |
| Overlap with Batcher | Spot handling only, and at a different layer: SkyPilot moves the *machine*, Batcher migrates the *work*. |
| Enterprise controls | Inherits the cloud's. |

## Claims not to make

1. "Batcher is faster than SageMaker / SkyPilot." Category error; neither ships an engine.
2. Any recovery-time comparison. Batcher's own recovery timings are `UNMEASURED`.
3. That Batcher's preemption handling is better than a platform's without a primary source for
   what that platform does — the ⚠️ above is not a source.
4. That Batcher replaces either. It is a layer in the same stack.

## What would move this file

1. Run Batcher under SkyPilot on real spot instances and record what happens on reclamation.
   That is the only measurement on this page that would mean anything.
2. Primary sources for SageMaker's and SkyPilot's preemption behavior.
3. Finish Phase 1E so preemption on a large cluster refetches a replica instead of recomputing
   a stage.

## See also

- `platform_parity_scorecard.md` — the hub, including the layer-mapping table.
- `anyscale_parity.md` — the cluster-scheduling layer, which these two sit above.
- `competitive_architecture.md` — the engine-vs-engine authority.
