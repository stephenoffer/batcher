# How much do batch and streaming actually share?

A code audit, 2026-09-11, of the claim `CLAUDE.md` makes in one line -- "batch is the bounded
special case of streaming over Arrow batches" -- and of the rule that follows from it: *keep
batch and micro-batch paths on the same operator semantics*.

The question is not whether the two modes produce the same answers. It is whether they do so
because they run the same code, or because two implementations currently agree. The second
is the failure mode worth auditing for: it passes every test until one side is changed.

**This is a reading of the code, not a measurement.** Where a divergence is claimed below, the
code that enforces or documents it is cited.

## What is shared, with the evidence

**The operator semantics are shared, and the streaming module is a scheduler over them.**
`bc-interp/src/stream/` contains no relational kernel of its own: it calls `ops::` for the
work (`folds.rs` alone references it 24 times) and its own files are pipelines, breakers,
fanout and metering. A pipeline operator is "a lazy adapter over its child's stream"; the
transform it applies is the batch one.

**The mergeable primitives are the streaming state.** A streaming aggregate is
`partial -> combine -> finalize` over the same Rust kernels the batch aggregate uses, with the
running state held as an Arrow `RecordBatch` -- which is also what the engine snapshots for
checkpoint recovery (`core/streaming/folds/running.py`). There is no second aggregate. This
is invariant #7 doing the work it was written for: the form that makes one core, many cores
and many machines the same operator also makes *time* the same operator.

**The JIT is reached from the streaming path.** `stream/folds.rs` threads an
`ops::AggJit` through the fold, so a streaming aggregate compiles its expressions exactly as a
batch one does rather than falling back to the interpreter.

**The classification rules are single-sourced, deliberately.** Two in particular:

* `is_partition_independent` owns the per-node rule and `is_streamable` is its recursive
  whole-tree form. Its docstring states the reason outright -- *"so the streaming and
  distributed paths cannot drift apart on it"* -- and `dist/` does call it
  (`flight_broadcast.py`, `executors/plan_analysis.py`).
* `streaming_fold_target` answers *"is this operator `partial -> combine -> finalize` over a
  breaker-free input?"* for every path that runs a stateful stream: the single-node
  processor, the distributed streaming runner, and the writer.

**Type coercion is shared even where scheduling is not.** `stream/union_all.rs` is a genuine
streaming `UNION ALL` -- it chains branch streams instead of concatenating them -- but it
settles column types through the same `coerce_union_branches` the batch path uses. The
difference is *when* the coercion is settled (one peeked morsel per branch, rather than every
batch of every branch), not what it decides.

## Where they differ, and whether the difference is earned

| Difference | Earned? |
|---|---|
| Breakers materialize (`Aggregate`, `Sort`, `Distinct`, `Window`, `Sample`, `AsofJoin`, `UNION DISTINCT`, a hash join's build) | **Yes** -- their semantics require the whole input, and they are the points the adaptive layer measures and re-plans at. Streaming the linear runs *between* them is the whole point. |
| `is_streamable` admits `MapBatches`; the distributed classification does not | **Yes, and stated.** A batch UDF cannot observe how its input was split, so streaming it per batch is sound; the distributed path excludes it because it schedules UDFs through its own operator (GPU placement, actor pools), not because the operator is stateful. |
| The streaming-query engine never re-optimizes (Kyber ran once, at `start()`) | **Yes** -- a long-running query optimizes against a plan, not against each micro-batch, and re-optimizing per epoch would charge every epoch the optimizer's latency. |
| Watermarks, output modes and triggers exist only in streaming | **Yes** -- they are answers to questions a bounded relation does not pose. |

## The one divergence that is a gap

**Distributed streaming does not implement event-time watermarks.** The distributed runner has
no window eviction, no late-row drop and no append output mode, so a watermarked aggregation
would degrade to an unbounded complete-mode aggregate that re-emits the whole running result
every epoch and grows state forever.

What makes this worth recording rather than filing is how it was found: the same query,
single-node against distributed, *"produced different results with no error and no warning"*.
It now raises at `api/io_namespace/writer.py` with the reason named. So the current state is
correct-by-refusal rather than correct-by-implementation, which is the right order to fix it
in, and the refusal is the thing to preserve if anyone implements the distributed half.

This is the streaming mirror of the gap `single_node_scaleout_audit.md` records for
common-subplan reuse: an optimization or a semantic that exists on one axis and not the
other. Both are the same class of defect, and neither is visible from a passing test suite.

## What this audit did not check

The **device tier** (`core/gpu_plan/`) is a translator rather than a consumer of the shared
`Expr`, for the reason `.claude/rules/device-tier.md` gives, so "shared with batch" means
something different there and is audited by that rule's own contract instead.

It also did not verify the **window** and **top-N** streaming paths at the same depth as the
aggregate. Both appear to be schedulers over `bc-runtime` kernels (`window/`, `topn.rs`), and
both are listed as breakers, but the aggregate is the one traced end to end here.
