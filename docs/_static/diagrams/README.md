# Diagrams

The documentation diagrams are **SVG**, and the SVG is what the pages embed. It stays
sharp at any zoom, it is a fraction of the size of the equivalent raster, and it can
restyle itself for the reader's theme, which a PNG cannot.

## Theme awareness

Every diagram carries a `prefers-color-scheme: dark` block inside its own `<style>`
element. CSS overrides SVG presentation attributes, so the authored light palette stays
the default and dark mode remaps the surfaces and text in place.

This matters more than it sounds. Before it, `custom.css` forced a white mat behind
every diagram so the light PNGs would not sit oddly on the page, which meant each one
became a glaring white slab in dark mode. The mat now uses the theme's own surface color,
and only non-SVG images still get a white background.

## Authoring a new diagram

Two ways in, both fine:

- **Write the SVG by hand.** Match the visual language: blue `#2563eb` for the primary
  subject, amber `#f59e0b` for the accent or highlighted path, slate `#1e293b` for text,
  white cards with a soft drop shadow, and a label on every arrow. Copy the `<style>`
  block from an existing file so the diagram is theme-aware.
- **Build it from `tools/diagrams/_authoring.py`.** That module holds the same language as
  functions (`band`, `card`, `arrow`, `curve`, `label`, `note`) with the theme-aware defs
  already wired. Write a small script beside it in `tools/diagrams/`, run it, and commit
  both the script and the SVG it emits here. `transfer_modes.py`, `adaptive_loop.py`, and
  `inference_stages.py` are the examples to copy.

**The scripts live in `tools/diagrams/`, not here, and that separation is load-bearing.**
Sphinx copies `html_static_path` wholesale, so any `.py` in this directory is published as
a website asset. `exclude_patterns` does not filter static files, so moving them out is the
only thing that works.

Design constraints that matter: label every arrow, because an unlabeled one only says
"related"; cap a diagram at roughly seven boxes and split into two zoom levels past that;
never encode meaning in color alone; and keep every label short enough to survive being
scaled to a phone column. A diagram must agree with the prose around it, and it must
never be the only carrier of a fact.

Diagrams that assert something the code decides should name their source module in the
generating script's docstring, so the two can be kept in step. `adaptive_loop.py` and
`transfer_modes.py` both do this.

## Raster output

`tools/diagrams/render.py` rasterizes every `*.svg` to a retina PNG with `rsvg-convert` (librsvg). The
docs no longer need those PNGs, so this is only for contexts that cannot take SVG, such
as a slide deck or a PDF export. It is not part of the docs build, and `rsvg-convert` is
not required to work on the documentation.

## Charts

`gpu_utilization`, `stage_overlap`, and `tpch_sf10` are charts rather than diagrams, so they
answer to `.claude/rules/documentation.md`'s charts rule as well: every figure traces to a
committed benchmark, named in the generating script's docstring, and the axes carry their
units. All were color-checked against each surface rather than eyeballed. The light "before"
bar in `stage_overlap` failed the 3:1 contrast floor at its first value and was re-stepped.

`tpch_sf10` plots suite *ratios* around a 1.0x parity line rather than absolute totals,
because only two of the four engines in that run have a recorded total. Back-solving the
other two and presenting them as measured is exactly what the contract forbids, so the
script says so in its docstring and the caption repeats it on the page.

## Diagrams that carry a disciplined claim

Two figures state something the project has deliberately narrowed, and both name their
source so the picture cannot drift away from the audit:

- `adaptive_loop` and `adaptive_positioning` are drawn to the wording
  `docs/architecture/internals/competitive_architecture.md` sanctions, **not** to the retired claim that
  Batcher re-optimizes more finely than Spark AQE. It does not: the within-query loop is
  stage-boundary adaptation at the same granularity, gated off below the thresholds in
  `python/batcher/api/adaptive/gating.py`. The differentiator those diagrams draw is that
  the loop runs single-node as well as distributed, and that what it measured survives into
  the next run. `adaptive_positioning` is a capability matrix rather than a timeline for
  precisely this reason: a timeline invites the "more marks means better" reading the
  retired claim was made of.
- `execution_tiers` draws the JIT fallback edge explicitly, because "falls back rather than
  diverges" is the load-bearing half of the parity contract and the half a reader forgets.

Current diagrams: `hub`, `lifecycle`, `mergeable`, `two_planes`, `layer_stack`,
`data_flow`, `pipeline_breakers`, `carbonite_loop`, `adaptive_loop`, `transfer_modes`,
`inference_stages`. Charts: `gpu_utilization`, `stage_overlap`.

## Where each diagram belongs

Every diagram is drawn by the script of the same name in `tools/diagrams/`, and the
script's docstring names the source files it was read from. A diagram not yet embedded
on its page is not finished, so this table is the checklist as well as the index.

### Operator internals

| Diagram | Page it belongs on |
| --- | --- |
| `morsel_scheduling.svg` | `docs/architecture/deep-dives/operators/morsel-parallelism.md` |
| `hash_join_spill.svg` | `docs/architecture/deep-dives/operators/join-algorithms.md` |
| `join_strategy_choice.svg` | `docs/architecture/deep-dives/operators/join-algorithms.md` |
| `sort_run_merge.svg` | `docs/architecture/deep-dives/operators/sort-internals.md` |
| `topn_heap.svg` | `docs/architecture/deep-dives/operators/sort-internals.md` |
| `agg_spill_states.svg` | `docs/architecture/deep-dives/operators/aggregation-internals.md` |
| `window_frame_eval.svg` | `docs/architecture/deep-dives/operators/window-internals.md` |
| `mergeable_algebra.svg` | `docs/architecture/deep-dives/operators/mergeable-algebra.md` |

### Query and plan

| Diagram | Page it belongs on |
| --- | --- |
| `plan_lowering.svg` | `docs/architecture/deep-dives/query/plan-ir.md` |
| `ir_wire_contract.svg` | `docs/architecture/deep-dives/query/plan-ir.md` |
| `query_lifecycle.svg` | `docs/architecture/deep-dives/query/query-lifecycle.md` |
| `jit_fallback.svg` | `docs/architecture/deep-dives/query/jit-compilation.md` |
| `expr_eval_nulls.svg` | `docs/architecture/deep-dives/query/expression-evaluation.md` |
| `physical_properties.svg` | `docs/architecture/deep-dives/query/physical-properties.md` |
| `pushdown_before_after.svg` | `docs/architecture/deep-dives/query/plan-ir.md` |
| `join_order_search.svg` | `docs/architecture/deep-dives/adaptive/cost-model.md` |

### Adaptive and learning

| Diagram | Page it belongs on |
| --- | --- |
| `reopt_at_breaker.svg` | `docs/architecture/deep-dives/adaptive/adaptive-reoptimization.md` |
| `adaptive_gating.svg` | `docs/architecture/deep-dives/adaptive/adaptive-reoptimization.md` |
| `cardinality_sketches.svg` | `docs/architecture/deep-dives/adaptive/cardinality-estimation.md` |
| `cost_model_inputs.svg` | `docs/architecture/deep-dives/adaptive/cost-model.md` |
| `cross_run_learning.svg` | `docs/architecture/deep-dives/adaptive/learned-metadata.md` |
| `bandit_tuning.svg` | `docs/architecture/deep-dives/adaptive/learned-metadata.md` |
| `hardware_awareness.svg` | `docs/architecture/deep-dives/adaptive/hardware-awareness.md` |

### Memory and spill

| Diagram | Page it belongs on |
| --- | --- |
| `buffer_pool_zones.svg` | `docs/architecture/deep-dives/memory/buffer-pool.md` |
| `memory_envelope.svg` | `docs/architecture/deep-dives/memory/buffer-pool.md` |
| `spill_ladder.svg` | `docs/architecture/deep-dives/memory/spilling.md` |
| `arrow_memory_layout.svg` | `docs/architecture/deep-dives/memory/arrow-memory.md` |
| `spill_artifacts.svg` | `docs/architecture/deep-dives/memory/on-disk-artifacts.md` |
| `tensor_column_layout.svg` | `docs/architecture/deep-dives/memory/tensor-columns.md` |
| `credit_backpressure.svg` | `docs/architecture/deep-dives/distribution/credit-flow-control.md` |

### Distribution and the device tier

| Diagram | Page it belongs on |
| --- | --- |
| `distributed_stages.svg` | `docs/architecture/deep-dives/distribution/distributed-scheduling.md` |
| `shuffle_dataflow.svg` | `docs/architecture/deep-dives/distribution/shuffle-flight.md` |
| `single_node_equals_distributed.svg` | `docs/architecture/deep-dives/distribution/index.md` |
| `partition_aware_planning.svg` | `docs/architecture/deep-dives/distribution/partition-aware-planning.md` |
| `gpu_tier_decision.svg` | `docs/architecture/deep-dives/distribution/gpu-execution.md` |
| `gpu_shadow_verify.svg` | `docs/architecture/deep-dives/distribution/gpu-execution.md` |
| `gpu_fabric_topology.svg` | `docs/architecture/deep-dives/distribution/gpu-fabric.md` |
| `fault_recovery.svg` | `docs/architecture/fault-tolerance.md` |

### Streaming and the lakehouse

| Diagram | Page it belongs on |
| --- | --- |
| `streaming_microbatch.svg` | `docs/user-guide/moving-data/streaming.md` |
| `watermark_late_data.svg` | `docs/user-guide/moving-data/streaming-stateful.md` |
| `window_types.svg` | `docs/user-guide/moving-data/streaming-stateful.md` |
| `state_store.svg` | `docs/user-guide/moving-data/streaming-stateful.md` |
| `exactly_once.svg` | `docs/user-guide/moving-data/streaming-stateful.md` |
| `output_modes.svg` | `docs/api/operations/streaming.md` |
| `delta_commit_log.svg` | `docs/user-guide/moving-data/lakehouse.md` |
| `merge_into_branches.svg` | `docs/user-guide/moving-data/lakehouse.md` |
| `scd_type2_timeline.svg` | `docs/user-guide/moving-data/lakehouse.md` |

### Models, governance and quality

| Diagram | Page it belongs on |
| --- | --- |
| `fit_transform_leakage.svg` | `docs/ml/preparing/preprocessors/index.md` |
| `inference_actor_pool.svg` | `docs/ml/inference/batch-scoring.md` |
| `vector_index_search.svg` | `docs/ml/retrieval/vector-search.md` |
| `data_loader_shards.svg` | `docs/ml/training/data-loaders.md` |
| `metrics_as_aggregates.svg` | `docs/ml/evaluation/evaluation.md` |
| `policy_plan_rewrite.svg` | `docs/user-guide/trust/governance.md` |
| `column_lineage.svg` | `docs/user-guide/trust/governance.md` |
| `dq_actions.svg` | `docs/user-guide/trust/data-quality.md` |
