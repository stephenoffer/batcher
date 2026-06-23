---
name: add-kyber-optimizer-pass
description: Recipe to add an optimization to the Kyber optimizer as a new Rule (predicate/projection pushdown, fusion, join ordering, pruning, etc.), using sketch-based cardinality and cost, while respecting layer boundaries and proving the rewrite preserves results. Invoke when adding or changing an optimization rule, cost model, or cardinality estimate.
---

# Add a Kyber optimizer rule

Kyber runs its optimizations as **phased `Rule`s** over a `LogicalPlan`. A rule is a
pure `plan → plan` (or `node → node`) function that may read shared analysis
(cardinality, cost, learned metadata) and record decisions. Rules are grouped into
**phases** that run in order; rewrite phases iterate to a fixpoint. Adding an
optimization means writing a rule and registering it — the driver
(`kyber/optimizer.py`) and phase pipeline are unchanged.

Read `.claude/rules/architecture.md` (Kyber's lane), `.claude/rules/maintainability.md`
(structure limits — group rules by family, never one file per rule), and
`.claude/rules/testing.md` first.

- **Framework:** `kyber/rule.py` (`Rule`, `Phase`, `RuleCategory`, `node_rule`,
  `plan_rule`), `kyber/registry.py` (`@rule` decorator, `DEFAULT_REGISTRY`).
- **Rule families (add yours to the matching module):**
  `kyber/rules/{normalize,pushdown,projections,fusion,selection,join_order,algebraic}.py`.
- **Phases, in order:** `NORMALIZE → REWRITE → PUSHDOWN → JOIN_REORDER → FUSION →
  SELECTION → ENFORCE`. Rewrite phases run to a fixpoint; `JOIN_REORDER`/`SELECTION`/
  `ENFORCE` run once.

## The invariants Kyber rules must honor

- **Decide, never execute.** A rule MUST NOT run the engine or collect runtime
  metadata — that's Core's job. Rules *consume* metadata and *decide*.
- **Semantics-preserving.** The optimized plan MUST produce the identical result to
  the unoptimized one. Optimization changes the plan, never the answer.
- **Layer-clean.** `kyber` may import `plan`, `metadata`, `config`, `_internal`
  only — never `carbonite`, `core`, or `api`. Estimates flow in; decisions flow out.
- **Pure.** A rule reads the `OptimizerContext` (config, sources, hub, estimator) and
  returns a new plan; side effects are limited to recording notes on the context for
  explain/telemetry. Nodes are frozen dataclasses — build new ones, never mutate.

## Steps

1. **Pick the rule shape.**
   - **Node-local rewrite (the default):** write `f(node, ctx) -> node | None`
     (return `None` for "no change") and decorate it with
     `@rule(name=..., phase=Phase.X, matches=(NodeType, ...))`. The driver supplies
     bottom-up traversal and fixpoint; `matches` is both the index key and the
     per-node guard, and it's what keeps optimization sub-linear as rules grow.
     Examples: `kyber/rules/algebraic.py`, `kyber/rules/pushdown.py::push_filter_*`.
   - **Whole-plan / cost-based transform:** write `f(plan, ctx) -> plan` and register
     it with `registry.add(plan_rule(name, phase, f, matches=(...), category=...))`
     in the module body. Use for holistic rewrites and cost-based search. Examples:
     `kyber/rules/join_order.py` (join reordering), `kyber/rules/selection.py`
     (build-side). Make sure the module is imported from `kyber/rules/__init__.py`.

2. **Use sketch-based estimates for decisions.** If the rewrite depends on sizes /
   selectivities / distinct counts, get them from the shared `CardinalityEstimator`
   (`kyber/cardinality.py`), backed by `bc-sketches` and learned stats from the
   `MetadataHub`. Don't compute statistics by scanning data in Python — wrong layer
   and a hot-path tuple touch. Carry estimate provenance/confidence; respect it.

3. **Cost it if choosing between alternatives.** Use / extend `kyber/cost.py` rather
   than hard-coding magic numbers. Keep the cost model in `kyber`.

4. **Choose the right phase** so the rule sees the plan shape it expects (e.g. expr
   normalization in `NORMALIZE`, predicate/projection pushdown in `PUSHDOWN`, cost-
   based ordering in `JOIN_REORDER`, physical/algorithm choice in `SELECTION`).
   Within a phase, rewrite rules iterate to a fixpoint, so ordering between rules in
   the same phase rarely matters — but the fixpoint must converge (make the rule
   idempotent: it should not re-fire on its own output).

5. **Record cost-based decisions** on `OptimizerContext.notes` (the way
   `selection.py` records build-side choices) so they appear in explain/telemetry.

## Tests — the hard gate

- **Plan-shape unit test** (`tests/unit/`, e.g. `test_algebraic_rules.py`,
  `test_join_reorder.py`): assert the rule produces the expected optimized plan
  (compare `.to_ir()` — never `==`, since `Expr.__eq__` builds a comparison
  expression), that it's a **no-op when it shouldn't fire**, and that the full
  `Optimizer().optimize(...)` is **idempotent/deterministic**.
- **Differential test** (`tests/differential/`): a query that triggers the rule must
  still match DuckDB (`assert_same`) — proving the rewrite is semantics-preserving.
- If you changed cardinality/cost, add a unit test pinning the estimate behavior.

## Don't

- Don't make the rule execute, measure, or allocate resources (Core/Carbonite lanes).
- Don't push optimization logic into the operator itself or into `plan` (keep `plan`
  neutral).
- Don't add a flag/knob with no caller, and don't make one file per rule — group by
  family (`.claude/rules/{python-quality,maintainability}.md`).
- Don't rely on Python `==` to detect plan change (use `.to_ir()`); the fixpoint
  driver already does this.

## Done

Rule registered (decorator or `registry.add`), module imported from
`kyber/rules/__init__.py`, plan-shape + differential tests green, `just lint-py`,
`just lint-structure`, and `just lint-layers` green, then `/run-quality-gate`. For
perf-relevant rewrites, benchmark to show the win (`.claude/rules/performance.md`).
