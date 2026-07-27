"""The rule-application engine: phases, fixpoint, and the two levels of fusion.

Kyber's driver runs a phase's rules over a plan until it stops changing. It stays affordable
as the rule set grows to hundreds through three devices, in increasing depth:

  * **pattern indexing** — a rule whose `matches` node types are absent from the plan never
    runs at all;
  * **node-rule fusion** — consecutive node-local rules share one bottom-up plan traversal;
  * **expression-rule fusion** — every rule whose body is a leaf `Expr -> Expr` rewrite shares
    one traversal of a node's *expressions*, instead of each walking the whole tree itself;
  * **a no-op memo** — a rule that did nothing to a node object will do nothing to it again.

The last two are what keep a hundred-odd expression rules from costing a hundred expression
walks per node per fixpoint iteration.
"""

from __future__ import annotations

from batcher._internal.logging import get_logger
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rule import Phase, Rule
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import LogicalPlan
from batcher.plan.visitor import children, transform_up, walk

__all__: list[str] = []


# Confluent rewrite phases iterate to a fixpoint (bounded by `_fixpoint_bound`, which caps
# pathological non-convergence); every other phase makes a single decision and runs once.
_FIXPOINT_PHASES = frozenset({Phase.NORMALIZE, Phase.REWRITE, Phase.PUSHDOWN, Phase.FUSION})

# Headroom above the plan's depth for rules whose convergence isn't purely depth-linear
# (a rewrite that deepens the plan, then pushes through the new level).
_FIXPOINT_DEPTH_SLACK = 4


def _depth(plan: LogicalPlan) -> int:
    """Longest root-to-leaf path in `plan`."""
    stack: list[tuple[LogicalPlan, int]] = [(plan, 1)]
    deepest = 1
    while stack:
        node, d = stack.pop()
        deepest = max(deepest, d)
        stack.extend((c, d + 1) for c in children(node))
    return deepest


def _fixpoint_bound(plan: LogicalPlan, configured: int) -> int:
    """How many iterations a fixpoint phase may take, given the plan it is rewriting.

    The pushdown rules are node-local over a bottom-up traversal, so a predicate descends
    exactly **one level per iteration** — the distance to a fixpoint is the plan's depth, not a
    constant. A fixed cap therefore doesn't bound pathological non-convergence (its purpose);
    it silently truncates *healthy, linear* convergence on any plan deeper than the cap, leaving
    predicates un-pushed and the engine scanning rows it should never have read. TPC-H q8 needs
    9 iterations and the shipped cap is 8, so this fires on the benchmark's own plans.

    Scaling the bound with depth makes the cap do its actual job: a plan that converges early
    still exits early (the loop breaks on structural identity), so this costs nothing for the
    plans that were already fine, while deep plans get the fixpoint they were promised.
    """
    return max(configured, _depth(plan) + _FIXPOINT_DEPTH_SLACK)


def _applicable(rules: list[Rule], present: frozenset[type]) -> list[Rule]:
    """Rules that could fire on a plan containing exactly `present` node types.

    A rule with `matches is None` always applies; otherwise it applies only if its
    matched node types intersect the plan. This is the indexing that keeps per-plan
    cost proportional to the applicable rules, not the total rule count.
    """
    return [r for r in rules if r.matches is None or (r.matches & present)]


def _fingerprint(plan: LogicalPlan) -> object:
    """A structural fingerprint of `plan`, for the fixpoint's "did anything change" test.

    The JSON IR is the fingerprint of choice: it is cheap, already memoized on the
    immutable nodes, and compares by value. But it is only defined for nodes that lower
    to Rust, and the streaming operators (`WatermarkDedup`, `WatermarkStreamJoin`)
    deliberately define no `to_ir()` — they are executed by the Python driver. Calling
    `to_ir()` on a plan containing one raises `NotImplementedError`, which is why *no*
    optimizer entry point could accept a streaming plan and why the streaming path had to
    optimize around those nodes rather than through them.

    The fallback is `repr`, deliberately, and **not** the node itself. `LogicalPlan`s are
    frozen dataclasses, so comparing two of them looks like a structural comparison — but
    it recurses into the `Expr`s they carry, and `Expr.__eq__` *builds an expression*
    rather than comparing (that is what makes `col("a") == 1` a predicate). Taking its
    truth value raises `PlanError: the truth value of an Expr is ambiguous`, so a plan
    equality test does not return False, it explodes. `repr` is structural, total, and
    cannot re-enter operator overloading.

    It is used only on the path where rule application already returned a *new* tree, so
    the common converged case still costs one identity check.

    Args:
        plan: The plan to fingerprint.

    Returns:
        The plan's IR when it has one, else a structural `repr` of the plan.
    """
    try:
        return plan.to_ir()
    except NotImplementedError:
        return repr(plan)


def _run_phase(
    plan: LogicalPlan,
    rules: list[Rule],
    ctx: OptimizerContext,
    max_iterations: int,
    present: frozenset[type] | None = None,
) -> tuple[LogicalPlan, dict | None]:
    """Run one phase's rules, up to `max_iterations` (1 = once, >1 = to fixpoint).

    Fixpoint is detected by **object identity first**: `transform_up` shares structure
    (an untouched subtree keeps its identity), and node rules return their input on a
    no-op, so a phase that changed nothing returns the *same* plan object — an O(1)
    check. Only when identity says "changed" do we fall back to comparing lowered IR
    (`to_ir()`), because a whole-plan rule may rebuild an equal-but-new tree
    unconditionally; the IR comparison (not Python `==`, which `Expr.__eq__` overloads
    to build a comparison expression) confirms a *real* change. So semantics are
    exactly as before, just without serializing the plan every iteration.

    `_present` (the node-type set for rule indexing) is likewise computed once and
    refreshed only after an iteration that actually changed the plan.

    Also returns the lowered IR of the returned plan **when this phase computed it**
    (during fixpoint change-detection), else `None` meaning "the plan is unchanged —
    reuse the caller's last IR". This lets the final lowering reuse the IR the
    fixpoint loop already built instead of serializing the plan a second time.
    """
    if not rules:
        return plan, None
    if present is None:  # standalone callers (tests) don't precompute it
        present = _present(plan)
    # A phase that iterates to a fixpoint may fuse its node rules into one traversal: a
    # rewrite the fused pass skipped (a rule that built a new subtree) is picked up on the
    # next iteration. A once-run phase has no next iteration, so it must not fuse.
    fuse = max_iterations > 1
    # Survives this phase's fixpoint iterations; see `_apply_node_rules`.
    noop: dict[tuple[str, int], LogicalPlan] = {}
    current_ir = None  # lazily computed, only on the identity-says-changed path
    changed = False
    converged = False
    for _ in range(max_iterations):
        updated = _apply_rules(plan, _applicable(rules, present), ctx, fuse=fuse, noop=noop)
        if updated is plan:  # structural sharing → confirmed fixpoint, O(1)
            converged = True
            break
        if current_ir is None:
            current_ir = _fingerprint(plan)
        updated_ir = _fingerprint(updated)
        if updated_ir == current_ir:  # equal-but-new tree (an unconditional rebuilder)
            converged = True
            break
        plan, current_ir, present = updated, updated_ir, _present(updated)
        changed = True
    if fuse and not converged:
        # The iteration cap was hit while the plan was still changing. The plan a query
        # gets then depends on `fixpoint_iterations`, which means some rule in this phase
        # is non-confluent or oscillating. Results stay correct (every rule is
        # semantics-preserving) but plan quality is silently non-reproducible, so say so.
        get_logger("kyber").warning(
            "phase did not reach a fixpoint in %d iterations; plan quality may depend on "
            "`OptimizerConfig.fixpoint_iterations` (a non-confluent rule?)",
            max_iterations,
        )
    # `current_ir` tracks the latest plan's IR, so when the phase changed the plan it
    # is exactly the returned plan's IR; on a no-op phase we computed nothing new.
    return plan, (current_ir if changed else None)


def _present(plan: LogicalPlan) -> frozenset[type]:
    """The set of node types in `plan`, for the per-plan rule pattern-index."""
    return frozenset(type(n) for n in walk(plan))


def _apply_rules(
    plan: LogicalPlan,
    rules: list[Rule],
    ctx: OptimizerContext,
    *,
    fuse: bool,
    noop: dict[tuple[str, int], LogicalPlan] | None = None,
) -> LogicalPlan:
    """Apply a phase's rules in registered order.

    With `fuse`, each maximal run of consecutive node-local rules shares a *single*
    bottom-up traversal instead of one walk per rule. That is cheaper but **not**
    observationally identical, contrary to what this once claimed: `transform_up` has
    already visited a node's children by the time a rule fires on it, so when a rule
    rewrites a node into a *new subtree* the later rules in that run never see the new
    children. A fixpoint phase recovers them on its next iteration, which is why fusing is
    sound there — and only there.

    The once-run phases (SELECTION, ENFORCE) have no next iteration, so a rewrite the fused
    pass skipped would be lost outright. They run unfused: one `transform_up` per rule,
    exactly the sequential semantics. That costs a few extra walks over the handful of
    rules those phases hold, and nothing at all in the hot fixpoint phases.

    Whole-plan rules (join reorder, projection pruning, build-side selection) always run
    individually; the registered order is preserved in every case.
    """
    out = plan
    i, n = 0, len(rules)
    while i < n:
        if rules[i].node_fn is None:
            out = rules[i].apply(out, ctx)
            i += 1
            continue
        j = i
        while j < n and rules[j].node_fn is not None:
            j += 1
        run = rules[i:j]
        if fuse:
            out = _apply_node_rules(out, run, ctx, noop, fuse_exprs=True)
        else:
            for r in run:
                out = _apply_node_rules(out, [r], ctx, noop)
        i = j
    return out


def _apply_node_rules(
    plan: LogicalPlan,
    node_rules: list[Rule],
    ctx: OptimizerContext,
    noop: dict[tuple[str, int], LogicalPlan] | None = None,
    fuse_exprs: bool = False,
) -> LogicalPlan:
    """One bottom-up pass applying every node-local rule at each node, in order.

    **A rule that did nothing to a node will do nothing to it again.** Rules are pure in
    `(node, ctx)`, `ctx` is fixed for the run, and plan nodes are immutable — so a node's
    identity determines its content. `noop` records "this rule was a no-op on this exact node
    object" and skips it on later fixpoint iterations; structural sharing keeps untouched
    nodes identical, so only the subtree that actually changed is re-examined. It holds a
    strong reference to the node it keyed on, so a recycled `id()` cannot produce a stale
    hit."""

    # Which leaf rewrites apply to a node depends only on the node's *type*, and the rule
    # list is fixed for this pass — but the filter ran per node, per fixpoint iteration,
    # scanning every one of the several hundred registered rules to rebuild the same handful
    # of answers. There are a dozen plan node types, so one entry each covers the whole run.
    leaves_by_type: dict[type, list] = {}

    def visit(node: LogicalPlan) -> LogicalPlan:
        # Leaf `Expr -> Expr` rules share ONE traversal of the node's expressions rather than
        # each walking the whole tree. Sound only where node rules already fuse (a fixpoint
        # phase, which recovers next iteration anything a fused pass stepped over).
        if fuse_exprs:
            node_type = type(node)
            leaves = leaves_by_type.get(node_type)
            if leaves is None:
                leaves = [
                    r.expr_fn
                    for r in node_rules
                    if r.expr_fn is not None and (r.matches is None or node_type in r.matches)
                ]
                leaves_by_type[node_type] = leaves
            if leaves:
                node = _apply_expr_leaves(node, leaves)
        for r in node_rules:
            if fuse_exprs and r.expr_fn is not None:
                continue  # already applied, in the shared traversal above
            # Re-check against the *current* node: an earlier rule may have replaced it with
            # a different type (`eliminate_identity_project` returns its input).
            if r.matches is not None and type(node) not in r.matches:
                continue
            key = (r.name, id(node))  # a rule's *name* is its unique, position-independent id
            if noop is not None:
                seen = noop.get(key)
                if seen is not None and seen is node:
                    continue  # this rule already proved itself a no-op on this exact node
            rewritten = r.node_fn(node, ctx)
            # `None` is the documented "no change" signal, but a rule may equivalently hand
            # back the very node it was given. That is definitionally a no-op too, and without
            # memoizing it the rule is re-run against the same node on every fixpoint
            # iteration, forever — the exact cost this memo exists to remove.
            if rewritten is None or rewritten is node:
                if noop is not None:
                    noop[key] = node  # strong ref: pins the id against reuse
                continue
            node = rewritten
        return node

    return transform_up(plan, visit)


def _apply_expr_leaves(node: LogicalPlan, leaves: list) -> LogicalPlan:
    """Apply every leaf `Expr -> Expr` rewrite to `node`'s expressions in ONE traversal.

    Each leaf is offered every expression node, bottom-up, in registered order. This is the
    expression-level analogue of the node-rule fusion the driver already does, and it turns
    "one full expression walk per rule" — two thirds of planning time, per the profiler —
    into one walk for all of them."""

    def combined(expr):
        for leaf in leaves:
            expr = leaf(expr)
        return expr

    return map_node_expressions(node, lambda e: transform_expr_up(e, combined))
