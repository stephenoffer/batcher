"""The rule-application engine: phases, fixpoint, and the levels of fusion.

Kyber's driver runs a phase's rules over a plan until it stops changing. It stays affordable
as the rule set grows to hundreds through four devices, in increasing depth:

  * **pattern indexing** — a rule whose `matches` node types are absent from the plan never
    runs at all;
  * **vocabulary indexing** — a rule that declared the expression shapes it needs, and finds
    none of them anywhere in the plan, is dropped for that plan too. This is the filter that
    bites: nearly every expression rule matches `Filter` and `Project`, so the node-type
    index barely separates them, and a query over two integer columns was still paying for
    the string, temporal, list, and struct families. It lives in `expr_dispatch`;
  * **node-rule fusion** — consecutive node-local rules share one bottom-up plan traversal;
  * **expression-rule fusion** — every rule whose body is a leaf `Expr -> Expr` rewrite shares
    one traversal of a node's *expressions*, instead of each walking the whole tree itself,
    with each expression dispatched only to the leaves that declared its type and operator;
  * **a no-op memo** — a rule that did nothing to a node object will do nothing to it again.

The last three are what keep a hundred-odd expression rules from costing a hundred expression
walks per node per fixpoint iteration. Every one of them is a strict filter: it can only skip
a rule that would have returned its input unchanged, and `BATCHER_VERIFY_EXPR_MATCHES=1`
re-runs both the chain and the whole phase unfiltered to prove it.
"""

from __future__ import annotations

from batcher._internal.logging import get_logger
from batcher.kyber.optimizer.expr_dispatch import (
    VERIFY_EXPR_MATCHES,
    apply_expr_leaves,
    bind_schema,
    expr_shapes,
    expr_type_index,
)
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.rule import Phase, Rule
from batcher.plan.logical import LogicalPlan
from batcher.plan.visitor import children, transform_up, walk

__all__: list[str] = []


# Confluent rewrite phases iterate to a fixpoint (bounded by `_fixpoint_bound`, which caps
# pathological non-convergence); every other phase makes a single decision and runs once.
#
# **SELECTION is deliberately absent, and the empty-relation rules deliberately are not in it.**
# That phase used to hold both kinds of rule at once: ones that *produce* the canonical empty
# marker `Limit(_, 0)` (`zonemap_prune_filter`, `empty_sample_n`,
# `empty_on_impossible_null_check`) and ones that *consume* one (`project_over_empty`,
# `aggregate_over_empty`, `window_over_empty`, `propagate_empty_relation`). Run once and unfused
# -- one bottom-up traversal per rule, in registration order -- a marker emitted by a producer
# registered after its consumer was never picked up, and there was no next iteration to catch it.
# The same logical emptiness then collapsed or did not purely on nesting order:
# `Aggregate(Project(Filter(false)))` reached `Limit 0` at the root while
# `Project(Aggregate(Filter(false)))` stalled one level short.
#
# Those eleven rules now live in FUSION, which iterates, and SELECTION keeps only the three
# genuine physical choices (`adaptive_build_side`, `split_expensive_filter`,
# `size_gpu_map_batches`). Splitting them that way rather than simply adding SELECTION here is
# what the phase's own semantics require: `build_side_rule` must run **exactly once**. It is not
# idempotent *as a decision* -- on a second pass it re-derives from the join it already swapped,
# sees the small side correctly on the right, and records `swapped=False` over the
# `swapped=True` that describes what actually happened. The plan stays right; the
# `BuildSideDecision` telemetry feeding explain and the learned broadcast crossover does not, and
# `test_build_side_after_swap` catches it.
#
# Two prerequisites had to be fixed before the empty rules could iterate anywhere.
# `adaptive_build_side` rebuilt the whole join tree on every pass, so the O(1) identity check
# could never fire; it now shares structure. And the family was not confluent -- nothing
# collapsed a *nested* marker, so the hoisting rules stacked `Limit(0)` on `Limit(0)`
# indefinitely, a fourteen-deep tower on TPC-DS q4 growing by one per pass;
# `propagate_empty_relation` now folds that pair.
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


#: `(id(rules), present, shapes) -> (rules, applicable)`. Selecting the applicable rules is a
#: pure function of those three, and the same three recur constantly: a phase's rule list is
#: fixed, and consecutive fixpoint iterations (and repeated queries of the same shape) ask
#: about identical plans. Without this the selection is rebuilt from all ~700 rules on every
#: iteration of every phase, which is O(rules) work that a small query cannot amortize -- it
#: made a two-column filter measurably *slower*, the trade `performance.md` rules out. The
#: rule list is stored alongside to pin the id against reuse.
_APPLICABLE_CACHE: dict[tuple, tuple[list[Rule], list[Rule]]] = {}
_APPLICABLE_CACHE_MAX = 512


def _applicable(
    rules: list[Rule],
    present: frozenset[type],
    shapes: frozenset[tuple[type, object]] | None = None,
) -> list[Rule]:
    """Rules that could fire on a plan with exactly `present` node types and `shapes`.

    A rule with `matches is None` always applies; otherwise it applies only if its
    matched node types intersect the plan. This is the indexing that keeps per-plan
    cost proportional to the applicable rules, not the total rule count.

    `shapes` extends the same idea one level down, to the `(Expr type, operator)` pairs the
    plan's expressions actually contain. The node-type index alone barely discriminates
    among the expression rules — nearly all of them match `Filter` and `Project`, which
    almost every plan has — so a query over two integer columns still paid to run the
    string, temporal, list, and struct families. A rule that declared the expression types
    it rewrites and finds none of them in the plan cannot fire, so it is dropped for this
    plan entirely: it never traverses, and it never joins the fused chain.

    The skip condition is exactly `apply_expr_leaves`'s per-expression dispatch test, lifted
    from one expression to the whole plan, so the two cannot disagree about what a
    declaration means — and `BATCHER_VERIFY_EXPR_MATCHES=1` checks that meaning against
    running the chain unfiltered. An undeclared rule (`expr_matches is None`) is never
    dropped, which keeps the default safe.
    """
    key = (id(rules), present, shapes)
    cached = _APPLICABLE_CACHE.get(key)
    if cached is not None and cached[0] is rules:
        return cached[1]
    by_node = [r for r in rules if r.matches is None or (r.matches & present)]
    if shapes is None:
        _remember(key, rules, by_node)
        return by_node
    index = expr_type_index(rules)
    reachable: set[int] = set()
    for expr_type, op in shapes:
        for i in index.get(expr_type, ()):
            if i in reachable:
                continue
            ops = rules[i].expr_ops
            if ops is None or op is None or op in ops:
                reachable.add(i)
    keep = {id(rules[i]) for i in reachable}
    selected = [r for r in by_node if r.expr_matches is None or id(r) in keep]
    _remember(key, rules, selected)
    return selected


def _remember(key: tuple, rules: list[Rule], selected: list[Rule]) -> None:
    """Record one rule selection, clearing the cache wholesale if it has grown too large."""
    if len(_APPLICABLE_CACHE) >= _APPLICABLE_CACHE_MAX:
        _APPLICABLE_CACHE.clear()
    _APPLICABLE_CACHE[key] = (rules, selected)


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
    _verifying: bool = False,
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
    # Both survive this phase's fixpoint iterations; see `_apply_node_rules` and
    # `apply_expr_leaves`.
    noop: dict[tuple[str, int], LogicalPlan] = {}
    expr_noop: dict[tuple[int, type], dict[int, object]] = {}
    current_ir = None  # lazily computed, only on the identity-says-changed path
    changed = False
    converged = False
    # The plan's expression vocabulary, refreshed alongside `present` — a rewrite can
    # introduce an expression type the plan did not contain (a range rule builds `Binary`s
    # out of a `MathExpr` call), and the rules that act on the new shape must become
    # applicable on the next iteration exactly as node-type rules do.
    # Only phases that actually hold expression-declared rules pay for the walk; for the
    # rest there is nothing the vocabulary could filter, so `None` means "don't ask".
    uses_shapes = bool(expr_type_index(rules)) and not _verifying
    shapes = expr_shapes(plan) if uses_shapes else None
    original = plan  # kept for the cross-check below, which re-runs the phase unfiltered
    for _ in range(max_iterations):
        updated = _apply_rules(
            plan,
            _applicable(rules, present, shapes),
            ctx,
            fuse=fuse,
            noop=noop,
            expr_noop=expr_noop,
        )
        if updated is plan:  # structural sharing → confirmed fixpoint, O(1)
            converged = True
            break
        if current_ir is None:
            current_ir = _fingerprint(plan)
        updated_ir = _fingerprint(updated)
        if updated_ir == current_ir:  # equal-but-new tree (an unconditional rebuilder)
            converged = True
            break
        refreshed = _present(updated)
        refreshed_shapes = expr_shapes(updated) if uses_shapes else None
        if refreshed != present or refreshed_shapes != shapes:
            # `_applicable` is a pure function of `present`, so a changed node-type set is
            # exactly when the leaf-rule list this memo was built against can differ. The
            # entries would still be *true* ("these leaves were a no-op here"), but they
            # would be answers about the previous iteration's leaves, so they are dropped.
            expr_noop.clear()
        plan, current_ir, present, shapes = updated, updated_ir, refreshed, refreshed_shapes
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
    if VERIFY_EXPR_MATCHES and uses_shapes and not _verifying:
        _verify_prefilter_kept_the_phase(original, plan, rules, ctx, max_iterations)
    # `current_ir` tracks the latest plan's IR, so when the phase changed the plan it
    # is exactly the returned plan's IR; on a no-op phase we computed nothing new.
    return plan, (current_ir if changed else None)


def _verify_prefilter_kept_the_phase(
    original: LogicalPlan,
    optimized: LogicalPlan,
    rules: list[Rule],
    ctx: OptimizerContext,
    max_iterations: int,
) -> None:
    """Assert the phase reached the same fixpoint it would have with no vocabulary filter.

    The prefilter drops whole rules, so a wrong `expr_matches` is invisible to the
    per-expression cross-check in `apply_expr_leaves` for any rule that walks the plan itself
    instead of joining the fused chain. This is the check that covers those.

    It compares the *phase's* output rather than a single iteration's, and that distinction is
    the whole point. A rule can create an expression type the plan did not contain -- rewriting
    `NOT (x IS NULL)` to `x IS NOT NULL` introduces an `IsNotNull` where there was none -- and
    the vocabulary was read before the iteration started, so rules keyed on the new type are
    unavailable until the next one. Per iteration the filtered and unfiltered runs therefore
    differ *by design*; across the fixpoint they must not, because the refreshed vocabulary
    picks the new type up and the loop runs again. Comparing per iteration reports that lag as
    a bug, which is exactly the false alarm this function exists to avoid.
    """
    reference, _ = _run_phase(original, rules, ctx, max_iterations, _verifying=True)
    if reference is optimized or _fingerprint(reference) == _fingerprint(optimized):
        return
    raise AssertionError(
        "the expression-vocabulary prefilter changed this phase's fixpoint: some rule's "
        "`expr_matches` declaration is too narrow.\n"
        f"  with the filter: {optimized!r}\n"
        f"  without it:      {reference!r}"
    )


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
    expr_noop: dict[tuple[int, type], dict[int, object]] | None = None,
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
            # The memo is scoped to `(this run, this node type)`: each maximal run of node
            # rules has its own leaf list, and within a run the list is filtered by node
            # type. A "nothing matched" answer is only about the leaves that were offered,
            # so an expression object shared between two runs — or between two node types —
            # must not carry one run's answer into the other. `i` is the run's start index,
            # which is stable for as long as the applicable rule list is (see `_run_phase`).
            out = _apply_node_rules(
                out, run, ctx, noop, fuse_exprs=True, expr_noop=expr_noop, run_key=i
            )
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
    expr_noop: dict[tuple[int, type], dict[int, object]] | None = None,
    run_key: int = 0,
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
    # The rules the per-node loop below can actually run. When expressions are fused, every
    # leaf rule has already been applied in the shared traversal, so iterating the full run
    # only to `continue` past it costs one Python loop step per rule per node — and a
    # NORMALIZE phase now holds several hundred leaf rules against a handful of node-local
    # ones. The list is independent of the node, so it is built once for the whole pass; the
    # per-rule `matches` test stays inside the loop, where it must be re-evaluated against
    # the *current* node.
    dispatch = [
        r
        for r in node_rules
        if not (fuse_exprs and (r.expr_fn is not None or r.expr_schema_fn is not None))
    ]
    # Schema-dependent leaves, indexed the same way. Kept separate from `leaves_by_type`
    # because they need the node's schema bound in before they can join the shared chain.
    schema_leaves_by_type: dict[type, list] = {}
    # The `(Expr type, operator) -> applicable leaf slots` index, cached per plan node type
    # for the whole pass, per leaf-list composition. Which leaves apply to an expression
    # depends only on the *shape* of the expression and on the leaf declarations, both fixed
    # here — the declarations come from `leaves_by_type`/`schema_leaves_by_type`, built once and
    # concatenated in a fixed order. Rebuilding it per node cost a scan of all several hundred
    # declarations per distinct shape per node, which is the work the index exists to avoid.
    expr_index_by_type: dict[tuple[type, int], dict] = {}
    # Imported here rather than at module scope on purpose. `rules.exprs.guards` lives
    # inside the rules package, and importing it from the driver's top level pulls that
    # package in *before* `registry` finishes populating it — which reorders every rule's
    # registration, and registration order is run order. `just surface-diff` reports it as
    # all 717 rules moving. A function-level import happens after registration is done.
    from batcher.kyber.rules.exprs.guards import node_schema

    def visit(node: LogicalPlan) -> LogicalPlan:
        # Leaf `Expr -> Expr` rules share ONE traversal of the node's expressions rather than
        # each walking the whole tree. Sound only where node rules already fuse (a fixpoint
        # phase, which recovers next iteration anything a fused pass stepped over).
        if fuse_exprs:
            node_type = type(node)
            leaves = leaves_by_type.get(node_type)
            if leaves is None:
                leaves = [
                    (r.expr_fn, r.expr_matches, r.expr_ops)
                    for r in node_rules
                    if r.expr_fn is not None and (r.matches is None or node_type in r.matches)
                ]
                leaves_by_type[node_type] = leaves
                schema_leaves_by_type[node_type] = [
                    (r.expr_schema_fn, r.expr_matches, r.expr_ops)
                    for r in node_rules
                    if r.expr_schema_fn is not None
                    and (r.matches is None or node_type in r.matches)
                ]
            schema_leaves = schema_leaves_by_type[node_type]
            if schema_leaves:
                # One schema resolution for every schema-dependent rule on this node,
                # instead of one per rule. `available_schema` rebuilds a pyarrow schema up
                # the plan, so this is the expensive half of what those rules used to do.
                schema = node_schema(node)
                if schema is not None:
                    leaves = [
                        *leaves,
                        *((bind_schema(leaf, schema), m, o) for leaf, m, o in schema_leaves),
                    ]
            if leaves:
                memo = None if expr_noop is None else expr_noop.setdefault((run_key, node_type), {})
                # Keyed on the leaf-list *length* as well as the node type: a node whose
                # schema cannot be resolved gets only the plain leaves, while its sibling of
                # the same type gets the schema leaves appended too. The two lists share the
                # plain prefix but not their length, so one cached slot index is out of
                # range for the other -- an `IndexError` inside the chain, caught by the
                # differential suite. Length identifies the composition exactly here,
                # because the schema leaves are an all-or-nothing suffix.
                index = expr_index_by_type.setdefault((node_type, len(leaves)), {})
                node = apply_expr_leaves(node, leaves, memo, index)
        for r in dispatch:
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
