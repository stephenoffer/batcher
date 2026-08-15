"""Expression-level rule dispatch: the vocabulary index, the fused chain, and its memo.

The driver runs a phase's rules over a plan; this module is how it avoids running most of
them. Three devices stack, each a strict filter that can only skip a rule which would have
returned its input unchanged:

  * **the vocabulary index** (`expr_shapes` + `expr_type_index`) collects every
    `(Expr type, operator)` pair a plan contains, so a rule declaring shapes the plan does
    not have is dropped for that plan entirely and never traverses it;
  * **the fused chain** (`apply_expr_leaves`) runs every rule whose body is a leaf
    `Expr -> Expr` rewrite in ONE bottom-up traversal of a node's expressions, dispatching
    each expression to only the leaves that declared its type and operator;
  * **the no-op memo** records that the whole chain left an expression object untouched, so
    a later fixpoint iteration skips it wholesale.

All three read the same `Rule.expr_matches`/`Rule.expr_ops` declarations, and
`BATCHER_VERIFY_EXPR_MATCHES=1` cross-checks every one of them against running undeclared.
"""

from __future__ import annotations

from bisect import bisect_right

from batcher.config.env import env_flag
from batcher.kyber.rule import Rule
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import LogicalPlan
from batcher.plan.visitor import walk

__all__ = [
    "VERIFY_EXPR_MATCHES",
    "apply_expr_leaves",
    "bind_schema",
    "discriminator",
    "expr_shapes",
    "expr_type_index",
]


# Paranoid cross-check for the `expr_matches` type index, off in production and switched on
# by the test suite (`tests/unit/test_kyber_expr_matches_verified.py`).
#
# A leaf that declares too *narrow* a set silently stops firing: the rule is
# semantics-preserving, so results stay correct and only plan quality quietly degrades —
# nothing a normal test would notice. With this on, every filtered chain is also run
# unfiltered and the two results compared, so a declaration that skips a leaf which *would*
# have rewritten the expression fails loudly on whatever expression the corpus produced.
VERIFY_EXPR_MATCHES = env_flag("BATCHER_VERIFY_EXPR_MATCHES")


#: `id(rules) -> (rules, {Expr type: rule indices})`, the expression-type inversion of a
#: phase's rule list. Phase rule lists are built once and reused for every plan, so this is
#: bounded by the number of phases; the list itself is stored alongside to pin the id.
_EXPR_TYPE_INDEX: dict[int, tuple[list[Rule], dict[type, list[int]]]] = {}
#: Bounded like its sibling caches. A process only ever has the handful of phase rule lists,
#: so this never fills in production -- but each entry holds the list alive to pin its id, so
#: a caller that builds rule lists ad hoc (a test driving one rule) would otherwise accumulate
#: them for the life of the process.
_EXPR_TYPE_INDEX_MAX = 64


def expr_type_index(rules: list[Rule]) -> dict[type, list[int]]:
    """`Expr type -> indices of the rules that declared it`, built once per rule list."""
    cached = _EXPR_TYPE_INDEX.get(id(rules))
    if cached is not None and cached[0] is rules:
        return cached[1]
    index: dict[type, list[int]] = {}
    for i, r in enumerate(rules):
        if r.expr_matches is None:
            continue
        for t in r.expr_matches:
            index.setdefault(t, []).append(i)
    if len(_EXPR_TYPE_INDEX) >= _EXPR_TYPE_INDEX_MAX:
        _EXPR_TYPE_INDEX.clear()
    _EXPR_TYPE_INDEX[id(rules)] = (rules, index)
    return index


#: `id(plan) -> (plan, shapes)`. A plan object survives every phase that does not rewrite
#: it, and the seven phases each ask for its shapes, so without this the walk is repeated
#: for a plan already known. The plan is stored alongside to pin the id against reuse.
_SHAPES_CACHE: dict[int, tuple[LogicalPlan, frozenset[tuple[type, object]]]] = {}
_SHAPES_CACHE_MAX = 512


def expr_shapes(plan: LogicalPlan) -> frozenset[tuple[type, object]]:
    """Every `(Expr type, operator)` pair appearing in `plan`'s expressions.

    Collected through the same `map_node_expressions` / `transform_expr_up` pair the rules
    themselves use, so a node that carries expressions in an unusual slot cannot be missed
    here while being rewritten there. Both are called with an identity rewrite, so this
    reads the plan without rebuilding it.
    """
    cached = _SHAPES_CACHE.get(id(plan))
    if cached is not None and cached[0] is plan:
        return cached[1]
    shapes: set[tuple[type, object]] = set()

    def note(expr):
        shapes.add((type(expr), discriminator(expr)))
        return expr

    def per_node(expr):
        transform_expr_up(expr, note)
        return expr

    for node in walk(plan):
        map_node_expressions(node, per_node)
    found = frozenset(shapes)
    if len(_SHAPES_CACHE) >= _SHAPES_CACHE_MAX:
        _SHAPES_CACHE.clear()
    _SHAPES_CACHE[id(plan)] = (plan, found)
    return found


def bind_schema(leaf, schema):
    """Bind a node's schema into a `(Expr, SchemaRef) -> Expr` leaf, yielding a plain leaf.

    This is what lets a schema-dependent rule join the shared expression traversal: the
    schema is constant for the node, so currying it in turns the rule into exactly the
    `Expr -> Expr` shape the fused chain runs.
    """

    def bound(expr):
        return leaf(expr, schema)

    return bound


#: Which attribute names an `Expr` type's operator, resolved once per type. `op` covers the
#: arithmetic/comparison/boolean `Binary` and `Unary`; `fn` covers the function-node families
#: (`DateFunc`, `ListFunc`, `StrFunc`, `Call`, ...). `Col.name` is deliberately *not* a
#: discriminator: it is a user-chosen column name, so indexing on it would make the cache
#: grow with the query's vocabulary while separating nothing the rules distinguish.
_DISCRIMINATOR_ATTR: dict[type, str | None] = {}


def discriminator(expr) -> str | None:
    """The operator/function name an expression dispatches on, or `None` if it has neither."""
    expr_type = type(expr)
    attr = _DISCRIMINATOR_ATTR.get(expr_type, "")
    if attr == "":
        attr = "op" if hasattr(expr, "op") else ("fn" if hasattr(expr, "fn") else None)
        _DISCRIMINATOR_ATTR[expr_type] = attr
    return None if attr is None else getattr(expr, attr)


def apply_expr_leaves(
    node: LogicalPlan,
    leaves: list,
    expr_noop: dict[int, object] | None = None,
    index_by_key: dict | None = None,
) -> LogicalPlan:
    """Apply every leaf `Expr -> Expr` rewrite to `node`'s expressions in ONE traversal.

    Each leaf is offered every expression node, bottom-up, in registered order. This is the
    expression-level analogue of the node-rule fusion the driver already does, and it turns
    "one full expression walk per rule" — two thirds of planning time, per the profiler —
    into one walk for all of them.

    `expr_noop` extends the node-level no-op memo one level down, and it is what makes a
    *fixpoint* phase affordable rather than only a single pass. Fusing collapses the walks
    within one iteration, but every later iteration still offered all several hundred leaves
    to every expression node of every plan node — and after the first iteration almost all of
    those nodes are the identical, structurally-shared objects the previous iteration already
    proved nothing matched. The memo records "the whole leaf chain was a no-op on this exact
    expression object" and skips it wholesale next time. It is keyed on `id()` with the
    expression itself as the value, so the strong reference pins the id against reuse, and it
    is cleared whenever the applicable rule set can change (see `_run_phase`).

    Soundness rests on the same three facts as the node-level memo: the leaves are pure
    functions of the expression, `Expr` nodes are immutable, and `transform_expr_up`
    preserves the identity of a subtree it did not rewrite — so an unchanged object has
    unchanged content, and a rewritten child yields a *new* parent that misses the memo."""

    # Index the chain by the `Expr` type each leaf declares it can rewrite, so an
    # expression is offered only to the handful of leaves that could match it rather than
    # to all several hundred. An undeclared leaf (`None`) is in every type's list, so this
    # is a pure filter: it can only skip a leaf that would have returned its input.
    # A leaf may be given bare or as a `(leaf, expr_matches, expr_ops)` triple. Bare means
    # "no declaration", which is the safe reading: the leaf then runs on every expression.
    leaves = [entry if isinstance(entry, tuple) else (entry, None, None) for entry in leaves]
    if index_by_key is None:
        index_by_key = {}

    def indices_for(expr_type: type, op: object) -> tuple[int, ...]:
        key = (expr_type, op)
        found = index_by_key.get(key)
        if found is None:
            found = tuple(
                i
                for i, (_, matched, ops) in enumerate(leaves)
                if (matched is None or expr_type in matched)
                and (ops is None or op is None or op in ops)
            )
            index_by_key[key] = found
        return found

    def combined(expr):
        if expr_noop is not None:
            key = id(expr)
            if expr_noop.get(key) is expr:
                return expr
        original, out = expr, expr
        shape = (type(out), discriminator(out))
        order = indices_for(*shape)
        position = 0
        while position < len(order):
            slot = order[position]
            rewritten = leaves[slot][0](out)
            if rewritten is not out:
                # A leaf may have changed the expression's *type* or its *operator*, and
                # either moves it into a different bucket, so the leaves after this one must
                # be re-selected for the new shape — exactly what running the whole chain
                # unfiltered would do. Re-selecting on the operator matters as much as on the
                # type: a leaf that mirrors `lt` to `gt` leaves the type alone while making
                # every `gt` leaf newly applicable. `bisect` resumes immediately after this
                # leaf, so no leaf runs twice and none is skipped.
                new_shape = (type(rewritten), discriminator(rewritten))
                out = rewritten
                if new_shape != shape:
                    shape = new_shape
                    order = indices_for(*shape)
                    position = bisect_right(order, slot)
                    continue
            out = rewritten
            position += 1
        if VERIFY_EXPR_MATCHES:
            unfiltered = original
            for leaf, _, _ in leaves:
                unfiltered = leaf(unfiltered)
            if unfiltered is not out and unfiltered.to_ir() != out.to_ir():
                raise AssertionError(
                    "expr_matches skipped a leaf that would have rewritten this expression: "
                    f"{original!r} -> filtered {out!r} vs unfiltered {unfiltered!r}"
                )
        if expr_noop is not None and out is original:
            expr_noop[id(original)] = original  # strong ref: pins the id against reuse
        return out

    return map_node_expressions(node, lambda e: transform_expr_up(e, combined))
