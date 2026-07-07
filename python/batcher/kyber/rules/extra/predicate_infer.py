"""Syntactic predicate inference — simplify a Filter's conjunction from its literals alone.

These NORMALIZE-phase rules reason about the top-level conjunction of a `Filter`
predicate, leaning on Batcher's 3VL filter semantics: a `Filter` keeps a row iff the
predicate is **TRUE** (NULL and FALSE rows alike are dropped), so `WHERE p1 AND … AND pn`
keeps exactly the rows where every conjunct is TRUE. That lets each conjunct be reasoned
about independently — a conjunct TRUE on every row the others keep is redundant
(`a > 5 AND a > 3 → a > 5`); two conjuncts that can never both be TRUE make the filter
empty (`a > 5 AND a < 3` → constant FALSE); an `IN` list narrows against a sibling
constraint (`a IN (1,2,3) AND a > 1 → a IN (2,3)`); and a strict-order chain implies its
transitive closure (`a < b AND b < c → a < c`), added for pushdown/pruning to use.

Everything works purely on the predicate's structure and literal values — no statistics,
unlike `zonemap_pruning` (which proves emptiness from column bounds). Each rule fires
only where the rewrite is provably result-preserving under 3VL, is idempotent, and skips
any pair whose literals are not mutually comparable.
"""

from __future__ import annotations

import json
import math
import operator

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Binary, Col, Expr, InList, IsNotNull, Lit
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts
from batcher.plan.logical import Filter, LogicalPlan

__all__ = [
    "drop_bound_dominated_neq",
    "drop_redundant_is_not_null",
    "filter_eq_neq_contradiction",
    "filter_range_contradiction",
    "infer_transitive_comparisons",
    "intersect_in_lists",
    "refine_in_list_by_comparison",
    "refine_in_list_by_equality",
    "refine_in_list_by_neq",
    "remove_duplicate_conjuncts",
    "singleton_in_list_to_eq",
    "tighten_comparison_bounds",
]

_ORDER = frozenset({"lt", "le", "gt", "ge"})
_LOWER = frozenset({"gt", "ge"})
_UPPER = frozenset({"lt", "le"})
_ALL_CMP = _ORDER | {"eq", "ne"}
# Flip a comparison when the column is written on the right (`lit < col` ≡ `col > lit`).
_FLIP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le", "eq": "eq", "ne": "ne"}
_OPS = {"lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}


# --- shared helpers ---------------------------------------------------------


def _key(expr: Expr) -> str:
    """A hashable structural identity for an expression (its IR rendered as JSON)."""
    return json.dumps(expr.to_ir(), sort_keys=True)


def _bad_literal(value: object) -> bool:
    """Whether a literal is NaN — unsafe to reason about by value (never equal to itself)."""
    return isinstance(value, float) and math.isnan(value)


def _col_op_lit(conj: Expr) -> tuple[str, str, object] | None:
    """`(column, op, literal)` for a `col OP literal`/`literal OP col`, the op oriented
    column-on-left (`5 < a` → `("a", "gt", 5)`); None for anything else or a NaN literal."""
    if not (isinstance(conj, Binary) and conj.op in _ALL_CMP):
        return None
    left, right = conj.left, conj.right
    if isinstance(left, Col) and isinstance(right, Lit) and not _bad_literal(right.value):
        return left.name, conj.op, right.value
    if isinstance(right, Col) and isinstance(left, Lit) and not _bad_literal(left.value):
        return right.name, _FLIP[conj.op], left.value
    return None


def _in_list(conj: Expr) -> tuple[str, tuple] | None:
    """`(column, values)` if `conj` is a bare `col IN (values)`, else None."""
    if isinstance(conj, InList) and isinstance(conj.input, Col):
        return conj.input.name, conj.values
    return None


def _empty(node: Filter) -> Filter:
    """A schema-preserving empty filter — the input under a constant-`FALSE` predicate
    (a filter keeps only TRUE rows, so this yields zero rows)."""
    return Filter(node.input, Lit(False))


def _rebuild(node: Filter, conjuncts: list[Expr]) -> Filter:
    return Filter(node.input, combine_conjuncts(conjuncts))


def _tighter(direction: str, op1: str, v1: object, op2: str, v2: object) -> bool:
    """Whether bound 1 is at least as tight as bound 2 (its truth-set a subset). For a
    lower bound a larger threshold is tighter (`>` beats `>=` at a tie); mirror for upper."""
    if v1 == v2:
        if direction == "lo":
            return op1 == "gt" or op2 == "ge"
        return op1 == "lt" or op2 == "le"
    return v1 > v2 if direction == "lo" else v1 < v2


def _bounds_of(conjuncts: list[Expr]) -> dict[str, tuple[list, list]]:
    """Per column, its `(lower, upper)` bound lists as `(op, value)` pairs. An equality
    `a = v` contributes both a lower `>= v` and an upper `<= v` (its exact-value range)."""
    out: dict[str, tuple[list, list]] = {}
    for conj in conjuncts:
        parsed = _col_op_lit(conj)
        if parsed is None:
            continue
        name, op, value = parsed
        lo, up = out.setdefault(name, ([], []))
        if op in _LOWER:
            lo.append((op, value))
        elif op in _UPPER:
            up.append((op, value))
        elif op == "eq":
            lo.append(("ge", value))
            up.append(("le", value))
    return out


# --- redundant-conjunct elimination -----------------------------------------


@rule(name="remove_duplicate_conjuncts", phase=Phase.NORMALIZE, matches=(Filter,))
def remove_duplicate_conjuncts(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop structurally-identical duplicate conjuncts: `p AND p AND q → p AND q`.

    `X AND X ≡ X` for every truth value, so repeated conjuncts (common after filter
    merging) are pure overhead. The first occurrence is kept; fires only on a real
    duplicate, so it is idempotent."""
    conjuncts = split_conjuncts(node.predicate)
    seen: set[str] = set()
    kept = [c for c in conjuncts if (k := _key(c)) not in seen and not seen.add(k)]
    return None if len(kept) == len(conjuncts) else _rebuild(node, kept)


@rule(name="tighten_comparison_bounds", phase=Phase.NORMALIZE, matches=(Filter,))
def tighten_comparison_bounds(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Keep only the tightest same-direction bound per column: `a > 3 AND a > 5 → a > 5`,
    `a >= 5 AND a > 5 → a > 5`, `a < 8 AND a <= 8 → a < 8`.

    The tightest lower bound (`>`/`>=`) implies every other lower bound on the column, so
    the rest are redundant; likewise for upper bounds. Non-comparable literals are left
    untouched. Fires only when a bound is dropped, so it is idempotent."""
    conjuncts = split_conjuncts(node.predicate)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, conj in enumerate(conjuncts):
        parsed = _col_op_lit(conj)
        if parsed is None:
            continue
        name, op, _v = parsed
        direction = "lo" if op in _LOWER else "up" if op in _UPPER else None
        if direction is not None:
            groups.setdefault((name, direction), []).append(i)

    drop: set[int] = set()
    for (_name, direction), idxs in groups.items():
        if len(idxs) < 2:
            continue
        keep = _tightest(conjuncts, idxs, direction)
        if keep is not None:
            drop.update(i for i in idxs if i != keep)
    if not drop:
        return None
    return _rebuild(node, [c for i, c in enumerate(conjuncts) if i not in drop])


def _tightest(conjuncts: list[Expr], idxs: list[int], direction: str) -> int | None:
    """The index of the single bound implying every other in the group, or None if the
    literals are not all mutually comparable (so no total ordering exists)."""
    best = idxs[0]
    _n, best_op, best_v = _col_op_lit(conjuncts[best])  # type: ignore[misc]
    for i in idxs[1:]:
        _n, op, v = _col_op_lit(conjuncts[i])  # type: ignore[misc]
        try:
            if _tighter(direction, op, v, best_op, best_v):
                best, best_op, best_v = i, op, v
            elif not _tighter(direction, best_op, best_v, op, v):
                return None  # incomparable pair — leave the whole group alone
        except TypeError:
            return None
    return best


# --- contradiction → empty --------------------------------------------------


@rule(name="filter_range_contradiction", phase=Phase.NORMALIZE, matches=(Filter,))
def filter_range_contradiction(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Empty out a filter with disjoint bounds on one column: `a > 5 AND a < 3`,
    `a = 1 AND a > 5`, `a = 1 AND a = 2` → constant FALSE.

    A lower bound `a op vL` and upper bound `a op vU` never both hold when `vL > vU`, or
    `vL == vU` with either side strict. Equalities count as their exact-value range, so
    conflicting equalities and equality-vs-range conflicts are caught too. A
    `FALSE`-reduced filter has no bounds left to conflict, so it is idempotent."""
    for _name, (lowers, uppers) in _bounds_of(split_conjuncts(node.predicate)).items():
        for lop, lv in lowers:
            for uop, uv in uppers:
                try:
                    disjoint = lv > uv or (lv == uv and (lop == "gt" or uop == "lt"))
                except TypeError:
                    continue
                if disjoint:
                    return _empty(node)
    return None


@rule(name="filter_eq_neq_contradiction", phase=Phase.NORMALIZE, matches=(Filter,))
def filter_eq_neq_contradiction(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Empty out `a = v AND a <> v` → constant FALSE. A row cannot have `a` both equal to
    and different from the same value; matched by value, so operand order is irrelevant."""
    eqs: set[tuple[str, str]] = set()
    neqs: set[tuple[str, str]] = set()
    for conj in split_conjuncts(node.predicate):
        parsed = _col_op_lit(conj)
        if parsed is None:
            continue
        name, op, value = parsed
        if op == "eq":
            eqs.add((name, _key(Lit(value))))
        elif op == "ne":
            neqs.add((name, _key(Lit(value))))
    return _empty(node) if eqs & neqs else None


@rule(name="drop_bound_dominated_neq", phase=Phase.NORMALIZE, matches=(Filter,))
def drop_bound_dominated_neq(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `<>` a range already excludes: `a > 5 AND a <> 3 → a > 5`.

    When a bound puts the excluded value strictly outside its range, `a <> v` is always
    TRUE on the surviving rows and redundant: a lower bound dominates when `v` is below
    its floor, an upper bound when `v` is above its ceiling. Idempotent — fires only when
    a `<>` is actually dropped."""
    conjuncts = split_conjuncts(node.predicate)
    bounds = _bounds_of(conjuncts)
    keep: list[Expr] = []
    changed = False
    for conj in conjuncts:
        parsed = _col_op_lit(conj)
        if parsed is not None and parsed[1] == "ne" and _neq_dominated(parsed, bounds):
            changed = True
        else:
            keep.append(conj)
    return _rebuild(node, keep) if changed else None


def _neq_dominated(parsed: tuple[str, str, object], bounds: dict[str, tuple[list, list]]) -> bool:
    name, _op, v = parsed
    lowers, uppers = bounds.get(name, ([], []))
    try:
        # Below a lower floor (`> c` with v <= c, `>= c` with v < c) or above an upper
        # ceiling (`< c` with v >= c, `<= c` with v > c) — the value can never be hit.
        return any((v <= c if op == "gt" else v < c) for op, c in lowers) or any(
            (v >= c if op == "lt" else v > c) for op, c in uppers
        )
    except TypeError:
        return False


# --- redundant IS NOT NULL --------------------------------------------------


@rule(name="drop_redundant_is_not_null", phase=Phase.NORMALIZE, matches=(Filter,))
def drop_redundant_is_not_null(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop `col IS NOT NULL` when a sibling conjunct already rejects a null `col`.

    A top-level comparison (`col OP …`) or `col IN (…)` is NULL — never TRUE — when `col`
    is NULL, so any row it keeps already has `col` non-null, making a sibling
    `col IS NOT NULL` always TRUE there and redundant. Only these directly null-rejecting
    forms count (an `OR`/`NOT` could still admit a null), keeping it conservative."""
    conjuncts = split_conjuncts(node.predicate)
    proven: set[str] = set()
    for conj in conjuncts:
        if isinstance(conj, Binary) and conj.op in _ALL_CMP:
            proven.update(o.name for o in (conj.left, conj.right) if isinstance(o, Col))
        elif isinstance(conj, InList) and isinstance(conj.input, Col):
            proven.add(conj.input.name)
    keep = [
        c
        for c in conjuncts
        if not (isinstance(c, IsNotNull) and isinstance(c.input, Col) and c.input.name in proven)
    ]
    return _rebuild(node, keep) if len(keep) != len(conjuncts) else None


# --- IN-list refinement -----------------------------------------------------


@rule(name="refine_in_list_by_comparison", phase=Phase.NORMALIZE, matches=(Filter,))
def refine_in_list_by_comparison(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Narrow `a IN (…) AND a <cmp> lit` to the members satisfying the comparison, then
    drop it: `a IN (1,2,3) AND a > 1 → a IN (2,3)`.

    A kept row needs both TRUE, so the survivors are exactly the list members satisfying
    the ordering comparison; once narrowed to them the comparison is redundant, and an
    empty result → constant FALSE. Only `< <= > >=` refine here; `=`/`<>` are separate."""
    conjuncts = split_conjuncts(node.predicate)
    for i, conj in enumerate(conjuncts):
        found = _in_list(conj)
        if found is None:
            continue
        name, values = found
        for j, other in enumerate(conjuncts):
            parsed = _col_op_lit(other)
            if parsed is None or parsed[0] != name or parsed[1] not in _ORDER:
                continue
            op, lit = parsed[1], parsed[2]
            try:
                kept = tuple(v for v in values if _OPS[op](v, lit))
            except TypeError:
                continue
            if len(kept) == len(values):
                continue  # comparison already implied by the list — no narrowing
            if not kept:
                return _empty(node)
            rest = [c for k, c in enumerate(conjuncts) if k not in (i, j)]
            return _rebuild(node, [InList(Col(name), kept), *rest])
    return None


@rule(name="refine_in_list_by_equality", phase=Phase.NORMALIZE, matches=(Filter,))
def refine_in_list_by_equality(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Collapse `a IN (…) AND a = v` to `a = v` when `v` is a member, else constant FALSE.

    The equality pins `a` to `v`; if `v` is in the list the membership is redundant (drop
    it, keep the sharper equality), and if `v` is absent no row satisfies both. Idempotent
    — with the `IN` gone there is nothing left to collapse."""
    conjuncts = split_conjuncts(node.predicate)
    for i, conj in enumerate(conjuncts):
        found = _in_list(conj)
        if found is None:
            continue
        name, values = found
        member_keys = {_key(Lit(v)) for v in values}
        for other in conjuncts:
            parsed = _col_op_lit(other)
            if parsed is None or parsed[0] != name or parsed[1] != "eq":
                continue
            if _key(Lit(parsed[2])) in member_keys:
                return _rebuild(node, [c for k, c in enumerate(conjuncts) if k != i])
            return _empty(node)
    return None


@rule(name="refine_in_list_by_neq", phase=Phase.NORMALIZE, matches=(Filter,))
def refine_in_list_by_neq(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Remove a `<>` value from a co-located `IN` list and drop the `<>`:
    `a IN (1,2,3) AND a <> 2 → a IN (1,3)`.

    A kept row needs `a` in the list minus `v`, so dropping `v` captures the constraint
    and the `<>` becomes redundant; an emptied list → constant FALSE. Fires only when `v`
    is a list member (else the `<>` is handled elsewhere), so it is idempotent."""
    conjuncts = split_conjuncts(node.predicate)
    for i, conj in enumerate(conjuncts):
        found = _in_list(conj)
        if found is None:
            continue
        name, values = found
        for j, other in enumerate(conjuncts):
            parsed = _col_op_lit(other)
            if parsed is None or parsed[0] != name or parsed[1] != "ne":
                continue
            drop_key = _key(Lit(parsed[2]))
            kept = tuple(v for v in values if _key(Lit(v)) != drop_key)
            if len(kept) == len(values):
                continue  # `v` not in the list — not a member constraint
            if not kept:
                return _empty(node)
            rest = [c for k, c in enumerate(conjuncts) if k not in (i, j)]
            return _rebuild(node, [InList(Col(name), kept), *rest])
    return None


@rule(name="intersect_in_lists", phase=Phase.NORMALIZE, matches=(Filter,))
def intersect_in_lists(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Intersect two `IN` lists on one column: `a IN (1,2,3) AND a IN (2,3,4) → a IN (2,3)`;
    an empty intersection → constant FALSE.

    A row satisfies both memberships iff `a` is in both sets, so the intersection is the
    single equivalent list (order follows the first). Reduces two conjuncts to one each
    time, so repeated `IN`s collapse pairwise to a fixpoint."""
    conjuncts = split_conjuncts(node.predicate)
    for i, ci in enumerate(conjuncts):
        first = _in_list(ci)
        if first is None:
            continue
        name, values = first
        for j in range(i + 1, len(conjuncts)):
            second = _in_list(conjuncts[j])
            if second is None or second[0] != name:
                continue
            other_keys = {_key(Lit(v)) for v in second[1]}
            kept = tuple(v for v in values if _key(Lit(v)) in other_keys)
            if not kept:
                return _empty(node)
            rest = [c for k, c in enumerate(conjuncts) if k not in (i, j)]
            return _rebuild(node, [InList(Col(name), kept), *rest])
    return None


@rule(name="singleton_in_list_to_eq", phase=Phase.NORMALIZE, matches=(Filter,))
def singleton_in_list_to_eq(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Canonicalize a one-element `IN` list to an equality: `a IN (v) → a = v`.

    The two are identical under 3VL (a NULL `a` makes both NULL, a non-null matches iff it
    equals `v`), and the equality is the sharper form for join-key derivation and constant
    propagation — often exposed after the other `IN`-refinement rules narrow a list."""
    conjuncts = split_conjuncts(node.predicate)
    changed = False
    out: list[Expr] = []
    for conj in conjuncts:
        found = _in_list(conj)
        if found is not None and len(found[1]) == 1:
            out.append(Binary("eq", Col(found[0]), Lit(found[1][0])))
            changed = True
        else:
            out.append(conj)
    return _rebuild(node, out) if changed else None


# --- transitive comparison inference ----------------------------------------


@rule(name="infer_transitive_comparisons", phase=Phase.NORMALIZE, matches=(Filter,))
def infer_transitive_comparisons(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Add the transitive closure of a column-to-column order chain: `a < b AND b < c`
    gains `a < c` (and `a <= b AND b <= c` gains `a <= c`).

    Under the engine's total order `<`/`<=` are transitive, so on every surviving row (all
    chained comparisons TRUE, hence all columns non-null) the derived comparison is TRUE
    too — adding it removes no row and, being an AND, adds none. The fresh `col OP col`
    aids predicate pushdown and join-key derivation (like `or_to_in_and_range` adds a
    bound). Restricted to column-to-column edges (never touching the literal-bound rules)
    and to genuinely new pairs, so it is idempotent; a cycle (`a < b AND b < a`) → FALSE."""
    edges, pairs = _order_edges(split_conjuncts(node.predicate))
    if not edges:
        return None
    closure = dict(edges)
    added: dict[tuple[str, str], bool] = {}
    progress = True
    while progress:
        progress = False
        for (x, y), s1 in list(closure.items()):
            for (y2, z), s2 in list(closure.items()):
                if y != y2:
                    continue
                strict = s1 or s2
                if x == z:
                    if strict:
                        return _empty(node)  # a < … < a is unsatisfiable
                    continue
                cur = closure.get((x, z))
                if cur is None or (strict and not cur):
                    closure[(x, z)] = strict
                    if (x, z) not in pairs:
                        added[(x, z)] = strict
                    progress = True
    if not added:
        return None
    new = [Binary("lt" if s else "le", Col(x), Col(z)) for (x, z), s in added.items()]
    return _rebuild(node, [*split_conjuncts(node.predicate), *new])


def _order_edges(conjuncts: list[Expr]) -> tuple[dict[tuple[str, str], bool], set]:
    """Directed `x < y` edges (value = strict?) from every column-to-column ordering
    conjunct, plus the set of ordered pairs already present (so nothing is re-derived)."""
    edges: dict[tuple[str, str], bool] = {}
    pairs: set = set()
    for conj in conjuncts:
        if not (isinstance(conj, Binary) and conj.op in _ORDER):
            continue
        left, right = conj.left, conj.right
        if not (isinstance(left, Col) and isinstance(right, Col)):
            continue
        if conj.op in _LOWER:  # x > y / x >= y  ==>  y < x
            edge, strict = (right.name, left.name), conj.op == "gt"
        else:  # x < y / x <= y
            edge, strict = (left.name, right.name), conj.op == "lt"
        pairs.add(edge)
        edges[edge] = strict or bool(edges.get(edge, False))
    return edges, pairs
