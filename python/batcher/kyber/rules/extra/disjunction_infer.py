"""NORMALIZE-phase implied-predicate inference from a multi-column disjunction.

A join-spanning disjunction of conjunctions — the DNF shape SQL writes as
``(a = 1 AND b = 2) OR (a = 3 AND b = 4)`` — is opaque to predicate pushdown: it
references two tables at once, so it cannot sink below the join that brings them
together, and every dimension row flows through the join before the disjunction
finally rejects it. But the disjunction *implies* a single-column constraint on
each column that is equality-pinned in **every** disjunct: any row it keeps has
``a`` equal to one of ``{1, 3}`` and ``b`` to one of ``{2, 4}``. Adding
``a IN (1, 3)`` and ``b IN (2, 4)`` — ANDed with the original disjunction — removes
no row (each is a provable superset of the disjunction) and adds none (it is an
extra conjunct), yet each derived ``IN`` references a single table, so pushdown
sinks it onto that dimension's scan and shrinks the join inputs before the join.

This is the classic win on TPC-H Q7 (the ``(n1=FRANCE AND n2=GERMANY) OR
(n1=GERMANY AND n2=FRANCE)`` nation pair, which without it joins all 25 nations
through the fact table) and Q19 (three OR-ed brand/quantity/container cubes). It is
the multi-column companion to ``or_to_in_and_range`` (which handles the
*single*-column ``c = v1 OR c = v2`` case with range bounds); the two are disjoint —
this rule fires only when a disjunct is itself a conjunction, which the single-column
rule never sees.

Sound under three-valued logic: a ``Filter`` keeps a row iff the predicate is TRUE,
so a kept row makes some disjunct TRUE, meaning every leaf of that disjunct — in
particular its ``col = v`` — is TRUE, so ``col`` is non-null and equals a value in
the derived list, making ``col IN (…)`` TRUE there too. NULL/bool literals are
excluded (never safe to reason about by equality).
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Binary, Col, Expr, InList, Lit
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts
from batcher.plan.logical import Filter, LogicalPlan

__all__ = ["infer_disjunction_in_lists"]


def _split_or(expr: Expr) -> list[Expr]:
    """Flatten a top-level ``OR`` tree into its disjuncts (a non-``OR`` node is one)."""
    if isinstance(expr, Binary) and expr.op == "or":
        return [*_split_or(expr.left), *_split_or(expr.right)]
    return [expr]


def _col_eq_lit(expr: Expr) -> tuple[str, object] | None:
    """``(column, value)`` for a ``col = literal`` / ``literal = col`` leaf, else None.
    NULL and bool literals are rejected — neither is safe to reason about by equality."""
    if not (isinstance(expr, Binary) and expr.op == "eq"):
        return None
    left, right = expr.left, expr.right
    if isinstance(left, Col) and isinstance(right, Lit):
        name, value = left.name, right.value
    elif isinstance(right, Col) and isinstance(left, Lit):
        name, value = right.name, left.value
    else:
        return None
    return None if value is None or isinstance(value, bool) else (name, value)


def _column_in_lists(conj: Expr) -> list[Expr]:
    """Per-column ``IN`` predicates implied by a DNF ``conj``, or ``[]`` if none apply.

    Fires only on a disjunction of ≥2 disjuncts where at least one disjunct is itself
    a conjunction (so the single-column ``or_to_in_and_range`` does not already cover
    it). For each column equality-pinned in *every* disjunct, the union of its values
    is the implied membership list; a column absent (or only range-bounded) in any
    disjunct is left out, since that disjunct places no finite bound on it."""
    disjuncts = _split_or(conj)
    if len(disjuncts) < 2:
        return []
    per_disjunct: list[dict[str, list[object]]] = []
    saw_conjunction = False
    for disjunct in disjuncts:
        leaves = split_conjuncts(disjunct)
        saw_conjunction = saw_conjunction or len(leaves) > 1
        pinned: dict[str, list[object]] = {}
        for leaf in leaves:
            parsed = _col_eq_lit(leaf)
            if parsed is not None:
                pinned.setdefault(parsed[0], []).append(parsed[1])
        per_disjunct.append(pinned)
    if not saw_conjunction:
        return []  # a plain `c = v1 OR c = v2` — left to `or_to_in_and_range`
    shared = set.intersection(*(set(p) for p in per_disjunct))
    out: list[Expr] = []
    for name in sorted(shared):
        values: list[object] = []
        for pinned in per_disjunct:
            values.extend(pinned[name])
        deduped = tuple(dict.fromkeys(values))
        if len(deduped) == 1:
            out.append(Binary("eq", Col(name), Lit(deduped[0])))
        else:
            out.append(InList(Col(name), deduped))
    return out


@rule(name="infer_disjunction_in_lists", phase=Phase.NORMALIZE, matches=(Filter,))
def infer_disjunction_in_lists(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Add a per-column ``IN`` alongside a multi-column DNF conjunct so pushdown can
    sink it: ``(a = 1 AND b = 2) OR (a = 3 AND b = 4)`` gains ``a IN (1, 3)`` and
    ``b IN (2, 4)``.

    Each derived membership is a provable superset of the disjunction (see the module
    docstring), so ANDing it in changes no result — but it references a single table,
    so the pushdown phase drops it below the join and shrinks the join inputs. Fires
    only when a disjunct is itself a conjunction (the multi-column case), keeping it
    disjoint from ``or_to_in_and_range``. Idempotent: a derived predicate already
    present is not re-added, and once added the enclosing disjunction is unchanged."""
    conjuncts = split_conjuncts(node.predicate)
    existing = [c.to_ir() for c in conjuncts]  # IR dicts are unhashable → list + `in`
    added: list[Expr] = []
    for conj in conjuncts:
        for pred in _column_in_lists(conj):
            if pred.to_ir() not in existing:
                added.append(pred)
                existing.append(pred.to_ir())
    if not added:
        return None
    return Filter(node.input, combine_conjuncts([*conjuncts, *added]))
