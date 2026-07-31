"""Narrow a plan tree to the columns it actually reads, at every level rather than at the leaves.

The device path reads a leaf, moves it, joins it, and only then narrows it. On a wide fact table
that is most of the cost of the query: TPC-H's `lineitem` has sixteen columns and q1 reads six
of them, q6 four. The extra bytes are paid three times over — off storage, across the host link,
and as resident device memory a shard is then sized against — and device memory is the resource
a GPU query actually runs out of.

Narrowing is derived from the plan, **top-down**: each node is asked what its parent needs of
it and answers by asking the same of its own inputs. A join needs its keys plus whichever of its
declared output columns survive; a projection needs whatever its surviving expressions
reference; a filter needs its predicate's columns on top of everything above it.

It is a *rewrite* and not a report, and that is the part worth stating. A join names the columns
it emits, so pruning the read of one of its inputs without pruning the join's own output list
leaves the join asking for a column that is no longer there — a missing-column failure, not a
wrong answer, but a failure on exactly the queries this was written to speed up. The two have to
move together, so one function does both.

Two operators refuse to narrow and say so rather than guess. `distinct` decides row identity
from *every* column it is given, and a `window` frame carries its whole input forward, so a
pruned column changes the answer instead of failing. Both answer "all of them", which every
caller reads as "read the relation as it is". That direction of caution is the only safe one:
reading a column nobody wanted is slow, and dropping one somebody did is wrong.
"""

from __future__ import annotations

__all__ = ["ALL_COLUMNS", "leaf_projections", "prune_tree"]

#: What a node answers when it cannot narrow: read every column the source has. Distinct from
#: an empty set, which is a real (and legal) answer meaning the node reads no column at all.
ALL_COLUMNS = None

#: Operators that consume every column of their input, so nothing below them can be pruned away.
_OPAQUE_OPS = frozenset({"distinct", "window"})


def prune_tree(spec: dict) -> tuple[dict, dict[int, list[str] | None]]:
    """`spec` narrowed to the columns it reads, plus each leaf's column list.

    Args:
        spec: A GPU plan-tree spec from `gpu_tree_spec`.

    Returns:
        `(pruned_spec, projections)` — the spec with every join's output list narrowed to the
        columns something above it uses, and leaf index -> the sorted column names that leaf
        must read (`None` for a leaf whose columns cannot be narrowed safely).

    Examples:
        .. doctest::

            >>> from batcher.core.gpu_plan.pruning import prune_tree
            >>> scan = {"kind": "scan", "leaf": 0, "ops": [
            ...     {"op": "project", "exprs": [{"alias": "a", "expr": {"e": "col", "name": "a"}}]}
            ... ]}
            >>> prune_tree(scan)[1]
            {0: ['a']}
    """
    projections: dict[int, list[str] | None] = {}
    pruned = _prune(spec, ALL_COLUMNS, projections)
    return pruned, projections


def leaf_projections(spec: dict) -> dict[int, list[str] | None]:
    """The columns each leaf of `spec` must read, keyed by leaf index.

    Only valid for a spec that will be executed **as given**. A caller that also wants the
    narrowing applied to the joins — which is what makes the leaf projections usable on a tree
    with more than one node — wants `prune_tree`, whose two halves cannot come apart.

    Args:
        spec: A GPU plan-tree spec from `gpu_tree_spec`.

    Returns:
        Leaf index -> the sorted column names that leaf needs, or `None` for a leaf whose
        columns cannot be narrowed safely.
    """
    return prune_tree(spec)[1]


def _prune(spec: dict, wanted: set[str] | None, out: dict) -> dict:
    """Narrow one node, given what its parent wants of it, recording what its leaves must read."""
    below = _through_ops(spec["ops"], wanted)
    kind = spec["kind"]
    if kind == "scan":
        _record_leaf(spec["leaf"], below, out)
        return spec
    if kind == "union":
        # Every input of a union carries the same columns, so they narrow together or not at all.
        share = None if below is None else set(below)
        return {**spec, "inputs": [_prune(child, share, out) for child in spec["inputs"]]}
    return _prune_join(spec, below, out)


def _prune_join(spec: dict, wanted: set[str] | None, out: dict) -> dict:
    """A join narrowed to the outputs still wanted, with both inputs narrowed to match it.

    The output list is narrowed *first* and the inputs are then asked for what the narrowed list
    needs, so the join and its inputs cannot disagree about which columns exist. Doing it the
    other way — deriving the inputs from the parent's want-set and leaving the output list alone
    — is what makes a join ask for a column its own input no longer reads.
    """
    join_ir = spec["join"]
    kept = [o for o in join_ir["output"] if wanted is None or o["alias"] in wanted]
    join_ir = {**join_ir, "output": kept}
    return {
        **spec,
        "join": join_ir,
        "left": _prune(spec["left"], _join_side(join_ir, "left"), out),
        "right": _prune(spec["right"], _join_side(join_ir, "right"), out),
    }


def _join_side(join_ir: dict, side: str) -> set[str]:
    """What one side of a join must supply: its keys, plus the outputs it still contributes.

    Always a concrete set, never "all": a join names every column it emits and every column it
    matches on, so its inputs' needs are fully known even when its own parent wants everything.
    That is what lets a projection reach a fact table under three joins instead of stopping at
    the first one whose parent could not narrow.
    """
    keys = set(join_ir["left_keys"] if side == "left" else join_ir["right_keys"])
    return keys | {o["name"] for o in join_ir["output"] if o["side"] == side}


def _record_leaf(leaf: int, wanted: set[str] | None, out: dict) -> None:
    """Record what one leaf must read, unioning it with any earlier use of the same leaf."""
    previous = out.get(leaf, ...)
    if previous is ...:
        out[leaf] = None if wanted is None else sorted(wanted)
    elif previous is not None:
        out[leaf] = None if wanted is None else sorted(set(previous) | wanted)


def _through_ops(ops: list[dict], wanted: set[str] | None) -> set[str] | None:
    """Push a want-set down through a bottom-up operator chain, returning what the chain's own
    input must supply."""
    for op in reversed(ops):
        wanted = _through_op(op, wanted)
    return wanted


def _through_op(op: dict, wanted: set[str] | None) -> set[str] | None:
    """What one operator's input must supply, given what its output is wanted for."""
    kind = op.get("op")
    if kind in _OPAQUE_OPS:
        return None
    if kind == "limit":
        return wanted
    if kind == "project":
        # A projection replaces the frame's columns outright, so only the surviving expressions
        # matter — and their *references*, which are the input's columns rather than the output's.
        return _refs_of([p["expr"] for p in op["exprs"] if wanted is None or p["alias"] in wanted])
    if kind == "filter":
        # A filter keeps every column, so it adds its predicate's references to whatever the
        # operators above it wanted.
        return None if wanted is None else wanted | _refs_of([op["predicate"]])
    if kind == "sort":
        keys = _refs_of([k["expr"] for k in op["keys"]])
        return None if wanted is None else wanted | keys
    if kind == "aggregate":
        # An aggregate replaces its input's columns with the keys and the reductions. Every key
        # is needed (they decide the grouping even when the parent drops one from the output),
        # and a reduction's input is needed when its alias survives.
        keys = _refs_of([gk["expr"] for gk in op["group_keys"]])
        inputs = _refs_of(
            [
                a["input"]
                for a in op["aggregates"]
                if "input" in a and (wanted is None or a["alias"] in wanted)
            ]
        )
        return keys | inputs
    return None


def _refs_of(exprs: list) -> set[str]:
    """Every column name any of these expression IR documents reads."""
    out: set[str] = set()
    for expr in exprs:
        _refs(expr, out)
    return out


def _refs(expr, out: set[str]) -> None:
    """Collect column references from one expression IR document.

    Written against the raw IR rather than the `Expr` objects `plan.expr_ir.walk` handles,
    because the tree spec carries IR: it has to survive being a Ray task argument, and a plan
    node does not. The traversal is deliberately structure-blind — anything that is a dict with
    an `e` of `col` is a reference, and anything else is searched for one — so a new expression
    kind is pruned correctly the day it is added rather than the day someone remembers to list
    it here.
    """
    if isinstance(expr, dict):
        if expr.get("e") == "col" and isinstance(expr.get("name"), str):
            out.add(expr["name"])
            return
        for value in expr.values():
            _refs(value, out)
        return
    if isinstance(expr, (list, tuple)):
        for value in expr:
            _refs(value, out)
