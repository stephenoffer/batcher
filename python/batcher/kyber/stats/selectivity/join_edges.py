"""Join selectivity measured by past runs, keyed by the *edge* a join applies, not by its tree.

Every learned cardinality the estimator reads is keyed by a plan signature, and a join's
signature is its whole subtree: which inputs, in which order, on which side. That is the right
key for "how many rows did this exact operator produce", and the wrong one for join ordering.
Reordering asks about joins that have never run -- the point of a search is to consider trees
nobody executed -- so a correction learned on the tree that *did* run never reaches the
alternatives it was meant to steer between. JOB q10c is the case in point: `cast_info` joined to
`char_name` on `person_role_id` was estimated at 747,786 rows and produced 10, run after run,
because the plan that measured it joined `char_name` last, and the tree that would join it first
had no signature anybody had measured.

What does transfer across trees is the join's **selectivity** -- the fraction of the cross
product its key equalities keep, ``|L ⋈ R| / (|L| · |R|)``. Under the same independence
assumption every Selinger estimate already makes, that fraction is a property of the edge, not
of the inputs it happened to meet, so a selectivity measured once applies to any candidate join
over the same edge. That is the estimate this module supplies.

## How the loop closes

Kyber cannot hook Core's recording (the subsystems are independent), so the measurement is
derived from the history Core already writes, as `kyber.measured_selectivity` does for filters:

1. When Kyber hands Core a plan, `register_join_edges` records, for each inner equi-join in it,
   the join's signature, its two inputs' signatures, and its **edge key** -- the base
   `(source, column)` each key column traces back to. This is plan structure, not a
   measurement, and it is written once per signature.
2. Core measures every operator's output rows under those same signatures.
3. `measured_edge_selectivities` folds the two: for a registered join whose output and both
   inputs have consistent measurements (`measured_fold`'s gates), the edge's selectivity is
   ``rows(join) / (rows(left) · rows(right))``; several joins over one edge combine by
   geometric mean, because a selectivity spans orders of magnitude.

The estimator then prices any inner join over a measured edge as ``selectivity · |L| · |R|``.

## Which edges qualify

Only a **key-to-foreign-key** edge -- one side's base column is a key of its table -- and that
restriction is what makes the transfer sound rather than hopeful. Joining a foreign key to a
key keeps each foreign-key row whose value the key side still holds, so under independence
the fraction is ``1 / |key table|`` whatever either side was filtered to: a property of the
edge. Two foreign keys meeting on a shared value (`movie_info.movie_id = movie_keyword.movie_id`)
have no such constant. Their fraction is ``Σ_v p_L(v)·p_R(v)``, which depends entirely on which
values the inputs hold: measured inside JOB q29a, where `title = 'Shrek 2'` confines every input
to one movie, it is 1.0, and that 1.0 applied to two whole tables priced their join as a cross
product and sent the plan from 79 ms to 3.9 s. Those edges are not learned.

The key side also bounds the estimate: each row of the other side meets at most one key-side
row, so the learned price never exceeds the other side's rows.

## Known limits

The selectivity is measured on the inputs the join actually saw. A runtime key-range filter
(`rules.joins.runtime_range`) removes only rows that cannot match, so it leaves the join's
output unchanged and shrinks an input: the measured fraction is then larger than the same edge
over unfiltered inputs, by that filter's selectivity. The error is an over-estimate, the safe
direction, and bounded by the filter's own effect. An edge key also cannot tell two instances
of the same table and column apart -- `title.kind_id = kind_type.id` written twice in one query
over differently filtered `kind_type`s shares one entry -- which is the same collision a
literal-normalized plan signature already has, bounded the same way by the concentration gate.
"""

from __future__ import annotations

import math
import weakref
from typing import NamedTuple, Protocol

from batcher._internal.logging import note_suppressed
from batcher.kyber.measured_fold import fold_measured
from batcher.metadata import MetadataHub
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Filter, Join, LogicalPlan, Project, Scan

__all__ = [
    "EDGE_PREFIX",
    "Edge",
    "EdgeNames",
    "OriginMemo",
    "edge_key",
    "measured_edge_selectivities",
    "register_join_edges",
]

#: The keyed-parameter namespace the registered join structures live under.
_NAMESPACE = "kyber.join_edges"

#: Prefix an edge key carries wherever it shares a map with plan signatures -- the estimator's
#: `consulted` set and `plan_deps`' measured-selectivity view -- so the two can never collide.
EDGE_PREFIX = "edge:"

# Join signatures already registered per hub, so a hot query does not re-read the store for
# structures it wrote on its first run.
_REGISTERED: weakref.WeakKeyDictionary[MetadataHub, set[str]] = weakref.WeakKeyDictionary()

# `(hub.version, hub.params_version) -> result` per hub: the fold reads both histories.
_MEMO: weakref.WeakKeyDictionary[MetadataHub, tuple[tuple[int, int], dict[str, float]]] = (
    weakref.WeakKeyDictionary()
)

# `(relation, column, whether the column is a key of its relation)`.
Origin = tuple[str, str, bool]
OriginMemo = dict[tuple[int, str], tuple[LogicalPlan, Origin | None]]


class EdgeNames(Protocol):
    """What edge identification asks of the estimator about a base relation."""

    def scan_identity(self, node: Scan) -> str | None:
        """The relation `node` reads, or None when it has no identity."""

    def key_is_unique(self, node: Scan, column: str) -> bool:
        """Whether `column` holds a distinct value in (nearly) every row of `node`."""


class Edge(NamedTuple):
    """A learnable join edge: its key and which side's base columns are a key."""

    key: str
    left_unique: bool
    right_unique: bool


def _origin(node: LogicalPlan, column: str, memo: OriginMemo, names: EdgeNames) -> Origin | None:
    """The base `(relation, column, is a key)` a column of `node`'s output is read from, or None.

    Traced through the nodes that carry a column's values unchanged -- filters, bare-column
    projections, and a join's output -- down to a `Scan`, which `names` identifies. Anything else
    (an aggregate, a computed column, a source with no identity) ends the trace: what such a
    column holds is not the base column's values, so an edge through it is not the edge
    measured elsewhere.
    """
    key = (id(node), column)
    hit = memo.get(key)
    if hit is not None and hit[0] is node:
        return hit[1]
    found: Origin | None = None
    if isinstance(node, Scan):
        name = names.scan_identity(node)
        found = (name, column, names.key_is_unique(node, column)) if name else None
    elif isinstance(node, Filter):
        found = _origin(node.input, column, memo, names)
    elif isinstance(node, Project):
        for item in node.items:
            if item.alias == column:
                if isinstance(item.expr, Col):
                    found = _origin(node.input, item.expr.name, memo, names)
                break
    elif isinstance(node, Join):
        for out in node.output:
            if out.alias == column:
                child = node.left if out.side == "left" else node.right
                found = _origin(child, out.name, memo, names)
                break
    memo[key] = (node, found)
    return found


def edge_key(join: Join, names: EdgeNames, memo: OriginMemo | None = None) -> Edge | None:
    """The learnable edge an inner equi-join applies, or None when it has none.

    Each key pair becomes its two base columns, written in a fixed order, and the pairs are
    sorted, so the key names the same edge whichever side of which tree it sits on.

    Args:
        join: The join to identify.
        names: Identifies each base relation and its key columns (the estimator).
        memo: A per-estimator memo of column origins, keyed by node identity.

    Returns:
        The edge, or None for a non-inner or keyless join, a key column that does not trace to
        a named source, or a key pair neither of whose base columns is a key (see the module
        notes on which edges qualify).
    """
    if join.join_type != "inner" or not join.left_keys:
        return None
    memo = {} if memo is None else memo
    pairs: list[str] = []
    left_unique = right_unique = False
    for lk, rk in zip(join.left_keys, join.right_keys, strict=True):
        left = _origin(join.left, lk, memo, names)
        right = _origin(join.right, rk, memo, names)
        if left is None or right is None or not (left[2] or right[2]):
            return None
        left_unique, right_unique = left_unique or left[2], right_unique or right[2]
        a, b = sorted((f"{left[0]}:{left[1]}", f"{right[0]}:{right[1]}"))
        pairs.append(f"{a}={b}")
    return Edge("&".join(sorted(pairs)), left_unique, right_unique)


def register_join_edges(hub: MetadataHub | None, plan: LogicalPlan, estimator) -> None:
    """Record the edge structure of every inner equi-join in a plan about to execute.

    Best-effort: a failure is noted and planning carries on, since the structure only lets a
    later run learn from this one.

    Args:
        hub: The metadata hub Core will record this plan's measurements into.
        plan: The optimized plan, whose nodes carry the signatures Core measures under.
        estimator: The plan's estimator: its `signature_of` is what `annotate_ops` stamps on
            each operator, and it identifies the relations edges are keyed by (`EdgeNames`).
    """
    if hub is None:
        return
    from batcher.plan.visitor import walk

    try:
        known = _REGISTERED.setdefault(hub, set())
        memo: OriginMemo = {}
        for node in walk(plan):
            if not isinstance(node, Join):
                continue
            edge = edge_key(node, estimator, memo)
            if edge is None:
                continue
            sig = estimator.signature_of(node)
            if sig in known:
                continue
            known.add(sig)
            left, right = estimator.signature_of(node.left), estimator.signature_of(node.right)
            entry = {"edge": edge.key, "l": left, "r": right}
            if hub.get_keyed_param(_NAMESPACE, sig) != entry:
                hub.put_keyed_param(_NAMESPACE, sig, entry)
    except Exception as exc:  # learning must never break planning
        note_suppressed("kyber", "register join edges", exc)


def _rows_of(row: dict) -> float | None:
    """The output rows one feedback row measured, or None when it measured nothing."""
    rows = row.get("n_actual")
    return float(rows) if isinstance(rows, (int, float)) and rows >= 0 else None


def measured_edge_selectivities(hub: MetadataHub | None) -> dict[str, float]:
    """`{edge key: measured selectivity}` for the edges past runs have measured consistently.

    A join that emitted nothing contributes half a row, so its edge is learned as very
    selective rather than as impossible: a measured zero is a fact about those inputs, and a
    zero selectivity would price every later join over the edge at nothing whatever it meets.

    Args:
        hub: The metadata hub holding the registered structures and the measured history.

    Returns:
        The measured selectivity per edge key; empty without a hub or a measurement.
    """
    if hub is None:
        return {}
    stamp = (hub.version, hub.params_version)
    cached = _MEMO.get(hub)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    out: dict[str, float] = {}
    try:
        rows = fold_measured(hub, _rows_of, what="operator output rows")
        structures = hub.load_keyed_params(_NAMESPACE) if rows else {}
        logs: dict[str, list[float]] = {}
        for sig, entry in structures.items():
            joined, left, right = rows.get(sig), rows.get(entry["l"]), rows.get(entry["r"])
            if joined is None or not left or not right:
                continue
            logs.setdefault(entry["edge"], []).append(math.log(max(joined, 0.5) / (left * right)))
        out = {edge: math.exp(sum(v) / len(v)) for edge, v in logs.items()}
    except Exception as exc:  # learning must never break planning
        note_suppressed("kyber", "read measured join edges", exc)
        out = {}
    _MEMO[hub] = (stamp, out)
    return out
