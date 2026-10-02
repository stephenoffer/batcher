"""Which part of a plan can run whole on each worker, over one key range of its inputs.

A table written in key order -- `lineitem` and `orders` in `orderkey` order, a lakehouse
table clustered on its join key, any append-only log keyed by time -- already has the
co-partitioning a shuffle join would build: rows that can join sit in files whose footer
ranges line up. Assign each worker a key range and the files that overlap it, and the join,
and anything grouped by the key above it, complete on that worker with nothing exchanged.

This module is the *decision*, a pure function of the plan: find the key column class the
plan's equi-joins share, and the largest subtree whose result is the disjoint union of its
per-range results. It reads no data. `units` supplies the footer facts that say whether the
files really are laid out that way; `run` executes the result.

The rules are the ones partition-wise joins have always had, stated per operator in
`_kind`. What matters for correctness is that every operator inside the subtree maps a
key-range restriction of its aligned inputs to the same restriction of its output, with
its other inputs whole:

* an equi-join of two aligned inputs must join *on* the key, so matching rows share a range;
* an aligned input joined to a whole (broadcast) one keeps only row-preserving shapes -- an
  inner join, or an outer/semi/anti join whose preserved side is the aligned one. A
  broadcast side that is preserved would emit its unmatched rows once per range;
* an aggregate or distinct must group by the key, or it is the top of the subtree and runs as
  partial state that is combined afterwards (the mergeable algebra every executor shares).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Scan,
    Sort,
)

__all__ = ["AlignedCut", "AlignedPlan", "KeyClass", "cut_signature", "find_plan", "key_classes"]


# (source_id, physical column) -- one column of one bound source.
SourceCol = tuple[int, str]


@dataclasses.dataclass(frozen=True)
class KeyClass:
    """Source columns an equi-join chain makes equal: one candidate alignment key.

    A `keyless` class names one source and no column: it is split by file, with no range
    filter, which is correct only because nothing else is aligned beside it -- every other
    input is read whole, so no row of it needs a partner in the same range.
    """

    columns: frozenset[SourceCol]
    keyless: bool = False

    def column_of(self, source_id: int) -> str | None:
        """The class's column in `source_id`, or None when that source is not in it."""
        names = sorted(c for s, c in self.columns if s == source_id)
        return names[0] if names else None

    @classmethod
    def by_file(cls, source_id: int) -> KeyClass:
        """The keyless class of one source, split by file."""
        return cls(frozenset({(source_id, "")}), keyless=True)


@dataclasses.dataclass(frozen=True)
class AlignedCut:
    """One subtree evaluated per key range: its body, and the aggregate it feeds, if any.

    `body` is evaluated once per range, with every scan of an `aligned` source restricted
    to that range and every other scan read whole. When `aggregate` is set the body is its
    input and each range yields partial state for it; otherwise the ranges' results are
    concatenated.
    """

    body: LogicalPlan
    aggregate: Aggregate | None
    key: KeyClass
    aligned: frozenset[int]
    broadcast: frozenset[int]

    @property
    def node(self) -> LogicalPlan:
        """The plan node this cut computes: the aggregate when there is one, else the body."""
        return self.aggregate if self.aggregate is not None else self.body

    def pushdown(self, body: LogicalPlan, source_id: int) -> tuple[list[str] | None, dict | None]:
        """The projection and predicate to read `source_id` with, for this cut over `body`.

        Asked of the aggregate over `body`, not of `body` alone: a body that is a bare scan
        requires every column it has, and it is the aggregate above it that needs two.
        """
        from batcher.dist.executors.partition_io import source_pushdown

        scope = body if self.aggregate is None else dataclasses.replace(self.aggregate, input=body)
        return source_pushdown(scope, source_id)


@dataclasses.dataclass(frozen=True)
class AlignedPlan:
    """A plan split into aligned cuts and a residual the driver runs over their results.

    `residual` is the original plan with each cut's node replaced by a scan of source id
    `placeholders[i]`, past the query's own sources; it reads only those results and the
    small sources no cut covers.
    """

    cuts: tuple[AlignedCut, ...]
    residual: LogicalPlan
    placeholders: tuple[int, ...]

    @property
    def key(self) -> KeyClass:
        return self.cuts[0].key

    @property
    def aligned(self) -> frozenset[int]:
        return self.cuts[0].aligned


@dataclasses.dataclass(frozen=True)
class _Kind:
    """How a subtree's rows relate to the key ranges: restricted, or whole on every range."""

    aligned: bool
    keys: frozenset[str] = frozenset()  # output columns carrying the key (aligned only)


_WHOLE = _Kind(aligned=False)


def _scan_names(scan: Scan) -> list[str]:
    arrow = getattr(scan.schema, "arrow", None)
    return list(arrow.names) if arrow is not None else []


def _origin(node: LogicalPlan, name: str) -> SourceCol | None:
    """The source column `name` passes through from unchanged, if it is one."""
    if isinstance(node, Scan):
        return (node.source_id, name) if name in _scan_names(node) else None
    if isinstance(node, (Filter, Sort, Limit, Distinct)):
        return _origin(node.input, name)
    if isinstance(node, Project):
        for item in node.items:
            if item.alias == name:
                return _origin(node.input, item.expr.name) if isinstance(item.expr, Col) else None
        return None
    if isinstance(node, Join):
        for out in node.output:
            if out.alias == name:
                return _origin(node.left if out.side == "left" else node.right, out.name)
        return None
    if isinstance(node, Aggregate):
        for gk in node.group_keys:
            if gk.alias == name and isinstance(gk.expr, Col):
                return _origin(node.input, gk.expr.name)
        return None
    return None


def _walk(node: LogicalPlan) -> Iterable[LogicalPlan]:
    from batcher.plan.visitor import walk

    return walk(node)


def key_classes(plan: LogicalPlan) -> list[KeyClass]:
    """The equality classes of source columns the plan's equi-joins induce, largest first."""
    parent: dict[SourceCol, SourceCol] = {}

    def find(x: SourceCol) -> SourceCol:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for node in _walk(plan):
        if not isinstance(node, Join):
            continue
        for lk, rk in zip(node.left_keys, node.right_keys, strict=False):
            a, b = _origin(node.left, lk), _origin(node.right, rk)
            if a is not None and b is not None:
                parent[find(a)] = find(b)
    # A relation the query uses twice is bound twice, as two source ids over one table; a
    # key column of one is the same key column of the other, so they align together.
    same_table: dict[str, set[int]] = {}
    for node in _walk(plan):
        if isinstance(node, Scan) and node.source_key:
            same_table.setdefault(node.source_key, set()).add(node.source_id)
    twins = {sid: ids for ids in same_table.values() for sid in ids if len(ids) > 1}
    for sid, column in list(parent):
        for twin in twins.get(sid, ()):
            if twin != sid:
                parent[find((twin, column))] = find((sid, column))
    groups: dict[SourceCol, set[SourceCol]] = {}
    for member in list(parent):
        groups.setdefault(find(member), set()).add(member)
    classes = [KeyClass(frozenset(g)) for g in groups.values() if len(g) > 1]
    return sorted(classes, key=lambda k: -len(k.columns))


def _key_outputs(node: LogicalPlan, key: KeyClass, aligned: frozenset[int]) -> frozenset[str]:
    """The output columns of `node` that carry the aligned key, by origin."""
    names = _output_names(node)
    out = set()
    for n in names:
        o = _origin(node, n)
        if o is not None and o[0] in aligned and o in key.columns:
            out.add(n)
    return frozenset(out)


def _output_names(node: LogicalPlan) -> list[str]:
    try:
        return list(node.available_columns())
    except Exception:
        return []


def _kind(node: LogicalPlan, key: KeyClass, aligned: frozenset[int]) -> _Kind | None:
    """`node`'s relation to the key ranges, or None when it cannot run per range."""
    if isinstance(node, Scan):
        if node.source_id in aligned:
            return _Kind(True, _key_outputs(node, key, aligned))
        return _WHOLE
    if isinstance(node, (Filter, Project)):
        inner = _kind(node.input, key, aligned)
        if inner is None or not inner.aligned:
            return inner
        return _Kind(True, _key_outputs(node, key, aligned))
    if isinstance(node, Join):
        return _join_kind(node, key, aligned)
    if isinstance(node, Aggregate):
        inner = _kind(node.input, key, aligned)
        if inner is None or not inner.aligned:
            return inner
        grouped = {gk.alias for gk in node.group_keys if isinstance(gk.expr, Col)}
        on_key = {
            gk.alias
            for gk in node.group_keys
            if isinstance(gk.expr, Col) and gk.expr.name in inner.keys
        }
        return _Kind(True, frozenset(on_key)) if on_key and on_key <= grouped else None
    if isinstance(node, Distinct):
        inner = _kind(node.input, key, aligned)
        if inner is None or not inner.aligned:
            return inner
        if node.limit is not None:
            return None
        covered = inner.keys if not node.keys else inner.keys & set(node.keys)
        return _Kind(True, frozenset(covered)) if covered else None
    return None


def _join_kind(node: Join, key: KeyClass, aligned: frozenset[int]) -> _Kind | None:
    left, right = _kind(node.left, key, aligned), _kind(node.right, key, aligned)
    if left is None or right is None:
        return None
    how = node.join_type
    if not left.aligned and not right.aligned:
        return _WHOLE
    if left.aligned and right.aligned:
        # Both sides restricted: they must meet on the key, so a row's partners share its range.
        on_key = any(
            lk in left.keys and rk in right.keys
            for lk, rk in zip(node.left_keys, node.right_keys, strict=False)
        )
        return _Kind(True, _key_outputs(node, key, aligned)) if on_key else None
    # One side restricted, one whole: only the shapes that never emit a whole-side row alone.
    preserved_ok = {
        "inner": True,
        "left": left.aligned,
        "semi": left.aligned,
        "anti": left.aligned,
        "right": right.aligned,
    }.get(how, False)
    return _Kind(True, _key_outputs(node, key, aligned)) if preserved_ok else None


def _has_aligned_join(node: LogicalPlan, aligned: frozenset[int]) -> bool:
    from batcher.plan.visitor import scanned_source_ids

    return any(isinstance(n, Join) and bool(scanned_source_ids(n) & aligned) for n in _walk(node))


#: Rounds of blocking aggregates computed ahead of the final cuts.
_MAX_STAGES = 3


def find_plan(
    plan: LogicalPlan,
    key: KeyClass,
    aligned: frozenset[int],
    n_sources: int,
    exclude: frozenset[int] = frozenset(),
    single_table: bool = False,
) -> AlignedPlan | None:
    """`plan` cut into per-range subtrees under `key` and a driver residual, or None.

    Cuts are found in rounds. An aggregate that a join consumes but that is not grouped by
    the key -- TPC-H q17's per-part average, q15's per-supplier revenue -- blocks every
    per-range subtree above it, so it is computed first, as a cut of its own, and its result
    read by the rest as a whole (broadcast) input. Then every aligned scan left must fall
    inside a final cut: each maximal subtree that runs per range whole (concatenated) or as
    an aggregate's input (combined) is one, and identical subtrees are one cut. What is left,
    the residual, joins, filters and sorts the cuts' results on the driver.

    Declined when an aligned scan cannot be covered, when a final cut would ship rows it did
    not reduce, or when the plan has no join at all: a single-table aggregate already has a
    distributed path with no driver residual.

    `exclude` names sources no final cut may read, which leaves their joins to the residual:
    a broadcast too large to hold on every node, joined above an aggregate that has already
    reduced the facts it meets (TPC-H q18's `customer`, over its few thousand large orders).
    `single_table` lifts the no-join rule, for a broadcast input evaluated on the fleet.
    """
    # `single_table` admits a plan with no join: a broadcast input being evaluated, whose
    # result the driver needs whole anyway (the distinct keys of an anti join's side).
    if not single_table and not any(isinstance(n, Join) for n in _walk(plan)):
        return None
    cuts: list[AlignedCut] = []
    ids: list[int] = []
    current = plan
    for _ in range(_MAX_STAGES):
        stage = _unique([_cut(a.input, a, key, aligned) for a in _blocking(current, key, aligned)])
        if not stage:
            break
        stage_ids = [n_sources + len(ids) + i for i in range(len(stage))]
        current = _replace_cuts(current, stage, tuple(stage_ids))
        if current is None:
            return None
        cuts += stage
        ids += stage_ids
    final = _final_cuts(current, key, aligned, exclude)
    if final is None:
        return None
    final_ids = [n_sources + len(ids) + i for i in range(len(final))]
    residual = _replace_cuts(current, final, tuple(final_ids))
    if residual is None or not (cuts or final):
        return None
    return AlignedPlan(tuple(cuts + final), residual, tuple(ids + final_ids))


def _cut(
    body: LogicalPlan, aggregate: Aggregate | None, key: KeyClass, aligned: frozenset[int]
) -> AlignedCut:
    """One cut, aligning only what it scans.

    Two scans of one table in different cuts are split independently, and a cut never reads
    files for a source it does not scan.
    """
    from batcher.plan.visitor import scanned_source_ids

    scans = scanned_source_ids(body)
    return AlignedCut(body, aggregate, key, frozenset(scans & aligned), frozenset(scans - aligned))


def _unique(cuts: list[AlignedCut]) -> list[AlignedCut]:
    out: list[AlignedCut] = []
    for cut in cuts:
        if cut not in out:
            out.append(cut)
    return out


def _blocking(plan: LogicalPlan, key: KeyClass, aligned: frozenset[int]) -> list[Aggregate]:
    """Aggregates over a per-range input, not grouped by the key, that a join consumes."""
    from batcher.plan.visitor import children

    out: list[Aggregate] = []

    def visit(node: LogicalPlan, under_join: bool) -> None:
        if isinstance(node, Aggregate) and under_join:
            inner = _kind(node.input, key, aligned)
            own = _kind(node, key, aligned)
            if inner is not None and inner.aligned and not (own is not None and own.aligned):
                out.append(node)
                return
        for child in children(node):
            visit(child, under_join or isinstance(node, Join))

    visit(plan, False)
    return out


def _final_cuts(
    plan: LogicalPlan, key: KeyClass, aligned: frozenset[int], exclude: frozenset[int]
) -> list[AlignedCut] | None:
    """The cuts covering every aligned scan left in `plan`, or None when one is uncovered."""
    from batcher.plan.visitor import children, scanned_source_ids

    found: list[AlignedCut] = []

    def visit(node: LogicalPlan) -> bool:
        scanned = scanned_source_ids(node)
        if not (scanned & aligned):
            return True
        kids = children(node)
        if scanned & exclude:
            return bool(kids) and all(visit(child) for child in kids)
        kind = _kind(node, key, aligned)
        if kind is not None and kind.aligned:
            found.append(_cut(node, None, key, aligned))
            return True
        if isinstance(node, Aggregate):
            inner = _kind(node.input, key, aligned)
            if inner is not None and inner.aligned:
                found.append(_cut(node.input, node, key, aligned))
                return True
        return bool(kids) and all(visit(child) for child in kids)

    if not visit(plan):
        return None
    cuts = _unique(found)
    # A cut that only scans, filters and projects reduces nothing: every row it reads crosses
    # to the driver for the residual. Each cut must join or aggregate, or it is not worth one.
    if not all(cut.aggregate is not None or _reduces(cut.body) for cut in cuts):
        return None
    return cuts


def _reduces(body: LogicalPlan) -> bool:
    """Whether `body` joins, aggregates or deduplicates somewhere below its root."""
    return any(isinstance(n, (Join, Aggregate, Distinct)) for n in _walk(body))


def _replace_cuts(plan: LogicalPlan, cuts: list[AlignedCut], ids: tuple[int, ...]):
    """`plan` with each cut's node replaced by a scan of its placeholder id."""
    nodes = [cut.node for cut in cuts]
    schemas = [cut.node.available_schema() for cut in cuts]
    if any(schema is None for schema in schemas):
        return None

    def swap(node: LogicalPlan) -> LogicalPlan:
        for cut_node, sid, schema in zip(nodes, ids, schemas, strict=True):
            if node == cut_node:
                return Scan(sid, schema)
        return node

    return _transform_down(plan, swap)


def _transform_down(node: LogicalPlan, fn) -> LogicalPlan:
    """Top-down rewrite that does not descend into a node `fn` replaced."""
    from batcher.plan.visitor import children, with_children

    replaced = fn(node)
    if replaced is not node:
        return replaced
    kids = children(node)
    if not kids:
        return node
    return with_children(node, [_transform_down(child, fn) for child in kids])


def cut_signature(cut: AlignedCut, sources: list[Source]) -> object | None:
    """`cut`'s computation with each scan named by its table's identity, or None."""
    from batcher.plan.visitor import transform_up

    def canonical(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Scan) and node.source_id < len(sources):
            identity = getattr(sources[node.source_id], "identity", None)
            if identity is None:
                raise LookupError("a source with no identity")
            first = next(
                i
                for i, s in enumerate(sources)
                if getattr(s, "identity", None) is not None and s.identity() == identity()
            )
            return dataclasses.replace(node, source_id=first)
        return node

    try:
        return transform_up(cut.node, canonical).content_key()
    except Exception:  # an unkeyable cut is simply computed
        return None
