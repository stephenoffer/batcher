"""Split a chain of operators into a per-shard stage and a merge stage, in the plan IR.

This is the mergeable algebra stated at the level of the wire contract: `partial -> combine ->
finalize` for an aggregate, and the shard/merge pair for the other reducers. Each shard runs
the chain's reducing prefix over the rows it holds, and the small results are folded once, so a
distributed result is identical to a single-node one by construction rather than by testing.

It lives in `plan` because two layers need it and they cannot import each other. `dist`
*executes* the split (fanning shards across GPUs); `kyber` *decides* on it (a plan that shards
is bounded by its shard size, not by one device's memory, which changes where the optimizer
routes it). Stating the algebra twice is the one way those two could ever disagree, and a
disagreement here is a wrong answer rather than a slow one.

Everything here is expressed as **more plan IR**, not as a second set of kernels: a partial
aggregate and its combine are both `aggregate` nodes, a finalize is a `project`, and a sharded
distinct is a `distinct` on each side. They run through the same translator every other
operator does, so a fix to a kernel fixes the distributed path with it, and the mergeable
invariant is checkable on the host.

Two questions decide whether a chain shards, and conflating them is how a distributed engine
gets a plausible wrong answer:

* **Which operators may run per shard at all?** Only the row-local ones — `filter` and
  `project`. Every other operator reads rows its shard does not have. Running a `sort ... LIMIT
  10` per shard and reducing afterwards looks like a top-N and is not one, because each shard
  contributes its own ten rows to a global ten that may all have come from one shard.
* **Which reducer has a mergeable form?** `aggregate` (for the reductions whose partials fold),
  `distinct` (idempotent, so deduplicating twice is deduplicating once), and a `sort` carrying
  a limit (a global top-N is the top-N of the shards' top-Ns). `median`, `quantile`, `var`,
  `stddev` and `count_distinct` are not: each needs the group's whole value set, so they stay
  on one device. A reducer that cannot shard is a scale ceiling, never a wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batcher.plan.ir_tags import Op

__all__ = [
    "ROW_LOCAL_OPS",
    "ShardSplit",
    "decompose",
    "flatten_ops",
    "nest_ops",
    "shard_plan",
]


@dataclass(frozen=True, slots=True)
class ShardSplit:
    """How one chain divides across workers.

    `shard_ops` runs on every worker over its own shard, `merge_ops` folds the shards' results,
    and `tail_ops` runs once on that folded result. `ordered` says the merge is a plain
    concatenation **in shard order** rather than a fold — true exactly when the chain is
    row-local, where each shard's output is its slice of the answer and the slices reassemble.

    `ordered` has to be carried rather than inferred from an empty `merge_ops`, because the two
    mean different things to the executor: a fold may collect its shards in any order, and a
    concatenation may not. Reading one as the other reorders the result of every filter that
    ever fans out.
    """

    shard_ops: list[dict]
    merge_ops: list[dict]
    tail_ops: list[dict]
    ordered: bool = False


#: Operators whose output for a row depends only on that row, so a shard can run them over the
#: rows it holds and get the same answer it would have as part of the whole.
ROW_LOCAL_OPS = frozenset({Op.FILTER, Op.PROJECT})

# Reductions that combine with *themselves*: applying them to the partials gives the answer.
_SELF_COMBINING = {"min": "min", "max": "max", "product": "product",
                   "bool_and": "bool_and", "bool_or": "bool_or"}  # fmt: skip

# Reductions whose combine is a `sum` of the per-shard partials, whatever the partial was.
_SUM_COMBINING = {"sum": "sum", "count": "count", "count_star": "count_star"}

_PREFIX = "__bt_pa"


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


def shard_plan(ops: list[dict]) -> ShardSplit | None:
    """Split a chain of operators into a per-shard stage and a merge stage.

    Two chains divide, for different reasons.

    A chain with a **mergeable reducer** folds: each shard reduces what it holds and the small
    results combine. Anything above the reducer becomes a tail that runs once on the folded
    result, which is what lets the ordinary analytical shape — group by, then sort, then limit
    — fan out at all; requiring the reducer to be the chain's last operator excluded almost
    every real query.

    A **row-local** chain (filter and project only) concatenates: every shard's output is
    already its slice of the answer, in order, so reassembling them in shard order is the
    answer. That is the most trivially divisible shape there is, and excluding it because it
    had "nothing to fold" left the largest scans — the ones a filter is written for — bounded
    by a single worker's memory.

    Args:
        ops: The bottom-up operator IR chain.

    Returns:
        The split, or `None` for an empty chain or one whose first non-row-local operator has
        no mergeable form.
    """
    if not ops:
        return None
    cut = next((i for i, op in enumerate(ops) if op.get("op") not in ROW_LOCAL_OPS), None)
    if cut is None:
        return ShardSplit(list(ops), [], [], ordered=True)
    split = _split_reducer(ops[cut], ops[cut + 1 :])
    if split is None:
        return None
    shard_stage, merge_stage, consumed = split
    return ShardSplit(
        [*ops[:cut], *shard_stage], merge_stage, ops[cut + 1 + consumed :], ordered=False
    )


def _split_reducer(op: dict, rest: list[dict]) -> tuple[list[dict], list[dict], int] | None:
    """The per-shard and merge forms of one reducing operator, plus how many followers it ate.

    Returns `None` when the operator has no mergeable form.
    """
    kind = op.get("op")
    if kind == Op.AGGREGATE:
        parts = decompose(op)
        if parts is None:
            return None
        partial_ir, combine_ir, finalize_ir = parts
        return [partial_ir], [combine_ir, finalize_ir], 0
    if kind == Op.DISTINCT:
        # Idempotent: deduplicating each shard and then the concatenation gives the same set as
        # deduplicating once. Row order is not preserved, which no distinct promises.
        return [op], [op], 0
    if kind != Op.SORT:
        return None
    # A global top-N is the top-N of the shards' top-Ns, and the merge re-sorts, so the result
    # is ordered exactly as the single-device sort would leave it. A sort with NO limit is
    # excluded on purpose: it is still correct, but the merge would carry every row, so it buys
    # parallel sorting at the cost of moving the whole dataset.
    if op.get("limit"):
        return [op], [op], 0
    # ...and the limit is as often a *separate* operator above the sort, which is the shape
    # `sort(...).limit(n)` lowers to when nothing fuses them. Matching only the fused form left
    # the most common top-N spelling unable to use more than one device. An `offset` breaks it —
    # a shard's rows 10..20 are not the global rows 10..20 — so only the offset-free form pairs.
    if rest and rest[0].get("op") == Op.LIMIT and not rest[0].get("offset"):
        pair = [op, rest[0]]
        return pair, pair, 1
    return None


def decompose(ir: dict) -> tuple[dict, dict, dict] | None:
    """Split an `aggregate` node into its partial, combine and finalize stages.

    Args:
        ir: The `aggregate` node's JSON IR.

    Returns:
        `(partial_ir, combine_ir, finalize_ir)` — two `aggregate` nodes and a `project` — or
        `None` when any reduction in the node has no mergeable partial form.
    """
    key_aliases = [gk["alias"] for gk in ir["group_keys"]]
    partials: list[dict] = []
    combines: list[dict] = []
    projections: list[dict] = []

    for slot, spec in enumerate(ir["aggregates"]):
        plan = _plan_one(spec, slot)
        if plan is None:
            return None
        stage_partials, stage_combines, projection = plan
        partials.extend(stage_partials)
        combines.extend(stage_combines)
        projections.append(projection)

    partial_ir = {"op": Op.AGGREGATE, "group_keys": ir["group_keys"], "aggregates": partials}
    # The combine groups by the partial stage's OUTPUT columns, which carry the key *aliases* —
    # the source names are gone by then, and grouping by a name that no longer exists would
    # fail on a renamed key rather than silently, but only on the renamed case.
    combine_ir = {
        "op": Op.AGGREGATE,
        "group_keys": [{"expr": _col(a), "alias": a} for a in key_aliases],
        "aggregates": combines,
    }
    finalize_ir = {
        "op": Op.PROJECT,
        "exprs": [{"expr": _col(a), "alias": a} for a in key_aliases] + projections,
    }
    return partial_ir, combine_ir, finalize_ir


def _plan_one(spec: dict, slot: int):
    """The partial(s), combine(s) and final projection for one reduction, or `None`."""
    func = spec["func"]
    alias = spec["alias"]
    if func in _SELF_COMBINING:
        name = f"{_PREFIX}{slot}"
        return (
            [{**spec, "alias": name}],
            [{"func": _SELF_COMBINING[func], "alias": alias, "input": _col(name)}],
            {"expr": _col(alias), "alias": alias},
        )
    if func in _SUM_COMBINING:
        name = f"{_PREFIX}{slot}"
        return (
            [{**spec, "alias": name}],
            [{"func": "sum", "alias": alias, "input": _col(name)}],
            {"expr": _col(alias), "alias": alias},
        )
    if func == "mean":
        # A mean is not mergeable, but the pair it is a ratio of is: sum the sums, sum the
        # counts, divide once at the end. Averaging the shards' averages would weight each
        # shard equally regardless of how many rows it held.
        total, n = f"{_PREFIX}{slot}s", f"{_PREFIX}{slot}n"
        return (
            [
                {"func": "sum", "alias": total, "input": spec["input"]},
                {"func": "count", "alias": n, "input": spec["input"]},
            ],
            [
                {"func": "sum", "alias": total, "input": _col(total)},
                {"func": "sum", "alias": n, "input": _col(n)},
            ],
            {
                "expr": {"e": "binary", "op": "div", "left": _col(total), "right": _col(n)},
                "alias": alias,
            },
        )
    return None


def nest_ops(ops: list[dict], source_id: int = 0) -> dict:
    """A flat bottom-up operator chain as one nested `RelOp` IR document over a scan.

    A chain is convenient to *rewrite* as a list and required to be *nested* to run, so both
    shapes exist and one function converts each way. The pair matters beyond convenience: the
    CPU substitute for a lost GPU shard is the same chain handed to the engine, so a drift
    between the two forms would have the substitute answer a different question.

    Args:
        ops: The bottom-up operator IR chain.
        source_id: The scan's source index within the executing task's input list.

    Returns:
        A nested `RelOp` IR document whose leaf is a `scan`.
    """
    node: dict[str, Any] = {"op": Op.SCAN, "source_id": source_id}
    for op in ops:
        node = {**op, "input": node}
    return node


def flatten_ops(ir: dict) -> list[dict] | None:
    """A nested single-source `RelOp` document as a flat bottom-up chain, or `None`.

    `None` when the document is not a linear chain over one scan — a join or a union has two
    inputs, so there is no single list of operators it is equivalent to.

    Args:
        ir: A nested `RelOp` IR document.

    Returns:
        The bottom-up operator chain, or `None` when the document branches.
    """
    ops: list[dict] = []
    node = ir
    while node.get("op") != Op.SCAN:
        child = node.get("input")
        if not isinstance(child, dict):
            return None
        ops.append({k: v for k, v in node.items() if k != "input"})
        node = child
    ops.reverse()
    return ops
