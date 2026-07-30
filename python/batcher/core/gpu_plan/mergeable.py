"""Split a translated chain into a per-shard stage and a merge stage, in the plan IR.

This is what lets the GPU backend use more than one device. Each device runs the chain's
reducing prefix over the shard it read itself, and the small results are folded once — the
same mergeable algebra the CPU engine's distributed path uses, so the multi-device result is
identical to the single-device one by construction rather than by testing.

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

from batcher.plan.ir_tags import Op

__all__ = ["decompose", "shard_plan"]

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


def shard_plan(ops: list[dict]) -> tuple[list[dict], list[dict], list[dict]] | None:
    """Split a translated chain into `(shard_ops, merge_ops, tail_ops)`.

    `shard_ops` runs on every device over its own shard; `merge_ops` folds the shards' results
    into the answer; `tail_ops` runs once on that answer, which is small by construction
    because a reducer produced it. Splitting out a tail is what lets the *ordinary* analytical
    shape — group by, then sort, then limit — fan out at all: requiring the reducer to be the
    chain's last operator excluded almost every real query.

    Args:
        ops: The bottom-up operator IR chain.

    Returns:
        The three stages, or `None` when the chain has no shardable reducer — a map-only chain
        (nothing to fold), or one whose first non-row-local operator has no mergeable form.
    """
    cut = next((i for i, op in enumerate(ops) if op.get("op") not in ROW_LOCAL_OPS), None)
    if cut is None:
        return None  # map-only: there is no reduction to distribute
    split = _split_reducer(ops[cut], ops[cut + 1 :])
    if split is None:
        return None
    shard_stage, merge_stage, consumed = split
    return [*ops[:cut], *shard_stage], merge_stage, ops[cut + 1 + consumed :]


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
