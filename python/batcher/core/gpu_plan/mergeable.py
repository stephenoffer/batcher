"""Decompose a group-by aggregate into `partial → combine → finalize`, in the plan IR.

This is what lets a GPU aggregate use more than one device. Each GPU reduces its own shard to
one row per group it saw, the small partials are concatenated, and a second aggregate folds
them into the answer — the same mergeable algebra the CPU engine's distributed path uses, so
the multi-device result is identical to the single-device one by construction rather than by
testing.

The decomposition is expressed as **more plan IR**, not as a second set of kernels: the
partial and the combine are both `aggregate` nodes, and the finalize is a `project`. They run
through the same translator every other operator does, so a fix to the aggregate kernel fixes
the distributed path with it, and the mergeable invariant can be checked on the host.

`decompose` returns `None` for a reduction with no mergeable partial form. `median`,
`quantile`, `var`, `stddev` and `count_distinct` are all in that class: each needs the group's
whole value set (or a Welford state whose combine is not a fold over columns), so a partial
per shard cannot be folded into the answer. Those stay on one device — an aggregate that
cannot shard is a scale ceiling, never a wrong number.
"""

from __future__ import annotations

from batcher.plan.ir_tags import Op

__all__ = ["decompose"]

# Reductions that combine with *themselves*: applying them to the partials gives the answer.
_SELF_COMBINING = {"min": "min", "max": "max", "product": "product",
                   "bool_and": "bool_and", "bool_or": "bool_or"}  # fmt: skip

# Reductions whose combine is a `sum` of the per-shard partials, whatever the partial was.
_SUM_COMBINING = {"sum": "sum", "count": "count", "count_star": "count_star"}

_PREFIX = "__bt_pa"


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


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
