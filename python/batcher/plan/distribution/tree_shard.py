"""Which leaf of a multi-way plan tree may be split across workers, and which must be replicated.

`mergeable` answers this for a *chain*: the operators divide or they do not. A tree of joins
asks a second question first — a fan-out over a tree splits exactly one of its leaves and gives
every worker the whole of the others, so before anything can fold, the leaf being split has to
be one whose rows each appear in the answer exactly once.

That is not true of every leaf, and the way it fails is silent. Split the *build* side of a
LEFT join and give every worker the whole probe side: a probe row that matches nothing in one
worker's slice of the build side is emitted by that worker as an unmatched row, and matched by
whichever worker holds its partner. The result gains a spurious null row per non-matching
worker and keeps the real one. Every worker's output is individually correct; only the union is
wrong, which is why no single-shard test finds it.

The rule is the one `BROADCAST_SAFE_JOINS` already states, read from the other direction and
applied at every join between the leaf and the root:

* an **inner** join drives its output from both sides, so either may be split;
* **left**, **semi** and **anti** are driven by their left input, so only the left may be;
* **right** is the mirror;
* **full** is driven by both and so neither may be split — an unmatched row on either side has
  to be emitted exactly once, and every worker would emit it.

A **union** below the root disqualifies everything under it. A union's other inputs would be
read whole by every worker, so each worker contributes them again, and the concatenation
duplicates them once per worker. The failure is the same shape as the LEFT join's, and it is
worth naming separately because a union looks like the most trivially splittable operator there
is — it is, but only when *every* input is split together, which a single-leaf fan-out is not.
"""

from __future__ import annotations

from batcher.plan.ir_tags import LEFT_DRIVEN_JOINS

__all__ = ["LEFT_DRIVEN_JOINS", "RIGHT_DRIVEN_JOINS", "ir_divides", "shardable_leaves"]

#: Join types whose output is driven by the LEFT input, so splitting it is safe: each left row
#: is seen by exactly one worker and contributes to the answer exactly as many times as the
#: matches in the (whole) right side it sees.

#: The mirror. `inner` is in both because it is driven by neither side alone.
RIGHT_DRIVEN_JOINS = frozenset({"inner", "right"})


def shardable_leaves(spec: dict) -> set[int]:
    """Leaf indices of `spec` that a fan-out may split, replicating every other leaf.

    Args:
        spec: A GPU plan-tree spec — nested dicts with a `kind` of `scan`, `join` or `union`.

    Returns:
        The leaf indices that are safe to split. Empty when none are, which the caller reads as
        "this tree cannot fan out"; it is never a reason to split one anyway.

    Examples:
        .. doctest::

            >>> from batcher.plan.distribution import shardable_leaves
            >>> leaf = lambda i: {"kind": "scan", "leaf": i, "ops": []}
            >>> tree = {"kind": "join", "left": leaf(0), "right": leaf(1),
            ...         "join": {"join_type": "left"}, "ops": []}
            >>> sorted(shardable_leaves(tree))
            [0]
    """
    out: set[int] = set()
    _walk(spec, out)
    return out


def _walk(spec: dict, out: set[int]) -> None:
    """Collect the splittable leaves under a node already known to be splittable."""
    kind = spec["kind"]
    if kind == "scan":
        out.add(spec["leaf"])
        return
    if kind != "join":
        # A union: replicating its siblings duplicates them once per worker. Nothing below it
        # may be split on its own.
        return
    join_type = spec["join"].get("join_type")
    if join_type in LEFT_DRIVEN_JOINS:
        _walk(spec["left"], out)
    if join_type in RIGHT_DRIVEN_JOINS:
        _walk(spec["right"], out)


#: Plan IR tags that are a *branch* rather than a chain step. A chain above one of these is what
#: a tree fan-out folds; the branch itself is what it splits a leaf of.
_BRANCH_OPS = frozenset({"hash_join", "union"})


def ir_divides(ir: dict) -> bool:
    """Whether a whole plan — join tree included — can be run a shard at a time.

    `shard_plan` answers this for a **chain**: the operators divide or they do not. It answers
    `None` for anything containing a join, because `flatten_ops` cannot flatten a branch — and
    the router read that as "this plan does not divide", so **every join plan was routed as
    though one device were enough for it**. On a six-T4 fleet at TPC-H sf10 that put a
    60 M x 15 M join on a single board: q4 and q12 took 8.7 s and 8.5 s against CPU-engine
    answers of 0.33 s and 1.28 s, with five devices idle.

    A plan divides when both halves of what a tree fan-out actually does are available:

    * the run of operators **above** the outermost branch has a mergeable decomposition, so the
      shards' outputs fold (`shard_plan`); and
    * some leaf of the branch is safe to **split** while every other leaf is replicated
      (`shardable_leaves`), which is the left/right-driven rule this module states.

    A plan with no branch at all is exactly `shard_plan`'s question, and is answered by it.

    Args:
        ir: A logical plan's JSON IR, as `LogicalPlan.to_ir()` produces it.

    Returns:
        True when the plan divides. False whenever it does not, or cannot be read — which every
        caller treats as "keep this on one device", the conservative direction.

    Examples:
        .. doctest::

            >>> from batcher.plan.distribution import ir_divides
            >>> scan = lambda i: {"op": "scan", "source_id": i}
            >>> ir_divides({"op": "hash_join", "join_type": "inner",
            ...             "left": scan(0), "right": scan(1)})
            True
            >>> ir_divides({"op": "hash_join", "join_type": "outer",
            ...             "left": scan(0), "right": scan(1)})
            False
    """
    from batcher.plan.distribution.mergeable import shard_plan

    above, branch = _above_branch(ir)
    if branch is None:
        return shard_plan(above) is not None
    if shard_plan(above) is None:
        return False
    spec = _as_spec(branch)
    return bool(spec is not None and shardable_leaves(spec))


def _above_branch(ir: dict) -> tuple[list[dict], dict | None]:
    """The chain above the outermost branch, bottom-up, and the branch itself.

    `(ops, None)` for a plan with no branch, which is `shard_plan`'s own question.
    """
    ops: list[dict] = []
    node: dict | None = ir
    while isinstance(node, dict):
        op = node.get("op")
        if op == "scan":
            return list(reversed(ops)), None
        if op in _BRANCH_OPS:
            return list(reversed(ops)), node
        ops.append(node)
        node = node.get("input")
    return list(reversed(ops)), None


def _as_spec(ir: dict) -> dict | None:
    """The branch's IR as the `{"kind": ...}` shape `shardable_leaves` walks.

    A translation rather than a second walk, so the left/right-driven rule is stated once. Leaf
    indices are assigned in encounter order, matching how a tree spec numbers them — only their
    *existence* matters here, never which source they came from.

    `None` for IR that cannot be read as a tree — a branch missing an input. Reading such a node
    as a join of two scans would answer "this divides" about a plan nobody can execute.
    """
    counter = [0]

    def convert(node) -> dict | None:
        if not isinstance(node, dict):
            return {"kind": "scan", "leaf": _next()}
        op = node.get("op")
        if op == "hash_join":
            # A join whose inputs are absent is malformed IR, not a join of two scans. Reading
            # it as one would answer "this divides" about a plan nobody can execute, which is
            # the opposite of the conservative direction every other decline here takes.
            if node.get("left") is None or node.get("right") is None:
                return None
            left, right = convert(_below(node["left"])), convert(_below(node["right"]))
            if left is None or right is None:
                return None
            return {
                "kind": "join",
                "join": {"join_type": node.get("join_type")},
                "left": left,
                "right": right,
            }
        if op == "union":
            inputs = [convert(_below(i)) for i in node.get("inputs", [])]
            if not inputs or any(i is None for i in inputs):
                return None
            return {"kind": "union", "inputs": inputs}
        return {"kind": "scan", "leaf": _next()}

    def _next() -> int:
        counter[0] += 1
        return counter[0] - 1

    return convert(ir)


def _below(node):
    """Skip the chain of linear operators over a branch's input, down to the next branch or scan.

    A pushed-down filter or projection sits between a join and its input in essentially every
    optimized plan, and it changes neither which leaves exist nor which of them may be split.
    """
    while isinstance(node, dict) and node.get("op") not in _BRANCH_OPS and node.get("op") != "scan":
        nxt = node.get("input")
        if nxt is None:
            return node
        node = nxt
    return node
