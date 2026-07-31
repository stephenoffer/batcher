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

__all__ = ["LEFT_DRIVEN_JOINS", "RIGHT_DRIVEN_JOINS", "shardable_leaves"]

#: Join types whose output is driven by the LEFT input, so splitting it is safe: each left row
#: is seen by exactly one worker and contributes to the answer exactly as many times as the
#: matches in the (whole) right side it sees.
LEFT_DRIVEN_JOINS = frozenset({"inner", "left", "semi", "anti"})

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
