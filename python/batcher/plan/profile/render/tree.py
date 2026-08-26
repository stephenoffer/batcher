"""Recovering the shape of a plan tree from a flat, pre-ordered operator list.

`OpProfile` carries a depth and a position, not parent links, so every structural question
the renderer asks — which node is a last child, where does a subtree end, how much time is
under here, which chain is hot, what can be folded — is answered from the depth sequence.
Doing that once, here, is what keeps the spine, the fold, and the critical path from
disagreeing about the same tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from batcher.plan.profile.render.options import FOLD_ABOVE_OPS, FOLD_BELOW_SHARE, RenderOptions

if TYPE_CHECKING:
    from batcher.plan.profile.types import OpProfile

__all__ = [
    "critical_path",
    "folded",
    "has_branch",
    "last_child_flags",
    "spine",
    "subtree_end",
    "subtree_ms",
]


def last_child_flags(ops: Sequence[OpProfile]) -> list[bool]:
    """For each operator, whether it is the last child of its parent.

    In a pre-order walk, node *i* is its parent's last child when no later node appears at
    the same depth before one appears at a shallower depth. That is what turns a ``├─``
    into a ``└─`` and closes the vertical bar under it.
    """
    flags = [True] * len(ops)
    for i, op in enumerate(ops):
        for j in range(i + 1, len(ops)):
            if ops[j].depth < op.depth:
                break
            if ops[j].depth == op.depth:
                flags[i] = False
                break
    return flags


def subtree_end(ops: Sequence[OpProfile], i: int) -> int:
    """The index one past the last descendant of operator `i`."""
    end = i + 1
    while end < len(ops) and ops[end].depth > ops[i].depth:
        end += 1
    return end


def subtree_ms(ops: Sequence[OpProfile]) -> list[float]:
    """Total measured milliseconds in each operator's subtree, itself included.

    The fold and the critical path both need "how much time is *under* here", which is not
    an operator's own time: a cheap exchange above an expensive join must not be folded.
    """
    totals = [0.0] * len(ops)
    for i in range(len(ops) - 1, -1, -1):
        own = ops[i].elapsed_ms if ops[i].measured else 0.0
        totals[i] = own + sum(
            totals[j] for j in range(i + 1, subtree_end(ops, i)) if ops[j].depth == ops[i].depth + 1
        )
    return totals


def critical_path(ops: Sequence[OpProfile], subtree: Sequence[float]) -> set[int]:
    """Indices on the hottest root-to-leaf chain — where the time actually goes.

    Descends from each root into whichever child holds the most subtree time. On a join
    tree this is the answer to "which side is costing me", which no per-operator column
    shows: both sides look individually modest while one of them carries the run.
    """
    marked: set[int] = set()
    roots = [i for i, op in enumerate(ops) if op.depth == 0]
    for root in roots:
        node = root
        while True:
            marked.add(node)
            kids = [
                j
                for j in range(node + 1, subtree_end(ops, node))
                if ops[j].depth == ops[node].depth + 1
            ]
            if not kids:
                break
            node = max(kids, key=lambda j: subtree[j])
    return marked


def folded(ops: Sequence[OpProfile], opts: RenderOptions, subtree: Sequence[float]) -> set[int]:
    """Indices to hide: whole subtrees too cold to hold the answer, on a large plan.

    Never folds a root, never folds anything on the critical path, and never folds at all
    below `FOLD_ABOVE_OPS` operators or without measurements to judge coldness by. A fold
    that could hide the interesting operator would be worse than a long tree.
    """
    if not opts.fold or not opts.analyze or len(ops) <= FOLD_ABOVE_OPS:
        return set()
    total = max(subtree[i] for i in range(len(ops)) if ops[i].depth == 0) if ops else 0.0
    if total <= 0:
        return set()
    keep = critical_path(ops, subtree)
    hidden: set[int] = set()
    i = 0
    while i < len(ops):
        if i in hidden or ops[i].depth == 0 or i in keep:
            i += 1
            continue
        end = subtree_end(ops, i)
        if subtree[i] / total < FOLD_BELOW_SHARE and not any(j in keep for j in range(i, end)):
            hidden.update(range(i, end))
            i = end
        else:
            i += 1
    return hidden


def spine(ops: Sequence[OpProfile], i: int, flags: Sequence[bool], glyphs: dict[str, str]) -> str:
    """The box-drawing prefix for operator `i`: ancestor bars, then its own branch."""
    if ops[i].depth == 0:
        return ""
    bars: list[str] = []
    # One segment per *strict* ancestor below the root: a depth-1 node is drawn flush
    # against a depth-0 root, so ancestor depths run 1 .. depth-1 and never include 0.
    # Getting this wrong indents every node by one level too many and, worse, gives
    # siblings at different depths the same visual indent — which is exactly the confusion
    # a tree is drawn to remove.
    wanted = ops[i].depth - 1
    for j in range(i - 1, -1, -1):
        if wanted < 1:
            break
        if ops[j].depth == wanted:
            bars.append(glyphs["gap"] if flags[j] else glyphs["pipe"])
            wanted -= 1
    return "".join(reversed(bars)) + (glyphs["last"] if flags[i] else glyphs["tee"])


def has_branch(ops: Sequence[OpProfile]) -> bool:
    """Whether any operator has more than one child — whether the plan is a tree at all.

    A hot-path mark on every row of a straight-line plan is decoration, not information:
    the hot path *is* the plan. The mark earns its column only where a reader has to choose
    which side to look at.

    Args:
        ops: The operators, in pre-order.

    Returns:
        True when at least one operator has two or more children.
    """
    return any(
        sum(1 for j in range(i + 1, subtree_end(ops, i)) if ops[j].depth == ops[i].depth + 1) > 1
        for i in range(len(ops))
    )
