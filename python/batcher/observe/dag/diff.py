"""What the optimizer actually did — the logical plan against the one that ran.

Batcher's whole claim rests on the plan it chose being better than the plan you wrote, and
until now the dashboard showed only the second of those. Both documents are already on
every profile (`logical_ir` and `optimized_ir`), so the difference between them is a fact
sitting unread, not a thing to model: this module reads it.

The comparison is structural, never textual. Operators are matched by ``(kind, detail)``
and then by where they sit in the tree, so "the filter moved below the join" is reported as
one move rather than as a removal plus an unrelated addition. Anything that cannot be
matched is reported as added or removed — never guessed at, and never smoothed over.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from batcher.observe.dag.describe import describe, kind_of
from batcher.plan.profile import walk_ir

__all__ = ["plan_diff"]


def plan_diff(logical: dict[str, Any] | None, optimized: dict[str, Any] | None) -> dict[str, Any]:
    """Compare the plan as written against the plan the engine ran.

    Args:
        logical: The plan IR before optimization, or None when it was not recorded.
        optimized: The plan IR the engine executed, or None.

    Returns:
        ``{"available": bool, "identical": bool, "changes": [...], "counts": [...],
        "before_ops": int, "after_ops": int}``. `available` is False when either document
        is missing, so the caller renders "not recorded" rather than "no changes" — those
        are different statements and conflating them is a lie about the optimizer.
    """
    if not logical or not optimized:
        return {
            "available": False,
            "identical": False,
            "changes": [],
            "counts": [],
            "before_ops": 0,
            "after_ops": 0,
        }
    before = _entries(logical)
    after = _entries(optimized)
    changes = _match(before, after)
    return {
        "available": True,
        "identical": not changes,
        "changes": changes,
        "counts": _counts(before, after),
        "before_ops": len(before),
        "after_ops": len(after),
        "summary": _summary(changes, len(before), len(after)),
    }


def _summary(changes: list[dict[str, Any]], before_ops: int, after_ops: int) -> str:
    """One sentence a reader can stop at, built only from what the diff actually found."""
    if not changes:
        return "The optimizer left the plan as written."
    primary = [c for c in changes if c["primary"]]
    parts: list[str] = []
    moved = [c for c in primary if c["change"] == "moved"]
    added = [c for c in primary if c["change"] == "added"]
    removed = [c for c in primary if c["change"] == "removed"]
    if moved:
        parts.append(f"reordered {_plural(len(moved), 'step')}")
    if removed:
        parts.append(f"removed {_plural(len(removed), 'step')}")
    if added:
        parts.append(f"added {_plural(len(added), 'step')}")
    if not parts:
        # Every change was a knock-on of another; say that rather than claiming nothing
        # happened, which the `identical` flag already covers and this is not.
        return "The optimizer reshaped the plan without changing which steps run."
    net = after_ops - before_ops
    tail = "" if net == 0 else f", {abs(net)} {'fewer' if net < 0 else 'more'} in total"
    return f"The optimizer {_and_list(parts)}{tail}."


def _plural(n: int, noun: str) -> str:
    """``1 step`` / ``3 steps`` — never ``1 steps``."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _entries(ir: dict[str, Any]) -> list[dict[str, Any]]:
    """Every operator with its identity and the chain of operators above it."""
    out: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    for op_id, (depth, node) in enumerate(walk_ir(ir)):
        # walk_ir is pre-order, so anything on the stack at a depth >= this node's is a
        # sibling or a cousin, never an ancestor.
        del stack[depth:]
        kind = kind_of(node)
        out.append(
            {
                "op_id": op_id,
                "depth": depth,
                "kind": kind,
                "detail": describe(kind, node),
                "path": [k for _d, k in stack],
            }
        )
        stack.append((depth, kind))
    return out


def _match(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair up operators between the two plans and describe every unpaired or moved one."""
    lhs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rhs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in before:
        lhs[(entry["kind"], entry["detail"])].append(entry)
    for entry in after:
        rhs[(entry["kind"], entry["detail"])].append(entry)

    changes: list[dict[str, Any]] = []
    for key in sorted(set(lhs) | set(rhs)):
        old, new = lhs.get(key, []), rhs.get(key, [])
        kind, detail = key
        for a, b in zip(old, new, strict=False):
            move = _move(a["path"], b["path"])
            if move:
                changes.append(
                    {
                        "change": "moved",
                        "kind": kind,
                        "detail": detail,
                        "op_id": b["op_id"],
                        "before_path": a["path"],
                        "after_path": b["path"],
                        **move,
                    }
                )
        for a in old[len(new) :]:
            changes.append(
                {
                    "change": "removed",
                    "kind": kind,
                    "detail": detail,
                    "op_id": None,
                    "note": _removed_note(kind),
                    "primary": True,
                    "before_path": a["path"],
                    "after_path": [],
                }
            )
        for b in new[len(old) :]:
            changes.append(
                {
                    "change": "added",
                    "kind": kind,
                    "detail": detail,
                    "op_id": b["op_id"],
                    "note": _added_note(kind),
                    "primary": True,
                    "before_path": [],
                    "after_path": b["path"],
                }
            )
    # Primary first, then moves before edits: one pushdown shows up as a move for the step
    # that was pushed *and* as a knock-on for everything it passed, and a reader scanning
    # the top of the list must meet the rewrite rather than its consequences.
    order = {"moved": 0, "removed": 1, "added": 2}
    changes.sort(key=lambda c: (not c["primary"], order[c["change"]], c["kind"]))
    return changes


def _move(before_path: list[str], after_path: list[str]) -> dict[str, Any] | None:
    """Describe a relocation, or `None` when the operator did not move.

    Stated in execution order, not tree order. An operator gaining an ancestor now runs
    *before* that ancestor, which is what "pushed below" means to a reader and the opposite
    of what "deeper in the tree" sounds like.

    Only a step that *gained* an ancestor is `primary`. A pushdown moves one operator and
    drags every operator it passed into a different position, so reporting all of them as
    equals turns one rewrite into four findings and buries the one that explains the others.
    """
    if before_path == after_path:
        return None
    gained = [k for k in after_path if k not in before_path]
    lost = [k for k in before_path if k not in after_path]
    if gained:
        return {"note": f"now runs before {_and_list(gained)}", "primary": True}
    if lost:
        return {"note": f"no longer runs before {_and_list(lost)}", "primary": False}
    return {"note": "reordered within the plan", "primary": False}


def _added_note(kind: str) -> str:
    """Why the optimizer inserted an operator that was not written."""
    return _ADDED.get(kind, "the optimizer inserted this step")


def _removed_note(kind: str) -> str:
    """Why the optimizer dropped an operator that was written."""
    return _REMOVED.get(kind, "the optimizer proved this step was unnecessary")


_ADDED = {
    "project": "the optimizer narrowed the columns carried through the plan",
    "filter": "the optimizer derived this predicate from the ones you wrote",
    "limit": "the optimizer pushed a row cap into this branch",
    "distinct": "the optimizer proved duplicates could be dropped here",
}

_REMOVED = {
    "project": "every column it selected was already the only one carried",
    "filter": "the predicate was folded into another step or proved always true",
    "sort": "the order it established was already guaranteed",
    "distinct": "the input was already unique on those keys",
    "limit": "the cap was absorbed into the step beneath it",
}


def _and_list(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` — a human list, not a repr."""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _counts(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-operator-kind tallies on both sides, biggest change first."""
    lhs = Counter(e["kind"] for e in before)
    rhs = Counter(e["kind"] for e in after)
    rows = [
        {
            "kind": kind,
            "before": lhs.get(kind, 0),
            "after": rhs.get(kind, 0),
            "delta": rhs.get(kind, 0) - lhs.get(kind, 0),
        }
        for kind in sorted(set(lhs) | set(rhs))
    ]
    rows.sort(key=lambda r: (-abs(r["delta"]), r["kind"]))
    return rows
