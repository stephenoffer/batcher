"""The plan as text — the EXPLAIN output every SQL engine's users already know how to read.

A graph shows shape; a text tree shows *nesting*, and it is the form that pastes into an
issue, diffs in a terminal, and is searchable with the browser's own find. DuckDB, Spark,
Postgres, and Polars all ship one, and a reader arriving from any of them looks for it
first. The graph and this tree are the same walk over the same IR, so an operator's id, its
subtitle, and its numbers are identical in both by construction rather than by care.

Rendered here rather than in the browser for the same reason the layout is: one answer to
"what does this plan say", shared by the dashboard, a future CLI, and anything else that
asks. Nothing is invented — an operator with no measurement prints no timing rather than a
zero, because a printed `0.0ms` is a claim that the step was instant.
"""

from __future__ import annotations

from typing import Any

from batcher.observe.dag.describe import children, describe, kind_of
from batcher.plan.profile import walk_ir

__all__ = ["explain_rows", "explain_text"]

#: Box-drawing for the tree spine. Plain ASCII is available via `ascii=True` for a terminal
#: that cannot encode these — the same fallback the console reporter makes.
_GLYPHS = {"tee": "├─ ", "last": "└─ ", "pipe": "│  ", "gap": "   "}
_ASCII = {"tee": "|- ", "last": "`- ", "pipe": "|  ", "gap": "   "}


def explain_rows(
    ir: dict[str, Any] | None, ops: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """The plan as flat rows in pre-order, each with the spine prefix already computed.

    Flat rather than nested so the browser can render it as a plain list — an interactive
    tree needs to filter, highlight, and virtualize rows, all of which a nested structure
    makes harder for nothing. `prefix` carries the box-drawing so the client never
    re-derives which branches are still open.

    Args:
        ir: A relational plan IR document, or None.
        ops: The per-operator profile dicts, joined by ``op_id``. Optional — a plan that
            never ran still explains, it just carries no measurements.

    Returns:
        One dict per operator: ``op_id``, ``depth``, ``kind``, ``detail``, ``prefix``,
        ``last``, and the measured fields when `ops` supplied them.
    """
    if not ir:
        return []
    by_id = {int(op.get("op_id", -1)): op for op in (ops or [])}
    walked = list(walk_ir(ir))
    index = {id(node): op_id for op_id, (_depth, node) in enumerate(walked)}
    # Which nodes are the last child of their parent: that is what turns a "├─" into a "└─"
    # and closes the vertical bar beneath it.
    last_child: set[int] = set()
    for _depth, node in walked:
        kids = [index[id(c)] for c in children(node) if id(c) in index]
        if kids:
            last_child.add(kids[-1])
    roots = {op_id for op_id, (depth, _node) in enumerate(walked) if depth == 0}
    last_child |= roots

    rows: list[dict[str, Any]] = []
    # `open_at[d]` is True while depth `d` still has siblings below it to draw a bar for.
    open_at: list[bool] = []
    for op_id, (depth, node) in enumerate(walked):
        while len(open_at) <= depth:
            open_at.append(False)
        del open_at[depth + 1 :]
        open_at[depth] = op_id not in last_child
        kind = kind_of(node)
        op = by_id.get(op_id)
        rows.append(
            {
                "op_id": op_id,
                "depth": depth,
                "kind": kind,
                "detail": describe(kind, node),
                "last": op_id in last_child,
                "ancestors": list(open_at[:depth]),
                "measured": bool(op and op.get("measured")),
                "elapsed_ms": float(op.get("elapsed_ms", 0.0)) if op else None,
                "rows_out": int(op.get("rows_out", 0)) if op else None,
                "est_rows": op.get("est_rows") if op else None,
                "spilled": bool(op and op.get("spilled")),
                "algorithm": str(op.get("algorithm", "")) if op else "",
                "provenance": str(op.get("provenance", "")) if op else "",
            }
        )
    return rows


def explain_text(
    ir: dict[str, Any] | None,
    ops: list[dict[str, Any]] | None = None,
    *,
    ascii_only: bool = False,
) -> str:
    """The plan as a single copy-pasteable text tree.

    Args:
        ir: A relational plan IR document, or None.
        ops: The per-operator profile dicts, joined by ``op_id``.
        ascii_only: Use ASCII spine glyphs instead of box drawing.

    Returns:
        The rendered tree, or ``""`` when there is no plan.
    """
    rows = explain_rows(ir, ops)
    if not rows:
        return ""
    glyphs = _ASCII if ascii_only else _GLYPHS
    out: list[str] = []
    for row in rows:
        spine = "".join(glyphs["pipe"] if open_ else glyphs["gap"] for open_ in row["ancestors"])
        branch = "" if row["depth"] == 0 else (glyphs["last"] if row["last"] else glyphs["tee"])
        head = f"{spine}{branch}{row['kind']}"
        if row["detail"]:
            head += f"  [{row['detail']}]"
        out.append(head)
        note = _annotation(row)
        if note:
            pad = "".join(glyphs["pipe"] if open_ else glyphs["gap"] for open_ in row["ancestors"])
            pad += "" if row["depth"] == 0 else glyphs["gap"]
            out.append(f"{pad}   {note}")
    return "\n".join(out)


def _annotation(row: dict[str, Any]) -> str:
    """The measurement line under an operator, or ``""`` when nothing was measured.

    Every clause appears only when its value is real. A plan explained without `analyze`
    prints its estimate and nothing else, rather than a row of confident zeroes.
    """
    parts: list[str] = []
    if row["measured"]:
        parts.append(f"{row['elapsed_ms']:.3g} ms")
        parts.append(f"{row['rows_out']:,} rows")
    if row["est_rows"] is not None:
        parts.append(f"est {round(float(row['est_rows'])):,}")
    if row["algorithm"]:
        parts.append(row["algorithm"])
    if row["spilled"]:
        parts.append("spilled")
    return " · ".join(parts)
