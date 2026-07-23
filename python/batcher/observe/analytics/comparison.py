"""Run-against-run comparison — what changed between two executions.

Steps are matched by `op_id`, which is only meaningful when both runs share a plan shape;
the report says so explicitly rather than lining up two unrelated plans side by side and
letting the reader assume the rows correspond.
"""

from __future__ import annotations

from typing import Any

__all__ = ["compare_runs"]


def compare_runs(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """A field-by-field and step-by-step diff of two runs.

    Matching is by `op_id`, which is the same pre-order index the engine measured against —
    so "the join" in one run and "the join" in the other are genuinely the same plan
    position, not two operators that happen to share a name. Runs with different plan shapes
    are reported as such rather than diffed into nonsense.

    Args:
        a: The baseline run's detail document.
        b: The run being compared against it.

    Returns:
        ``{"ok": bool, "reason": str, "totals": [...], "steps": [...]}``.
    """
    if not a or not b:
        return {
            "ok": False,
            "reason": "One of the runs is no longer retained.",
            "totals": [],
            "steps": [],
        }
    if a.get("signature") and a["signature"] != b.get("signature"):
        return {
            "ok": False,
            "reason": "These runs have different plan shapes, so their steps do not correspond.",
            "totals": _total_deltas(a, b),
            "steps": [],
        }
    nodes_a = {n["op_id"]: n for n in (a.get("dag") or {}).get("nodes", [])}
    nodes_b = {n["op_id"]: n for n in (b.get("dag") or {}).get("nodes", [])}
    steps = []
    for op_id in sorted(set(nodes_a) | set(nodes_b)):
        left, right = nodes_a.get(op_id), nodes_b.get(op_id)
        steps.append(
            {
                "op_id": op_id,
                "kind": (right or left or {}).get("kind", "?"),
                "detail": (right or left or {}).get("detail", ""),
                "a_ms": (left or {}).get("elapsed_ms"),
                "b_ms": (right or {}).get("elapsed_ms"),
                "a_rows": (left or {}).get("rows_out"),
                "b_rows": (right or {}).get("rows_out"),
                "delta_ms": _delta((left or {}).get("elapsed_ms"), (right or {}).get("elapsed_ms")),
                "ratio": _ratio((left or {}).get("elapsed_ms"), (right or {}).get("elapsed_ms")),
            }
        )
    steps.sort(key=lambda s: abs(s["delta_ms"] or 0.0), reverse=True)
    return {"ok": True, "reason": "", "totals": _total_deltas(a, b), "steps": steps}


def _total_deltas(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Whole-run metrics side by side, with their change."""
    fields = [
        ("Duration", "total_ms", "ms"),
        ("Rows returned", "rows", "rows"),
    ]
    out = []
    for label, key, unit in fields:
        left, right = a.get(key), b.get(key)
        out.append(
            {
                "label": label,
                "unit": unit,
                "a": left,
                "b": right,
                "delta": _delta(left, right),
                "ratio": _ratio(left, right),
            }
        )
    return out


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else float(b) - float(a)


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not a:
        return None
    return float(b) / float(a)
