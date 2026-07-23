"""Translate a pushed predicate into the message-index filters a robot log supports.

A drive log is a time-ordered multiplexed stream: one file carries every sensor, and a
query wants two of the hundred topics over five seconds of a two-hour drive. Both of those
are answerable from the container's *index* — MCAP records a per-channel offset index and a
message time range — so pushing them down turns a full-file scan into a seek.

This is a pure IR→filter translation with no reader in it, which is what lets the boundary
semantics below be tested as arithmetic rather than against a file.

**`end_time` is exclusive; `start_time` is inclusive.** Verified against the reader, not
assumed — the difference is one message at each boundary, and a scan that silently drops
the last message of a window is exactly the bug that never surfaces in a row count.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TimeRange", "message_filters"]

# Nanoseconds per microsecond — the IR carries timestamp literals in micros
# (`io/predicate.py::_literal`), MCAP indexes in nanos.
_NANOS_PER_MICRO = 1_000

TimeRange = tuple[int | None, int | None]


def message_filters(
    ir: dict[str, Any] | None, *, topic_column: str, time_column: str
) -> tuple[list[str] | None, TimeRange]:
    """The `(topics, (start_time, end_time))` a predicate justifies pushing to the reader.

    Only conjunctions are mined: under an `AND`, each conjunct that is understood tightens
    the filter and the rest are left to the engine. A disjunction is *not* mined at all —
    `topic = '/a' OR log_time > t` would otherwise push a topic filter that drops rows the
    predicate keeps. Ignoring a term is always safe (the engine's `Filter` re-checks every
    row); pushing one that is too narrow is not.

    Args:
        ir: The pushed predicate as its IR dictionary, or None.
        topic_column: The column whose equality/membership becomes a topic filter.
        time_column: The column whose range becomes a log-time filter.

    Returns:
        `(topics, (start_time, end_time))`, each None where the predicate says nothing.
        Times are nanoseconds; `start_time` is inclusive and `end_time` exclusive.
    """
    topics: list[str] | None = None
    start: int | None = None
    end: int | None = None
    for conjunct in _conjuncts(ir):
        got_topics = _topics_of(conjunct, topic_column)
        if got_topics is not None:
            # Two topic filters in one predicate intersect; keep the narrower.
            topics = got_topics if topics is None else [t for t in topics if t in set(got_topics)]
            continue
        lo, hi = _time_bounds(conjunct, time_column)
        if lo is not None:
            start = lo if start is None else max(start, lo)
        if hi is not None:
            end = hi if end is None else min(end, hi)
    return topics, (start, end)


def _conjuncts(ir: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten an `AND` tree into its terms; anything else is a single opaque term."""
    if not ir:
        return []
    if ir.get("e") == "binary" and ir.get("op") == "and":
        return _conjuncts(ir.get("left")) + _conjuncts(ir.get("right"))
    return [ir]


def _topics_of(ir: dict[str, Any], column: str) -> list[str] | None:
    """`topic = 'x'` or `topic IN ('x','y')` → the topics, else None."""
    if ir.get("e") == "binary" and ir.get("op") == "eq":
        for side, other in ((ir.get("left"), ir.get("right")), (ir.get("right"), ir.get("left"))):
            if _is_column(side, column):
                value = _string_literal(other)
                return None if value is None else [value]
        return None
    if ir.get("e") == "in_list" and _is_column(ir.get("input"), column):
        values = [_string_literal(item) for item in ir.get("list") or []]
        return None if any(v is None for v in values) else values  # type: ignore[misc]
    return None


def _time_bounds(ir: dict[str, Any], column: str) -> TimeRange:
    """A comparison against `column` → its `(start, end)` contribution in nanoseconds."""
    if ir.get("e") != "binary":
        return (None, None)
    op = ir.get("op")
    left, right = ir.get("left"), ir.get("right")
    if _is_column(left, column):
        nanos = _time_literal(right)
    elif _is_column(right, column):
        nanos = _time_literal(left)
        # `5 < log_time` is `log_time > 5`: flipping the operands flips the operator.
        op = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}.get(op, op)
    else:
        return (None, None)
    if nanos is None:
        return (None, None)
    # `start` is inclusive and `end` exclusive, so a strict/non-strict bound differs by one
    # nanosecond — the smallest step the index can express.
    if op == "ge":
        return (nanos, None)
    if op == "gt":
        return (nanos + 1, None)
    if op == "le":
        return (None, nanos + 1)
    if op == "lt":
        return (None, nanos)
    if op == "eq":
        return (nanos, nanos + 1)
    return (None, None)


def _is_column(ir: Any, name: str) -> bool:
    return isinstance(ir, dict) and ir.get("e") == "col" and ir.get("name") == name


def _string_literal(ir: Any) -> str | None:
    if not isinstance(ir, dict) or ir.get("e") != "lit":
        return None
    value = ir.get("value") or {}
    got = value.get("str")
    return got if isinstance(got, str) else None


def _time_literal(ir: Any) -> int | None:
    """A timestamp or integer literal as nanoseconds, or None if it is neither."""
    if not isinstance(ir, dict) or ir.get("e") != "lit":
        return None
    value = ir.get("value") or {}
    if "timestamp" in value:  # the IR carries timestamps in microseconds
        return int(value["timestamp"]) * _NANOS_PER_MICRO
    if "int" in value:  # a bare integer is taken as nanoseconds, as the column is
        return int(value["int"])
    return None
