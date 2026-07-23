"""Findings about rows: how many were read, discarded, or multiplied.

All three read row counts rather than timings, which is what lets them catch shapes that
are invisible in a plan's structure — an exploding join looks identical to a normal one
until you read its edges."""

from __future__ import annotations

from typing import Any

from .kinds import (
    _EXPLODING_JOIN_MIN_ROWS,
    _EXPLODING_JOIN_RATIO,
    _LATE_FILTER_KEEP,
    _LATE_FILTER_MIN_MS,
    _LATE_FILTER_MIN_ROWS,
    _WIDE_SCAN_MIN_ROWS,
    _WIDE_SCAN_RATIO,
    Insight,
    count,
)

__all__ = ["exploding_join", "late_filter", "wide_scan"]


def wide_scan(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """A scan feeding a highly selective filter — the classic missed-pushdown shape.

    Selectivity is measured **at the filter**, not at the query's output. Those differ
    whenever anything else reduces rows: a join + filter + `GROUP BY` returning 3 rows from
    420K looks maximally selective end-to-end, but its filter kept 380K of 400K — the
    aggregate did the reducing, and no pushdown would have helped. Comparing scan rows to
    final rows credited the filter for the aggregate's work and fired on almost every
    grouped query.

    Pushdown is also the only remedy this advice can offer, which is why a filter must exist
    at all: a top-N has to read everything to rank it, and a `GROUP BY` has to read
    everything to group it.
    """
    filters = [op for op in ops if op.get("kind") == "filter"]
    scans = [op for op in ops if op.get("kind") == "scan"]
    if not filters or not scans:
        return []
    read = sum(int(op.get("rows_out", 0)) for op in scans)
    kept = sum(int(op.get("rows_out", 0)) for op in filters)
    if read < _WIDE_SCAN_MIN_ROWS or kept <= 0 or read < kept * _WIDE_SCAN_RATIO:
        return []
    return [
        Insight(
            severity="info",
            rule="selective-query",
            title=f"Scanned {count(read)} rows to keep {count(kept)}",
            evidence=(
                f"The scan(s) read {count(read)} rows and the filter kept {count(kept)} "
                f"— a {read / kept:.0f}:1 ratio, so most of that read was discarded."
            ),
            action=(
                "If the data is partitioned or carries zone maps, a predicate on the "
                "partition/sort column lets the reader skip files without decoding them; "
                "otherwise consider storing it sorted on the filtered column."
            ),
            detail={"rows_scanned": read, "rows_kept": kept},
        )
    ]


def exploding_join(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """A join emitting far more rows than it consumed.

    A join whose output exceeds its input means the key was non-unique on both sides, so
    every match multiplied. It is the shape behind most accidental Cartesian products, and it
    is invisible in the *structure* of a plan — the node count is identical whether the join
    emitted a thousand rows or a billion. Only the row counts show it.
    """
    found: list[Insight] = []
    for op in ops:
        if op.get("kind") != "hash_join":
            continue
        rows_in = int(op.get("rows_in", 0) or 0)
        rows_out = int(op.get("rows_out", 0) or 0)
        if rows_in < _EXPLODING_JOIN_MIN_ROWS or rows_out < rows_in * _EXPLODING_JOIN_RATIO:
            continue
        found.append(
            Insight(
                severity="warning",
                rule="exploding-join",
                title=f"A join multiplied {count(rows_in)} rows into {count(rows_out)}",
                evidence=(
                    f"Step {op.get('op_id')} took in {count(rows_in)} rows and emitted "
                    f"{count(rows_out)} — {rows_out / max(rows_in, 1):.1f}x. That happens when "
                    f"the join key is not unique on either side, so every match multiplies."
                ),
                action=(
                    "Check the join key is as selective as you expect. If duplicates are "
                    "legitimate, deduplicate or aggregate the smaller side before joining, so "
                    "the multiplication happens on fewer rows."
                ),
                detail={"op_id": op.get("op_id"), "rows_in": rows_in, "rows_out": rows_out},
            )
        )
    return found


def late_filter(
    _profile: dict[str, Any], ops: list[dict[str, Any]], _total_ms: float
) -> list[Insight]:
    """A selective filter running after an expensive step.

    Rows discarded by a filter are rows every step beneath it processed for nothing. The
    cost is not the filter — it is everything that ran before it on rows that were never
    going to survive.
    """
    by_id = {int(op.get("op_id", -1)): op for op in ops}
    found: list[Insight] = []
    for op in ops:
        if op.get("kind") != "filter":
            continue
        rows_in = int(op.get("rows_in", 0) or 0)
        rows_out = int(op.get("rows_out", 0) or 0)
        if rows_in < _LATE_FILTER_MIN_ROWS or rows_out > rows_in * _LATE_FILTER_KEEP:
            continue
        # Only interesting when something costly ran below it — a selective filter directly
        # above a scan is exactly where it should be.
        below = [
            other
            for other in by_id.values()
            if int(other.get("op_id", -1)) > int(op.get("op_id", -1))
            and other.get("kind") in {"hash_join", "aggregate", "sort", "window"}
        ]
        if not below:
            continue
        wasted = sum(float(o.get("elapsed_ms", 0.0) or 0.0) for o in below)
        if wasted < _LATE_FILTER_MIN_MS:
            continue
        found.append(
            Insight(
                severity="warning",
                rule="late-filter",
                title=f"A filter discarding {1 - rows_out / max(rows_in, 1):.0%} runs late",
                evidence=(
                    f"Step {op.get('op_id')} kept {count(rows_out)} of {count(rows_in)} rows, "
                    f"but {len(below)} costly step(s) below it spent {wasted:.0f} ms on rows it "
                    f"then discarded."
                ),
                action=(
                    "Move the predicate earlier if the query allows it. A filter the optimizer "
                    "could not push down is often one wrapped in a function — comparing the "
                    "bare column instead usually lets it move."
                ),
                detail={"op_id": op.get("op_id"), "wasted_ms": round(wasted, 1)},
            )
        )
    return found
