#!/usr/bin/env python3
"""Draw `streaming_health.svg` -- the two ways a streaming query goes wrong, over batches.

Source of truth: `docs/user-guide/moving-data/streaming-monitoring.md`, and the fields it
reads: `python/batcher/plan/streaming/progress.py` (`StreamingQueryProgress.behind_by_ms`,
how far a micro-batch overran its trigger interval and `0.0` when it kept up;
`StateOperatorProgress.num_rows_total` / `num_rows_removed`, printed as "rows retained" and
"evicted") and `python/batcher/core/streaming_query/engine.py::_behind_by`.

The figure exists because both failure signals are *trends*, not readings. One long batch
is normal and a `behind_by_ms` that grows batch over batch is not; a retained-row count
that rises and falls with evictions is healthy and one that only rises ends in a
`ResourceError`. A single progress record cannot show either, which is the page's point.

The heights are illustrative shapes, not measurements, and the figure says so. It
deliberately draws no input-rate or processing-rate series: the page does not teach those
fields, and a throughput line is exactly what the page argues cannot answer the question.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, BLUE_MID, GREY, band, label, note, svg, write

W, H = 980, 700

N = 7
SLOT = 108
X0 = 176  # left edge of the first batch slot


def cx(i: int) -> float:
    """Horizontal centre of batch `i` (0-based)."""
    return X0 + SLOT * i + SLOT / 2


# --- panel A: duration against the trigger interval ---------------------------------------
A_BASE = 318  # baseline of the bars
UNIT = 104  # pixels for one trigger interval
INTERVAL_Y = A_BASE - UNIT
DURATIONS = (0.7, 0.78, 1.2, 0.8, 1.14, 1.32, 1.52)
BAR_W = 48


def bar(i: int, d: float) -> str:
    """One micro-batch's duration: blue up to the interval, amber for the overrun."""
    x = cx(i) - BAR_W / 2
    kept = min(d, 1.0) * UNIT
    out = (
        f'<rect x="{x}" y="{A_BASE - kept}" width="{BAR_W}" height="{kept}" rx="3" '
        f'fill="{BLUE_MID}" fill-opacity="0.85"/>'
    )
    if d > 1.0:
        over = (d - 1.0) * UNIT
        out += (
            f'<rect x="{x}" y="{A_BASE - kept - over}" width="{BAR_W}" height="{over}" rx="3" '
            f'fill="{AMBER_DEEP}" fill-opacity="0.9"/>'
            # A white hatch keeps the overrun readable without relying on colour.
            f'<path d="M {x} {A_BASE - kept - 2} L {x + BAR_W} {A_BASE - kept - 2}" '
            f'stroke="#ffffff" stroke-width="1.5" stroke-dasharray="4 3"/>'
        )
    return out


body: list[str] = [
    note(
        490,
        40,
        "Seven micro-batches of one query on a processing-time trigger. "
        "Shapes are illustrative, not measured.",
        anchor="middle",
    ),
    band(20, 62, 940, 322, "IS IT KEEPING UP?", "blue"),
    note(38, 106, "Each bar is one batch's duration, against the trigger interval."),
    # axis and interval line
    f'<path d="M {X0} {A_BASE} L {X0 + SLOT * N} {A_BASE}" stroke="{GREY}" stroke-width="1.5"/>',
    f'<path d="M {X0} {INTERVAL_Y} L {X0 + SLOT * N} {INTERVAL_Y}" stroke="{BLUE}" '
    f'stroke-width="1.8" stroke-dasharray="7 5"/>',
    label(X0 - 12, INTERVAL_Y - 2, "trigger", anchor="end", size=12),
    label(X0 - 12, INTERVAL_Y + 13, "interval", anchor="end", size=12),
    note(X0 - 12, A_BASE - 4, "0", anchor="end"),
]

for i, d in enumerate(DURATIONS):
    body.append(bar(i, d))
    body.append(note(cx(i), A_BASE + 20, f"batch {i + 1}", anchor="middle"))
    behind = "0" if d <= 1.0 else "> 0"
    body.append(label(cx(i), A_BASE + 40, behind, anchor="middle", size=12))

# annotations over the bars
TREND_Y = A_BASE - 1.62 * UNIT - 8
body += [
    label(cx(2), A_BASE - 1.2 * UNIT - 30, "one long batch", anchor="middle", size=12),
    note(cx(2), A_BASE - 1.2 * UNIT - 14, "normal", anchor="middle"),
    f'<path d="M {cx(4) - 24} {TREND_Y} L {cx(6) + 24} {TREND_Y}" '
    f'stroke="{AMBER_DEEP}" stroke-width="1.8"/>',
    label(cx(5), TREND_Y - 26, "overrun grows batch over batch", anchor="middle", size=12),
    note(cx(5), TREND_Y - 10, "falling behind its source", anchor="middle"),
    label(X0 - 12, A_BASE + 40, "behind_by_ms", anchor="end", size=12),
    f'<rect x="40" y="140" width="14" height="14" rx="2" fill="{AMBER_DEEP}"/>',
    label(62, 152, "overrun", size=12),
    note(40, 172, "is_behind is True"),
    note(40, 188, "for that batch"),
]

# --- panel B: rows retained -------------------------------------------------------------
B_BASE = 624
HEALTHY = (0.30, 0.52, 0.44, 0.56, 0.47, 0.58, 0.50)
STALLED = (0.30, 0.52, 0.68, 0.84, 1.0, 1.16, 1.30)
SCALE = 118


def line(values: tuple[float, ...], color: str, dash: str) -> str:
    """A polyline of retained-row counts across the batches, with a dot per batch."""
    pts = [(cx(i), B_BASE - v * SCALE) for i, v in enumerate(values)]
    path = " ".join(f"{'M' if i == 0 else 'L'} {x} {y}" for i, (x, y) in enumerate(pts))
    out = (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.6" '
        f'stroke-dasharray="{dash}" stroke-linejoin="round"/>'
    )
    for x, y in pts:
        out += f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>'
    return out


body += [
    band(20, 400, 940, 280, "IS ITS STATE BOUNDED?", "grey"),
    note(38, 444, "Rows retained by a stateful operator, read from state_operators on each batch."),
    f'<path d="M {X0} {B_BASE} L {X0 + SLOT * N} {B_BASE}" stroke="{GREY}" stroke-width="1.5"/>',
    note(X0 - 12, B_BASE - 4, "0", anchor="end"),
    line(STALLED, AMBER_DEEP, "8 5"),
    line(HEALTHY, BLUE, "0"),
]
for i in range(N):
    body.append(note(cx(i), B_BASE + 20, f"batch {i + 1}", anchor="middle"))

body += [
    f'<path d="M 40 478 L 72 478" stroke="{AMBER_DEEP}" stroke-width="2.6" '
    f'stroke-dasharray="8 5"/>',
    label(80, 482, "stalled", size=12.5),
    note(40, 502, "nothing evicted,"),
    note(40, 518, "only grows: ends"),
    note(40, 534, "in ResourceError"),
    f'<path d="M 40 566 L 72 566" stroke="{BLUE}" stroke-width="2.6"/>',
    label(80, 570, "bounded", size=12.5),
    note(40, 590, "evictions offset"),
    note(40, 606, "what arrives"),
]

write("streaming_health", svg(W, H, "".join(body)))
print("wrote streaming_health.svg")
