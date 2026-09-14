#!/usr/bin/env python3
"""Draw `window_types.svg` -- tumbling, sliding and session windows over one event-time axis.

Source of truth: `python/batcher/plan/functions/temporal.py::window` (``window(ts, '10m')``
is the window *start*, a tumbling grid; ``window(ts, '10m', '5m')`` is the *list* of
overlapping starts, which the caller explodes with `unnest` before grouping),
`python/batcher/core/streaming/folds/windowed.py::_WindowKey` (width and hop are separate,
and hop equals width for a tumbling window), and
`python/batcher/api/dataset/_build/sessions.py` (a session starts where the gap to the
previous event *exceeds* the configured gap, so an interval exactly equal to it does not
split a session).

The figure exists because the difference between the three is a difference in *shape over
time*, which a definition list restates rather than shows. Every row buckets the same eight
events, and each bucket carries the number of events it caught, so the reader can check the
picture rather than take it.
"""

from __future__ import annotations

from _authoring import AMBER, AMBER_DEEP, BLUE_MID, FONT, GREY, band, label, note, svg, write

W, H = 980, 534

X0, PX_PER_MIN = 190, 25
EVENTS = (1, 3, 4, 9, 11, 22, 24, 27)


def mx(minutes: float) -> float:
    """Screen x for an event time in minutes."""
    return X0 + PX_PER_MIN * minutes


def bar(m0: float, m1: float, y: float, h: float, count: int, color: str) -> str:
    """One window, drawn as the interval it covers, carrying the events it caught."""
    x, w = mx(m0) + 2, (m1 - m0) * PX_PER_MIN - 4
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{color}" '
        f'fill-opacity="0.16" stroke="{color}" stroke-width="1.8"/>'
        f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="13" font-weight="700" class="t-title">n = {count}</text>'
    )


body: list[str] = [
    note(
        490,
        62,
        "Eight events on one event-time axis. Each window shows how many of them it caught.",
        anchor="middle",
    ),
]

# Guides first, so the bars drawn over them stay readable while the alignment stays visible.
for m in EVENTS:
    body.append(
        f'<path d="M {mx(m)} 106 V 150 M {mx(m)} 190 V 404" stroke="{GREY}" stroke-width="1" '
        f'stroke-opacity="0.35" stroke-dasharray="3 5"/>'
    )
for m in EVENTS:
    body.append(f'<circle cx="{mx(m)}" cy="100" r="5.5" fill="{AMBER_DEEP}"/>')

body.append(
    f'<path d="M {X0} 124 H 952" stroke="{GREY}" stroke-width="1.6" marker-end="url(#arG)"/>'
)
for m in (0, 5, 10, 15, 20, 25, 30):
    body.append(
        f'<path d="M {mx(m)} 120 V 130" stroke="{GREY}" stroke-width="1.4"/>'
        f'<text x="{mx(m)}" y="144" text-anchor="middle" font-family="{FONT}" font-size="10.5" '
        f'class="t-sub">{m}</text>'
    )
body.append(note(948, 112, "event time (minutes)", anchor="end"))

body.append(band(20, 150, 940, 268, "THE SAME EIGHT EVENTS, BUCKETED THREE WAYS", "blue"))

# Tumbling: a fixed grid, every event in exactly one window.
body += [
    label(30, 210, "TUMBLING"),
    note(30, 228, "window(ts, '10m')"),
    bar(0, 10, 196, 36, 4, BLUE_MID),
    bar(10, 20, 196, 36, 1, BLUE_MID),
    bar(20, 30, 196, 36, 3, BLUE_MID),
]

# Sliding: the same grid plus the offset windows in between, so an event is counted twice.
body += [
    label(30, 276, "SLIDING"),
    note(30, 294, "window(ts,'10m','5m')"),
    note(30, 312, "then unnest"),
    bar(0, 10, 262, 30, 4, BLUE_MID),
    bar(10, 20, 262, 30, 1, BLUE_MID),
    bar(20, 30, 262, 30, 3, BLUE_MID),
    bar(5, 15, 298, 30, 2, AMBER),
    bar(15, 25, 298, 30, 2, AMBER),
]

# Session: no grid at all. The bounds are the data's.
body += [
    label(30, 382, "SESSION"),
    note(30, 400, "session_window(gap='5m')"),
    bar(1, 11, 364, 36, 5, AMBER),
    bar(22, 27, 364, 36, 3, AMBER),
]

body += [
    band(20, 438, 940, 76, "READING IT", "grey"),
    note(
        490,
        482,
        "A sliding window whose hop equals its width is a tumbling window. The second lane holds the offset windows in between, which is why one event is counted twice.",
        anchor="middle",
    ),
    note(
        490,
        500,
        "A session has no grid: it ends when nothing has arrived for the gap, so the data decides both bounds. The gap here is five minutes, and a gap of exactly five does not split one.",
        anchor="middle",
    ),
]

write("window_types", svg(W, H, "".join(body)))
print("wrote window_types.svg")
