#!/usr/bin/env python3
"""Draw `watermark_late_data.svg` -- event time against arrival order, and what is dropped.

Source of truth: `python/batcher/plan/streaming/tracker.py` (`WatermarkTracker`: the
watermark is the running maximum of the *minimum* over per-partition event-time maxima,
less the allowed lateness) and `python/batcher/core/streaming/folds/windowed.py`
(`_WindowedAggFold.push`, which reads the frontier *before* the batch contributes to it,
then keeps only rows whose event time is at or above it, counting the rest into
`num_late_inputs_dropped`). The allowed lateness itself is
`plan/streaming/spec.py::Watermark.of`.

The figure exists because this is the concept prose cannot hold: two different clocks on
two axes, a frontier that moves in steps, and a record whose fate depends on where the
frontier stood when it arrived rather than on anything about the record. The seven
arrivals are chosen so that one row is out of order and survives, and the next is out of
order by more than the lateness and does not -- which is the whole distinction.

Deliberately not drawn: any claim that a late row is buffered, side-output, or recoverable.
The engine filters it out and counts it. That is all it does.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, GREY, band, label, note, svg, write

W, H = 980, 520

X0, X1 = 130, 790          # plot horizontal extent
AXIS_Y = 345               # the arrival-order axis
PX_PER_MIN = 13            # vertical scale
LATENESS_PX = 5 * PX_PER_MIN


def ey(minutes: float) -> float:
    """Screen y for an event time, in minutes either side of 10:00."""
    return 330 - PX_PER_MIN * (minutes + 5)


#: (arrival x, event-time minutes past 10:00, clock label, kept?)
ARRIVALS = (
    (165, 0, "10:00", True),
    (255, 2, "10:02", True),
    (345, 5, "10:05", True),
    (435, 3, "10:03 · kept", True),
    (525, 7, "10:07", True),
    (615, -2, "09:58 · dropped", False),
    (705, 12, "10:12", True),
)

#: The highest event time the stream had seen when each region's arrivals landed, as
#: (x_from, x_to, minutes). The watermark is this line, `LATENESS_PX` lower.
MAX_SEEN = ((215, 345, 0), (345, 435, 2), (435, 615, 5), (615, 760, 7), (760, 790, 12))

body: list[str] = [
    note(490, 50, "Seven records, left to right in the order they arrived. The frontier trails the highest event time seen by the allowed lateness.", anchor="middle"),
    note(60, 78, "event time", anchor="start"),
]

# Horizontal gridlines, one per five minutes, with their clock labels.
for minutes, clock in ((15, "10:15"), (10, "10:10"), (5, "10:05"), (0, "10:00"), (-5, "09:55")):
    y = ey(minutes)
    body.append(
        f'<path d="M {X0} {y} H {X1}" stroke="{GREY}" stroke-width="1" stroke-opacity="0.35"/>'
        f'<text x="{X0 - 10}" y="{y + 4}" text-anchor="end" font-family="{FONT}" font-size="11" '
        f'class="t-sub">{clock}</text>'
    )

# The two step lines. `max event time seen` is grey and dashed; the watermark is the same
# line shifted down by the lateness, drawn solid in blue because it is the one that decides.
max_path = " ".join(f"M {a} {ey(m)} H {b}" for a, b, m in MAX_SEEN)
wm_path = " ".join(f"M {a} {ey(m) + LATENESS_PX} H {b}" for a, b, m in MAX_SEEN)
risers = "".join(
    f'M {MAX_SEEN[i + 1][0]} {ey(MAX_SEEN[i][2])} V {ey(MAX_SEEN[i + 1][2])}'
    for i in range(len(MAX_SEEN) - 1)
)
wm_risers = "".join(
    f'M {MAX_SEEN[i + 1][0]} {ey(MAX_SEEN[i][2]) + LATENESS_PX} '
    f'V {ey(MAX_SEEN[i + 1][2]) + LATENESS_PX}'
    for i in range(len(MAX_SEEN) - 1)
)
body += [
    f'<path d="{max_path} {risers}" fill="none" stroke="{GREY}" stroke-width="2.2" stroke-dasharray="7 4"/>',
    f'<path d="{wm_path} {wm_risers}" fill="none" stroke="{BLUE}" stroke-width="3"/>',
    f'<text x="798" y="{ey(12) + 4}" font-family="{FONT}" font-size="11.5" class="t-sub">max event time seen</text>',
    f'<text x="798" y="{ey(12) + LATENESS_PX + 4}" font-family="{FONT}" font-size="12.5" '
    f'font-weight="700" class="t-arrow">watermark</text>',
]

# The gap between the two lines is the allowed lateness, so measure it once, explicitly.
gap_x, top, bottom = 480, ey(5), ey(5) + LATENESS_PX
body += [
    f'<path d="M {gap_x} {top} V {bottom}" stroke="{AMBER_DEEP}" stroke-width="2"/>'
    f'<path d="M {gap_x - 6} {top} H {gap_x + 6} M {gap_x - 6} {bottom} H {gap_x + 6}" '
    f'stroke="{AMBER_DEEP}" stroke-width="2"/>',
    label(gap_x + 12, (top + bottom) / 2 + 4, "allowed lateness · 5 min"),
]

# The arrivals. A kept row is a filled dot; a dropped one is a struck-through ring, so the
# distinction survives a greyscale print and does not rest on colour.
for i, (x, minutes, clock, kept) in enumerate(ARRIVALS, start=1):
    y = ey(minutes)
    if kept:
        body.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{BLUE}"/>')
    else:
        body.append(
            f'<circle cx="{x}" cy="{y}" r="7.5" fill="none" stroke="{AMBER_DEEP}" stroke-width="2.4"/>'
            f'<path d="M {x - 7} {y + 7} L {x + 7} {y - 7}" stroke="{AMBER_DEEP}" stroke-width="2.4"/>'
        )
    ty = y + 22 if minutes in (3, -2) else y - 14
    body.append(
        f'<text x="{x}" y="{ty}" text-anchor="middle" font-family="{FONT}" font-size="11" '
        f'class="t-sub">{clock}</text>'
    )
    body.append(
        f'<text x="{x}" y="{AXIS_Y + 17}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="10.5" class="t-sub">{i}</text>'
    )

body += [
    f'<path d="M {X0} {AXIS_Y} H {X1 + 10}" stroke="{GREY}" stroke-width="1.6" marker-end="url(#arG)"/>',
    note(465, AXIS_Y + 37, "arrival order (processing time)", anchor="middle"),
    band(20, 396, 940, 104, "WHAT THE ENGINE DOES WITH EACH ROW", "grey"),
    note(490, 440, "A row is compared against the frontier as it stood before its own batch, so record 4 arrives out of order and still counts.", anchor="middle"),
    note(490, 458, "Across several partitions the watermark is the minimum of their maxima, so one slow partition holds the whole frontier back.", anchor="middle"),
    note(490, 476, "Record 6 is filtered out in Rust and never reaches the aggregate. It is counted: num_late_inputs_dropped on the progress record.", anchor="middle"),
]

write("watermark_late_data", svg(W, H, "".join(body)))
print("wrote watermark_late_data.svg")
