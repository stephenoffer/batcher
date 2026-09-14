#!/usr/bin/env python3
"""Draw `window_frame_eval.svg` -- what a tie in the ORDER BY key does to a frame.

Source of truth: `crates/bc-runtime/src/window/mod.rs` -- `ordered_partitions_by_global_sort`
(:652), which sorts every row once by (partition keys, then order keys) rather than sorting
each partition; and `crates/bc-runtime/src/window/frame/bounds.rs` -- `FrameUnit` (:50),
`frame_bounds` (:245), `frame_half_open` (:363, the ROWS arithmetic), the peer-group arm
(:262-300, the RANGE and GROUPS expansion), `PeerGroups::new` (:203) and `frame_ctx` (:420),
which builds peer groups for RANGE and GROUPS and returns None for ROWS.

The figure exists for one thing prose cannot hold: the same row, the same bound spelling,
and two different frames, because RANGE ends at the last row of the peer group and ROWS
ends at the row itself. Everything else on the page is a list.

Accuracy notes carried into the drawing:
  * ROWS builds no peer structure at all -- `frame_ctx` returns None for it.
  * `RANGE UNBOUNDED PRECEDING TO CURRENT ROW` is SQL's default frame, and it is handled
    by `window::running_aggregate` rather than by this kernel; the peer rule is the same.
  * A numeric RANGE offset is a different mechanism again: a binary search over the key's
    values (`value_range_bounds`, bounds.rs:314), not a walk over peer groups.
"""

from __future__ import annotations

from _authoring import AMBER, AMBER_DEEP, BLUE, BLUE_MID, FONT, band, label, note, svg, write

W, H = 980, 520

X0, CW, STEP, CY, CH = 85, 96, 102, 168, 58
VALUES = (1, 2, 2, 2, 5, 5, 7, 9)
CURRENT = 2  # the middle of the three tied rows


def cell(i: int, value: int, highlight: bool) -> str:
    """One ordered row of the partition, showing its ORDER BY key."""
    x = X0 + i * STEP
    stroke, width = (AMBER_DEEP, 2.6) if highlight else (BLUE_MID, 1.4)
    return (
        f'<rect x="{x}" y="{CY}" width="{CW}" height="{CH}" rx="5" fill="{BLUE_MID}" '
        f'fill-opacity="0.16" stroke="{stroke}" stroke-width="{width}"/>'
        f'<text x="{x + CW / 2}" y="{CY + 30}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="15" font-weight="700" class="t-title">t = {value}</text>'
        f'<text x="{x + CW / 2}" y="{CY + 48}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="10.5" class="t-sub">row {i}</text>'
    )


def span(i0: int, i1: int, y: float, color: str, caption: str) -> str:
    """A frame drawn as the extent it actually covers, with end ticks."""
    x1, x2 = X0 + i0 * STEP, X0 + i1 * STEP + CW
    return (
        f'<path d="M {x1} {y} H {x2}" stroke="{color}" stroke-width="6" stroke-linecap="round"/>'
        f'<path d="M {x1} {y - 9} V {y + 9} M {x2} {y - 9} V {y + 9}" stroke="{color}" stroke-width="2.4"/>'
        f'<text x="{x2 + 14}" y="{y + 5}" font-family="{FONT}" font-size="12.5" font-weight="700" '
        f'class="t-arrow">{caption}</text>'
    )


body: list[str] = [
    band(20, 24, 940, 96, "ONE GLOBAL SORT, NOT ONE SORT PER PARTITION", "grey"),
    note(
        490,
        62,
        "The rows are sorted once by the PARTITION BY keys and then the ORDER BY keys. "
        "The partition keys lead, so",
        anchor="middle",
    ),
    note(
        490,
        80,
        "every partition comes out contiguous, and a near-unique key does not cost "
        "a million tiny sorts.",
        anchor="middle",
    ),
    note(
        490,
        104,
        "Each function's output column is then scattered back to the row it came from, "
        "in the original row order.",
        anchor="middle",
    ),
    band(20, 136, 940, 248, "ONE ORDERED PARTITION, ONE ROW, TWO FRAMES", "blue"),
]

body += [cell(i, v, i == CURRENT) for i, v in enumerate(VALUES)]

# The tie that the whole figure is about.
body += [
    f'<path d="M 187 {CY - 12} H 487" stroke="{AMBER}" stroke-width="3"/>',
    f'<path d="M 187 {CY - 12} V {CY - 4} M 487 {CY - 12} V {CY - 4}" stroke="{AMBER}" stroke-width="3"/>',
    label(
        337, CY - 20, "peer group: three rows tied on the ORDER BY key", anchor="middle", size=11.5
    ),
    label(337, CY + CH + 22, "the current row", anchor="middle", size=11.5),
    f'<path d="M 337 {CY + CH + 2} V {CY + CH + 10}" stroke="{AMBER_DEEP}" stroke-width="2.4"/>',
]

# The two frames, same spelling of the bound, different extent.
body += [
    span(0, CURRENT, 278, BLUE, "ROWS ... CURRENT ROW"),
    note(
        X0,
        300,
        "stops at this row: pure position arithmetic, so each tied row gets a different answer",
    ),
    span(0, 3, 336, AMBER_DEEP, "RANGE ... CURRENT ROW"),
    note(X0, 358, "runs to the end of the peer group, so all three tied rows get the same answer"),
]

body += [
    band(20, 400, 940, 100, "THE REST OF THE FAMILY", "amber"),
    note(
        490,
        438,
        "GROUPS counts peer groups the way ROWS counts rows. A numeric RANGE offset is a "
        "binary search over the",
        anchor="middle",
    ),
    note(
        490,
        456,
        "key's values rather than a walk over peers, and a null order key frames only its "
        "own null peer group.",
        anchor="middle",
    ),
    note(
        490,
        480,
        "Both frame edges only ever slide right, so a frame is a FIFO queue and no frame is "
        "ever rescanned.",
        anchor="middle",
    ),
]

write("window_frame_eval", svg(W, H, "".join(body)))
print("wrote window_frame_eval.svg")
