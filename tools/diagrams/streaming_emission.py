#!/usr/bin/env python3
"""Draw `streaming_emission.svg` -- what three shapes emit on each trigger of one stream.

Source of truth: `docs/user-guide/moving-data/streaming-emission.md` (row-wise shapes emit
per window as rows arrive; a watermarked `window(...)` group key emits as the watermark
closes each window; `group_by(...).agg(...)` with no watermark emits once, at end of input)
and `python/batcher/core/streaming/folds/windowed.py` (`_WindowedAggFold`: a window is
closed exactly when `window_start <= watermark - width`, i.e. its end is at or below the
watermark; `flush` emits every still-open window when the stream ends). The watermark is the
highest event time seen less the allowed lateness (`plan/streaming/tracker.py`), which on a
single partition is what the header row shows.

The event times are illustrative and chosen so that exactly one window closes mid-stream.
The one-hour window and ten-minute lateness are the ones in the page's own example.

The figure exists because the three shapes differ only in *when*, and that is a grid: the
same four triggers, three lanes, and a last column that a Kafka topic never reaches.
"""

from __future__ import annotations

from _authoring import BLUE_MID, FONT, GREY, band, label, note, svg, write

W, H = 980, 572

LABEL_X = 38
COL_X = (240, 380, 520, 660)
END_X = 816
CW = 128
ROWS = (202, 302, 402)
RH = 76

TRIGGERS = (
    ("10:40", "10:30"),
    ("11:05", "10:55"),
    ("11:25", "11:15"),
    ("11:50", "11:40"),
)


def cell(x: float, y: float, lines: tuple[str, ...], live: bool, end: bool = False) -> str:
    """One trigger's output for one shape. Grey and 'nothing' when the shape emits no rows."""
    color = BLUE_MID if live else GREY
    dash = ' stroke-dasharray="6 4"' if end else ""
    out = (
        f'<rect x="{x}" y="{y}" width="{CW}" height="{RH}" rx="8" fill="{color}" '
        f'fill-opacity="{0.14 if live else 0.06}" stroke="{color}" stroke-width="1.6"{dash}/>'
    )
    top = y + RH / 2 - 8 * (len(lines) - 1) + 5
    weight = "700" if live else "400"
    cls = "t-title" if live else "t-sub"
    for i, text in enumerate(lines):
        out += (
            f'<text x="{x + CW / 2}" y="{top + 17 * i}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="12.5" font-weight="{weight}" '
            f'class="{cls}">{text}</text>'
        )
    return out


NOTHING = ("nothing",)

body: list[str] = [
    note(
        490,
        40,
        "One unbounded stream, four triggers. Event times are illustrative; the window is "
        "1 hour and the allowed lateness 10 minutes.",
        anchor="middle",
    ),
    band(20, 62, 940, 432, "WHAT EACH SHAPE EMITS, TRIGGER BY TRIGGER", "blue"),
    note(LABEL_X, 142, "max event time seen"),
    note(LABEL_X, 166, "watermark"),
]

for i, (seen, wm) in enumerate(TRIGGERS):
    cxm = COL_X[i] + CW / 2
    body += [
        label(cxm, 120, f"trigger {i + 1}", anchor="middle"),
        note(cxm, 142, seen, anchor="middle"),
        label(cxm, 166, wm, anchor="middle", size=12.5),
    ]

body += [
    f'<path d="M {END_X - 14} 104 L {END_X - 14} 482" stroke="{GREY}" stroke-width="1.4" '
    f'stroke-dasharray="4 4"/>',
    label(END_X + CW / 2, 120, "input ends", anchor="middle"),
    note(END_X + CW / 2, 142, "a drain reaches this;", anchor="middle"),
    note(END_X + CW / 2, 158, "a Kafka topic never", anchor="middle"),
]

# Row-wise shapes: every trigger's own rows, as they arrive.
y = ROWS[0]
body += [
    label(LABEL_X, y + 30, "filter / select"),
    note(LABEL_X, y + 48, "map_batches, with_columns"),
    note(LABEL_X, y + 64, "per window, as rows arrive"),
]
for i, x in enumerate(COL_X):
    body.append(cell(x, y, (f"trigger {i + 1}'s", "rows"), live=True))
body.append(cell(END_X, y, NOTHING, live=False, end=True))

# Watermarked window: a window leaves once its end is at or below the watermark.
y = ROWS[1]
body += [
    label(LABEL_X, y + 30, "with_watermark + window"),
    note(LABEL_X, y + 48, "a window, once the"),
    note(LABEL_X, y + 64, "watermark passes its end"),
    cell(COL_X[0], y, NOTHING, live=False),
    cell(COL_X[1], y, NOTHING, live=False),
    cell(COL_X[2], y, ("window", "10:00 to 11:00"), live=True),
    cell(COL_X[3], y, NOTHING, live=False),
    cell(END_X, y, ("window", "11:00 to 12:00"), live=True, end=True),
]

# Unwatermarked aggregate: one running state, finalized only when the input stops.
y = ROWS[2]
body += [
    label(LABEL_X, y + 30, "group_by().agg()"),
    note(LABEL_X, y + 48, "no watermark: once,"),
    note(LABEL_X, y + 64, "at end of input"),
]
for x in COL_X:
    body.append(cell(x, y, NOTHING, live=False))
body.append(cell(END_X, y, ("the whole", "result"), live=True, end=True))

body += [
    band(20, 510, 940, 48, "", "grey"),
    note(
        490,
        539,
        "Trigger 3's watermark, 11:15, is the first at or past 11:00, so that window closes "
        "and leaves state. Until then it is open and emits nothing.",
        anchor="middle",
    ),
]

write("streaming_emission", svg(W, H, "".join(body)))
print("wrote streaming_emission.svg")
