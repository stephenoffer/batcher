#!/usr/bin/env python3
"""Draw `time_series_align.svg` -- bucket, fill the empty buckets, and align a second clock.

Source of truth: `docs/user-guide/analyze/time-series.md`. The readings are sensor `a` from
the page's setup data (09:00, 09:01, 09:02 and 09:32 at 20.0, 21.0, 22.0 and 26.0). The
buckets are its `bt.window(bt.col("at"), "30m")` group-by output (09:00 mean 21.0 over 3,
09:30 mean 26.0 over 1). The grid is its `bt.date_range(09:00, 10:30, interval="30m")`
left join, and the filled row is its `forward_fill` on the mean and `fill_null(0)` on the
count (26.0 carried, n 0). The as-of panel is the page's `join_asof(quotes, on="t",
by="sym", tolerance=5)` example: the trade at t=10 takes the quote at t=8, and the trade at
t=40 is left unmatched because its nearest preceding quote, t=12, is 28 units old.
"""

from __future__ import annotations

from _authoring import (
    AMBER,
    AMBER_DEEP,
    BLUE,
    BLUE_MID,
    FONT,
    GREY,
    arrow,
    band,
    label,
    note,
    svg,
    write,
)

W, H = 980, 640

X0, PX = 200, 6.0  # screen x of 09:00, pixels per minute
READ_Y, BUCKET_Y, FILL_Y, BOX_H = 118, 176, 276, 46


def mx(minutes: float) -> float:
    """Screen x for minutes after 09:00."""
    return X0 + minutes * PX


def box(start: int, y: float, title: str, sub: str, kind: str) -> str:
    """One 30-minute bucket. `kind` is measured, carried, or empty."""
    x, w = mx(start) + 4, 30 * PX - 8
    if kind == "measured":
        rect = (
            f'<rect x="{x}" y="{y}" width="{w}" height="{BOX_H}" rx="8" fill="{BLUE_MID}" '
            f'fill-opacity="0.16" stroke="{BLUE}" stroke-width="1.4"/>'
        )
    elif kind == "carried":
        rect = (
            f'<rect x="{x}" y="{y}" width="{w}" height="{BOX_H}" rx="8" fill="{AMBER}" '
            f'fill-opacity="0.16" stroke="{AMBER_DEEP}" stroke-width="1.6" stroke-dasharray="6 4"/>'
        )
    else:
        rect = (
            f'<rect x="{x}" y="{y}" width="{w}" height="{BOX_H}" rx="8" fill="none" '
            f'stroke="{GREY}" stroke-width="1.4" stroke-dasharray="4 4"/>'
        )
    cx = x + w / 2
    out = rect + (
        f'<text x="{cx}" y="{y + (20 if sub else 28)}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="13" font-weight="700" class="t-title">{title}</text>'
    )
    if sub:
        out += note(cx, y + 37, sub, anchor="middle")
    return out


body = [band(20, 20, 940, 356, "BUCKET, THEN FILL THE BUCKETS WITH NO ROWS", "blue")]

# Bucket boundaries and their times.
for i, text in enumerate(("09:00", "09:30", "10:00", "10:30", "11:00")):
    x = mx(i * 30)
    body += [
        f'<path d="M {x} 84 V {FILL_Y + BOX_H + 8}" stroke="{GREY}" stroke-opacity="0.45" '
        f'stroke-width="1" stroke-dasharray="3 4"/>',
        note(x, 76, text, anchor="middle"),
    ]

# The raw readings.
body += [
    label(44, READ_Y + 5, "readings"),
    f'<path d="M {mx(0)} {READ_Y} H {mx(120)}" stroke="{GREY}" stroke-width="1.4"/>',
]
for minute in (0, 1, 2, 32):
    body.append(f'<circle cx="{mx(minute) + 3}" cy="{READ_Y}" r="5.5" fill="{BLUE}"/>')
body += [
    note(mx(1) + 3, READ_Y - 12, "20.0, 21.0, 22.0", anchor="start"),
    note(mx(32) + 3, READ_Y - 12, "26.0", anchor="middle"),
    note(mx(90), READ_Y - 12, "no readings after 09:32", anchor="middle"),
]

# group_by on the window start.
body += [
    label(44, BUCKET_Y + 20, "group_by"),
    note(44, BUCKET_Y + 37, 'window "30m"'),
    box(0, BUCKET_Y, "mean 21.0", "n 3", "measured"),
    box(30, BUCKET_Y, "mean 26.0", "n 1", "measured"),
    box(60, BUCKET_Y, "no row", "", "empty"),
    box(90, BUCKET_Y, "no row", "", "empty"),
]

# The grid, the left join, and the per-column fill rule.
body += [
    arrow(110, BUCKET_Y + BOX_H + 6, 110, FILL_Y - 6),
    label(122, BUCKET_Y + BOX_H + 32, "join onto a date_range grid, fill", size=12),
    label(44, FILL_Y + 20, "filled"),
    box(0, FILL_Y, "mean 21.0", "n 3", "measured"),
    box(30, FILL_Y, "mean 26.0", "n 1", "measured"),
    box(60, FILL_Y, "26.0 carried", "n 0", "carried"),
    box(90, FILL_Y, "26.0 carried", "n 0", "carried"),
    note(
        490,
        356,
        "The mean is carried forward. The count is filled with 0, "
        "so a reader can tell carried from measured.",
        anchor="middle",
    ),
]

# --- the as-of panel -------------------------------------------------------------------
T0, TPX, AX_Y = 200, 16.0, 500


def tx(t: float) -> float:
    """Screen x for a value of the numeric `on` key."""
    return T0 + t * TPX


def tolerance(t: float) -> str:
    """The span a backward as-of search with tolerance=5 may reach from a trade."""
    return (
        f'<rect x="{tx(t - 5)}" y="{AX_Y - 22}" width="{5 * TPX}" height="44" rx="6" '
        f'fill="{AMBER}" fill-opacity="0.18" stroke="{AMBER_DEEP}" stroke-width="1.2" '
        f'stroke-dasharray="4 3"/>'
    )


body += [
    band(20, 394, 940, 226, "ALIGN A SECOND CLOCK WITH JOIN_ASOF", "amber"),
    label(44, AX_Y + 5, "sym A"),
    tolerance(10),
    tolerance(40),
    f'<path d="M {tx(0)} {AX_Y} H {tx(45)}" stroke="{GREY}" stroke-width="1.4"/>',
]
for t, anchor, dx in ((8, "end", 6), (12, "start", -6)):
    body += [
        f'<rect x="{tx(t) - 7}" y="{AX_Y - 7}" width="14" height="14" rx="2" fill="{AMBER_DEEP}"/>',
        note(tx(t) + dx, AX_Y - 34, f"quote t={t}", anchor=anchor),
    ]
for t in (10, 40):
    body += [
        f'<circle cx="{tx(t)}" cy="{AX_Y}" r="7" fill="{BLUE}"/>',
        note(tx(t), AX_Y + 40, f"trade t={t}", anchor="middle"),
    ]
body += [
    label(tx(10) - 40, AX_Y + 62, "price 1.0: quote t=8 is inside the tolerance", size=12),
    label(tx(40), AX_Y - 34, "price null: nothing within 5", anchor="middle", size=12),
    note(
        490,
        600,
        "The shaded span is tolerance=5, looking back from each trade. "
        "Without it, t=40 takes t=12.",
        anchor="middle",
    ),
]

write("time_series_align", svg(W, H, "".join(body)))
print("wrote time_series_align.svg")
