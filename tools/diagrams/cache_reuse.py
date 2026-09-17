#!/usr/bin/env python3
"""Draw `cache_reuse.svg`: a shared subquery with two consumers, without and with `cache()`.

Source of truth: `docs/user-guide/operate/tuning/caching.md`. A `Dataset` is a plan, so
two terminals over the same `active` subquery run its scan and filter twice. `cache()` is
a marker: the first terminal that materializes the result (`collect`) executes the plan and
stores the Arrow result, and later terminals on the same cached dataset are served from it.
`count()` is drawn second on purpose: it is served from a warm cache but never fills a cold
one, so a `collect()` has to come first.

Form: a before/after at the same scale, with the scan and filter drawn once per execution,
so the duplication is the thing the eye counts.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, pill, svg, tint, write

W, H = 980, 520

CW, CH = 180, 54
ROW = (84, 184, 284)  # scan, filter, consumer (left) / cached result (right)


def column(cx: float, consumer: str, consumer_sub: str) -> list[str]:
    """One full execution: scan, filter, and the terminal that asked for it."""
    x = cx - CW / 2
    return [
        card(x, ROW[0], CW, CH, "scan events"),
        arrow(cx, ROW[0] + CH + 2, cx, ROW[1] - 6),
        label(cx + 10, ROW[0] + CH + 29, "rows", size=11.5),
        card(x, ROW[1], CW, CH, "filter", "status == active"),
        arrow(cx, ROW[1] + CH + 2, cx, ROW[2] - 6),
        label(cx + 10, ROW[1] + CH + 29, "active rows", size=11.5),
        tint(x, ROW[2], CW, CH, consumer, consumer_sub),
    ]


body = [
    band(20, 20, 456, 480, "WITHOUT CACHE()", "grey"),
    band(504, 20, 456, 480, "WITH CACHE()", "blue"),
]

# ---- Left: two terminals, two executions ------------------------------------------------
body += column(134, "collect()", "first terminal")
body += column(362, "count()", "second terminal")
body += [
    pill(248, 386, "scan + filter run twice", "grey", anchor="middle"),
    note(248, 426, "Each terminal executes the whole plan.", anchor="middle"),
    note(248, 446, "Five reports over one filtered scan", anchor="middle"),
    note(248, 464, "are five scans.", anchor="middle"),
]

# ---- Right: one execution, stored, then read twice --------------------------------------
RX = 732
body += [
    card(RX - CW / 2, ROW[0], CW, CH, "scan events"),
    arrow(RX, ROW[0] + CH + 2, RX, ROW[1] - 6),
    label(RX + 10, ROW[0] + CH + 29, "rows", size=11.5),
    card(RX - CW / 2, ROW[1], CW, CH, "filter", "status == active"),
    arrow(RX, ROW[1] + CH + 2, RX, ROW[2] - 6, "amber"),
    label(RX + 10, ROW[1] + CH + 29, "first collect() stores it", size=11.5),
    tint(RX - 110, ROW[2], 220, CH, "cached result", "Arrow, kept after first run", "amber"),
]

CONSUMER_Y = 404
for cx, title, sub, text in (
    (618, "collect() again", "served, no re-run", "hit"),
    (846, "count()", "served from warm cache", "hit"),
):
    body += [
        arrow(RX + (-40 if cx < RX else 40), ROW[2] + CH + 2, cx, CONSUMER_Y - 6, "amber"),
        label(
            (RX + cx) / 2 + (-26 if cx < RX else 26),
            ROW[2] + CH + 34,
            text,
            anchor="end" if cx < RX else "start",
            size=11.5,
        ),
        tint(cx - 105, CONSUMER_Y, 210, CH, title, sub),
    ]

body.append(pill(RX, 486, "scan + filter run once", "blue", anchor="middle"))

write("cache_reuse", svg(W, H, "".join(body)))
print("wrote cache_reuse.svg")
