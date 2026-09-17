#!/usr/bin/env python3
"""Draw `slow_query_loop.svg`: read the plan, measure, fix one operator, measure again.

Source of truth: `docs/getting-started/tutorials/foundations/optimizing-a-slow-query.md`.
`explain()` runs the optimizer and renders the plan without executing it; `stats()`
executes and reports per-operator rows, time, and the estimate beside the measurement
(`est_rows`, `rows_out`, `bottleneck`). The before and after figures are that page's
own: with a `map_batches` in the middle every operator reads `est≈?` and the fee
arithmetic touches 200,000 rows; rewritten as an expression, the filter is pushed into
the scan and the arithmetic touches 20,000, with an identical answer.
"""

from __future__ import annotations

from _authoring import arrow, band, card, curve, label, note, step, svg, tint, write

W, H = 980, 470

ROW_Y, CARD_H, CARD_W = 76, 92, 250
MID = ROW_Y + CARD_H / 2
XS = (44, 365, 686)  # card left edges; the gaps between them carry the arrow labels

steps = [
    ("Read the plan", "explain(), nothing runs"),
    ("Measure it", "stats(), per operator"),
    ("Fix that operator", "one change at a time"),
]

body = [band(20, 20, 940, 250, "THE LOOP", "blue")]
for i, (x, (title, sub)) in enumerate(zip(XS, steps, strict=True), start=1):
    body += [card(x, ROW_Y, CARD_W, CARD_H, title, sub), step(x + 4, ROW_Y + 4, i)]

body += [
    arrow(XS[0] + CARD_W, MID, XS[1] - 2, MID),
    label((XS[0] + CARD_W + XS[1]) / 2, MID - 12, "then run", anchor="middle", size=11.5),
    arrow(XS[1] + CARD_W, MID, XS[2] - 2, MID),
    label((XS[1] + CARD_W + XS[2]) / 2, MID - 12, "hot spot", anchor="middle", size=11.5),
    curve(XS[2] + CARD_W / 2, ROW_Y + CARD_H, 490, 262, XS[0] + CARD_W / 2, ROW_Y + CARD_H + 4),
    label(490, 246, "fixed? explain() and stats() again", anchor="middle"),
    band(20, 290, 940, 160, "WHAT THE TUTORIAL FINDS", "grey"),
    tint(44, 334, 370, 84, "Before: map_batches", "no row estimates, fee on 200,000 rows"),
    tint(
        566, 334, 370, 84, "After: an expression", "filter in the scan, fee on 20,000 rows", "amber"
    ),
    arrow(414, 376, 564, 376, "amber"),
    label(489, 364, "same answer", anchor="middle", size=11.5),
    note(489, 396, "10x less arithmetic", anchor="middle"),
]

write("slow_query_loop", svg(W, H, "".join(body)))
print("wrote slow_query_loop.svg")
