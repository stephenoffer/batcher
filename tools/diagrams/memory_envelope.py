#!/usr/bin/env python3
"""Draw `memory_envelope.svg` — what is resident at one moment, how much of the
envelope this query may plan against, and why exceeding it is a counter-offer rather
than a failure.

Source of truth: `python/batcher/carbonite/memory/estimator.py::peak_operator_bytes`
(the schedule walk and its worked figures), `policies/admission.py::BudgetingAdmission`
(the envelope arithmetic, the morsel floor, `_bytes_already_held`, `_rests_on_a_guess`)
and `policies/concurrency.py::query_memory_share`.

The peak rule is drawn as the formula the estimator implements rather than as a count
of live hash tables: the two do not say the same thing, and the formula is the half
that decides an admission.
"""

from __future__ import annotations

from _authoring import AMBER, BLUE_MID, FONT, arrow, band, card, label, note, svg, write

W, H = 980, 716

# The shrinking envelope. Each step is one reduction admission actually applies.
STEPS = (
    (900, "available: the process envelope, or total RAM when none is set"),
    (770, "x memory.soft_limit"),
    (640, "- what concurrent work already holds"),
    (430, "/ concurrent queries: 1 / min(active, slots), exactly 1 when unbounded"),
)

body = [
    band(30, 26, 920, 206, "WHAT A JOIN HOLDS WHILE ITS PROBE SIDE RUNS", "grey"),
    card(70, 68, 230, 72, "build subtree", "peaks, then collapses"),
    arrow(300, 104, 370, 104),
    label(335, 92, "materialize", anchor="middle"),
    card(370, 68, 230, 72, "the build table", "resident from here on"),
    arrow(600, 104, 670, 104),
    label(635, 92, "held while", anchor="middle"),
    card(670, 68, 230, 72, "probe subtree", "adds its own peak"),
    label(
        490, 178, "peak(join) = max( peak(build),  resident(join) + peak(probe) )", anchor="middle"
    ),
    note(
        490,
        202,
        "On one worked bushy plan the largest single operator reads 18.2 MB where the concurrent figure is 27.4 - a 1.5x under-count.",
        anchor="middle",
    ),
    band(30, 248, 920, 212, "THE ENVELOPE THIS QUERY MAY PLAN AGAINST", "blue"),
]

for i, (right, text) in enumerate(STEPS):
    y = 286 + i * 38
    fill = BLUE_MID if i < len(STEPS) - 1 else AMBER
    body += [
        f'<text x="62" y="{y}" font-family="{FONT}" font-size="11.5" class="t-sub">{text}</text>',
        f'<rect x="62" y="{y + 6}" width="{right - 62}" height="16" rx="4" fill="{fill}"/>',
    ]

body += [
    note(
        62,
        448,
        "Floored at one morsel, so a streaming plan is never refused for a budget smaller than a single batch.",
    ),
    band(30, 478, 920, 206, "OVER BUDGET IS A COUNTER-OFFER, NOT A FAILURE", "grey"),
    label(490, 514, "is the plan's peak inside that envelope?", anchor="middle"),
    arrow(430, 522, 280, 548),
    label(330, 528, "yes"),
    arrow(550, 522, 700, 548),
    label(636, 528, "no"),
    card(70, 552, 380, 76, "Runs in memory", "admitted with no bound imposed"),
    card(530, 552, 380, 76, "m_max_bytes = the envelope", "the operator goes out of core"),
    note(720, 652, "The verdict names the binding join, aggregate or sort.", anchor="middle"),
    note(
        720, 670, "Sized from a guess it is advisory: it routes, it never fails.", anchor="middle"
    ),
]

write("memory_envelope", svg(W, H, "".join(body)))
print("wrote memory_envelope.svg")
