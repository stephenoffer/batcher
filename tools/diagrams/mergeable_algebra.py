#!/usr/bin/env python3
"""Draw `mergeable_algebra.svg` -- the deep view of partial/combine/finalize.

Source of truth: `crates/bc-runtime/` (the three functions themselves),
`crates/bc-interp/src/{lib,par,dist}.rs` and `crates/bc-runtime/src/agg/spill.rs`
(the four callers), and `docs/architecture/deep-dives/operators/mergeable-algebra.md`.

This is deliberately *not* `mergeable.svg`, which draws the chain alone for the
concepts page. The claim here is the one the prose makes in a sentence and a reader
cannot hold: the four execution modes are not four algorithms. They differ only in
what carries a partial state from `partial` to `combine` -- nothing, a thread, a
spill file, or a Flight stream -- and `combine` being associative *and* commutative
is exactly what makes that substitution legal.

Layout: the algebra on one rail across the top, the four transports on a bus below
it feeding the single `combine`, and the invariant stated last.
"""

from __future__ import annotations

from _authoring import GREY, arrow, band, card, curve, label, note, svg, write

W, H = 980, 540

# The three functions share one vertical axis so the chain reads left to right.
CHAIN_Y, CHAIN_H = 62, 80
MID = CHAIN_Y + CHAIN_H / 2  # 102
RAIL_Y = 208  # the bus every mode publishes onto
MODE_Y, MODE_H = 268, 88

body = [
    band(20, 24, 940, 136, "ONE OPERATOR, WRITTEN ONCE", "blue"),
    card(44, CHAIN_Y, 226, CHAIN_H, "partial(batch)", "rows in, state out"),
    card(376, CHAIN_Y, 228, CHAIN_H, "combine(states)", "associative + commutative"),
    card(710, CHAIN_Y, 226, CHAIN_H, "finalize(state)", "state in, rows out"),
    arrow(270, MID, 370, MID),
    label(320, MID - 14, "partial state", anchor="middle", size=11.5),
    note(320, MID + 22, "not the answer", anchor="middle"),
    arrow(604, MID, 704, MID),
    label(654, MID - 14, "merged state", anchor="middle", size=11.5),
    note(654, MID + 22, "one per group", anchor="middle"),
]

# The bus: every mode publishes its partials onto one rail, and the rail feeds the
# single `combine` above. One line, so the shared destination is visual, not asserted.
body += [
    band(20, 184, 940, 248, "WHAT CARRIES A PARTIAL FROM partial TO combine", "amber"),
    f'<path d="M 146 {RAIL_Y} H 842" fill="none" stroke="{GREY}" stroke-width="2.4"/>',
    arrow(490, RAIL_Y, 490, CHAIN_Y + CHAIN_H + 6),
    label(502, 176, "the same combine, whatever carried the state", size=11.5),
]

modes = (
    (40, "One core", "bc-interp::execute", "nothing to carry", "one partial, nothing to merge"),
    (272, "Many cores", "bc-interp::par", "a thread hand-off", "one partial per morsel"),
    (504, "Bounded memory", "agg::spill", "an IPC spill file", "one partition at a time"),
    (736, "Many machines", "bc-interp::dist", "a Flight stream", "hash-partitioned by key"),
)
for x, title, where, carrier, gloss in modes:
    cx = x + 106
    body += [
        card(x, MODE_Y, 212, MODE_H, title, where),
        arrow(cx - 45, MODE_Y, cx - 45, RAIL_Y + 6),
        label(cx - 33, 236, carrier, size=11.5),
        note(cx, MODE_Y + MODE_H + 20, gloss, anchor="middle"),
    ]

# The invariant, and the two algebraic properties it rests on.
body += [
    band(20, 452, 940, 68, "THE TEST THAT MUST STAY GREEN", "grey"),
    label(
        490,
        492,
        "combine_finalize(partition(partial(p_k)))  ==  the single-node result",
        anchor="middle",
        size=14,
    ),
    curve(736, MODE_Y + MODE_H, 640, 470, 300, 462, "amber"),
    note(500, 446, "arrival order cannot change the answer", anchor="middle"),
]

write("mergeable_algebra", svg(W, H, "".join(body)))
print("wrote mergeable_algebra.svg")
