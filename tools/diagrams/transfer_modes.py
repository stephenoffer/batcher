#!/usr/bin/env python3
"""Draw `transfer_modes.svg` — how Carbonite routes one shuffle partition.

Source of truth: `python/batcher/carbonite/transfer/locality.py::select_mode`.
The diagram states the two comparisons the selector actually makes, in order, and
marks SHARED_MEMORY as selected-but-not-yet-executed, which is what that module's
docstring says today. Keep both in step.

Layout: three columns, each centered on its mode card, so the decision text, the
arrow, and the outcome share one vertical axis.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 500

# One x-axis per branch; every element in a column shares it.
COLS = (193, 490, 787)
BAND_BOTTOM = 116
CARD_TOP = 250

body = [
    band(20, 20, 940, 96, "ONE PARTITION, FETCHED BY A CONSUMER", "grey"),
    card(300, 44, 380, 56, "select_mode(source, local)", "pure: placement in, mode out"),
]

# The decision row. Each column asks its question, then drops an arrow to its outcome.
branches = [
    (COLS[0], "Same Flight address?", "one process", "yes", "blue"),
    (COLS[1], "Same node identity?", "one host, two processes", "no, but", "blue"),
    (COLS[2], "Neither is known equal", "including unknown placement", "no", "grey"),
]
for x, question, gloss, verdict, kind in branches:
    body += [
        label(x, 152, question, anchor="middle"),
        note(x, 172, gloss, anchor="middle"),
        arrow(x, 186, x, CARD_TOP - 6, kind),
        label(x + 12, 220, verdict, size=12),
    ]

body += [
    band(20, 208, 940, 262, "TRANSFER MODE, CHEAPEST FIRST", "blue"),
    card(48, CARD_TOP, 290, 92, "DIRECT_MEMORY", "read from the local store"),
    note(COLS[0], 372, "No serialization, no socket.", anchor="middle"),
    note(COLS[0], 390, "The concrete win over an object store.", anchor="middle"),
    card(358, CARD_TOP, 264, 92, "SHARED_MEMORY", "Arrow IPC over a memory map"),
    note(COLS[1], 372, "Selected today, not yet executed:", anchor="middle"),
    note(COLS[1], 390, "a planned Rust fast path.", anchor="middle"),
    card(642, CARD_TOP, 290, 92, "NETWORK", "credit-bounded Arrow Flight"),
    note(COLS[2], 372, "One credit is one batch slot.", anchor="middle"),
    note(COLS[2], 390, "The producer blocks at zero.", anchor="middle"),
    note(490, 440, "locality_ratio reports the fraction of transfers that stayed off the network.", anchor="middle"),
]

write("transfer_modes", svg(W, H, "".join(body)))
print("wrote transfer_modes.svg")
