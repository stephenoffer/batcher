#!/usr/bin/env python3
"""Draw `credit_backpressure.svg` — the credit protocol on one shuffle channel, with
the blocked state drawn rather than described.

Source of truth: `crates/bc-transport/src/handler.rs` (the `Semaphore`, the gated
source stream that awaits a permit before a batch reaches the Flight encoder, and the
clamp that keeps outstanding permits at or below the seeded window),
`crates/bc-transport/src/exchange.rs::credit_exchange_inner` (the seed in the first
message's `app_metadata`, and the low-watermark refill at `(credits / 2).max(1)`),
`crates/bc-transport/src/lib.rs::DEFAULT_CREDITS` (16), and
`python/batcher/carbonite/policies/flow_control.py` (who sizes the window).

Note what the refill is: grants are accumulated and sent in bulk once half the window
has freed, not one grant per consumed batch. The bound is unaffected - a grant only
ever lags consumption - and drawing it as one-per-batch would misstate the wire cost.
"""

from __future__ import annotations

from _authoring import AMBER, arrow, band, card, label, note, svg, write

W, H = 980, 524

LX, RX, CW = 49, 681, 250
LMID, RMID = LX + CW / 2, RX + CW / 2

body = [
    band(24, 30, 300, 420, "PRODUCER", "blue"),
    band(656, 30, 300, 420, "CONSUMER", "grey"),
    card(RX, 80, CW, 66, "opens the exchange", "names the ticket, seeds the window"),
    card(LX, 80, CW, 66, "credit semaphore", "seeded at 16 permits"),
    arrow(679, 113, 303, 113),
    label(491, 101, "seed W = 16 credit slots", anchor="middle"),
    arrow(LMID, 146, LMID, 176),
    label(LMID + 12, 168, "per batch"),
    card(LX, 180, CW, 66, "acquire one permit", "before the Flight encoder sees it"),
    arrow(303, 213, 679, 213),
    label(491, 201, "one batch, one permit spent", anchor="middle"),
    card(RX, 180, CW, 66, "consume the batch", "pending += 1"),
    arrow(LMID, 246, LMID, 278),
    label(LMID - 12, 270, "permits exhausted", anchor="end"),
    card(LX, 282, CW, 66, "BLOCKED at zero", "the next batch is never encoded"),
    f'<rect x="{LX}" y="293" width="5" height="44" rx="2.5" fill="{AMBER}"/>',
    arrow(RMID, 246, RMID, 278),
    label(RMID + 12, 270, "slots freed"),
    card(RX, 282, CW, 66, "pending reaches 8", "half the window has come back"),
    arrow(679, 315, 305, 315, "amber"),
    label(491, 303, "one grant for all 8, not one per batch", anchor="middle"),
    arrow(310, 280, 310, 222, "amber"),
    label(306, 266, "resume", anchor="end"),
    note(LMID, 376, "The top-up is clamped to the seeded", anchor="middle"),
    note(LMID, 394, "window, so an over-granting consumer", anchor="middle"),
    note(LMID, 412, "cannot make the producer buffer the", anchor="middle"),
    note(LMID, 430, "whole partition.", anchor="middle"),
    note(RMID, 376, "Batching the grants cuts control", anchor="middle"),
    note(RMID, 394, "traffic without loosening the bound:", anchor="middle"),
    note(RMID, 412, "a grant is deferred, never", anchor="middle"),
    note(RMID, 430, "anticipated.", anchor="middle"),
    note(
        490,
        482,
        "One credit is one in-flight batch slot, so the window is the channel's memory bound -",
        anchor="middle",
    ),
    note(
        490,
        500,
        "about 16 MiB at a 1 MiB morsel. Carbonite's AIMD controller sizes it per channel.",
        anchor="middle",
    ),
]

write("credit_backpressure", svg(W, H, "".join(body)))
print("wrote credit_backpressure.svg")
