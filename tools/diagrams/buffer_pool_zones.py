#!/usr/bin/env python3
"""Draw `buffer_pool_zones.svg` — the memory envelope's one soft line, and what a
refused reservation actually costs.

Source of truth: `crates/bc-resource/src/lib.rs` (`MemoryPool`, `Pressure`,
`DEFAULT_SOFT_BPS`, `try_reserve_cooperative`, `MAX_SPILL_ROUNDS`, `MemoryReservation`)
and `python/batcher/carbonite/memory/pool.py` (`BufferPool`, `engine_pool_stats`).

Two things the picture is careful about, because the code is:

* **`Critical` is a line, not a region.** `PoolStats::pressure` reports it at
  `used >= limit`, and `try_reserve_bytes` refuses any growth past `limit` — so the
  pool cannot sit inside a critical *band* the way it sits inside the elevated one.
* **The cooperative path does not make the pending reservation fit today.** Its one
  registered consumer frees shuffle-store bytes that were never charged to this pool,
  so the progress check stops after a round and the caller is refused into a process
  with room to run its own spill path. Drawn as "retry, then the caller spills"
  rather than as a rescue.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, arrow, band, card, label, note, svg, write

W, H = 960, 660

GAUGE_Y, GAUGE_H = 86, 46
X0, X_SOFT, X_LIMIT = 62, 652, 820

body = [
    band(30, 26, 900, 180, "THE ENVELOPE, DIVIDED BY ONE SOFT LINE", "grey"),
    # The gauge. Two segments, because the pool has exactly two thresholds.
    f'<rect x="{X0}" y="{GAUGE_Y}" width="{X_SOFT - X0}" height="{GAUGE_H}" rx="6" '
    f'class="band-blue" stroke-width="1.5"/>',
    f'<rect x="{X_SOFT}" y="{GAUGE_Y}" width="{X_LIMIT - X_SOFT}" height="{GAUGE_H}" rx="6" '
    f'class="band-amber" stroke-width="1.5"/>',
    f'<path d="M {X_LIMIT} {GAUGE_Y - 12} L {X_LIMIT} {GAUGE_Y + GAUGE_H + 12}" '
    f'stroke="{AMBER_DEEP}" stroke-width="4"/>',
    note(X0, 76, "0"),
    note(X_SOFT, 76, "soft line: 80% of the limit", anchor="middle"),
    note(X_LIMIT, 76, "limit", anchor="end"),
    label((X0 + X_SOFT) / 2, 108, "NOMINAL", anchor="middle"),
    note((X0 + X_SOFT) / 2, 126, "no throttling", anchor="middle"),
    label((X_SOFT + X_LIMIT) / 2, 108, "ELEVATED", anchor="middle"),
    note((X_SOFT + X_LIMIT) / 2, 126, "spill early", anchor="middle"),
    note(
        X_LIMIT,
        158,
        "CRITICAL is used == limit: a line, not a region, because growth past it is refused.",
        anchor="end",
    ),
    note(
        X0,
        182,
        "used moves right as operators reserve and back as they release. The pool counts bytes; it never allocates them.",
    ),
    arrow(480, 206, 480, 250),
    label(492, 234, "try_reserve(n bytes)"),
    band(30, 258, 900, 342, "WHAT A RESERVATION IS, AND WHAT A REFUSAL COSTS", "blue"),
    label(480, 302, "does used + n still fit under the limit?", anchor="middle"),
    arrow(430, 310, 250, 330),
    label(296, 316, "yes"),
    arrow(530, 310, 640, 330),
    label(556, 316, "no"),
    card(62, 330, 300, 84, "Granted", "an RAII guard on n bytes"),
    note(212, 436, "Every byte returns when the guard drops,", anchor="middle"),
    note(212, 454, "on a panic as much as on a clean finish.", anchor="middle"),
    card(560, 330, 300, 84, "Refused", "denied += 1; used is untouched"),
    arrow(710, 414, 710, 452),
    label(722, 438, "ask the largest other consumer"),
    card(560, 456, 300, 84, "Cooperative retry", "it spills, then re-reserve"),
    note(710, 560, "At most 32 rounds, and it stops the moment a round frees nothing.", anchor="middle"),
    arrow(556, 528, 368, 528),
    label(462, 514, "still short", anchor="middle"),
    card(62, 486, 300, 84, "The caller spills", "the refusal is the signal"),
    note(
        480,
        626,
        "Two pools count different bytes and are read side by side, never summed: Carbonite's holds one coarse",
        anchor="middle",
    ),
    note(
        480,
        644,
        "reservation per query, the engine's holds the operator state and the Flight transit buffers it became.",
        anchor="middle",
    ),
]

write("buffer_pool_zones", svg(W, H, "".join(body)))
print("wrote buffer_pool_zones.svg")
