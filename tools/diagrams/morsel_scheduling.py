#!/usr/bin/env python3
"""Draw `morsel_scheduling.svg` -- how a scan/filter/project chain reaches the cores.

Source of truth, checked line by line before drawing:
  * `crates/bc-arrow/src/lib.rs` -- `DEFAULT_MORSEL_ROWS` (16,384), `DEFAULT_MORSEL_BYTES`
    (1 MiB) and `MorselTarget`, which is full at **whichever bound trips first**.
  * `crates/bc-arrow/src/hardware.rs` -- `operator_cores()`, every physical core plus a
    third of the SMT siblings, which is what sizes the pool.
  * `crates/bc-interp/src/par.rs` -- `pool_for` (a pool cached per width, never rayon's
    global one), `auto_width`, and `exec_fused`, the single `par_iter` that runs a run of
    Filter/Project stages over one morsel at a time.
  * `crates/bc-interp/src/ops/morsel.rs` -- `morselize`, which splits and coalesces.

Two things this diagram is careful about, because the obvious picture is wrong:
  * **Batcher writes no work-stealing code.** The morsels are a `Vec<RecordBatch>` handed
    to rayon's `par_iter()`; the stealing is rayon's scheduler. There is no Batcher-owned
    queue, atomic morsel counter, or claim function anywhere in `bc-interp`.
  * **The scan materializes.** `exec_fused` fuses only Filter and Project; the base morsels
    are a fully built vector before the pool starts. Drawing the scan as a lazy pull would
    flatter the engine into something it is not.

Layout: three zoom levels top to bottom -- make the morsels, spread them over the pool,
then one worker's pass over one morsel.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE_MID, FONT, GREY, arrow, band, card, curve, label, note, svg, write

W, H = 980, 596


def chip(x: float, y: float, w: float, h: float, kind: str = "blue") -> str:
    """A small data rectangle: an input batch or a morsel. Not a card -- no title."""
    fill = {"blue": BLUE_MID, "grey": GREY}[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
        f'fill-opacity="0.28" stroke="{fill}" stroke-width="1.4"/>'
    )


body: list[str] = []

# ---- Zoom 1: the morsels themselves ------------------------------------------------
body.append(band(20, 24, 940, 130, "MAKE THE MORSELS", "grey"))
body.append(note(44, 64, "input: whatever the source emitted"))

x = 44
for w in (142, 30, 96, 16, 20, 58):          # row groups, arrivals, a filter's crumbs
    body.append(chip(x, 78, w, 34, "grey"))
    x += w + 6

body.append(arrow(456, 95, 516, 95))
body.append(label(486, 74, "morselize", anchor="middle", size=11.5))
body.append(note(486, 132, "split / coalesce", anchor="middle"))

body.append(note(733, 64, "full at 16,384 rows OR 1 MiB, whichever trips first", anchor="middle"))
x = 528
for _ in range(8):
    body.append(chip(x, 78, 46, 34))
    x += 52
body.append(note(733, 132, "one over-budget row becomes a one-row morsel", anchor="middle"))

# ---- Zoom 2: the pool ---------------------------------------------------------------
body.append(arrow(490, 156, 490, 194))
body.append(label(504, 180, "par_iter() over the morsel vector", size=11.5))

body.append(band(20, 200, 940, 190, "SCHEDULE: ONE POOL, ONE MORSEL PER TASK", "blue"))
body.append(note(490, 244, "W = operator_cores(), capped by the number of morsels the input can produce "
                           "(par::auto_width)", anchor="middle"))

WORKER_Y, WORKER_H = 258, 82
for i, wx in enumerate((44, 286, 528, 770)):
    body.append(card(wx, WORKER_Y, 196, WORKER_H, f"worker {i}", "one morsel at a time"))

body.append(curve(142, WORKER_Y + WORKER_H, 490, 386, 868, WORKER_Y + WORKER_H, "amber"))
body.append(label(490, 378, "an idle worker steals: rayon's scheduler, not Batcher's",
                  anchor="middle", size=11.5))

# ---- Zoom 3: what one worker does ---------------------------------------------------
body.append(arrow(490, 392, 490, 424))
body.append(label(504, 412, "one task = one morsel", size=11.5))

body.append(band(20, 430, 940, 116, "ONE WORKER, ONE MORSEL, ONE PASS", "amber"))
body.append(chip(46, 468, 74, 56))
body.append(note(83, 540, "morsel in", anchor="middle"))
body.append(arrow(126, 496, 186, 496))
body.append(label(156, 484, "rows", anchor="middle", size=11.5))
body.append(card(192, 468, 218, 56, "filter", "JIT compiled once"))
body.append(arrow(416, 496, 476, 496))
body.append(label(446, 484, "survivors", anchor="middle", size=11.5))
body.append(card(482, 468, 218, 56, "project", "never materialized"))
body.append(arrow(706, 496, 766, 496))
body.append(label(736, 484, "output", anchor="middle", size=11.5))
body.append(chip(772, 468, 74, 56))
body.append(note(809, 540, "morsel out", anchor="middle"))
body.append(
    f'<text x="864" y="492" font-family="{FONT}" font-size="11.5" class="t-sub">collected in</text>'
    f'<text x="864" y="508" font-family="{FONT}" font-size="11.5" class="t-sub">index order</text>'
)

body.append(note(490, 572, "Filter and project preserve row order because the morsels are collected in index "
                           "order. The hash operators do not.", anchor="middle"))

write("morsel_scheduling", svg(W, H, "".join(body)))
print("wrote morsel_scheduling.svg")
