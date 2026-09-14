#!/usr/bin/env python3
"""Draw `cross_run_learning.svg` -- the loop that outlives one query.

`carbonite_loop.svg`, already on the same page, draws the Kyber-Carbonite-Core hand-off
*inside* one query. This one draws the other axis, and the axis is the point: what Core
measured is written to the `MetadataHub` and read by the **next** run, in another
process, possibly days later. That is the half of the adaptive story with no DuckDB and
no Spark equivalent -- Spark AQE keeps nothing once the query ends -- and it is one of
the only two things `.claude/rules/performance.md` sanctions claiming. So the across-runs
axis is drawn as an axis, not implied by a curved arrow.

Source, and what to keep in step:

* `python/batcher/metadata/hub.py`, `store.py`, `backends/` -- the hub itself.
* `python/batcher/kyber/learning.py` -- what is recorded and read back: measured output
  cardinality per plan signature, per-column sketches (NDV, quantiles, most-common
  values, row bytes), measured selectivities and widths, and the q-error correction.
* `python/batcher/kyber/calibration.py` -- cost coefficients fitted from Core's
  `op_stats`.
* `python/batcher/kyber/learned_tuning/` -- the bandit arm statistics.
* `python/batcher/kyber/signature.py` and `metadata/hardware_scope.py` -- an entry is
  keyed by the plan signature, and anything measured in machine units is additionally
  scoped by the hardware fingerprint so unlike machines never blend.

Core measures, Kyber consumes. Nothing in this diagram may show Kyber collecting runtime
metadata or Core making an optimization decision.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 530

body = [
    # ---- run N -------------------------------------------------------------
    band(20, 56, 340, 280, "RUN N", "blue"),
    card(44, 110, 292, 80, "Kyber plans", "on whatever it knows now"),
    card(44, 222, 292, 80, "Core measures", "rows, times, column sketches"),
    arrow(190, 190, 190, 222, "blue"),
    label(202, 212, "executes it", anchor="start"),
    # ---- the thing in the middle -------------------------------------------
    card(390, 150, 220, 140, "MetadataHub", "keyed by plan signature"),
    note(500, 246, "and, for anything in machine", anchor="middle"),
    note(500, 264, "units, by hardware fingerprint", anchor="middle"),
    note(500, 338, "Core writes after every run; Kyber reads before every plan.", anchor="middle"),
    note(500, 366, "measured cardinalities  ·  operator wall times", anchor="middle"),
    note(
        500,
        386,
        "column sketches  ·  fitted cost coefficients  ·  bandit arm rewards",
        anchor="middle",
    ),
    # ---- run N + 1 ---------------------------------------------------------
    band(640, 56, 320, 280, "RUN N + 1", "blue"),
    card(664, 110, 272, 80, "Kyber plans again", "on measured numbers"),
    card(664, 222, 272, 80, "Core measures", "and records again"),
    arrow(800, 190, 800, 222, "blue"),
    label(812, 212, "executes it", anchor="start"),
    arrow(336, 258, 390, 232, "amber"),
    label(363, 222, "writes", anchor="middle"),
    arrow(610, 198, 664, 166, "amber"),
    label(637, 140, "reads", anchor="middle"),
    # ---- the axis that makes it different ----------------------------------
    arrow(40, 442, 940, 442, "grey"),
    label(40, 428, "the query ends; the hub does not", anchor="start"),
    note(940, 428, "a later query, another process, minutes or days on", anchor="end"),
    note(
        490,
        490,
        "This is the same stage-boundary mechanism Spark AQE uses, with one difference worth claiming: AQE keeps nothing once the",
        anchor="middle",
    ),
    note(
        490,
        510,
        "query finishes, so it re-learns the same shape every time. Core measures, Kyber consumes -- and never the other way round.",
        anchor="middle",
    ),
]

write("cross_run_learning", svg(W, H, "".join(body)))
print("wrote cross_run_learning.svg")
