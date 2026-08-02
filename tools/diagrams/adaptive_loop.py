#!/usr/bin/env python3
"""Draw `adaptive_loop.svg` — the two feedback loops, and where each one stops.

This diagram carries a claim the project is careful about, so it is drawn to the
wording `docs/architecture/internals/competitive_architecture.md` sanctions rather than the older
marketing line. The inner loop is stage-boundary re-optimization at the same
granularity as Spark AQE, gated off below `_ADAPTIVE_MIN_INPUT_ROWS`
(`python/batcher/api/adaptive/gating.py`, 20,000,000 rows). The outer loop is the
cross-query learned-stats loop, which is the half with no DuckDB or Spark equivalent.

If either the gate constant or the retired-claim wording changes, change this too.

Layout: the inner band reserves a clear channel under its cards for the return curve,
so the caveat text sits below the curve rather than across it.
"""

from __future__ import annotations

from _authoring import arrow, band, card, curve, label, note, svg, write

W, H = 980, 560

ROW1_Y = 76           # inner-loop card row
ROW1_BOTTOM = ROW1_Y + 84
RETURN_APEX = 214     # the feedback curve's low point, clear of the cards
ROW2_Y = 372          # outer-loop card row

body = [
    # ---- Inner loop: within one query -------------------------------------
    band(20, 20, 940, 274, "WITHIN ONE QUERY  ·  STAGE-BOUNDARY RE-OPTIMIZATION", "blue"),
    card(48, ROW1_Y, 190, 84, "Plan", "estimated rows"),
    card(268, ROW1_Y, 190, 84, "Execute a stage", "to a pipeline breaker"),
    card(488, ROW1_Y, 190, 84, "Measure", "actual cardinalities"),
    card(708, ROW1_Y, 224, 84, "Re-plan the rest", "on real numbers"),
    arrow(238, 118, 268, 118, "blue"),
    arrow(458, 118, 488, 118, "blue"),
    arrow(678, 118, 708, 118, "blue"),
    curve(820, ROW1_BOTTOM, 490, RETURN_APEX, 143, ROW1_BOTTOM, "blue"),
    label(490, 208, "the remaining stages, re-planned", anchor="middle"),
    note(490, 250, "Same mechanism and granularity as Spark AQE, and available single-node.", anchor="middle"),
    note(490, 270, "Gated off below 20,000,000 input rows, so most queries never reach it.", anchor="middle"),

    # ---- Outer loop: across runs ------------------------------------------
    band(20, 318, 940, 218, "ACROSS RUNS  ·  LEARNED STATISTICS", "amber"),
    card(48, ROW2_Y, 244, 84, "Record what happened", "sketches, not raw rows"),
    card(342, ROW2_Y, 244, 84, "MetadataHub", "cross-query learned stats"),
    card(636, ROW2_Y, 296, 84, "The next run plans better", "bandit picks what worked"),
    arrow(292, 414, 342, 414, "amber"),
    arrow(586, 414, 636, 414, "amber"),
    note(490, 500, "Core measures, Kyber consumes. This half is what neither DuckDB nor Spark has.", anchor="middle"),
]

write("adaptive_loop", svg(W, H, "".join(body)))
print("wrote adaptive_loop.svg")
