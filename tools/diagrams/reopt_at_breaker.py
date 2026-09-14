#!/usr/bin/env python3
"""Draw `reopt_at_breaker.svg` -- what happens at ONE pipeline breaker.

`adaptive_loop.svg` draws the loop as a loop. This one zooms into a single turn of it
and draws the thing a loop diagram cannot hold: the *branch*. A stage is executed, its
exact cardinality is compared against the estimate that planned it, and the two outcomes
go opposite ways -- one stops cutting the pipeline, the other re-plans the residual.

Source, and what must be kept in step with it:

* `python/batcher/api/adaptive/staging.py::_staged_loop` -- the loop itself: pick the
  lowest breaker whose inputs all stream (`plan_surgery.lowest_breaker`), run it, splice
  a `Scan` over its result (`plan_surgery.replace`), repeat.
* `python/batcher/api/adaptive/gating.py::_estimate_accurate` -- the **symmetric**
  q-error test, against `optimizer.reoptimize_error` (2.0), so "held" means the measured
  size is within a factor of 3.0 either way. If that constant moves, move the label.
* The "17 of 51" figure is `_staged_loop`'s own recorded measurement over the 22 TPC-H
  shapes: a breaker whose output size is already known exactly is not worth a
  materialization, so it runs inline, fused into the subplan staged above it.

The claim discipline in `CLAUDE.md` and `.claude/rules/performance.md` applies: this is
stage-boundary adaptation at the same granularity as Spark AQE, not something finer.
Nothing here re-plans *within* a stage, and the diagram must not suggest it does.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 560

body = [
    band(20, 20, 940, 216, "ONE TURN OF THE LOOP  ·  AT A SINGLE PIPELINE BREAKER", "blue"),
    card(48, 88, 252, 88, "Plan the query", "a join operand sized by a guess"),
    card(364, 88, 252, 88, "Execute one breaker", "the lowest whose inputs all stream"),
    card(680, 88, 252, 88, "Exact row count", "measured on the materialized result"),
    arrow(300, 132, 364, 132, "blue"),
    label(332, 122, "cut here", anchor="middle"),
    arrow(616, 132, 680, 132, "blue"),
    label(648, 122, "measure", anchor="middle"),
    note(
        490,
        208,
        "Provenance.DEFAULT: the optimizer knows it is guessing this one.",
        anchor="middle",
    ),
    # ---- the branch: did the estimate hold? --------------------------------
    card(80, 312, 330, 88, "Stop cutting", "finish the rest in one shot"),
    card(570, 312, 330, 88, "Re-plan the residual", "on the size just measured"),
    arrow(760, 180, 420, 308, "grey"),
    label(556, 252, "held: within 3x", anchor="middle"),
    arrow(830, 180, 760, 308, "amber"),
    label(872, 250, "missed by more", anchor="start"),
    card(570, 440, 330, 84, "Build side, broadcast, join order", "chosen on rows, not on a guess"),
    arrow(735, 400, 735, 440, "amber"),
    label(750, 426, "re-optimize", anchor="start"),
    note(80, 440, "The splice is a Scan over the stage's", anchor="start"),
    note(80, 458, "result, so the next stage's estimator", anchor="start"),
    note(80, 476, "reads an exact size rather than one", anchor="start"),
    note(80, 494, "more inherited guess.", anchor="start"),
    note(
        490,
        526,
        '"Held" is the symmetric q-error, against optimizer.reoptimize_error (2.0). Same mechanism and granularity as Spark AQE -- nothing re-plans inside a stage.',
        anchor="middle",
    ),
    note(
        490,
        546,
        "A breaker whose output size is already known exactly is not cut at all: 17 of 51 across the 22 TPC-H shapes ran inline, fused into the subplan above them.",
        anchor="middle",
    ),
]

write("reopt_at_breaker", svg(W, H, "".join(body)))
print("wrote reopt_at_breaker.svg")
