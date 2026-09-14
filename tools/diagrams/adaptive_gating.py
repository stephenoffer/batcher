#!/usr/bin/env python3
"""Draw `adaptive_gating.svg` -- when the within-query adaptive loop engages at all.

This is the diagram that stops the overclaim, so every number in it is read from
`python/batcher/api/adaptive/gating.py` and nothing is rounded in the drawing's favour:

* `_ADAPTIVE_MIN_ROWS_PER_STAGE = 5_000_000`
* `_ADAPTIVE_MIN_BYTES_PER_STAGE = _ADAPTIVE_MIN_ROWS_PER_STAGE * 64` -- 320,000,000
  bytes, about 320 MB. The two terms are OR'd, so a query clears whichever suits it.
* The floor is charged **per pipeline breaker the loop would cut at** (`_stage_count`,
  which is `max(1, <BREAKERS in the plan>)`), so a two-breaker plan qualifies at 10M
  rows, a four-breaker one at 20M and a six-breaker one at 30M. That table is
  `_ADAPTIVE_MIN_ROWS_PER_STAGE`'s own comment.
* `joins(plan)` must be non-empty, and both size terms read EXACT source row counts.

The flat 20,000,000-row whole-query gate this replaced is retired. Do not draw it back.

The order is `resolve_adaptive`'s, and so are the two bypasses drawn above the ladder.
An explicit `adaptive=` wins outright. On a distributed run, a plan the one-shot
dispatcher cannot route at all is staged whatever its size -- a join whose operand spans
two sources, or a breaker beneath a breaker (`dist.requires_staging`) -- because there
staging is the only execution path rather than an optimization, and that second shape
needs no join. So "no join, never adaptive" is a statement about this ladder, not about
every route into the loop, and the drawing has to say which.

The route bandit that can turn staging back off has its own diagram (`bandit_tuning`)
and appears here only as the override it is.
"""

from __future__ import annotations

from _authoring import FONT, arrow, band, card, label, note, svg, write

W, H = 980, 644

GATE_Y, GATE_H = 140, 104
G1, G2, G3 = 40, 350, 660
GW = 280

body = [
    band(20, 20, 940, 294, "adaptive=\"auto\"  ·  WHEN THE WITHIN-QUERY LOOP ENGAGES", "blue"),
    note(490, 72, "Two things skip the ladder: an explicit adaptive=True or False wins outright, and a distributed plan the one-shot dispatcher", anchor="middle"),
    note(490, 92, "cannot route is staged whatever its size, because there staging is the only execution path rather than an optimization.", anchor="middle"),

    card(G1, GATE_Y, GW, GATE_H, "Is there a join?", "no join: nothing to re-decide"),
    card(G2, GATE_Y, GW, GATE_H, "Does it clear the floor?", "per breaker, not per query"),
    card(G3, GATE_Y, GW, GATE_H, "Is an operand unsized?", "breaker-produced, and still a guess"),
    arrow(320, GATE_Y + 52, 350, GATE_Y + 52, "blue"),
    label(335, GATE_Y + 42, "yes", anchor="middle"),
    arrow(630, GATE_Y + 52, 660, GATE_Y + 52, "blue"),
    label(645, GATE_Y + 42, "yes", anchor="middle"),

    # ---- every "no" goes to the same place ---------------------------------
    band(20, 344, 600, 96, "ANY \"NO\"  ·  PLAN ONCE, RUN ONCE", "grey"),
    note(44, 392, "No staging, no per-stage cut, no re-plan. This is where the great", anchor="start"),
    note(44, 412, "majority of queries land, and it is the cheaper path for them.", anchor="start"),
    arrow(180, GATE_Y + GATE_H, 180, 344, "grey"),
    label(192, 294, "no", anchor="start"),
    arrow(490, GATE_Y + GATE_H, 490, 344, "grey"),
    label(502, 294, "no", anchor="start"),
    arrow(760, GATE_Y + GATE_H, 610, 340, "grey"),
    label(660, 294, "no", anchor="start"),

    # ---- the "yes" outcome --------------------------------------------------
    card(660, 348, 280, 88, "Stage, measure, re-plan", "one breaker per stage"),
    arrow(870, GATE_Y + GATE_H, 870, 348, "blue"),
    label(882, 294, "yes", anchor="start"),
    note(660, 462, "...unless the route bandit, having", anchor="start"),
    note(660, 482, "measured both arms for this plan", anchor="start"),
    note(660, 502, "signature, says one-shot was faster.", anchor="start"),
]

# ---- the per-stage floor, spelled out -------------------------------------
FX, FY = 20, 512
body += [
    band(FX, FY, 940, 112, "THE FLOOR IS CHARGED PER BREAKER, NOT PER QUERY", "amber"),
    note(FX + 24, FY + 46, "5,000,000 rows  OR  about 320 MB  --  times the pipeline", anchor="start"),
    note(FX + 24, FY + 64, "breakers the loop would cut at. A cut is what staging costs,", anchor="start"),
    note(FX + 24, FY + 82, "so a plan with more cuts must be larger to earn them.", anchor="start"),
    note(FX + 24, FY + 100, "The flat 20,000,000-row whole-query gate is retired.", anchor="start"),
]

ROWS = (("2 breakers", "10,000,000 rows"), ("4 breakers", "20,000,000 rows"), ("6 breakers", "30,000,000 rows"))
for i, (left, right) in enumerate(ROWS):
    y = FY + 40 + i * 26
    body += [
        f'<text x="672" y="{y}" font-family="{FONT}" font-size="12" font-weight="700" '
        f'class="t-arrow">{left}</text>',
        f'<text x="940" y="{y}" text-anchor="end" font-family="{FONT}" font-size="12" '
        f'class="t-sub">{right}</text>',
    ]
body.append(note(672, FY + 100, "or the byte floor, whichever suits the shape.", anchor="start"))

write("adaptive_gating", svg(W, H, "".join(body)))
print("wrote adaptive_gating.svg")
