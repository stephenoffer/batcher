#!/usr/bin/env python3
"""Draw `plan_lowering.svg` — the whole lowering chain, and where the planes meet.

Source of truth for each stage:

* `python/batcher/api/dataset/` — the fluent, lazy, immutable `Dataset`.
* `python/batcher/plan/logical/base.py` — `LogicalPlan`, validated at build time,
  with a memoized `to_ir()` per node.
* `python/batcher/kyber/rule.py::Phase` — the seven phases, in declared order.
  NORMALIZE, REWRITE, PUSHDOWN and FUSION iterate to a fixpoint; JOIN_REORDER,
  SELECTION and ENFORCE run once, because they make a decision rather than converge.
* `python/batcher/kyber/optimizer/facade.py::Optimizer.optimize_full` — builds the
  `PhysicalPlan` from the rewritten tree's `to_ir()` plus `annotate_ops`.
* `python/batcher/plan/physical.py::PhysicalPlan.to_json` — the one serialization.
* `crates/bc-py/src/lib.rs::execute_plan` — `(plan_json, sources)` in, Arrow
  `RecordBatch`es out, zero-copy through the Arrow C Data Interface.
* `crates/bc-ir/src/lib.rs::RelOp` — what serde deserializes the document into.

The single fact the diagram exists for is the divider: everything above it is a
*decision* about work, everything below it is the work. No row is touched above the
line, and no plan is chosen below it. A chain of stages drawn without that line reads
as seven equivalent steps, which is exactly the reading the architecture forbids.

Form: two bands, one per plane, with one labelled edge crossing between them.
"""

from __future__ import annotations

from _authoring import (
    AMBER_DEEP,
    BLUE,
    FONT,
    GREY,
    band,
    card,
    label,
    note,
    svg,
    write,
)

W, H = 980, 500

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

# Control-plane row: four narrow cards with wide gaps, so every edge gets a real label.
TOP_Y, TOP_H, TOP_W, TOP_GAP = 76, 84, 160, 92
TOP_X = [42 + i * (TOP_W + TOP_GAP) for i in range(4)]

# Data-plane row: three wider cards, centered under the first.
BOT_Y, BOT_H, BOT_W, BOT_GAP = 314, 84, 210, 110
BOT_X = [75 + i * (BOT_W + BOT_GAP) for i in range(3)]

DIVIDER_Y = 204


def mono(x: float, y: float, text: str) -> str:
    """A code-voiced line under a card title."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{MONO}" font-size="10.5" '
        f'fill="{BLUE}">{text}</text>'
    )


body = [
    band(20, 20, 940, 172, "CONTROL PLANE  ·  PYTHON  ·  DECIDES WHAT THE WORK IS", "blue"),
    card(TOP_X[0], TOP_Y, TOP_W, TOP_H, "Dataset", "lazy, immutable"),
    card(TOP_X[1], TOP_Y, TOP_W, TOP_H, "LogicalPlan", "validated tree"),
    card(TOP_X[2], TOP_Y, TOP_W, TOP_H, "Kyber", "seven phases"),
    card(TOP_X[3], TOP_Y, TOP_W, TOP_H, "PhysicalPlan", "IR + bounds"),
    mono(TOP_X[2] + TOP_W / 2, TOP_Y + TOP_H - 10, "NORMALIZE .. ENFORCE"),
]

TOP_MID = TOP_Y + TOP_H / 2
top_edges = [
    ("each operation", "returns a new node"),
    ("rules rewrite", "plan to plan"),
    ("to_ir() lowers", "the rewritten tree"),
]
for i, (line1, line2) in enumerate(top_edges):
    x1 = TOP_X[i] + TOP_W + 8
    x2 = TOP_X[i + 1] - 10
    mid = (x1 + x2) / 2
    body += [
        f'<path d="M {x1} {TOP_MID} L {x2} {TOP_MID}" fill="none" stroke="{BLUE}" '
        f'stroke-width="2.4" marker-end="url(#arB)"/>',
        label(mid, TOP_MID - 16, line1, anchor="middle", size=11.5),
        note(mid, TOP_MID + 28, line2, anchor="middle"),
    ]

# ---- The divider. This is the diagram. -----------------------------------
body += [
    f'<path d="M 20 {DIVIDER_Y} H 960" fill="none" stroke="{AMBER_DEEP}" stroke-width="2" '
    f'stroke-dasharray="9 6"/>',
    f'<text x="20" y="{DIVIDER_Y - 12}" font-family="{FONT}" font-size="12.5" '
    f'font-weight="700" fill="{AMBER_DEEP}">'
    f'The boundary: one JSON document and zero-copy Arrow, and nothing else.</text>',
]

# The crossing edge, routed down the right margin and back along under the divider.
cross_x = TOP_X[3] + TOP_W / 2
body += [
    f'<path d="M {cross_x} {TOP_Y + TOP_H + 8} V 244 H {BOT_X[0] + BOT_W / 2} V {BOT_Y - 10}" '
    f'fill="none" stroke="{AMBER_DEEP}" stroke-width="2.6" marker-end="url(#arA)"/>',
    f'<text x="{cross_x - 14}" y="238" text-anchor="end" font-family="{FONT}" font-size="12" '
    f'font-weight="700" fill="{AMBER_DEEP}">to_json(), plus the Arrow input batches</text>',
]

# ---- Data plane ----------------------------------------------------------
body += [
    band(20, 258, 940, 172, "DATA PLANE  ·  RUST + ARROW  ·  DOES THE WORK", "grey"),
    card(BOT_X[0], BOT_Y, BOT_W, BOT_H, "execute_plan", "the one FFI entry"),
    mono(BOT_X[0] + BOT_W / 2, BOT_Y + BOT_H - 10, "bc-py"),
    card(BOT_X[1], BOT_Y, BOT_W, BOT_H, "RelOp tree", "one relational type"),
    mono(BOT_X[1] + BOT_W / 2, BOT_Y + BOT_H - 10, "bc-ir"),
    card(BOT_X[2], BOT_Y, BOT_W, BOT_H, "Operators", "over Arrow morsels"),
    mono(BOT_X[2] + BOT_W / 2, BOT_Y + BOT_H - 10, "bc-interp / bc-runtime"),
]

BOT_MID = BOT_Y + BOT_H / 2
bot_edges = [
    ("serde deserializes", "unknown tag: hard error"),
    ("walked once", "16,384-row morsels"),
]
for i, (line1, line2) in enumerate(bot_edges):
    x1 = BOT_X[i] + BOT_W + 8
    x2 = BOT_X[i + 1] - 10
    mid = (x1 + x2) / 2
    body += [
        f'<path d="M {x1} {BOT_MID} L {x2} {BOT_MID}" fill="none" stroke="{BLUE}" '
        f'stroke-width="2.4" marker-end="url(#arB)"/>',
        label(mid, BOT_MID - 16, line1, anchor="middle", size=11.5),
        note(mid, BOT_MID + 28, line2, anchor="middle"),
    ]

# The result, returning across the same boundary.
body += [
    f'<path d="M {BOT_X[2] + BOT_W / 2} {BOT_Y - 10} V 276" fill="none" stroke="{GREY}" '
    f'stroke-width="2.4" stroke-dasharray="5 4" marker-end="url(#arG)"/>',
    note(BOT_X[2] + BOT_W / 2, 272, "result batches return the same way", anchor="middle"),
    note(490, 464, "Python never sees a row on the way down and never copies one on the way back.",
         anchor="middle"),
    note(490, 484, "A per-row loop above the line is the one thing this shape is designed to prevent.",
         anchor="middle"),
]

write("plan_lowering", svg(W, H, "".join(body)))
print("wrote plan_lowering.svg")
