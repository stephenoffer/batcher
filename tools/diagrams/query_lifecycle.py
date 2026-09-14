#!/usr/bin/env python3
"""Draw `query_lifecycle.svg` — the contract loop a terminal op actually drives.

This is the subsystem hand-off view. `lifecycle.svg` already covers the reader's
first question ("when does anything run?") as a flat lazy-to-collect chain; this one
answers the second, which is who does what to the plan on the way through, and it is
a *ring* rather than a chain because the last step feeds the first.

Source of truth:

* `python/batcher/api/orchestration/run.py::run_relational` — the sequence, and the
  module docstring that names it: "Kyber optimizes, Carbonite admits, Core executes,
  metadata flows back".
* `python/batcher/api/orchestration/phases.py::PHASE_LABELS` — the machine phase
  names printed under each card, in the order the query passes through them.
* `run.py::_admit` — the branch. A verdict that is infeasible *on memory* is a
  counter-offer: the plan routes out-of-core. Any other binding constraint raises
  `PlanError`, because spilling is no remedy for it.
* `.claude/rules/architecture.md` — the three verbs, which are the point of the
  picture: Core measures, Kyber decides, Carbonite protects.

The two things prose keeps losing, and the reason this is worth a diagram: admission
has a *third* outcome besides pass and fail, and the loop closes — what Core measured
is read by Kyber on the next run, not this one.

Form: a clockwise ring, with the out-of-core detour drawn as a real alternative route
between the same two cards rather than as a footnote.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, GREY, card, label, note, svg, write

W, H = 980, 540

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

CW, CH = 230, 96
LEFT_X, RIGHT_X = 150, 600
TOP_Y, BOT_Y = 90, 340

DET_X, DET_Y, DET_W, DET_H = 395, 206, 190, 76


def phase(x: float, y: float, text: str) -> str:
    """The machine phase name, under a card, in the engine's own spelling."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{MONO}" font-size="10.5" '
        f'fill="{BLUE}">{text}</text>'
    )


def verb(x: float, y: float, text: str) -> str:
    """The one word that keeps a subsystem in its lane."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{FONT}" font-size="12.5" '
        f'font-weight="700" fill="{AMBER_DEEP}" letter-spacing="1.2">{text}</text>'
    )


body = [
    # ---- The four stations -------------------------------------------------
    card(LEFT_X, TOP_Y, CW, CH, "Kyber", "rules, join order, bounds"),
    verb(LEFT_X + CW / 2, TOP_Y + 26, "DECIDES"),
    phase(LEFT_X + CW / 2, TOP_Y + CH - 12, "kyber.optimize_full"),
    card(RIGHT_X, TOP_Y, CW, CH, "Carbonite", "does this fit the envelope?"),
    verb(RIGHT_X + CW / 2, TOP_Y + 26, "PROTECTS"),
    phase(RIGHT_X + CW / 2, TOP_Y + CH - 12, "carbonite.validate"),
    card(RIGHT_X, BOT_Y, CW, CH, "Core", "runs it, reports what happened"),
    verb(RIGHT_X + CW / 2, BOT_Y + 26, "MEASURES"),
    phase(RIGHT_X + CW / 2, BOT_Y + CH - 12, "core.execute"),
    card(LEFT_X, BOT_Y, CW, CH, "MetadataHub", "sketches and measured rows"),
    verb(LEFT_X + CW / 2, BOT_Y + 26, "REMEMBERS"),
    phase(LEFT_X + CW / 2, BOT_Y + CH - 12, "collect_source_metadata"),
]

# ---- The ring, clockwise ---------------------------------------------------
MID_X = (LEFT_X + CW + RIGHT_X) / 2
body += [
    # Kyber -> Carbonite
    f'<path d="M {LEFT_X + CW + 8} {TOP_Y + CH / 2} L {RIGHT_X - 10} {TOP_Y + CH / 2}" '
    f'fill="none" stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arB)"/>',
    label(MID_X, TOP_Y + CH / 2 - 16, "a PhysicalPlan", anchor="middle", size=11.5),
    note(MID_X, TOP_Y + CH / 2 + 26, "with a resource bound per operator", anchor="middle"),
    # Carbonite -> Core, the direct route
    f'<path d="M {RIGHT_X + CW / 2} {TOP_Y + CH + 8} L {RIGHT_X + CW / 2} {BOT_Y - 10}" '
    f'fill="none" stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arB)"/>',
    label(RIGHT_X + CW / 2 + 14, TOP_Y + CH + 58, "it fits:", size=11.5),
    note(RIGHT_X + CW / 2 + 14, TOP_Y + CH + 76, "reserve, then run"),
    # Core -> MetadataHub
    f'<path d="M {RIGHT_X - 10} {BOT_Y + CH / 2} L {LEFT_X + CW + 8} {BOT_Y + CH / 2}" '
    f'fill="none" stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arB)"/>',
    label(MID_X, BOT_Y + CH / 2 - 16, "per-operator metrics", anchor="middle", size=11.5),
    note(MID_X, BOT_Y + CH / 2 + 26, "actual rows, time, peak bytes", anchor="middle"),
    # MetadataHub -> Kyber, the edge that makes it a loop
    f'<path d="M {LEFT_X + CW / 2} {BOT_Y - 10} L {LEFT_X + CW / 2} {TOP_Y + CH + 8}" '
    f'fill="none" stroke="{AMBER_DEEP}" stroke-width="2.4" stroke-dasharray="6 4" '
    f'marker-end="url(#arA)"/>',
    f'<text x="{LEFT_X + CW / 2 - 16}" y="{TOP_Y + CH + 58}" text-anchor="end" '
    f'font-family="{FONT}" font-size="11.5" font-weight="700" fill="{AMBER_DEEP}">'
    f"read on the</text>",
    f'<text x="{LEFT_X + CW / 2 - 16}" y="{TOP_Y + CH + 76}" text-anchor="end" '
    f'font-family="{FONT}" font-size="11.5" font-weight="700" fill="{AMBER_DEEP}">'
    f"next run, not this one</text>",
]

# ---- The third outcome of admission ---------------------------------------
body += [
    card(DET_X, DET_Y, DET_W, DET_H, "Out-of-core", "a counter-offer"),
    f'<path d="M {RIGHT_X + 30} {TOP_Y + CH + 8} L {DET_X + DET_W + 10} {DET_Y + 18}" '
    f'fill="none" stroke="{AMBER_DEEP}" stroke-width="2.2" marker-end="url(#arA)"/>',
    f'<path d="M {DET_X + DET_W + 10} {DET_Y + DET_H - 14} L {RIGHT_X + 46} {BOT_Y - 10}" '
    f'fill="none" stroke="{AMBER_DEEP}" stroke-width="2.2" marker-end="url(#arA)"/>',
    note(DET_X + DET_W / 2, DET_Y - 16, "won't fit in memory: spill, then run", anchor="middle"),
    note(
        DET_X + DET_W / 2,
        DET_Y + DET_H + 20,
        "Any other binding constraint raises instead:",
        anchor="middle",
    ),
    note(
        DET_X + DET_W / 2,
        DET_Y + DET_H + 38,
        "spilling only ever answers a memory constraint.",
        anchor="middle",
    ),
]

# ---- In and out ------------------------------------------------------------
body += [
    f'<path d="M {LEFT_X + CW / 2} 44 L {LEFT_X + CW / 2} {TOP_Y - 10}" fill="none" '
    f'stroke="{GREY}" stroke-width="2.4" marker-end="url(#arG)"/>',
    note(LEFT_X + CW / 2, 34, "collect() / write(): the first real work", anchor="middle"),
    f'<path d="M {RIGHT_X + CW / 2} {BOT_Y + CH + 8} L {RIGHT_X + CW / 2} {BOT_Y + CH + 52}" '
    f'fill="none" stroke="{GREY}" stroke-width="2.4" marker-end="url(#arG)"/>',
    note(RIGHT_X + CW / 2, BOT_Y + CH + 70, "Arrow batches, zero-copy", anchor="middle"),
]

write("query_lifecycle", svg(W, H, "".join(body)))
print("wrote query_lifecycle.svg")
