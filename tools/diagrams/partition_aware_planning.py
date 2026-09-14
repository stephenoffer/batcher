#!/usr/bin/env python3
"""Draw `partition_aware_planning.svg` -- the four tests that let a read replace a shuffle.

Source of truth, read before drawing:
  * `python/batcher/io/splits/clustering.py::declared_clustering` -- the columns *every* split
    in the set holds constant, or `()`. One split declaring `day` beside one declaring nothing
    leaves no column every row can be located by.
  * `python/batcher/dist/executors/partition_io/assignment.py` -- `group_by_clustering` groups
    are bin-packed as indivisible units, which is what *establishes* the co-location the split
    set can only declare.
  * `python/batcher/kyber/properties.py` -- `clustered_on` propagates a clustering through
    Filter / Project / unlimited Distinct only, and `satisfies` holds the containment: a
    delivered partitioning satisfies a grouping when its keys are a **subset** of the required
    ones.
  * `python/batcher/dist/executors/map.py` -- `scan_clustering_for`, and the two parallelism
    floors it applies: `_MIN_ALIGNED_TASKS` (2) and `_MIN_PARALLELISM_RETENTION` (4, i.e. keep
    at least a quarter of the shuffle's task count). Both capped by the fleet.

Why this is a figure. The decision has a property prose keeps losing: **three of the four
tests are correctness and the fourth is not.** Getting one of the first three wrong does not
make a query slow, it makes it wrong -- a group split across two workers comes back as two
rows, each a partial sum labelled as final, which no single-node test can see. The fourth is
a scheduling judgment where being wrong costs time. Drawing them in one row with the verdict
under each is the only form that holds both facts at once.

The last band is the half a reader needs and the decision itself cannot show: what puts a
layout's guarantee back out of reach.
"""

from __future__ import annotations

from _authoring import BLUE_MID, GREY, arrow, band, card, label, note, svg, write

W, H = 980, 640


def chip(x: float, y: float, w: float, h: float, kind: str = "blue") -> str:
    """A split, or a directory of them."""
    fill = {"blue": BLUE_MID, "grey": GREY}[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
        f'fill-opacity="0.28" stroke="{fill}" stroke-width="1.4"/>'
    )


body: list[str] = []

# ---- What the read already established ------------------------------------------------
body.append(band(20, 24, 940, 128, "WHAT THE LAYOUT ALREADY DID", "grey"))

for i, day in enumerate(("day=08-01", "day=08-02", "day=08-03")):
    x = 44 + i * 150
    body.append(chip(x, 58, 134, 44))
    body.append(note(x + 67, 85, day, anchor="middle"))
body.append(note(217, 126, "one directory per value", anchor="middle"))

body.append(arrow(500, 80, 560, 80))
body.append(label(530, 67, "assigned whole", anchor="middle", size=11.5))

for i in range(3):
    x = 566 + i * 134
    body.append(chip(x, 58, 118, 44))
    body.append(note(x + 59, 85, f"worker {i}", anchor="middle"))
body.append(note(763, 126, "every row for a value lands on one worker --", anchor="middle"))
body.append(note(763, 143, "exactly what a shuffle by that column arranges", anchor="middle"))

# ---- The four tests --------------------------------------------------------------------
body.append(arrow(490, 166, 490, 198))
body.append(label(504, 190, "so: may the exchange go?", size=11.5))

body.append(band(20, 204, 940, 214, "FOUR TESTS, AND ONLY THE LAST IS ABOUT SPEED", "blue"))

CARD_Y, CARD_H, CARD_W = 238, 60, 190
tests = (
    (44, "same columns?", "declared_clustering",
     ("Every split holds the same", "columns constant, or none do.")),
    (266, "assigned together?", "group_by_clustering",
     ("A value's splits are one unit.", "No split can promise this.")),
    (488, "keys contain them?", "properties.satisfies",
     ("Clustering must be a subset", "of the group keys.")),
    (710, "enough tasks left?", "scan_clustering_for",
     ("At least 2, and a quarter of", "the shuffle's. A judgment.")),
)
for i, (x, title, where, lines) in enumerate(tests):
    cx = x + CARD_W / 2
    body.append(card(x, CARD_Y, CARD_W, CARD_H, title, where))
    for j, line in enumerate(lines):
        body.append(note(cx, 318 + j * 17, line, anchor="middle"))
    drop = cx if i < 3 else cx - 46
    body.append(arrow(drop, 360, drop, 392, "grey"))
    body.append(label(drop + 9, 382, "no", size=11.5))
    if i < 3:
        body.append(arrow(x + CARD_W + 2, CARD_Y + CARD_H / 2, x + CARD_W + 28, CARD_Y + CARD_H / 2))
        body.append(label(x + CARD_W + 15, CARD_Y - 6, "yes", anchor="middle", size=11.5))

# The failure rail: any one test failing lands on the same plan.
body.append(f'<path d="M 139 394 H 759" fill="none" stroke="{GREY}" stroke-width="2.4"/>')
body.append(arrow(254, 394, 254, 444, "grey"))

# Passing all four is the only way out on the right.
body.append(arrow(851, 360, 851, 444))
body.append(label(864, 382, "yes, all four", size=11.5))

body.append(note(490, 404, "the first three are correctness -- a group split across two "
                           "workers returns two partial sums, each labelled final",
                 anchor="middle"))

# ---- The two outcomes ------------------------------------------------------------------
body.append(band(20, 428, 940, 110, "TWO PLANS, THE SAME ROWS", "amber"))
body.append(card(44, 452, 420, 60, "hash shuffle", "every row crosses the network"))
body.append(card(516, 452, 420, 60, "no exchange", "each worker folds its own directories"))
body.append(note(254, 530, "the fallback, and never wrong", anchor="middle"))
body.append(note(726, 530, "published to explain(analyze=True) as a core / exchange decision",
                 anchor="middle"))

# ---- What puts the guarantee back out of reach -----------------------------------------
body.append(band(20, 552, 940, 72, "WHAT UNCLAIMS THE LAYOUT", "grey"))
unclaims = (
    (44, "a glob path", "per-file splits record no partition value"),
    (370, "grouping below the split", "month= sits under every year= directory"),
    (680, "a Limit in the chain", "clustering places rows, it does not finish them"),
)
for x, title, why in unclaims:
    body.append(label(x, 590, title))
    body.append(note(x, 610, why))

write("partition_aware_planning", svg(W, H, "".join(body)))
print("wrote partition_aware_planning.svg")
