#!/usr/bin/env python3
"""Draw `tutorial_chooser.svg`: which tutorial to read, by what you are building.

Source of truth: `docs/getting-started/tutorials/index.md`. New readers start with "Your
first pipeline"; after that the tutorials are independent, grouped as Foundations, Data
pipelines and Machine learning, and the "Pick a tutorial" table maps each goal to one
tutorial. The learning paths sequence the tutorials for four roles for a reader who
wants an order.

Layout: the entry decision across the top, then three goal columns, one per group,
with each card naming the tutorial and the goal it answers.
"""

from __future__ import annotations

from _authoring import arrow, band, card, heading, hero, label, svg, tint, write

W, H = 980, 590

GROUPS = [
    (
        "FOUNDATIONS",
        [
            ("From SQL to DataFrames", "bring SQL habits"),
            ("Optimizing a slow query", "find why it's slow"),
        ],
    ),
    (
        "DATA PIPELINES",
        [
            ("Building a lakehouse", "a transactional table"),
            ("A streaming pipeline", "a source that never ends"),
            ("Synthetic data generation", "test data first"),
        ],
    ),
    (
        "MACHINE LEARNING",
        [
            ("Batch inference", "a model over a corpus"),
            ("RAG from scratch", "retrieval and generation"),
            ("Distributed training", "feed DDP ranks"),
            ("Feature engineering", "a feature matrix"),
        ],
    ),
]

BAND_Y, BAND_W, BAND_GAP = 222, 300, 20
ITEM_H, ITEM_GAP = 60, 12

body = [
    card(20, 30, 210, 76, "New to Batcher?", "learning the API"),
    arrow(230, 68, 304, 68, "amber"),
    label(267, 56, "yes", anchor="middle"),
    hero(310, 26, 300, 84, "Your first pipeline", "the lazy, expression-first model"),
    arrow(610, 68, 734, 68),
    label(672, 56, "want an order?", anchor="middle"),
    tint(740, 30, 220, 76, "Learning paths", "four roles, in order", "amber"),
    arrow(125, 106, 125, 176),
    label(137, 150, "no"),
    arrow(460, 110, 460, 176),
    label(472, 150, "then"),
    heading(490, 200, "PICK BY WHAT YOU ARE BUILDING", anchor="middle", kind="grey"),
]

for g, (title, items) in enumerate(GROUPS):
    x = 20 + g * (BAND_W + BAND_GAP)
    body.append(band(x, BAND_Y, BAND_W, 50 + len(items) * (ITEM_H + ITEM_GAP), title, "blue"))
    for i, (tutorial, goal) in enumerate(items):
        y = BAND_Y + 42 + i * (ITEM_H + ITEM_GAP)
        body.append(tint(x + 18, y, BAND_W - 36, ITEM_H, tutorial, goal))

write("tutorial_chooser", svg(W, H, "".join(body)))
print("wrote tutorial_chooser.svg")
