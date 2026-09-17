#!/usr/bin/env python3
"""Draw `streaming_tutorial_flow.svg`: the streaming tutorial's pipeline, one stage per step.

Source of truth: `docs/getting-started/tutorials/pipelines/streaming-pipeline.md`. The
numbers in the circles are that page's step numbers. Step 1 builds the unbounded source
with `bt.from_batches(..., bounded=False)`, which cannot `collect()`. Step 3 dedupes with
`drop_duplicates_within_watermark`, which forgets keys the watermark has passed. Step 4
sets `with_watermark` and groups by `bt.window`, where the watermark is
`max(event_time) - lateness` and a window closes once the watermark passes its end.
Step 5 writes with a `Trigger`; step 6 adds `checkpoint=`, so a restart resumes at the
last committed offset.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, step, svg, tint, write

W, H = 980, 640

CARD_X, CARD_W, CARD_H = 64, 330, 72
GAP = 40
TOP = 72
NOTE_X, NOTE_W = 470, 466
BAND_H = TOP + 5 * CARD_H + 4 * GAP + 22 - 20

stages = [
    (
        1,
        "Unbounded source",
        "from_batches(..., bounded=False)",
        "Never ends",
        "consume it with a sink or iter_batches, not collect()",
    ),
    (
        3,
        "Deduplicate",
        "drop_duplicates_within_watermark",
        "Bounded state",
        "keys the watermark has passed are forgotten",
    ),
    (
        4,
        "Window by event time",
        "with_watermark, then bt.window",
        "Watermark = max(event_time) - lateness",
        "a window closes once the watermark passes its end",
    ),
    (
        5,
        "Write with a trigger",
        "available_now or processing_time",
        "When micro-batches run",
        "drain what is there and stop, or run on a clock",
    ),
    (
        6,
        "Sink with a checkpoint",
        "memory, Parquet files, or Delta",
        "Survives a restart",
        "resumes at the last committed offset",
    ),
]
edges = ["Arrow batches", "first row per key", "window aggregates", "micro-batches"]

body = [
    band(20, 20, 400, BAND_H, "THE PIPELINE", "blue"),
    band(446, 20, 514, BAND_H, "WHAT TO KNOW AT EACH STAGE", "grey"),
]
for i, (n, title, sub, head, detail) in enumerate(stages):
    y = TOP + i * (CARD_H + GAP)
    body += [
        card(CARD_X, y, CARD_W, CARD_H, title, sub),
        step(CARD_X, y + 4, n),
        tint(NOTE_X, y, NOTE_W, CARD_H, head, detail),
    ]
    if i < len(edges):
        ay = y + CARD_H
        body += [
            arrow(CARD_X + CARD_W / 2, ay, CARD_X + CARD_W / 2, ay + GAP - 4),
            label(CARD_X + CARD_W / 2 + 12, ay + GAP / 2 + 4, edges[i], size=11.5),
        ]

write("streaming_tutorial_flow", svg(W, H, "".join(body)))
print("wrote streaming_tutorial_flow.svg")
