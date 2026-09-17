#!/usr/bin/env python3
"""Draw `training_data_flow.svg`: split, fit on train only, transform both, feed the loader.

Source of truth: `docs/getting-started/tutorials/ml/distributed-training-pipeline.md`,
steps 2, 3 and 6. `ds.ml.train_test_split(test_size=0.25, seed=7, key="user_id")` assigns
each row by a reproducible hash; `StandardScaler.fit(train)` takes its statistics from
`train` only and `transform` applies them to both parts; `ds.ml.stream_loader(world_size=2,
rank=...)` gives each rank a disjoint slice of one global order. Where the fit statistics
come from is also drawn, with the wrong arrow, in `fit_transform_leakage.svg`; this figure
is the whole last mile rather than the leak.
"""

from __future__ import annotations

from _authoring import arrow, card, heading, hero, label, note, svg, tint, write

W, H = 900, 610

LEFT, RIGHT, COL_W = 110, 510, 280
CARD_H = 70
ROWS = (40, 150, 260, 370, 480)
LC, RC = LEFT + COL_W / 2, RIGHT + COL_W / 2  # column centres

body = [
    heading(24, ROWS[0] + 40, "SHAPE", kind="grey"),
    card(290, ROWS[0], 320, CARD_H, "featured", "features computed in the engine"),
    heading(24, ROWS[1] + 40, "SPLIT", kind="grey"),
    arrow(400, ROWS[0] + CARD_H, LC + 20, ROWS[1] - 3),
    arrow(500, ROWS[0] + CARD_H, RC - 20, ROWS[1] - 3),
    label(450, ROWS[0] + CARD_H + 24, "hash of key", anchor="middle", size=11.5),
    card(LEFT, ROWS[1], COL_W, CARD_H, "train", "44 rows in the example"),
    card(RIGHT, ROWS[1], COL_W, CARD_H, "test", "20 rows, held out"),
    heading(24, ROWS[2] + 40, "FIT", kind="amber"),
    arrow(LC, ROWS[1] + CARD_H, LC, ROWS[2] - 3, "amber"),
    label(LC + 12, ROWS[1] + CARD_H + 24, "fit", size=11.5),
    tint(LEFT, ROWS[2], COL_W, CARD_H, "scaler.fit(train)", "statistics from train only", "amber"),
    heading(24, ROWS[3] + 40, "APPLY", kind="grey"),
    arrow(LC, ROWS[2] + CARD_H, LC, ROWS[3] - 3, "amber"),
    label(LC + 12, ROWS[2] + CARD_H + 24, "transform", size=11.5),
    arrow(LEFT + COL_W, ROWS[2] + CARD_H / 2, RIGHT + 60, ROWS[3] - 3, "amber"),
    label(470, ROWS[2] + CARD_H / 2 + 8, "same stats", size=11.5),
    arrow(RC, ROWS[1] + CARD_H, RC, ROWS[3] - 3, "grey"),
    label(RC + 12, ROWS[2] + CARD_H / 2 + 4, "transform", size=11.5),
    card(LEFT, ROWS[3], COL_W, CARD_H, "train_x", "scaled with train's stats"),
    card(RIGHT, ROWS[3], COL_W, CARD_H, "test_x", "scaled with train's stats too"),
    heading(24, ROWS[4] + 40, "LOAD", kind="blue"),
    arrow(LC, ROWS[3] + CARD_H, LC, ROWS[4] - 3),
    label(LC + 12, ROWS[3] + CARD_H + 24, "per rank", size=11.5),
    hero(LEFT, ROWS[4], COL_W, CARD_H + 6, "stream_loader", "rank 0, rank 1: disjoint slices"),
    note(RC, ROWS[4] + 30, "Split, then fit. A fit on test rows", anchor="middle"),
    note(RC, ROWS[4] + 48, "raises no error and still leaks.", anchor="middle"),
]

write("training_data_flow", svg(W, H, "".join(body)))
print("wrote training_data_flow.svg")
