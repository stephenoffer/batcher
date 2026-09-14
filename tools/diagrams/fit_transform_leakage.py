#!/usr/bin/env python3
"""Draw `fit_transform_leakage.svg` - fit on the training split, transform both.

Source of truth: `python/batcher/ml/preprocessors/base.py` (`Preprocessor.fit`,
`transform`, `fit_transform`, `fit_aggregate`) and `python/batcher/api/dataset/ml.py`
(`train_test_split`). `fit` executes one aggregate and reads a few scalars back to the
driver as the trailing-underscore state (`mean_`, `scale_`, `categories_`); `transform`
bakes those scalars into an `Expr` and stays lazy.

The picture exists for the arrow, not the boxes: the only difference between the two
halves is which rows reach the aggregate. Nothing in the code detects the wrong one --
there is no leakage guard anywhere under `ml/preprocessors/`, so `fit` runs the same
aggregate over whatever it is handed, with no error and no warning.
"""

from __future__ import annotations

from _authoring import arrow, band, card, curve, label, note, svg, write

W, H = 980, 680

body = [
    # The split, shared by both halves.
    band(20, 20, 230, 490, "YOUR SPLIT", "grey"),
    card(44, 90, 182, 110, "train split", "fit may see these"),
    card(44, 330, 182, 110, "test split", "held out"),
    note(135, 468, "ds.ml.train_test_split(0.3, seed=0)", anchor="middle"),
    note(135, 486, "a seeded hash of each row's content", anchor="middle"),
    # The correct half.
    band(290, 20, 670, 210, "FIT ON THE TRAINING SPLIT ONLY", "blue"),
    card(320, 74, 280, 110, "scaler.fit(train)", "mean_, scale_ from train rows"),
    card(690, 74, 250, 110, "transform both", "one scale, learned once"),
    arrow(226, 129, 314, 129, "blue"),
    label(270, 112, "fit", anchor="middle", size=12),
    arrow(600, 129, 684, 129, "blue"),
    label(642, 112, "transform", anchor="middle", size=12),
    note(625, 210, "The held-out statistics never existed.", anchor="middle"),
    # The test split reaches transform without ever reaching fit.
    curve(226, 340, 470, 290, 684, 176, "grey"),
    label(462, 264, "transform only, never fitted", anchor="middle", size=12),
    # The leaking half. Same boxes; two arrows into fit instead of one.
    band(290, 300, 670, 210, "FIT ON EVERYTHING", "amber"),
    card(320, 354, 280, 110, "scaler.fit(ds)", "mean_, scale_ from every row"),
    card(690, 354, 250, 110, "transform both", "the held-out rows set the scale"),
    arrow(226, 175, 314, 352, "amber"),
    label(256, 250, "train rows", size=12),
    arrow(226, 385, 314, 400, "amber"),
    label(268, 372, "test rows too", anchor="middle", size=12),
    arrow(600, 409, 684, 409, "amber"),
    label(642, 392, "transform", anchor="middle", size=12),
    note(625, 490, "Every offline score afterwards is optimistic.", anchor="middle"),
    # What the engine does, and what it will not do for you.
    band(20, 540, 940, 118, "WHAT THE ENGINE DOES, AND DOES NOT DO", "grey"),
    note(260, 584, "fit executes one aggregate and reads", anchor="middle"),
    note(260, 602, "a few scalars back to the driver;", anchor="middle"),
    note(260, 620, "transform bakes them in and stays lazy.", anchor="middle"),
    note(720, 584, "Nothing detects the leak. fit runs the same", anchor="middle"),
    note(720, 602, "aggregate over whatever you hand it:", anchor="middle"),
    note(720, 620, "no error, no warning, no flag.", anchor="middle"),
]

write("fit_transform_leakage", svg(W, H, "".join(body)))
print("wrote fit_transform_leakage.svg")
