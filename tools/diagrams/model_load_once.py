#!/usr/bin/env python3
"""Draw `model_load_once.svg`: a batch function that loads its model versus a class that loads once.

Source of truth: `python/batcher/core/udf/lifecycle.py::build_udf_callable`. A class passed
as `fn` is instantiated once per worker ("locally: once; distributed: once per actor") and
the instance handles each batch; any other callable is used directly on every batch. The
7 s load against about 1 s of generation is the gpt2 figure quoted in
`docs/getting-started/tutorials/ml/batch-inference.md` from the AI and GPU benchmark page.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, pill, svg, tint, write

W, H = 980, 480

COL_W = 400
LEFT_X, RIGHT_X = 50, 530
TOP_Y, TOP_H = 76, 70
ROW_H, ROW_GAP = 62, 26
ROWS_Y = [TOP_Y + TOP_H + 34 + i * (ROW_H + ROW_GAP) for i in range(3)]


def column(x: float, rows: list[tuple[str, str]], tag: str, kind: str) -> list[str]:
    """Three batch cards under a header card, each tagged with what that batch pays."""
    cx = x + COL_W / 2
    out: list[str] = []
    for i, (y, (title, sub)) in enumerate(zip(ROWS_Y, rows, strict=True)):
        out += [
            card(x, y, COL_W, ROW_H, title, sub),
            pill(x + COL_W - 42, y + 22, tag, kind, "middle"),
        ]
        if i:
            out += [
                arrow(cx, y - ROW_GAP, cx, y - 3, "grey"),
                label(cx + 12, y - ROW_GAP / 2 + 4, "next batch", size=11),
            ]
    return out


body = [
    band(20, 20, 460, 424, "A FUNCTION THAT LOADS ITS MODEL", "grey"),
    card(LEFT_X, TOP_Y, COL_W, TOP_H, "def score(batch)", "used as-is on every batch"),
    arrow(LEFT_X + COL_W / 2, TOP_Y + TOP_H, LEFT_X + COL_W / 2, ROWS_Y[0] - 3, "grey"),
    label(LEFT_X + COL_W / 2 + 12, TOP_Y + TOP_H + 22, "each batch", size=11.5),
    *column(
        LEFT_X,
        [
            ("batch 1", "load, then score"),
            ("batch 2", "load again, score"),
            ("batch 3", "load again, score"),
        ],
        "load",
        "amber",
    ),
    band(500, 20, 460, 424, "A CLASS, LOADED ONCE PER WORKER", "blue"),
    tint(
        RIGHT_X, TOP_Y, COL_W, TOP_H, "class Classifier", "__init__ loads the model once", "amber"
    ),
    arrow(RIGHT_X + COL_W / 2, TOP_Y + TOP_H, RIGHT_X + COL_W / 2, ROWS_Y[0] - 3),
    label(RIGHT_X + COL_W / 2 + 12, TOP_Y + TOP_H + 22, "one instance", size=11.5),
    *column(
        RIGHT_X,
        [
            ("batch 1", "__call__ scores"),
            ("batch 2", "__call__ scores"),
            ("batch 3", "__call__ scores"),
        ],
        "reuse",
        "blue",
    ),
    note(
        490,
        468,
        "A gpt2 load takes about 7 s against about 1 s of generation,"
        " so the load is most of the cost.",
        anchor="middle",
    ),
]

write("model_load_once", svg(W, H, "".join(body)))
print("wrote model_load_once.svg")
