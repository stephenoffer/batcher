#!/usr/bin/env python3
"""Draw `udf_choice.svg` -- whether to write a UDF at all, which form, and what it receives.

Source of truth: `docs/user-guide/transform/columns/udfs.md`, where it is embedded: an
expression first; `map_groups` for per-group work (never `map_batches` straight after
`group_by`); `map`/`flat_map`/`ds.ml.filter` when a row at a time is unavoidable; a class
passed to `map_batches` is built once per worker, a plain function is not. The batch formats
shown are the ones that page teaches; the full set is `FORMATS` in
`python/batcher/interop/formats.py`, which converts around the call while the engine
boundary stays Arrow.

Layout: the questions as a ladder in the order the page asks them, each exit to the right,
and the batch format as a strip underneath because it applies to every batch form.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, pill, step, svg, tint, write

W, H = 980, 682
CX = 245
CW = 340
X0 = CX - CW / 2
RX = 596  # left edge of the answers column
RW = 344
YS = (58, 156, 254, 352)  # question card tops
QH = 56

QUESTIONS = [
    ("Can an expression say it?", "bt.col, .str, .dt, .list ..."),
    ("Is the work per group?", "a session, a series, a document"),
    ("Must it see one row at a time?", "a Python dict per row"),
    ("Is there costly setup?", "a model, a client, a connection"),
]
ANSWERS = [
    ("Write the expression", "vectorized in Rust, no UDF at all", "amber"),
    ("group_by(k).map_groups(fn)", "every row of one group per call", "blue"),
    ("map, flat_map, ml.filter", "a Python object per row", "blue"),
    ("map_batches(MyClass)", "built once per worker", "blue"),
]

body: list[str] = [band(20, 20, 940, 504, "DO YOU NEED A UDF, AND WHICH ONE?", "grey")]
for i, (y, (q, qs), (a, asub, kind)) in enumerate(zip(YS, QUESTIONS, ANSWERS, strict=True)):
    mid = y + QH / 2
    body += [
        step(52, mid, i + 1),
        card(X0, y, CW, QH, q, qs),
        arrow(X0 + CW + 4, mid, RX - 6, mid, "amber" if kind == "amber" else "blue"),
        label((X0 + CW + RX) / 2, mid - 10, "yes", anchor="middle", size=11.5),
        tint(RX, y, RW, QH, a, asub, kind),
        arrow(CX, y + QH + 4, CX, y + 94),
        label(CX + 12, y + QH + 26, "no", size=11.5),
    ]
body += [
    tint(X0, 450, CW, QH - 6, "map_batches(fn)", "a plain function, per batch"),
    note(RX, 438, "Pass the class, not an instance, when construction"),
    note(RX, 456, "must happen on the worker, such as a CUDA context."),
    # ---- batch formats --------------------------------------------------------------
    band(20, 544, 940, 118, "WHAT A BATCH FUNCTION RECEIVES: batch_format", "blue"),
]
formats = [
    ("pyarrow", "RecordBatch, the default"),
    ("numpy", "a dict of ndarrays"),
    ("pandas", "a DataFrame"),
    ("torch", "a dict of tensors"),
]
for i, (name, what) in enumerate(formats):
    cx = 140 + i * 233
    body += [
        pill(cx, 596, f'"{name}"', "amber" if i == 0 else "blue", anchor="middle"),
        note(cx, 624, what, anchor="middle"),
    ]
body.append(
    note(
        490,
        648,
        "Converted around the call only. The engine boundary stays Arrow.",
        anchor="middle",
    )
)

write("udf_choice", svg(W, H, "".join(body)))
print("wrote udf_choice.svg")
