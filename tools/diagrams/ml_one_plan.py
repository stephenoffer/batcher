#!/usr/bin/env python3
"""Draw `ml_one_plan.svg`: raw files to predictions as one lazy plan.

Source of truth: `docs/ml/index.md`. The page states each fact drawn here: media decode
is an expression on the `.image`/`.audio`/`.video` namespaces implemented in Rust and
parallel across cores; `ds.ml.infer` takes a class whose constructor loads the weights
once per worker and whose `__call__` gets whole Arrow batches; the CPU stage and the GPU
stage overlap under `distributed.stream_inference=True`; and read, filter, join, score,
write all sit in one lazy plan, so scoring more data than fits in memory is the ordinary
case. `python/batcher/ml/pipeline.py` is the stage runner behind the overlap.

Layout: five stages on one axis with the model as the hero, a placement row beneath
naming where each stage runs, and a strip of the three consequences the page draws.
"""

from __future__ import annotations

from _authoring import arrow, band, hero, label, note, pill, svg, tint, write

W, H = 1000, 452

ROW_Y = 92
ROW_H = 96
MID = ROW_Y + ROW_H / 2
TINT_W = 128
HERO_W = 184
GAP = 61
X0 = 32

# Left edge of each stage, in order: read, decode, filter/join, infer (hero), write.
xs = []
x = X0
for w in (TINT_W, TINT_W, TINT_W, HERO_W, TINT_W):
    xs.append(x)
    x += w + GAP

body = [
    band(16, 20, 968, 262, "ONE LAZY PLAN  ·  NOTHING RUNS UNTIL A WRITE OR COLLECT", "blue"),
    tint(xs[0], ROW_Y, TINT_W, ROW_H, "Read", "Parquet, images"),
    tint(xs[1], ROW_Y, TINT_W, ROW_H, "Decode", ".image, in Rust"),
    tint(xs[2], ROW_Y, TINT_W, ROW_H, "Filter, join", "attach labels"),
    hero(xs[3], ROW_Y - 8, HERO_W, ROW_H + 16, "Infer", "ds.ml.infer(Model)"),
    tint(xs[4], ROW_Y, TINT_W, ROW_H, "Write", "or aggregate"),
]

# Arrows between stages, each named by what crosses it.
crossing = ("bytes", "tensors", "rows", "+ score")
for i, text in enumerate(crossing):
    left_w = HERO_W if i == 3 else TINT_W
    x1 = xs[i] + left_w + 6
    x2 = xs[i + 1] - 6
    kind = "amber" if i in (2, 3) else "blue"
    body += [
        arrow(x1, MID, x2, MID, kind),
        label((x1 + x2) / 2, MID - 12, text, anchor="middle", size=11.5),
    ]

# Where each stage runs. The bracket spans the stages that share CPU cores.
cpu_left, cpu_right = xs[1], xs[2] + TINT_W
body += [
    f'<path d="M {cpu_left} 216 L {cpu_left} 224 L {cpu_right} 224 L {cpu_right} 216" '
    'fill="none" stroke="#94a3b8" stroke-width="1.6"/>',
    pill((cpu_left + cpu_right) / 2, 248, "CPU CORES", "grey", anchor="middle"),
    f'<path d="M {xs[3]} 216 L {xs[3]} 224 L {xs[3] + HERO_W} 224 L {xs[3] + HERO_W} 216" '
    'fill="none" stroke="#d97706" stroke-width="1.6"/>',
    pill(xs[3] + HERO_W / 2, 248, "GPU ACTOR POOL", "amber", anchor="middle"),
]

# The three consequences, one column each.
cols = (180, 500, 820)
lines = (
    ("The model loads once per", "worker, in __init__."),
    ("The model scores partition k", "while CPU stages prepare k+1."),
    ("Batches stream, so more data than", "fits in memory is the ordinary case."),
)
heads = ("LOAD ONCE", "OVERLAP", "STREAM")
body.append(band(16, 300, 968, 132, "WHAT ONE PLAN BUYS", "grey"))
for cx, (a, b), head in zip(cols, lines, heads, strict=True):
    body += [
        pill(cx, 350, head, "blue", anchor="middle"),
        note(cx, 384, a, anchor="middle"),
        note(cx, 403, b, anchor="middle"),
    ]

write("ml_one_plan", svg(W, H, "".join(body)))
print("wrote ml_one_plan.svg")
