#!/usr/bin/env python3
"""Draw `tabular_predict_flow.svg`: what `ds.ml.predict` does once, and what it does per batch.

Source of truth: `docs/ml/inference/tabular-models.md` ("How it works", "Scale it out")
and `python/batcher/ml/tabular/predictor.py`. `predicted_column_names` runs on the
driver when the query is built: it checks `features=` against the model's recorded
feature names, raising on a mismatch, and derives how many output columns the model
produces. The predictor class loads the model once per worker in `__init__` and caps its
thread pool (`resolve_threads`); `__call__` turns one Arrow batch into one dense
matrix in `features=` order, with a null becoming `missing` (NaN by default)
(`python/batcher/ml/tabular/features.py`), makes one model call, and appends the
prediction columns.

Layout: the build-time checks in a band on top, then the worker: a load that happens
once, feeding a per-batch loop drawn with a return curve.
"""

from __future__ import annotations

from _authoring import arrow, band, card, curve, hero, label, note, svg, tint, write

W, H = 980, 500

body = [
    # ---- Driver, at plan time ---------------------------------------------------
    band(16, 20, 948, 136, "WHEN THE QUERY IS BUILT  ·  ON THE DRIVER", "grey"),
    card(36, 64, 250, 70, "ds.ml.predict(model)", "a fitted object, or a path"),
    card(360, 64, 250, 70, "Check feature names", "if recorded, raises on mismatch"),
    card(684, 64, 260, 70, "Resolve output columns", "from class or tree count"),
    arrow(292, 99, 354, 99, "blue"),
    label(323, 88, "features", anchor="middle", size=11.5),
    arrow(616, 99, 678, 99, "blue"),
    label(647, 88, "then", anchor="middle", size=11.5),
    # ---- Worker ------------------------------------------------------------------
    band(16, 176, 948, 304, "ON EACH WORKER", "blue"),
    hero(36, 240, 196, 112, "Load once", "__init__, threads capped"),
    arrow(238, 296, 294, 296, "blue"),
    label(266, 284, "ready", anchor="middle", size=11.5),
    tint(300, 254, 196, 84, "Dense matrix", "features= order, null=NaN"),
    tint(548, 254, 170, 84, "One model call", "method=", kind="amber"),
    tint(770, 254, 174, 84, "Append columns", "prediction, ..."),
    arrow(502, 296, 542, 296, "blue"),
    label(522, 284, "array", anchor="middle", size=11.5),
    arrow(724, 296, 764, 296, "blue"),
    label(744, 284, "output", anchor="middle", size=11.5),
    curve(857, 346, 630, 436, 398, 346, "blue"),
    label(628, 418, "the next Arrow batch", anchor="middle"),
    note(134, 376, "a path is fetched", anchor="middle"),
    note(134, 394, "once per worker", anchor="middle"),
    note(
        490,
        458,
        "Nothing crosses the boundary a row at a time, and nothing is materialized on the driver.",
        anchor="middle",
    ),
]

write("tabular_predict_flow", svg(W, H, "".join(body)))
print("wrote tabular_predict_flow.svg")
