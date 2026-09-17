#!/usr/bin/env python3
"""Draw `inference_actor_pool.svg` - the two nested pools behind batch inference.

Source of truth: `python/batcher/dist/executors/map.py` (`_MapActor`,
`_drive_actor_pool`, `gpu_aware_pool_default` in `python/batcher/ml/gpu.py`),
`python/batcher/ml/inference/pool.py` (`InferencePool`),
`python/batcher/ml/autobatch.py` (`ThroughputController`), and
`python/batcher/core/udf/apply.py` (`apply_udf`, the batch-first boundary).

The picture exists because the two pools are easy to conflate and are not the same
thing. The outer one is Ray actors and exists only on the distributed path; each actor
builds the model once in `__init__`. The inner one is a `ThreadPoolExecutor` whose slots
share one model object and one CUDA context, so it buys overlap with host work rather
than replicas. `ds.ml.infer` is `map_batches` with inference defaults, not a separate
operator. Keep this in step with those modules.
"""

from __future__ import annotations

from _authoring import arrow, band, card, curve, label, note, svg, write

W, H = 980, 750

body = [
    band(20, 20, 940, 96, "ONE PLAN NODE, NOTHING RUNS YET", "grey"),
    card(
        190,
        44,
        600,
        56,
        "ds.map_batches(Scorer, num_gpus=1, concurrency=(2, 8))",
        "ds.ml.infer is this call with inference defaults, not a separate operator",
    ),
    arrow(490, 102, 490, 190, "blue"),
    label(504, 146, "each partition goes to the emptiest actor", size=12),
    # Outer pool: real Ray actors, distributed only.
    band(20, 152, 940, 158, "ONE RAY ACTOR PER GPU  -  THE DISTRIBUTED PATH ONLY", "blue"),
    card(80, 200, 280, 78, "actor 1", "Scorer() built once, in __init__"),
    card(620, 200, 280, 78, "actor N", "Scorer() built once, in __init__"),
    note(490, 234, "num_gpus is a Ray reservation.", anchor="middle"),
    note(490, 252, "concurrency=(min, max) grows the pool", anchor="middle"),
    note(490, 270, "while work waits, reaps an idle actor.", anchor="middle"),
    # Inner pool: threads, one model.
    band(20, 342, 940, 278, "INSIDE ONE ACTOR", "amber"),
    label(337, 382, "in input order", anchor="middle", size=12),
    label(647, 382, "rows/s and VRAM", anchor="middle", size=12),
    card(60, 394, 250, 88, "InferencePool", "threads, one shared model"),
    card(370, 394, 250, 88, "your __call__(batch)", "a whole Arrow RecordBatch"),
    card(680, 394, 250, 88, "autobatch", "hill-climb under a VRAM cap"),
    arrow(310, 438, 364, 438, "amber"),
    arrow(620, 438, 674, 438, "amber"),
    curve(805, 482, 490, 544, 185, 482, "amber"),
    label(490, 562, "the next batch size", anchor="middle", size=12),
    note(
        490,
        586,
        "The threads share one model and one CUDA context, so they buy overlap with host work, not replicas.",
        anchor="middle",
    ),
    note(
        490,
        606,
        "An OOM bisects the batch and records a ceiling for the run; the size that worked is written back for the next one.",
        anchor="middle",
    ),
    # The boundary the whole surface is built on.
    band(20, 652, 940, 86, "THE BATCH-FIRST BOUNDARY", "grey"),
    note(
        490,
        692,
        "Your callable is handed a whole pyarrow.RecordBatch. batch_format reframes only the call; the data plane stays Arrow.",
        anchor="middle",
    ),
    note(
        490,
        712,
        "ds.map is the row-at-a-time escape hatch, and it is marked as one so a profile can price what it costs.",
        anchor="middle",
    ),
]

write("inference_actor_pool", svg(W, H, "".join(body)))
print("wrote inference_actor_pool.svg")
