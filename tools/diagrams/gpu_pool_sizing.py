#!/usr/bin/env python3
"""Draw `gpu_pool_sizing.svg`: the layers that size a GPU actor pool, from your call down.

Source of truth: `docs/ml/inference/gpu.md` ("Autoscale the pool", "Let the engine pack by
memory", "The engine packs across runs"). A value you set always wins, and Kyber fills
only what you leave unset. With `num_gpus` and `batch_size` unset, `model_memory_gb`
picks the GPU fraction (a light model gets several copies per device, a model larger
than one GPU gets whole GPUs) and seeds `batch_size` from the VRAM left over; the online
throughput controller refines it from measured VRAM and throughput. `concurrency=(min,
max)` adds actors while batches queue and releases them once the stage drains. With
`concurrency` unset, each run records utilization and peak device memory, and the next
run packs toward 90% utilization, bounded by that peak, capped at eight actors per
device, holding any device at or above 80% as fed.

Layout: four stacked tiers in the order they act, each joined to the next by a labeled
arrow down the middle.
"""

from __future__ import annotations

from _authoring import arrow, band, label, pill, svg, tint, write

W, H = 980, 606

LEFT, RIGHT, CW = 48, 502, 430
TIERS = (20, 160, 314, 468)  # top of each band
BAND_H = (100, 118, 118, 118)

body = [
    # ---- 1. The call ------------------------------------------------------------
    band(16, TIERS[0], 948, BAND_H[0], "YOU DECLARE  ·  ANY OF THESE, OR NONE", "grey"),
]
x = 44
for name in ("num_gpus", "concurrency", "batch_size", "model_memory_gb"):
    body.append(pill(x, TIERS[0] + 66, name, "blue"))
    x += 14 + 6.7 * len(name) + 18
body.append(label(940, TIERS[0] + 66, "a value you set always wins", anchor="end", size=12))

# ---- 2. Kyber, before the run ----------------------------------------------------
t = TIERS[1]
body += [
    band(16, t, 948, BAND_H[1], "KYBER  ·  BEFORE THE RUN", "blue"),
    tint(
        LEFT,
        t + 40,
        CW,
        62,
        "GPU fraction from model_memory_gb",
        "light: several per device; over one GPU: whole GPUs",
    ),
    tint(RIGHT, t + 40, CW, 62, "A starting batch_size", "seeded from the VRAM left over"),
]

# ---- 3. During the run -----------------------------------------------------------
t = TIERS[2]
body += [
    band(16, t, 948, BAND_H[2], "DURING THE RUN", "blue"),
    tint(
        LEFT,
        t + 40,
        CW,
        62,
        "concurrency=(min, max)",
        "adds actors while batches queue, drops to min",
    ),
    tint(
        RIGHT,
        t + 40,
        CW,
        62,
        "Throughput controller",
        "refines batch_size from VRAM and throughput",
    ),
]

# ---- 4. The next run -------------------------------------------------------------
t = TIERS[3]
body += [
    band(16, t, 948, BAND_H[3], "THE NEXT RUN  ·  WHEN concurrency IS UNSET", "amber"),
    tint(
        LEFT,
        t + 40,
        CW,
        62,
        "Pack toward 90% utilization",
        "bounded by measured peak memory, 8 per device max",
        kind="amber",
    ),
    tint(
        RIGHT,
        t + 40,
        CW,
        62,
        "Hold a fed device",
        "at 80% or more, density stays put",
        kind="amber",
    ),
]

# Tier-to-tier arrows down the gutter between the two card columns.
links = (
    (TIERS[0] + BAND_H[0], TIERS[1], "fills only what is unset", "blue"),
    (TIERS[1] + BAND_H[1], TIERS[2], "starts the pool", "blue"),
    (TIERS[2] + BAND_H[2], TIERS[3], "records utilization and peak memory", "amber"),
)
for y1, y2, text, kind in links:
    body += [
        arrow(490, y1 + 4, 490, y2 - 4, kind),
        label(504, (y1 + y2) / 2 + 5, text, size=12),
    ]

write("gpu_pool_sizing", svg(W, H, "".join(body)))
print("wrote gpu_pool_sizing.svg")
