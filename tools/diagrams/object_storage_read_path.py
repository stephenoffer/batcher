#!/usr/bin/env python3
"""Draw `object_storage_read_path.svg`: a distributed scan against object storage.

Source of truth:

* `docs/user-guide/operate/tuning/object-storage.md`, "Reading from object storage in
  parallel": a scan is bound by request latency, so each scan task keeps a bounded window of
  reads in flight, at most `BATCHER_SCAN_PREFETCH` (32 by default), and yields results in file
  order; the driver caches footer metadata and file schemas against each file's identity
  (path, size, modification time); each worker keeps the batches it decoded, bounded by
  `BATCHER_SCAN_CACHE_FRACTION` (0.3) of worker memory, so a repeated query skips both the
  fetch and the decode.
* `python/batcher/dist/executors/scan_read.py`: `_SCAN_PREFETCH = 32` from the environment,
  the worker scan cache of decoded Arrow batches (`_default_scan_cache_cap`), and the module
  docstring's statement that every reader streams and preserves file order.

Form: the driver's plan-time work on top, one worker's read path below, with the warm path
(a cache hit) drawn in amber so it visibly skips the object store.
"""

from __future__ import annotations

from _authoring import (
    arrow,
    band,
    card,
    curve,
    label,
    note,
    svg,
    tint,
    write,
)

W, H = 980, 490

body: list[str] = [
    # ---- Driver ----------------------------------------------------------------------------
    band(20, 20, 940, 128, "DRIVER: PLAN THE SCAN", "grey"),
    card(44, 58, 300, 64, "read footers and schema", "rows, bytes, column bounds"),
    arrow(350, 90, 420, 90, "grey"),
    label(385, 80, "cache", anchor="middle", size=11),
    tint(426, 58, 300, 64, "cached per file identity", "path, size, modification time"),
    note(750, 84, "A second query over the"),
    note(750, 102, "same files reuses it."),
    # ---- One worker --------------------------------------------------------------------------
    band(20, 176, 940, 294, "ONE SCAN TASK ON A WORKER", "blue"),
    arrow(310, 124, 310, 202),
    label(322, 166, "splits", size=11.5),
    card(44, 208, 300, 64, "scan task", "its splits, many files"),
    arrow(350, 240, 420, 240),
    label(385, 230, "lookup", anchor="middle", size=11),
    tint(426, 208, 250, 64, "worker scan cache", "decoded batches, 0.3 of memory", "amber"),
    arrow(682, 240, 740, 240, "amber"),
    label(711, 230, "hit", anchor="middle", size=11),
    card(746, 208, 190, 64, "downstream", "operators"),
    # miss: the read window against the object store
    arrow(551, 278, 551, 336),
    label(563, 314, "miss", size=11.5),
    card(426, 342, 250, 72, "up to 32 reads in flight", "BATCHER_SCAN_PREFETCH"),
    arrow(420, 368, 350, 368),
    label(385, 358, "GET", anchor="middle", size=11),
    arrow(350, 390, 420, 390, "grey"),
    label(385, 408, "bytes", anchor="middle", size=11),
    card(44, 342, 300, 72, "object storage", "S3, GCS, Azure"),
    arrow(682, 378, 740, 378),
    label(711, 368, "decode", anchor="middle", size=11),
    card(746, 342, 190, 72, "yield in file order", "reads overlap unseen"),
    arrow(841, 336, 841, 278),
    label(853, 314, "batches", size=11.5),
    curve(772, 338, 740, 292, 676, 278, "amber"),
    label(716, 318, "kept", size=11.5),
    note(
        490,
        448,
        "Latency, not bandwidth, caps the scan. The window keeps a task to a few files in memory.",
        anchor="middle",
    ),
]

write("object_storage_read_path", svg(W, H, "".join(body)))
print("wrote object_storage_read_path.svg")
