#!/usr/bin/env python3
"""Draw `spill_ladder.svg` — the two out-of-core mechanisms, and which operator
takes which.

Source of truth: `crates/bc-runtime/src/agg/spill/mod.rs` (the grace aggregate),
`crates/bc-interp/src/spill_split.rs` (`MAX_GRACE_FANOUT`, `MAX_GRACE_SPLIT_DEPTH`,
the salted re-split), `crates/bc-interp/src/join_par/mod.rs` (the grace join and its
streamed probe), `crates/bc-interp/src/window_spill.rs`, `distinct_on_spill.rs`,
`ops/quantile_spill/mod.rs`, and `crates/bc-interp/src/ops/external_sort.rs`
(`run_target_bytes`, the bounded-fan-in merge; the fan-in default is
`execution.sort_merge_fanin = 16` in `python/batcher/config/config.py`).

Drawn as two mechanisms rather than as three spilling operators, because that is the
shape of the code: aggregate, join, window and `DISTINCT ON` all reuse one grace
algebra and one `DiskSpillStore`, while sort — and the quantile aggregates built on
it — partition by arrival and size rather than by a key, which is why key skew is a
hazard for one family and not the other.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 664

LEGEND = (
    "aggregate - partial state: one row per group per morsel reaches the disk, not the input rows",
    "join - both sides co-partitioned by the join key; only the build bucket is resident, the probe streams through it",
    "window - whole rows, keyed by PARTITION BY, so every bucket holds complete partitions",
    "DISTINCT ON - reduced to one row per key per morsel before it is written",
    "sort - sorted runs; median and n_unique stream one pass over the sorted run instead of holding a per-group list",
)

body = [
    band(24, 20, 926, 74, "WHEN THE IN-MEMORY KERNEL WOULD NOT FIT ITS BUDGET", "grey"),
    note(
        487,
        70,
        "Which mechanism an operator takes follows from what its state is keyed on.",
        anchor="middle",
    ),
    arrow(400, 94, 300, 140),
    label(296, 128, "keyed state", anchor="end"),
    arrow(574, 94, 674, 140),
    label(678, 128, "ordered state"),
    # --- left: grace partitioning ------------------------------------------------
    band(24, 146, 450, 330, "GRACE PARTITIONING", "blue"),
    card(50, 186, 398, 64, "route by hash(key) into P buckets", "one Arrow IPC file per bucket"),
    arrow(249, 250, 249, 282),
    label(261, 272, "one bucket at a time"),
    card(50, 286, 398, 64, "read one bucket, run the kernel", "the union is the whole answer"),
    arrow(249, 350, 249, 382, "amber"),
    label(261, 372, "bucket still over budget"),
    card(50, 386, 398, 64, "re-split under a salted hash", "fan-out 256 max, depth 3 max"),
    note(249, 468, "Each sub-bucket is an independent instance of the same operator.", anchor="middle"),
    # --- right: external merge sort ----------------------------------------------
    band(500, 146, 450, 330, "EXTERNAL MERGE SORT", "grey"),
    card(526, 186, 398, 64, "sort into sized runs, spill each", "cut by size, never by key"),
    arrow(725, 250, 725, 282),
    label(737, 272, "pass 0 complete"),
    card(526, 286, 398, 64, "merge up to 16 runs at a time", "one batch per run resident"),
    arrow(725, 350, 725, 382, "amber"),
    label(737, 372, "more than one run left"),
    card(526, 386, 398, 64, "another pass over everything", "log16(runs) passes in all"),
    note(725, 468, "Runs are cut by size (64 MiB by default), not by key, so skew cannot defeat it.", anchor="middle"),
    # --- legend -------------------------------------------------------------------
    band(24, 492, 926, 140, "WHAT EACH OPERATOR ACTUALLY WRITES", "grey"),
]

for i, line in enumerate(LEGEND):
    body.append(note(48, 532 + i * 22, line))

body.append(
    note(
        487,
        656,
        "Both mechanisms return exactly what the in-memory kernel returns. Only peak memory differs.",
        anchor="middle",
    )
)

write("spill_ladder", svg(W, H, "".join(body)))
print("wrote spill_ladder.svg")
