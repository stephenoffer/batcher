#!/usr/bin/env python3
"""Draw `state_store.svg` -- what a streaming query keeps, writes down, and gets back.

Source of truth: `python/batcher/core/streaming/folds/running.py::_AggFold` (the running
state is one Arrow ``RecordBatch``, the output of the native ``combine``, bounded by the
group count rather than the input; ``take_delta`` hands back only the partial the last
push absorbed), `python/batcher/io/formats/streaming/checkpoint/store.py` (the three logs
under one directory and the order they are written), `checkpoint/state_store.py` (a
``batch-<id>.arrow`` snapshot against a ``batch-<id>.delta.arrow`` changelog entry, and
how an eviction rides a delta as a prefix bound), and `checkpoint/recovery.py`
(``recover``: resume at the first uncommitted batch, seek to the last committed batch's
positions, restore the newest snapshot plus every delta after it).

The figure exists because this is a *mapping*, not a sequence: three artifacts on disk
each answer one different question at restart, and the answer to one of them is a chain
rather than a file. A list restates the three; only the picture shows that the running
state in memory and the bytes on disk are not the same size.

Deliberately not drawn: a guarantee. The engine replays the uncommitted batch, and
whether that is exactly-once depends on the sink, not on this directory.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 632

# In-memory row.
MEM_Y, MEM_H = 100, 82
SRC_X, SRC_W = 110, 250
RUN_X, RUN_W = 560, 300

# Checkpoint row.
CK_Y, CK_H = 262, 96
CK = ((60, 260), (360, 260), (660, 250))

# Restart row.
RS_X, RS_W, RS_Y, RS_H = 250, 460, 462, 76

body = [
    note(
        490,
        52,
        "A streaming aggregate keeps one row per group. The checkpoint is what lets a restart pick that up rather than re-read the stream.",
        anchor="middle",
    ),
    band(20, 66, 940, 138, "BETWEEN MICRO-BATCHES, IN MEMORY", "blue"),
    card(SRC_X, MEM_Y, SRC_W, MEM_H, "micro-batch 8", "the rows this trigger read"),
    card(RUN_X, MEM_Y, RUN_W, MEM_H, "running state", "one partial row per group, in Arrow"),
    arrow(SRC_X + SRC_W, MEM_Y + MEM_H / 2, RUN_X - 6, MEM_Y + MEM_H / 2, "blue"),
    label(465, MEM_Y - 12, "partial, then combine, in Rust", anchor="middle"),
    note(
        465,
        MEM_Y + MEM_H + 18,
        "bounded by the group count, not by how many rows produced it",
        anchor="middle",
    ),
    # What the epoch writes down, and in what order.
    band(20, 222, 940, 160, "IN THE CHECKPOINT DIRECTORY", "grey"),
    card(CK[0][0], CK_Y, CK[0][1], CK_H, "offsets/", "each source's position"),
    card(CK[1][0], CK_Y, CK[1][1], CK_H, "state/", "batch-4.arrow plus a delta each"),
    card(CK[2][0], CK_Y, CK[2][1], CK_H, "commits/", "batch id and sink token"),
    note(CK[0][0] + CK[0][1] / 2, CK_Y + 88, "written before the batch runs", anchor="middle"),
    note(CK[1][0] + CK[1][1] / 2, CK_Y + 88, "batch 8's partial only", anchor="middle"),
    note(CK[2][0] + CK[2][1] / 2, CK_Y + 88, "written last of all", anchor="middle"),
    arrow(RUN_X + RUN_W / 2, MEM_Y + MEM_H + 26, CK[1][0] + CK[1][1] / 2 + 40, CK_Y - 6, "blue"),
    label(724, 238, "snapshot", anchor="start"),
    # What comes back.
    band(20, 402, 940, 206, "WHAT A RESTART REBUILDS", "amber"),
    card(
        RS_X,
        RS_Y,
        RS_W,
        RS_H,
        "batch 9 runs again, with batch 8's state",
        "a batch in offsets but not in commits is the one to replay",
    ),
    arrow(170, CK_Y + CK_H + 4, RS_X + 90, RS_Y - 6, "amber"),
    label(60, 444, "seek each source", anchor="start"),
    arrow(CK[1][0] + CK[1][1] / 2, CK_Y + CK_H + 4, RS_X + RS_W / 2, RS_Y - 6, "amber"),
    label(500, 432, "combine the snapshot with every delta after it", anchor="start"),
    arrow(790, CK_Y + CK_H + 4, RS_X + RS_W - 90, RS_Y - 6, "amber"),
    label(812, 452, "resume at 9", anchor="start"),
    note(
        490,
        568,
        "Combining a base snapshot with the deltas recorded after it reaches the same state a full snapshot would, because combine is associative and commutative.",
        anchor="middle",
    ),
    note(
        490,
        590,
        "A windowed aggregate evicts too, and its eviction is always a prefix, so one bound in each entry describes it.",
        anchor="middle",
    ),
]

write("state_store", svg(W, H, "".join(body)))
print("wrote state_store.svg")
