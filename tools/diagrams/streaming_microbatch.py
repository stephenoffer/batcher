#!/usr/bin/env python3
"""Draw `streaming_microbatch.svg` -- the order of one micro-batch, and why it is the order.

Source of truth: `python/batcher/core/streaming_query/engine.py` --
`StreamingQueryEngine._process_next` (stage, write-ahead the source position, publish,
commit), `_loop` (the trigger cadence, and `once`/`available_now`/`continuous` sharing
one drain loop), and `_commit_microbatch` (snapshot state, commit with the sink's token,
prune). The idempotent-commit half is `io/formats/lakehouse/delta/_commit.py`, whose
`already_committed` checks the log for this query's `(app_id, batch_id)` before writing.

The figure exists for one thing prose cannot hold: the *ordering* is the guarantee. Read
as a cycle, it shows that the only interval a crash can land in is between staging and
publishing, and that the write-ahead is what makes that interval replayable rather than
skippable. A numbered list states the four steps but not the window between them.

Deliberately not drawn: anything resembling a Flink delivery guarantee. What is here is
what the loop does -- replay into an idempotent sink -- and nothing stronger.
"""

from __future__ import annotations

from _authoring import arrow, band, card, curve, label, note, svg, write

W, H = 980, 508

ROW1, ROW2, CH = 104, 268, 88
COL = (90, 390, 690)
CW = 200
MID1, MID2 = ROW1 + CH / 2, ROW2 + CH / 2

body = [
    band(20, 20, 940, 372, "ONE MICRO-BATCH · THE ORDER IS THE GUARANTEE", "blue"),
    # Forward path, left to right along the top row.
    card(COL[0], ROW1, CW, CH, "Trigger fires", "processing_time · available_now"),
    card(COL[1], ROW1, CW, CH, "Stage the epoch", "read, compute, publish nothing"),
    card(COL[2], ROW1, CW, CH, "Write ahead", "record the source position"),
    arrow(COL[0] + CW, MID1, COL[1] - 6, MID1, "blue"),
    label(340, 92, "one poll of the source", anchor="middle"),
    arrow(COL[1] + CW, MID1, COL[2] - 6, MID1, "blue"),
    label(640, 92, "nothing is published yet", anchor="middle"),
    # Down the right-hand side into the second row.
    arrow(COL[2] + CW / 2, ROW1 + CH, COL[2] + CW / 2, ROW2 - 6, "blue"),
    label(776, 232, "the position is durable first", anchor="end"),
    # Return path, right to left along the bottom row.
    card(COL[2], ROW2, CW, CH, "Publish", "hand the rows to the sink"),
    card(COL[1], ROW2, CW, CH, "Commit", "snapshot state, then commit"),
    arrow(COL[2] - 6, MID2, COL[1] + CW + 6, MID2, "blue"),
    label(640, 254, "rows land in the sink", anchor="middle"),
    # And back to the trigger. Dashed, because this edge is a wait, not work.
    curve(COL[1] - 6, MID2, COL[0] + CW / 2, MID2, COL[0] + CW / 2, ROW1 + CH + 6, "amber"),
    label(370, 340, "sleep the rest of the interval", anchor="end"),
    note(
        190,
        364,
        "A draining trigger (once · available_now) skips the wait and stops when the source is spent.",
        anchor="start",
    ),
    # What the ordering buys, stated as the window it leaves open.
    band(20, 412, 940, 76, "IF THE PROCESS DIES", "grey"),
    note(
        490,
        456,
        "The only epoch a crash can lose is one that was staged and not published.",
        anchor="middle",
    ),
    note(
        490,
        474,
        "The next run replays it, and a sink that records its own query name and batch id commits nothing the second time.",
        anchor="middle",
    ),
]

write("streaming_microbatch", svg(W, H, "".join(body)))
print("wrote streaming_microbatch.svg")
