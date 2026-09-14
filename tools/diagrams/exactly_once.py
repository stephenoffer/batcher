#!/usr/bin/env python3
"""Draw `exactly_once.svg` -- why a replayed append duplicates and a replayed merge does not.

Source of truth: `examples/streams/exactly_once_semantics.py`, which asserts both halves
of this picture on real data. The same 5,000-row batch written twice with
``mode="append"`` asserts ``count() == 10_000`` over 5,000 distinct keys; written three
times with ``merge_on="o_orderkey"`` it asserts ``count() == 5_000``, 5,000 distinct, and
a sum equal to the batch's own rather than doubled. The streaming form of the same
mechanism is `io/formats/lakehouse/delta/_commit.py`, whose ``already_committed`` checks
the log for this query's ``(app_id, batch_id)`` before writing a micro-batch again.

The figure exists because the point is a *comparison of two outcomes from the same
input*, which is what a picture holds and a paragraph restates. A reader who is told
"make the write idempotent" does not see that the two writes are the same write, run the
same number of times, against the same table.

Deliberately not drawn: anything resembling a Flink delivery guarantee, and any claim
that the source delivers a record once. What is here is what the code does -- a keyed
write absorbs its own replay -- and nothing stronger.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 524

CW, CH = 200, 84
COL = (190, 430, 670)
ROW_A, ROW_B = 118, 258
MID_A, MID_B = ROW_A + CH / 2, ROW_B + CH / 2

body = [
    note(490, 52, "One batch of 5,000 rows, delivered to the same table twice. The write is what decides whether that is a problem.", anchor="middle"),
    band(20, 76, 940, 300, "THE SAME BATCH, DELIVERED TWICE", "blue"),
    # Lane A: an unkeyed append. The replay is a second, indistinguishable insert.
    label(38, ROW_A + 38, "APPEND"),
    note(38, ROW_A + 58, "write.delta(table)"),
    note(38, ROW_A + 74, "mode='append'"),
    card(COL[0], ROW_A, CW, CH, "first delivery", "5,000 rows land"),
    card(COL[1], ROW_A, CW, CH, "same batch again", "5,000 more rows land"),
    card(COL[2], ROW_A, CW, CH, "10,000 rows", "5,000 distinct keys"),
    arrow(COL[0] + CW, MID_A, COL[1] - 6, MID_A, "amber"),
    label(415, ROW_A - 12, "replay after a crash", anchor="middle"),
    arrow(COL[1] + CW, MID_A, COL[2] - 6, MID_A, "amber"),
    label(655, ROW_A - 12, "every key now appears twice", anchor="middle"),
    # Lane B: the same two deliveries, matched on a key. The second writes what is there.
    label(38, ROW_B + 38, "KEYED MERGE"),
    note(38, ROW_B + 58, "write.delta(table,"),
    note(38, ROW_B + 74, "merge_on='o_orderkey')"),
    card(COL[0], ROW_B, CW, CH, "first delivery", "5,000 rows land"),
    card(COL[1], ROW_B, CW, CH, "same batch again", "each key already there"),
    card(COL[2], ROW_B, CW, CH, "5,000 rows", "5,000 distinct keys"),
    arrow(COL[0] + CW, MID_B, COL[1] - 6, MID_B, "blue"),
    label(415, ROW_B - 12, "replay after a crash", anchor="middle"),
    arrow(COL[1] + CW, MID_B, COL[2] - 6, MID_B, "blue"),
    label(655, ROW_B - 12, "each row is rewritten, not added", anchor="middle"),
    # What the comparison is actually about.
    band(20, 396, 940, 112, "WHAT MAKES THE SECOND LANE SAFE", "grey"),
    note(490, 438, "The merge matches on a key, so a redelivered row overwrites the row it already wrote. Running it a third time changes nothing again:", anchor="middle"),
    note(490, 456, "the example asserts 5,000 rows and the batch's own total, not a doubled one. The append has no key to match on and cannot tell the two apart.", anchor="middle"),
    note(490, 480, "A streaming write gets the same property from the Delta txn action: the commit records the query name and batch id, and a replayed id commits nothing.", anchor="middle"),
    note(490, 498, "Resuming also needs the position the write reached, recorded with it. An idempotent write alone does not say where to restart.", anchor="middle"),
]

write("exactly_once", svg(W, H, "".join(body)))
print("wrote exactly_once.svg")
